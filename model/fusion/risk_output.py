"""
PairwiseInteractionFusionModel 출력값 → 타입화된 객체.

raw 모델 출력은 (B, N=worker수, K=2) ndarray ∈ [0, 1].
  K[0] = vs forklift collision prob
  K[1] = vs dropzone overlap prob

이 모듈은 그 raw tensor를 코드 어디에서도 명시적으로 다룰 수 있도록
다음 4개의 객체로 변환한다:

  RiskLevel       (Enum)  : 확률 → SAFE / WARNING / DANGER 분류
  ThreatType      (Enum)  : FORKLIFT / DROPZONE (graph_input 인덱스와 매핑)
  PairRisk        (DTO)   : 한 worker × 한 threat 단위 위험
  FusionPrediction(DTO)   : 단일 시점 모델 출력 전체 (worker×threat 묶음)

사용 예:
  risk = model(nodes, adj, scene)            # torch.Tensor (1, 1, 2)
  pred = FusionPrediction.from_model_output(
      risk[0].cpu().numpy(),                  # (N, K) ndarray
      worker_ids=["W01"],
  )
  if pred.has_alert():
      for p in pred.triggered():
          print(p.alert_topic, p.prob, p.level)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .graph_input import THREAT_FORKLIFT, THREAT_DROPZONE, K_THREATS


# ── 임계값 (inference.py의 DEFAULT_THRESHOLD와 일관) ────────
WARN_THRESHOLD = 0.4
DANGER_THRESHOLD = 0.8

# 기본 ID (단일 worker / 단일 threat 인스턴스 시나리오)
DEFAULT_WORKER_ID = "W01"
DEFAULT_FORKLIFT_ID = "Forklift_A"
DEFAULT_DROPZONE_ID = "DropZone_A"


# ── Enums ──────────────────────────────────────────────────
class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"

    @classmethod
    def from_prob(
        cls,
        prob: float,
        # warn 임계값 : 0.4
        warn: float = WARN_THRESHOLD,
        # danger 임계값 : 0.8
        danger: float = DANGER_THRESHOLD,
    ) -> "RiskLevel":
        if prob >= danger:
            return cls.DANGER
        if prob >= warn:
            return cls.WARNING
        return cls.SAFE


class ThreatType(str, Enum):
    FORKLIFT = "forklift"
    DROPZONE = "dropzone"

    @property
    def index(self) -> int:
        # graph_input.THREAT_FORKLIFT / THREAT_DROPZONE과 동일.
        # THREAT_FORKLIFT : 0 DROPZONE : 1로 반환 
        return _THREAT_INDEX[self]

    @classmethod
    def from_index(cls, idx: int) -> "ThreatType":
        return _INDEX_TO_THREAT[idx]

    # method를 사용하는데 괄호를 안 쓰고 호출 할 수 있도록.. 
    @property
    def metric_name(self) -> str:
        # JSON 키에 쓰는 동사 (collision_prob / overlap_prob).
        # 해당 객체가 ThreatType.FORKLIFT 사고면 collision_prob으로, 아니면 overlap_prob으로 
        return "collision_prob" if self is ThreatType.FORKLIFT else "overlap_prob"


_THREAT_INDEX = {
    ThreatType.FORKLIFT: THREAT_FORKLIFT,
    ThreatType.DROPZONE: THREAT_DROPZONE,
}
_INDEX_TO_THREAT = {v: k for k, v in _THREAT_INDEX.items()}


# ── PairRisk : 단일 (worker × threat) ──────────────────────
# 한 개의 쌍 데이터를 담는 immutable 객체 ㅇㅇ W01과 Forklift_A와 0.85 충돌 확률 존재 
# frozen : 객체의 값을 변경할 수 없도록 (JAVA의 private와 같은 비슷한 역할)
# dataclass : 생성자, 출력문, 비교연산자를 한번에 완성해주는 데코레이션
@dataclass(frozen=True)
class PairRisk:
    worker_id: str
    threat_type: ThreatType
    threat_id: str
    prob: float

    @property
    def level(self) -> RiskLevel:
        return RiskLevel.from_prob(self.prob)

    @property
    def json_key(self) -> str:
        # 기존 inference.risk_matrix_to_json 형식과 동일:
        # "vs_Forklift_A_collision_prob", "vs_DropZone_A_overlap_prob"
        return f"vs_{self.threat_id}_{self.threat_type.metric_name}"

    @property
    def alert_topic(self) -> str:
        return f"worker/{self.worker_id}/vibration"

    def is_triggered(self, threshold: float = DANGER_THRESHOLD) -> bool:
        return self.prob >= threshold


# ── FusionPrediction : 단일 시점 모델 출력 전체 ────────────
@dataclass
class FusionPrediction:
    timestamp: datetime
    pairs: list[PairRisk]
    raw_matrix: np.ndarray              # (N, K) — 원본 보존

    # ── 팩토리 ──
    @classmethod
    def from_model_output(
        cls,
        risk_matrix: np.ndarray,        # (N, K) — 모델 출력에서 batch 차원 제거된 것
        worker_ids: Sequence[str] = (DEFAULT_WORKER_ID,),
        forklift_id: str = DEFAULT_FORKLIFT_ID,
        dropzone_id: str = DEFAULT_DROPZONE_ID,
        timestamp: Optional[datetime] = None,
    ) -> "FusionPrediction":
        # 입력 검증 : shape 확인 
        # 위험 요소는 2개임 
        # 인원수는 똑같고 
        if risk_matrix.ndim != 2 or risk_matrix.shape[1] != K_THREATS:
            raise ValueError(
                f"risk_matrix shape must be (N, {K_THREATS}); got {risk_matrix.shape}"
            )
        # 입력 검증 : worker_ids 개수 일치
        # 행의 개수 == 워커 id의 갯수 ㅇㅇ 
        if len(worker_ids) != risk_matrix.shape[0]:
            raise ValueError(
                f"worker_ids length ({len(worker_ids)}) must match "
                f"risk_matrix N ({risk_matrix.shape[0]})"
            )

        # threat_id 딕셔너리 만들기 
        threat_ids = {
            ThreatType.FORKLIFT: forklift_id,
            ThreatType.DROPZONE: dropzone_id,
        }


        pairs: list[PairRisk] = []
        for i, w_id in enumerate(worker_ids):
            for threat in (ThreatType.FORKLIFT, ThreatType.DROPZONE):
                pairs.append(PairRisk(
                    worker_id=w_id,
                    threat_type=threat,
                    threat_id=threat_ids[threat],
                    prob=float(risk_matrix[i, threat.index]),
                ))
        return cls(
            timestamp=timestamp or datetime.now(),
            pairs=pairs,
            raw_matrix=risk_matrix.copy(),
        )

    # ── 조회 ──
    @property
    def worker_ids(self) -> list[str]:
        seen: list[str] = []
        for p in self.pairs:
            if p.worker_id not in seen:
                seen.append(p.worker_id)
        return seen

    def for_worker(self, worker_id: str) -> list[PairRisk]:
        return [p for p in self.pairs if p.worker_id == worker_id]

    def get(self, worker_id: str, threat_type: ThreatType) -> PairRisk:
        for p in self.pairs:
            if p.worker_id == worker_id and p.threat_type is threat_type:
                return p
        raise KeyError(f"({worker_id}, {threat_type}) not in prediction")

    @property
    def max_prob(self) -> float:
        return max((p.prob for p in self.pairs), default=0.0)

    @property
    def max_level(self) -> RiskLevel:
        return RiskLevel.from_prob(self.max_prob)

    def has_alert(self, threshold: float = DANGER_THRESHOLD) -> bool:
        return any(p.is_triggered(threshold) for p in self.pairs)

    def triggered(self, threshold: float = DANGER_THRESHOLD) -> list[PairRisk]:
        return [p for p in self.pairs if p.is_triggered(threshold)]

    # ── 직렬화 ──
    def to_json(self) -> dict:
        """기존 inference.risk_matrix_to_json과 호환되는 dict 반환."""
        predictions: dict[str, dict] = {}
        for p in self.pairs:
            predictions.setdefault(p.worker_id, {})[p.json_key] = p.prob
        return {
            "timestamp": self.timestamp.isoformat(timespec="milliseconds") + "Z",
            "predictions": predictions,
        }


# ── Sanity check ───────────────────────────────────────────
def _sanity_check():
    # 가짜 raw 출력으로 객체 변환 테스트
    fake_matrix = np.array([[0.92, 0.15]], dtype=np.float32)   # (N=1, K=2)
    pred = FusionPrediction.from_model_output(fake_matrix)

    print(f"timestamp : {pred.timestamp}")
    print(f"workers   : {pred.worker_ids}")
    print(f"max prob  : {pred.max_prob:.3f}  → {pred.max_level.value}")
    print(f"has_alert : {pred.has_alert()}\n")

    for p in pred.pairs:
        print(f"  [{p.worker_id} × {p.threat_type.value:8s}] "
              f"prob={p.prob:.3f}  level={p.level.value:7s}  "
              f"key='{p.json_key}'")

    print(f"\ntriggered (≥ {DANGER_THRESHOLD}):")
    for p in pred.triggered():
        print(f"  → {p.alert_topic}  scenario={p.threat_type.value}  prob={p.prob:.3f}")

    print(f"\nto_json():")
    import json
    print(json.dumps(pred.to_json(), indent=2))


if __name__ == "__main__":
    _sanity_check()
