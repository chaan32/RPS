"""
실시간 추론 래퍼.

3가지 사용 패턴 지원:
1. predict_scenario()    : Scenario 전체 timeline에 대해 매 step risk_matrix
2. predict_window()      : 미리 변환된 단일 윈도우 텐서 (학습 코드와 호환)
3. RealtimeInference     : JSON 메시지 stream을 ring buffer로 관리

출력 JSON 형식 (팀원 제안):
{
  "timestamp": "...",
  "predictions": {
    "W01": {
      "vs_Forklift_A_collision_prob": 0.85,
      "vs_DropZone_A_overlap_prob":   0.10
    }
  }
}
"""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from scenario_generator import Scenario, RATE, DZ_CENTER, DZ_RADIUS, N_STEPS
import numpy as _np  # noqa
from graph_input import (
    to_graph_input,
    THREAT_FORKLIFT,
    THREAT_DROPZONE,
    V_NODES,
    F_NODE,
    F_SCENE,
    NODE_WORKER,
    NODE_FORKLIFT,
    NODE_DROPZONE,
)
from model import PairwiseInteractionFusionModel
from dataset import T_WIN

# 알림 정책
DEFAULT_THRESHOLD = 0.8


# ── 모델 로드 ──────────────────────────────────────
def load_model(
    ckpt_path: str,
    device: str = "cpu",
) -> PairwiseInteractionFusionModel:
    """체크포인트에서 모델 로드 (eval 모드)."""
    model = PairwiseInteractionFusionModel()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    model.to(device)
    return model


# ── 페어별 dual 모델 ───────────────────────────────
class DualPairModel(nn.Module):
    """
    forklift / dropzone 각자 best 체크포인트를 들고 있다가, 추론 시
    각 페어의 출력 값만 자기 모델에서 가져와 합친다.

    출력 shape는 단일 모델과 동일: (B, N, K=2)
      - K=THREAT_FORKLIFT 채널 ← model_forklift
      - K=THREAT_DROPZONE 채널 ← model_dropzone
    """
    def __init__(
        self,
        model_forklift: PairwiseInteractionFusionModel,
        model_dropzone: PairwiseInteractionFusionModel,
    ):
        super().__init__()
        self.m_forklift = model_forklift
        self.m_dropzone = model_dropzone

    def forward(self, n, a, s):
        r_f = self.m_forklift(n, a, s)   # (B, N, 2)
        r_d = self.m_dropzone(n, a, s)   # (B, N, 2)
        # forklift 채널은 m_forklift, dropzone 채널은 m_dropzone
        out = r_f.clone()
        out[..., THREAT_DROPZONE] = r_d[..., THREAT_DROPZONE]
        return out


def load_dual_model(
    ckpt_dir: str,
    device: str = "cpu",
    forklift_ckpt: str = "best_forklift.pt",
    dropzone_ckpt: str = "best_dropzone.pt",
) -> DualPairModel:
    """
    페어별 best 체크포인트 두 개를 로드해 DualPairModel 반환.

    누락 시 단일 best.pt로 폴백.
    """
    ckpt_f = os.path.join(ckpt_dir, forklift_ckpt)
    ckpt_d = os.path.join(ckpt_dir, dropzone_ckpt)
    fallback = os.path.join(ckpt_dir, "best.pt")

    if not os.path.exists(ckpt_f) or not os.path.exists(ckpt_d):
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"per-pair ckpts not found and no fallback best.pt at {ckpt_dir}"
            )
        print(f"[DualPairModel] per-pair ckpts 미존재 → best.pt 폴백 사용")
        m = load_model(fallback, device=device)
        return DualPairModel(m, m).eval().to(device)

    m_f = load_model(ckpt_f, device=device)
    m_d = load_model(ckpt_d, device=device)
    dual = DualPairModel(m_f, m_d).eval().to(device)

    # 정보 출력
    sf = torch.load(ckpt_f, map_location="cpu", weights_only=True)
    sd = torch.load(ckpt_d, map_location="cpu", weights_only=True)
    print(f"[DualPairModel] forklift ← {forklift_ckpt} (epoch {sf.get('epoch')})")
    print(f"[DualPairModel] dropzone ← {dropzone_ckpt} (epoch {sd.get('epoch')})")
    return dual


# ── Scenario 단위 추론 (오프라인) ──────────────────
def predict_scenario(
    model: PairwiseInteractionFusionModel,
    scenario: Scenario,
    device: str = "cpu",
    t_win: int = T_WIN,
) -> np.ndarray:
    """
    Scenario 전체에 대해 매 시점 risk_matrix 계산 (윈도우 슬라이딩).

    Returns:
      risk_timeline: (T_total, N=1, K=2) — 첫 (t_win-1) 스텝은 NaN
    """
    nodes, adj, scene = to_graph_input(scenario)
    T_total = nodes.shape[1]

    risks = np.full((T_total, 1, 2), np.nan, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for end in range(t_win, T_total + 1):
            start = end - t_win
            n = torch.from_numpy(nodes[:, start:end, :]).float().unsqueeze(0).to(device)
            a = torch.from_numpy(adj[start:end, :, :]).float().unsqueeze(0).to(device)
            s = torch.from_numpy(scene[start:end, :]).float().unsqueeze(0).to(device)
            risk = model(n, a, s)        # (1, 1, 2)
            risks[end - 1] = risk[0].cpu().numpy()
    return risks


# ── JSON 변환 ───────────────────────────────────────
def risk_matrix_to_json(
    risk_matrix: np.ndarray,
    worker_ids: list[str] = ("W01",),
    forklift_id: str = "Forklift_A",
    dropzone_id: str = "DropZone_A",
    timestamp: Optional[str] = None,
) -> dict:
    """
    risk_matrix (N, K) → JSON dict (팀원 제안 형식).
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="milliseconds") + "Z"
    out = {"timestamp": timestamp, "predictions": {}}
    for i, w_id in enumerate(worker_ids):
        out["predictions"][w_id] = {
            f"vs_{forklift_id}_collision_prob": float(risk_matrix[i, THREAT_FORKLIFT]),
            f"vs_{dropzone_id}_overlap_prob":   float(risk_matrix[i, THREAT_DROPZONE]),
        }
    return out


# ── 알림 트리거 (지게차/드롭존 분기) ───────────────
def build_alerts(
    json_pred: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    JSON 예측 → 진동 알림 메시지 리스트.

    각 알림:
      {
        "worker_id": "W01",
        "topic":     "worker/W01/vibration",
        "scenario":  "forklift" | "dropzone",
        "direction": "left" | "right" | "rear" | "front" | "all",
        "prob":      0.87,
        "intensity": "strong"
      }
    """
    alerts = []
    for w_id, risks in json_pred["predictions"].items():
        for key, prob in risks.items():
            if prob < threshold:
                continue
            scenario = "forklift" if "Forklift" in key else "dropzone"
            # 방향 계산은 inference 단계에선 placeholder
            # (실제로는 worker pose + threat position으로 계산 — direction_router 모듈)
            direction = "all" if scenario == "dropzone" else "front"
            alerts.append({
                "worker_id": w_id,
                "topic": f"worker/{w_id}/vibration",
                "scenario": scenario,
                "direction": direction,
                "prob": float(prob),
                "intensity": "strong",
                "source": key,
            })
    return alerts


# ── 실시간 ring buffer 추론 ─────────────────────────
class RealtimeInference:
    """
    JSON 메시지 stream을 받아 매 step risk_matrix 출력.

    사용 예:
      rt = RealtimeInference(model, device='cpu')
      for frame in stream:
          rt.push(forklift_xy, worker1_xy, audio, crane_state)
          if rt.ready():
              risk = rt.predict()  # (N, K)
              json_pred = risk_matrix_to_json(risk)
              alerts = build_alerts(json_pred)
    """
    def __init__(
        self,
        model: PairwiseInteractionFusionModel,
        device: str = "cpu",
        t_win: int = T_WIN,
        dist_sigma: float = 2.0,
    ):
        self.model = model
        self.device = device
        self.t_win = t_win
        self.dist_sigma = dist_sigma
        self.dt = 1.0 / RATE

        # ring buffer
        self._buf_w = deque(maxlen=t_win)         # worker (T, 2)
        self._buf_f = deque(maxlen=t_win)         # forklift (T, 2) — NaN 허용
        self._buf_audio = deque(maxlen=t_win)     # (T,)
        self._buf_crane = deque(maxlen=t_win)     # (T,) int
        # 동적 dropzone (인양물 위치). None이면 학습 default(DZ_CENTER) 사용.
        self._dropzone_center = np.array(DZ_CENTER, dtype=np.float32)
        self._dropzone_radius = float(DZ_RADIUS)

    def push(
        self,
        forklift_xy: Optional[tuple[float, float]],
        worker1_xy: tuple[float, float],
        audio_score: float,
        crane_active: int,
    ) -> None:
        """프레임 1개 추가 (5Hz 간격 권장)."""
        f = (np.nan, np.nan) if forklift_xy is None else forklift_xy
        self._buf_f.append(np.array(f, dtype=np.float32))
        self._buf_w.append(np.array(worker1_xy, dtype=np.float32))
        self._buf_audio.append(float(audio_score))
        self._buf_crane.append(int(crane_active))

    def update_dropzone(
        self,
        center: Optional[tuple[float, float]] = None,
        radius: Optional[float] = None,
    ) -> None:
        """동적 dropzone 갱신 (인양물 좌표가 들어올 때마다 호출).

        center=None이면 변경 없음 (마지막 알려진 값 유지).
        검출이 일시 끊겨도 직전 위치 유지.
        """
        if center is not None:
            self._dropzone_center = np.array(center, dtype=np.float32)
        if radius is not None:
            self._dropzone_radius = float(radius)

    def ready(self) -> bool:
        return len(self._buf_w) >= self.t_win

    def predict(self) -> np.ndarray:
        """
        Returns risk_matrix: (N=1, K=2)
        """
        assert self.ready(), "buffer not full yet"

        # ring buffer → Scenario-like 텐서 (윈도우 단일)
        f_arr = np.stack(list(self._buf_f), axis=0)        # (T_WIN, 2)
        w_arr = np.stack(list(self._buf_w), axis=0)        # (T_WIN, 2)
        audio_arr = np.array(self._buf_audio, dtype=np.float32)  # (T_WIN,)
        crane_arr = np.array(self._buf_crane, dtype=np.int32)    # (T_WIN,)

        # 미니 Scenario 만들어서 to_graph_input 재사용
        mini = Scenario(
            name="_realtime",
            forklift=f_arr,
            worker1=w_arr,
            crane_state=crane_arr,
            audio=audio_arr,
            labels=None,
        )
        # 주의: to_graph_input은 N_STEPS 가정이 아니라 입력 길이를 그대로 사용함
        nodes, adj, scene = to_graph_input(
            mini,
            dt=self.dt,
            dist_sigma=self.dist_sigma,
            dropzone_center=self._dropzone_center,
            dropzone_radius=self._dropzone_radius,
        )

        n = torch.from_numpy(nodes).float().unsqueeze(0).to(self.device)
        a = torch.from_numpy(adj).float().unsqueeze(0).to(self.device)
        s = torch.from_numpy(scene).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            risk = self.model(n, a, s)        # (1, N, K)
        return risk[0].cpu().numpy()


# ── Sanity check ───────────────────────────────────
def _sanity_check():
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(here, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"checkpoint dir not found: {ckpt_dir}")
        print("→ run train.py first")
        return

    # 페어별 best 체크포인트로 dual 모델 로드 (없으면 best.pt 폴백)
    model = load_dual_model(ckpt_dir, device="cpu")

    # 1) Scenario 전체 타임라인 추론
    from scenarios_synthetic import build_synthetic_24
    scenarios = build_synthetic_24()

    print(f"=== Scenario timeline inference ===\n")
    targets = [
        "s05_fork_slow_approach_static_worker",
        "s11_fork_frontal_full",
        "s15_dz_enter_from_south_lifting",
        "s19_dz_idle_to_lifting_worker_inside",
        "s01_safe_fork_pass_worker_far",
    ]
    for name in targets:
        s = next(s for s in scenarios if s.name == name)
        timeline = predict_scenario(model, s)   # (100, 1, 2)
        # 5초 간격 샘플링
        sample_idx = [19, 25, 50, 75, 99]
        fork_seq = [timeline[i, 0, 0] for i in sample_idx]
        dz_seq = [timeline[i, 0, 1] for i in sample_idx]
        print(f"[{s.name}]")
        print(f"  vs_forklift @t=4,5,10,15,20s: "
              f"{['{:.2f}'.format(v) for v in fork_seq]}")
        print(f"  vs_dropzone @t=4,5,10,15,20s: "
              f"{['{:.2f}'.format(v) for v in dz_seq]}")
        print()

    # 2) Realtime ring buffer 시뮬레이션
    print(f"=== Realtime inference (s11_fork_frontal_full) ===\n")
    s = next(s for s in scenarios if s.name == "s11_fork_frontal_full")
    rt = RealtimeInference(model, device="cpu")

    for t in range(N_STEPS):
        f_xy = (None if np.isnan(s.forklift[t, 0])
                else (float(s.forklift[t, 0]), float(s.forklift[t, 1])))
        rt.push(
            forklift_xy=f_xy,
            worker1_xy=(float(s.worker1[t, 0]), float(s.worker1[t, 1])),
            audio_score=float(s.audio[t]),
            crane_active=int(s.crane_state[t]),
        )
        if rt.ready() and t in [19, 50, 80, 99]:
            risk = rt.predict()
            json_pred = risk_matrix_to_json(risk)
            alerts = build_alerts(json_pred)
            print(f"t={t} ({t * 0.2:.1f}s):")
            print(f"  risk = forklift {risk[0,0]:.3f}, dropzone {risk[0,1]:.3f}")
            if alerts:
                for a in alerts:
                    print(f"  ALERT → {a['topic']}  "
                          f"scenario={a['scenario']} dir={a['direction']} "
                          f"prob={a['prob']:.3f}")
            else:
                print(f"  (no alert)")
            print()


if __name__ == "__main__":
    _sanity_check()
