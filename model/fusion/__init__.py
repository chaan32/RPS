"""Pairwise Interaction Fusion Model 패키지.

폴더 구성:
    model.py         ⭐ 모델 정의 (PairwiseInteractionFusionModel)
    inference.py     ⭐ 추론 래퍼 (RealtimeInference)
    risk_output.py   ⭐ 출력 타입 (FusionPrediction / ThreatType)
    graph_input.py   ⭐ Scenario → 그래프 텐서 변환

    data/            데이터 정의 (Scenario, 합성 시나리오, 라벨 룰)
    training/        학습 + 시각화 스크립트
    runtime/         실시간 운영 통합 (server 가 subprocess 로 spawn)
    checkpoints/     학습된 가중치

Public API:
    from model.fusion import PairwiseInteractionFusionModel
    from model.fusion import FusionPrediction, ThreatType
"""

from .model import PairwiseInteractionFusionModel
from .risk_output import FusionPrediction, PairRisk, RiskLevel, ThreatType

__all__ = [
    "PairwiseInteractionFusionModel",
    "FusionPrediction",
    "PairRisk",
    "RiskLevel",
    "ThreatType",
]
