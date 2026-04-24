"""
Pairwise Interaction Fusion Model 모듈.

구성:
- scenario_generator.py : 24개 합성 시나리오 (SAFE 4 + 지게차 위험 10 + 드롭존 위험 10)
- graph_input.py        : 좌표 → 그래프 노드 텐서 변환
- pair_labels.py        : (T, N, K) pair-level 위험 라벨 생성
- model.py              : PairwiseInteractionFusionModel 클래스
- dataset.py            : PyTorch Dataset + 슬라이딩 윈도우
- train.py              : 학습 루프
- inference.py          : 추론 래퍼 (JSON → risk_matrix)
"""
