"""Fusion 모델의 데이터 정의.

scenario_generator    : Scenario 데이터클래스 + 궤적 primitive (linear / piecewise / still ...)
scenarios_synthetic   : 합성 24 시나리오 (SAFE 4 + 지게차 10 + 드롭존 10)
pair_labels           : Scenario → (T, N, K) pair-level 위험 라벨
fusion_train_24.npz   : 합성 시나리오 캐시
"""
