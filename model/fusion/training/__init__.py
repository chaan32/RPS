"""Fusion 모델 학습 + 시각화.

dataset              : PyTorch Dataset + 슬라이딩 윈도우 (T_WIN, STRIDE)
train                : 메인 학습 루프 (BCE on soft labels, EarlyStopping)
train_with_history   : 학습 + history.json 기록
plot_history         : 학습 곡선 (loss / F1 변화)
plot_summary         : 결과 요약 시각화

학습 실행:
    python -m model.fusion.training.train
    python -m model.fusion.training.train_with_history
"""
