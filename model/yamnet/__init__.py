"""YAMNet 기반 음향 이상 감지.

Public API:
    from model.yamnet import YamnetDetector

폴더 구성:
    detector.py            ⭐ 운영 — YamnetDetector 클래스
    anomaly_centroid.npy   ⭐ 학습된 anomaly 임베딩 평균 (1024차원)
    anomaly_config.json    ⭐ threshold / frame size 등 메타
    data_prep/             1회성 데이터 준비 스크립트 (녹음 / 포맷 / 폴더 정리)
    tools/                 단독 실행 검증 도구 (마이크 / WAV 테스트)
    notebooks/             학습 노트북
    dataset/               학습 wav (DVC 관리)
"""

from .detector import YamnetDetector

__all__ = ["YamnetDetector"]
