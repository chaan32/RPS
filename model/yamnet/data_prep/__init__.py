"""YAMNet 학습용 데이터 준비 스크립트 (1회성).

운영 코드와 무관하며, 학습 데이터셋 구축 / 정리 시점에만 직접 실행한다:

    record.py              마이크 녹음 → dataset/{category}/{label}/*.wav
    convert_to_wav.py      mp3/m4a/ogg/flac/aac/wma → 16kHz mono WAV
    organize_dataset.py    rope_stress + rope_release + winch → anomaly/normal 이진 분류
    reorganize_dataset.py  기존 anomaly/normal → 카테고리별 폴더 재구성

사용 예:
    python -m model.yamnet.data_prep.record
"""
