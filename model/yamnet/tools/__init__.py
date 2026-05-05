"""YAMNet 단독 실행 검증 도구.

운영 흐름과 격리된 테스트/디버깅용:

    realtime_detect.py    노트북 마이크로 실시간 anomaly 감지 (운영용은 input/audio/esp32_ws.py)
    test_with_wav.py      wav 파일 / 폴더로 모델 동작 검증

사용 예:
    python -m model.yamnet.tools.realtime_detect
    python -m model.yamnet.tools.test_with_wav <wav파일_또는_폴더>
"""
