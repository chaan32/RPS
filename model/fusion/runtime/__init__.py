"""Fusion 실시간 운영 통합.

realtime_camera : 메인 진입점 (server lifespan 에서 subprocess 로 spawn).
                  카메라 + ArUco + Fusion 추론 + BEV 시각화 + 알림 통합.
publisher       : MQTT 발행 (forklift/4/vibration)
db_logger       : POST /incident-logs / /incident-logs/with-snapshot

실행:
    python -m model.fusion.runtime.realtime_camera --no-prompt
"""
