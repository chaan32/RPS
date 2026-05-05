"""파이프라인 공용 상수.

값 변경 시 이 파일 한 곳만 수정.
"""

# ── 작업자 식별용 ArUco ────────────────────────────────────────────────
# 작업자별 고유 ArUco 마커 (조끼 등에 부착). 검출되면 해당 worker_id 부여.
# workspace 코너 마커(22/24/27/38)와 ID 충돌 방지: 5, 10, 15 사용.
WORKER_ARUCO_MAP: dict[int, str] = {
    5: "W01",
    10: "W02",
    15: "W03",
}

# ── Worker ID persistence ──────────────────────────────────────────────
# Cross-camera 전파 시 같은 사람으로 인정할 world 좌표 반경 (m).
# Homography 오차 + 좌우 카메라 시점차로 동일인이라도 0.3~0.8m 차이 흔함.
WORLD_MATCH_RADIUS_M = 1.0

# 이 시간(초)을 넘긴 worker_id는 cross-camera 매칭 후보에서 제외.
WORKER_STATE_TTL_S = 5.0


# ── Pose 키포인트 ──────────────────────────────────────────────────────
# YOLO11n-pose 의 발목 keypoint 인덱스 (COCO 17 keypoint 규격).
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# keypoint conf 이 임계 미만이면 invalid 로 간주.
KPT_CONF_THRESHOLD = 0.3


# ── Custom 모델 클래스 ────────────────────────────────────────────────
# fusion 의 dropzone 위치 갱신에 쓰이는 인양물 클래스 이름.
BOX_CLASS_NAMES = ("box_1", "box_2")
