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
# 작업자가 작게 보이는 고정 CCTV 샷에서는 기본 0.25가 너무 보수적이다.
# 낮은 confidence 후보를 살린 뒤 world 좌표 범위로 false positive를 제거한다.
POSE_CONF_THRESHOLD = 0.01

# Unity blindspot 벤치마크의 worker 통로 world 좌표 범위.
# 현재 Fab/T-junction 씬은 작업자가 좌우 복도에 동시에 존재할 수 있으므로
# 기본 범위를 넓게 잡고, 필요하면 WORKER_WORLD_BOUNDS env 로 덮어쓴다.
# 예: WORKER_WORLD_BOUNDS=none 또는 WORKER_WORLD_BOUNDS=-10,2,-2,9
WORKER_WORLD_BOUNDS = (-10.5, 2.5, -2.0, 8.5)  # x_min, x_max, y_min, y_max

# YOLO11 pose 의 발목 keypoint 인덱스 (COCO 17 keypoint 규격).
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# keypoint conf 이 임계 미만이면 invalid 로 간주.
KPT_CONF_THRESHOLD = 0.3


# ── Custom 모델 클래스 ────────────────────────────────────────────────
# fusion 의 dropzone 위치 갱신에 쓰이는 인양물 클래스 이름.
BOX_CLASS_NAMES = ("box_1", "box_2")

# DetectionPipeline 이 custom YOLO 결과에서 유지할 클래스.
CUSTOM_OBJECT_CLASS_NAMES = ("forklift", *BOX_CLASS_NAMES)

# Forklift bbox 하단 전체는 포크/그림자/가려짐에 흔들려 ground-plane 좌표가 튄다.
# Unity 도로 시나리오 기준으로 bbox 중앙 x, 높이 75% 지점이 앞바퀴/포크 루트 기준점에 가깝다.
FORKLIFT_REF_X_RATIO = 0.5
FORKLIFT_REF_Y_RATIO = 0.75

# ── Lifted box/dropzone 좌표 보정 ───────────────────────────────────────
# 공중에 떠 있는 인양물은 bbox 하단점을 ground-plane homography에 넣으면
# 바닥으로 투영되어 큰 오차가 난다. 현재 Unity 벤치마크에서는 cam1 bbox 중심을
# Box1 중심 높이 평면에 ray-cast한 좌표가 가장 안정적이다.
LIFTED_BOX_CLASS_NAMES = ("box_1",)
LIFTED_BOX_PRIMARY_CAM_ID = "cam1"
LIFTED_BOX_CENTER_UNITY_Y = 2.2
