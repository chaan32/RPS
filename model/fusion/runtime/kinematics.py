"""작업자 운동학 + 위협 방향 결정.

- WorkerKinematics: 위치 이력으로 heading(facing 방향) 추정.
- _avg_speed: 위치 deque 의 평균 속도 (m/frame).
- resolve_direction: fusion 결과 + 워커 heading + forklift 정지 여부 등을
  종합해 ESP32 펌웨어 진동 방향("back"/"left"/"right"/"all"/None) 결정.
"""

from __future__ import annotations

import math
import os
from collections import deque as _deque

from ..risk_output import FusionPrediction, ThreatType


class WorkerKinematics:
    """worker별 위치 이력으로 heading(facing 방향) 추정.

    - 최근 N 프레임 위치 평균 속도 → heading 갱신 (속도 < 0.05m/s면 직전 값 유지)
    - default heading = π/2 (북쪽, +Y) — 검출 직후 첫 frame 처리용
    - collision_direction(threat_xy)로 worker body frame 기준 4방향(front/left/right/rear)
      반환
    """
    HISTORY_LEN = 5
    MIN_SPEED = 0.05    # m/frame 미만이면 정지 — heading 갱신 X

    def __init__(self):
        self.history = _deque(maxlen=self.HISTORY_LEN)
        self.heading = math.pi / 2     # 기본값: +Y (북쪽)
        self.last_xy = None

    def update(self, xy: tuple):
        self.history.append(xy)
        self.last_xy = xy
        if len(self.history) >= 3:
            pts = list(self.history)
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            mean_step = math.hypot(dx, dy) / max(1, len(pts) - 1)
            if mean_step > self.MIN_SPEED:
                self.heading = math.atan2(dy, dx)

    def collision_direction(self, threat_xy) -> str:
        """worker body frame 기준 위협 방향. 'front'|'left'|'right'|'rear'."""
        if self.last_xy is None:
            return "front"
        dx = threat_xy[0] - self.last_xy[0]
        dy = threat_xy[1] - self.last_xy[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return "front"
        threat_angle = math.atan2(dy, dx)
        rel = threat_angle - self.heading
        # [-π, π]로 정규화
        while rel > math.pi:
            rel -= 2 * math.pi
        while rel < -math.pi:
            rel += 2 * math.pi
        deg = math.degrees(rel)
        # body frame: 정면=0°, 좌측=+90°, 우측=-90°, 후방=±180°
        if -45 < deg <= 45:
            return "front"
        if 45 < deg <= 135:
            return "left"
        if -135 <= deg <= -45:
            return "right"
        return "rear"


# 워커 body frame 4방향 → ESP32 펌웨어 진동 명령 매핑.
# 펌웨어가 인식하는 4종: "back" / "left" / "right" / "all"
#   - rear  → back  (위협이 뒤에 있음 = 뒤로 물러나라)
#   - left  → left  (좌측 진동)
#   - right → right (우측 진동)
#   - front → None  (작업자가 이미 시야로 인지하고 있다고 보고 알림 X)
# None 매핑 시 resolve_direction 도 None 반환 → maybe_publish 호출이 스킵됨
# (MQTT/DB 모두 발생 X). 정책 바꾸려면 이 dict 한 곳만 손대면 된다.
_BODY_TO_PAYLOAD: dict[str, str | None] = {
    "rear":  "back",
    "left":  "left",
    "right": "right",
    "front": None,
}

# forklift 가 이 속도(m/frame) 미만이면 "정지"로 간주해 forklift trigger 를 무시한다.
# 사람이 정지된 forklift 옆을 지나가는 건 위협이 아님.
# 5Hz 기준 0.10 m/frame ≈ 0.5 m/s (사람 평지 걷기 1.4 m/s 의 약 1/3).
FORKLIFT_STATIC_SPEED = 0.10

# Forklift collision risk is driven by its forward footprint/fork area, not only
# the bbox-derived reference point.  We project a small hazard point ahead of the
# tracked forklift motion and feed that point to fusion/early-warning logic.
FORKLIFT_HAZARD_FORWARD_OFFSET_M = float(
    os.getenv("FORKLIFT_HAZARD_FORWARD_OFFSET_M", "1.0")
)
FORKLIFT_HAZARD_MIN_DISPLACEMENT_M = float(
    os.getenv("FORKLIFT_HAZARD_MIN_DISPLACEMENT_M", "0.25")
)

# 드롭존 중심으로부터 이 반경(m) 안에 워커가 있으면 fusion 출력과 무관하게
# danger 로 격상한다. 학습 모델은 인양 idle 시 dropzone 을 0/0.5 정도로 출력하지만,
# 실제 운영에선 idle 이어도 접근 자체가 위험 인지 대상이라는 정책.
DROPZONE_ALERT_RADIUS = float(os.getenv("DROPZONE_ALERT_RADIUS_M", "0.5"))


def avg_speed(history) -> float:
    """history(deque of (x, y)) 의 평균 한 step 이동거리(m/frame).

    history 가 부족하면 0.0 을 반환 → 정지로 간주(검출 직후 알림 안 울림).
    """
    if len(history) < 2:
        return 0.0
    pts = list(history)
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    return math.hypot(dx, dy) / max(1, len(pts) - 1)


def forklift_hazard_point(
    forklift_xy: tuple[float, float] | None,
    history,
    *,
    forward_offset_m: float = FORKLIFT_HAZARD_FORWARD_OFFSET_M,
    min_displacement_m: float = FORKLIFT_HAZARD_MIN_DISPLACEMENT_M,
) -> tuple[float, float] | None:
    """Return a forward hazard point for forklift-vs-worker risk.

    `forklift_xy` is the YOLO bbox reference point.  The collision-relevant point
    is usually closer to the moving front/fork area, so we estimate heading from
    recent BEV positions and project a configurable offset forward.  If heading
    is not reliable yet, fall back to the original point.
    """
    if forklift_xy is None:
        return None
    if len(history) < 2:
        return forklift_xy

    pts = list(history)
    dx = float(pts[-1][0]) - float(pts[0][0])
    dy = float(pts[-1][1]) - float(pts[0][1])
    displacement = math.hypot(dx, dy)
    if displacement < min_displacement_m:
        return forklift_xy

    ux = dx / displacement
    uy = dy / displacement
    return (
        float(forklift_xy[0]) + ux * forward_offset_m,
        float(forklift_xy[1]) + uy * forward_offset_m,
    )


def resolve_direction(
    pred: FusionPrediction,
    threshold: float,
    kin: WorkerKinematics,
    forklift_xy: tuple | None,
    dropzone_xy: tuple | None,
    forklift_speed: float = 0.0,
    dz_force: bool = False,
) -> str | None:
    """위험 발생 시점의 펌웨어 진동 방향 결정.

    Returns:
      "back" | "left" | "right" | "all" | None
      None 이면 알림 X (정지 forklift 만 trigger / 정면 충돌 / 위협 없음).

    정책:
      1) forklift trigger 라도 forklift 가 정지(< FORKLIFT_STATIC_SPEED)면 무시.
      2) forklift 가 움직이는 경우 워커 facing 기준 4방향 매핑.
         - front 는 None (작업자 시야 → 알림 불필요)
      3) dropzone trigger 또는 dz_force(거리 기반 강제 격상) 가 있으면 "all".
    """
    triggered_types = {p.threat_type for p in pred.triggered(threshold)}

    fork_active = (
        ThreatType.FORKLIFT in triggered_types
        and forklift_xy is not None
        and forklift_speed >= FORKLIFT_STATIC_SPEED
    )
    dz_active = ThreatType.DROPZONE in triggered_types or dz_force

    if fork_active:
        body_dir = kin.collision_direction(forklift_xy)
        payload = _BODY_TO_PAYLOAD.get(body_dir)
        if payload is not None:
            return payload
        # front 는 차단 → dz_active 가 있으면 거기로 fall through, 아니면 None.

    if dz_active:
        return "all"

    return None
