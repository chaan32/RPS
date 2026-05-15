"""TTC-based early warning layer for worker/forklift interactions.

This module is intentionally separate from the learned fusion model.  It uses
recent world-coordinate history to estimate relative motion and warns before the
model reaches its high-confidence collision state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal


AlertLevel = Literal["safe", "warning", "danger"]

WARNING_TTC_SEC = 7.0
WARNING_CLOSEST_DISTANCE_M = 1.2
DANGER_TTC_SEC = 2.0
DANGER_CLOSEST_DISTANCE_M = 0.8
PREDICTION_HORIZON_SEC = 8.0
MIN_SPEED_MPS = 0.05
MIN_HISTORY_POINTS = 5
MIN_HISTORY_DURATION_SEC = 0.6


@dataclass(frozen=True)
class EarlyWarning:
    level: AlertLevel
    current_distance_m: float | None = None
    ttc_s: float | None = None
    closest_distance_m: float | None = None
    approaching: bool = False
    reason: str = ""

    @property
    def score(self) -> float:
        if self.level == "danger":
            return 1.0
        if self.level == "warning":
            return 0.5
        return 0.0


class MotionHistory:
    """Small timestamped position history for velocity estimation."""

    def __init__(self, maxlen: int = 12):
        self.points: deque[tuple[float, tuple[float, float]]] = deque(maxlen=maxlen)

    def update(self, time_s: float, xy: tuple[float, float] | None) -> None:
        if xy is None:
            return
        self.points.append((float(time_s), (float(xy[0]), float(xy[1]))))

    def velocity(self) -> tuple[float, float] | None:
        if len(self.points) < MIN_HISTORY_POINTS:
            return None
        t0, p0 = self.points[0]
        t1, p1 = self.points[-1]
        dt = t1 - t0
        if dt < MIN_HISTORY_DURATION_SEC:
            return None
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        if math.hypot(vx, vy) < MIN_SPEED_MPS:
            return None
        return (vx, vy)


def _closest_approach(
    worker_xy: tuple[float, float],
    worker_v: tuple[float, float],
    forklift_xy: tuple[float, float],
    forklift_v: tuple[float, float],
    horizon_s: float,
) -> tuple[float, float, bool]:
    """Return (ttc_s, closest_distance_m, approaching)."""
    rx = worker_xy[0] - forklift_xy[0]
    ry = worker_xy[1] - forklift_xy[1]
    rvx = worker_v[0] - forklift_v[0]
    rvy = worker_v[1] - forklift_v[1]
    rel_speed_sq = rvx * rvx + rvy * rvy
    current_dot = rx * rvx + ry * rvy
    approaching = current_dot < 0.0

    if rel_speed_sq <= 1e-9:
        return (0.0, math.hypot(rx, ry), False)

    t_star = -current_dot / rel_speed_sq
    ttc_s = max(0.0, min(float(horizon_s), float(t_star)))
    closest_x = rx + rvx * ttc_s
    closest_y = ry + rvy * ttc_s
    return (ttc_s, math.hypot(closest_x, closest_y), approaching)


def evaluate_worker_forklift(
    worker_xy: tuple[float, float] | None,
    forklift_xy: tuple[float, float] | None,
    worker_history: MotionHistory,
    forklift_history: MotionHistory,
    fusion_risk: float | None = None,
    fusion_threshold: float = 0.8,
    horizon_s: float = PREDICTION_HORIZON_SEC,
    warning_ttc_s: float = WARNING_TTC_SEC,
    warning_distance_m: float = WARNING_CLOSEST_DISTANCE_M,
    danger_ttc_s: float = DANGER_TTC_SEC,
    danger_distance_m: float = DANGER_CLOSEST_DISTANCE_M,
) -> EarlyWarning:
    """Combine learned fusion risk with TTC/closest-approach early warning."""
    if worker_xy is None or forklift_xy is None:
        return EarlyWarning("safe", reason="missing_position")

    current_distance = math.hypot(
        worker_xy[0] - forklift_xy[0],
        worker_xy[1] - forklift_xy[1],
    )

    if fusion_risk is not None and fusion_risk >= fusion_threshold:
        return EarlyWarning(
            "danger",
            current_distance_m=current_distance,
            reason="fusion_threshold",
        )

    worker_v = worker_history.velocity()
    forklift_v = forklift_history.velocity()
    if worker_v is None or forklift_v is None:
        return EarlyWarning(
            "safe",
            current_distance_m=current_distance,
            reason="insufficient_motion",
        )

    ttc_s, closest_distance, approaching = _closest_approach(
        worker_xy,
        worker_v,
        forklift_xy,
        forklift_v,
        horizon_s,
    )
    if not approaching:
        return EarlyWarning(
            "safe",
            current_distance_m=current_distance,
            ttc_s=ttc_s,
            closest_distance_m=closest_distance,
            approaching=False,
            reason="not_approaching",
        )

    if ttc_s <= danger_ttc_s and closest_distance <= danger_distance_m:
        return EarlyWarning(
            "danger",
            current_distance_m=current_distance,
            ttc_s=ttc_s,
            closest_distance_m=closest_distance,
            approaching=True,
            reason="ttc_danger",
        )

    if ttc_s <= warning_ttc_s and closest_distance <= warning_distance_m:
        return EarlyWarning(
            "warning",
            current_distance_m=current_distance,
            ttc_s=ttc_s,
            closest_distance_m=closest_distance,
            approaching=True,
            reason="ttc_warning",
        )

    return EarlyWarning(
        "safe",
        current_distance_m=current_distance,
        ttc_s=ttc_s,
        closest_distance_m=closest_distance,
        approaching=True,
        reason="below_threshold",
    )
