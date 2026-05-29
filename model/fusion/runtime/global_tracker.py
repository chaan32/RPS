"""Global world-coordinate tracker for fusion inputs.

This layer sits after multi-view detection/refinement and before fusion.  It
keeps a short motion state per object so one-frame coordinate jumps from
occlusion do not immediately enter the BEV/fusion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


Point = tuple[float, float]


@dataclass(frozen=True)
class TrackConfig:
    alpha: float = 0.65
    outlier_alpha: float = 0.12
    velocity_alpha: float = 0.35
    outlier_distance_m: float = 0.90
    max_speed_mps: float = 2.5
    max_hold_s: float = 0.60
    extended_max_hold_s: float | None = None
    min_measurements_for_extended_hold: int = 4
    reset_after_outliers: int | None = None


@dataclass
class TrackUpdate:
    xy: Point
    raw_xy: Point | None
    predicted_xy: Point
    residual_m: float
    smoothed: bool
    outlier: bool
    stale: bool


class PointTrack:
    def __init__(self, config: TrackConfig):
        self.config = config
        self.xy: Point | None = None
        self.velocity: Point = (0.0, 0.0)
        self.last_ts: float | None = None
        self.last_measurement_ts: float | None = None
        self.missed = 0
        self.measurements = 0
        self.consecutive_outliers = 0

    def update(self, ts: float, measurement: Point | None) -> TrackUpdate | None:
        if self.xy is None:
            if measurement is None:
                return None
            self.xy = (float(measurement[0]), float(measurement[1]))
            self.last_ts = float(ts)
            self.last_measurement_ts = float(ts)
            self.velocity = (0.0, 0.0)
            self.missed = 0
            self.measurements = 1
            self.consecutive_outliers = 0
            return TrackUpdate(
                xy=self.xy,
                raw_xy=measurement,
                predicted_xy=self.xy,
                residual_m=0.0,
                smoothed=False,
                outlier=False,
                stale=False,
            )

        assert self.last_ts is not None
        dt = max(1e-3, min(float(ts) - self.last_ts, 0.5))
        predicted = (
            self.xy[0] + self.velocity[0] * dt,
            self.xy[1] + self.velocity[1] * dt,
        )

        if measurement is None:
            self.missed += 1
            measurement_age = (
                float(ts) - self.last_measurement_ts
                if self.last_measurement_ts is not None
                else float("inf")
            )
            hold_s = self.config.max_hold_s
            if (
                self.config.extended_max_hold_s is not None
                and self.measurements >= self.config.min_measurements_for_extended_hold
            ):
                hold_s = self.config.extended_max_hold_s
            stale = measurement_age > hold_s
            if stale:
                return None
            self.xy = predicted
            self.last_ts = float(ts)
            return TrackUpdate(
                xy=self.xy,
                raw_xy=None,
                predicted_xy=predicted,
                residual_m=0.0,
                smoothed=True,
                outlier=False,
                stale=True,
            )

        measurement = (float(measurement[0]), float(measurement[1]))
        self.last_measurement_ts = float(ts)
        residual = _dist(measurement, predicted)
        dynamic_gate = self.config.outlier_distance_m + self.config.max_speed_mps * dt
        outlier = residual > dynamic_gate

        if outlier:
            self.consecutive_outliers += 1
            if (
                self.config.reset_after_outliers is not None
                and self.consecutive_outliers >= self.config.reset_after_outliers
            ):
                previous_xy = self.xy
                self.xy = measurement
                self.velocity = _clamp_velocity(
                    (
                        (self.xy[0] - previous_xy[0]) / dt,
                        (self.xy[1] - previous_xy[1]) / dt,
                    ),
                    self.config.max_speed_mps,
                )
                self.last_ts = float(ts)
                self.missed = 0
                self.measurements = 1
                self.consecutive_outliers = 0
                return TrackUpdate(
                    xy=self.xy,
                    raw_xy=measurement,
                    predicted_xy=predicted,
                    residual_m=residual,
                    smoothed=False,
                    outlier=True,
                    stale=False,
                )
        else:
            self.consecutive_outliers = 0

        alpha = self.config.outlier_alpha if outlier else self.config.alpha
        new_xy = (
            predicted[0] + alpha * (measurement[0] - predicted[0]),
            predicted[1] + alpha * (measurement[1] - predicted[1]),
        )

        measured_v = (
            (new_xy[0] - self.xy[0]) / dt,
            (new_xy[1] - self.xy[1]) / dt,
        )
        measured_v = _clamp_velocity(measured_v, self.config.max_speed_mps)
        self.velocity = (
            (1.0 - self.config.velocity_alpha) * self.velocity[0]
            + self.config.velocity_alpha * measured_v[0],
            (1.0 - self.config.velocity_alpha) * self.velocity[1]
            + self.config.velocity_alpha * measured_v[1],
        )
        self.xy = new_xy
        self.last_ts = float(ts)
        self.missed = 0
        if not outlier:
            self.measurements += 1
        return TrackUpdate(
            xy=new_xy,
            raw_xy=measurement,
            predicted_xy=predicted,
            residual_m=residual,
            smoothed=True,
            outlier=outlier,
            stale=False,
        )


class GlobalTrackManager:
    """Track worker/forklift/dropzone positions in world coordinates."""

    def __init__(
        self,
        worker_config: TrackConfig | None = None,
        forklift_config: TrackConfig | None = None,
        dropzone_config: TrackConfig | None = None,
    ):
        self.worker_config = worker_config or TrackConfig(
            alpha=0.70,
            outlier_alpha=0.10,
            outlier_distance_m=0.95,
            max_speed_mps=2.2,
            max_hold_s=0.65,
            extended_max_hold_s=1.35,
            min_measurements_for_extended_hold=4,
            reset_after_outliers=3,
        )
        self.forklift_config = forklift_config or TrackConfig(
            alpha=0.75,
            outlier_alpha=0.18,
            outlier_distance_m=0.85,
            max_speed_mps=3.0,
        )
        self.dropzone_config = dropzone_config or TrackConfig(
            alpha=0.45,
            outlier_alpha=0.15,
            outlier_distance_m=1.40,
            max_speed_mps=2.0,
        )
        self.workers: dict[str, PointTrack] = {}
        self.forklift = PointTrack(self.forklift_config)
        self.dropzone = PointTrack(self.dropzone_config)
        self.last_updates: dict[str, TrackUpdate] = {}

    def update(
        self,
        ts: float,
        workers_xy: dict[str, Point],
        forklift_xy: Point | None,
        dropzone_xy: Point | None,
    ) -> tuple[dict[str, Point], Point | None, Point | None]:
        smoothed_workers: dict[str, Point] = {}

        active_worker_ids = set(workers_xy)
        for wid, xy in workers_xy.items():
            track = self.workers.setdefault(wid, PointTrack(self.worker_config))
            update = track.update(ts, xy)
            if update is not None:
                smoothed_workers[wid] = update.xy
                self.last_updates[f"worker:{wid}"] = update

        for wid in list(self.workers.keys()):
            if wid in active_worker_ids:
                continue
            update = self.workers[wid].update(ts, None)
            if update is None:
                self.workers.pop(wid, None)
                self.last_updates.pop(f"worker:{wid}", None)
            else:
                smoothed_workers[wid] = update.xy
                self.last_updates[f"worker:{wid}"] = update

        forklift_update = self.forklift.update(ts, forklift_xy)
        smoothed_forklift = forklift_update.xy if forklift_update is not None else None
        if forklift_update is not None:
            self.last_updates["forklift"] = forklift_update

        dropzone_update = self.dropzone.update(ts, dropzone_xy)
        smoothed_dropzone = dropzone_update.xy if dropzone_update is not None else None
        if dropzone_update is not None:
            self.last_updates["dropzone"] = dropzone_update

        return smoothed_workers, smoothed_forklift, smoothed_dropzone

    def update_for(self, key: str) -> TrackUpdate | None:
        return self.last_updates.get(key)


def _dist(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _clamp_velocity(v: Point, max_speed: float) -> Point:
    speed = hypot(v[0], v[1])
    if speed <= max_speed or speed <= 1e-9:
        return v
    scale = max_speed / speed
    return v[0] * scale, v[1] * scale
