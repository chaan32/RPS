"""Realtime coordinate-window adapter for Fusion V2.

Fusion V2 is trained on 24-frame coordinate feature windows.  The V1 realtime
loop already converts camera detections into BEV/world coordinates; this module
keeps a per-worker rolling window in the same feature schema and calls the V2
GRU checkpoint.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from model.fusion.data.scenario_generator import DZ_CENTER

from .inference import load_checkpoint, predict_window
from .model import TemporalRiskPredictor
from .schema import FEATURE_COLUMNS


MISSING_DISTANCE_M = 20.0


def _xy_or_none(xy: Optional[tuple[float, float]]) -> tuple[float, float] | None:
    if xy is None:
        return None
    x, y = float(xy[0]), float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _velocity(
    current: tuple[float, float],
    previous: tuple[float, float] | None,
    dt_s: float,
) -> tuple[float, float]:
    if previous is None or dt_s <= 1e-6:
        return (0.0, 0.0)
    return (
        float((current[0] - previous[0]) / dt_s),
        float((current[1] - previous[1]) / dt_s),
    )


class FusionV2RealtimeInference:
    """Rolling-window V2 inference for one worker.

    The public API intentionally mirrors the V1 `RealtimeInference` shape used
    by `realtime_camera.py`: `push()`, `ready()`, `predict()`, and
    `update_dropzone()`.
    """

    def __init__(
        self,
        model: TemporalRiskPredictor,
        payload: dict,
        *,
        device: str = "cpu",
        window_size: int | None = None,
    ):
        self.model = model
        self.payload = payload
        self.device = device
        self.window_size = int(window_size or payload.get("window_size") or 24)
        self._features: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._last_ts: float | None = None

        self._last_worker_xy: tuple[float, float] | None = None
        self._last_forklift_xy: tuple[float, float] | None = None
        self._last_hazard_xy: tuple[float, float] | None = None
        self._last_dropzone_xy: tuple[float, float] | None = None

        self._default_dropzone_xy = (
            float(DZ_CENTER[0]),
            float(DZ_CENTER[1]),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: str = "cpu",
        window_size: int | None = None,
    ) -> tuple["FusionV2RealtimeInferenceFactory", dict]:
        model, payload = load_checkpoint(Path(checkpoint), device=device)
        return FusionV2RealtimeInferenceFactory(
            model=model,
            payload=payload,
            device=device,
            window_size=window_size,
        ), payload

    def update_dropzone(
        self,
        center: Optional[tuple[float, float]] = None,
        radius: Optional[float] = None,
    ) -> None:
        del radius
        xy = _xy_or_none(center)
        if xy is not None:
            self._last_dropzone_xy = xy

    def push(
        self,
        *,
        now_ts: float,
        worker_xy: tuple[float, float],
        forklift_xy: Optional[tuple[float, float]],
        forklift_hazard_xy: Optional[tuple[float, float]],
        dropzone_xy: Optional[tuple[float, float]],
        has_forklift: bool,
        has_dropzone: bool,
        worker_tracker_outlier: float = 0.0,
        forklift_tracker_outlier: float = 0.0,
    ) -> None:
        worker_xy = _xy_or_none(worker_xy)
        if worker_xy is None:
            return

        forklift_xy = _xy_or_none(forklift_xy)
        forklift_hazard_xy = _xy_or_none(forklift_hazard_xy) or forklift_xy
        dropzone_xy = _xy_or_none(dropzone_xy)

        dt_s = 0.0 if self._last_ts is None else float(now_ts - self._last_ts)
        self._last_ts = float(now_ts)

        prev_forklift_xy = self._last_forklift_xy
        prev_hazard_xy = self._last_hazard_xy
        prev_dropzone_xy = self._last_dropzone_xy

        effective_forklift = (
            forklift_xy
            or prev_forklift_xy
            or (0.0, 0.0)
        )
        effective_hazard = (
            forklift_hazard_xy
            or prev_hazard_xy
            or effective_forklift
        )
        effective_dropzone = (
            dropzone_xy
            or prev_dropzone_xy
            or self._default_dropzone_xy
        )

        worker_v = _velocity(worker_xy, self._last_worker_xy, dt_s)
        forklift_v = _velocity(effective_forklift, prev_forklift_xy, dt_s)
        hazard_v = _velocity(effective_hazard, prev_hazard_xy, dt_s)
        dropzone_v = _velocity(effective_dropzone, prev_dropzone_xy, dt_s)

        self._last_worker_xy = worker_xy
        if forklift_xy is not None:
            self._last_forklift_xy = forklift_xy
        if forklift_hazard_xy is not None:
            self._last_hazard_xy = forklift_hazard_xy
        if dropzone_xy is not None:
            self._last_dropzone_xy = dropzone_xy

        worker_forklift_dist = (
            _dist(worker_xy, effective_forklift) if has_forklift else MISSING_DISTANCE_M
        )
        worker_hazard_dist = (
            _dist(worker_xy, effective_hazard) if has_forklift else MISSING_DISTANCE_M
        )
        worker_dropzone_dist = (
            _dist(worker_xy, effective_dropzone) if has_dropzone else MISSING_DISTANCE_M
        )

        row = np.array([
            worker_xy[0],
            worker_xy[1],
            effective_forklift[0],
            effective_forklift[1],
            effective_hazard[0],
            effective_hazard[1],
            effective_dropzone[0],
            effective_dropzone[1],
            worker_forklift_dist,
            worker_hazard_dist,
            worker_dropzone_dist,
            worker_v[0],
            worker_v[1],
            forklift_v[0],
            forklift_v[1],
            hazard_v[0],
            hazard_v[1],
            dropzone_v[0],
            dropzone_v[1],
            1.0 if has_forklift else 0.0,
            1.0 if has_dropzone else 0.0,
            float(worker_tracker_outlier),
            float(forklift_tracker_outlier),
        ], dtype=np.float32)

        if len(row) != len(FEATURE_COLUMNS):
            raise RuntimeError(
                f"Fusion V2 feature size mismatch: {len(row)} != {len(FEATURE_COLUMNS)}"
            )
        self._features.append(row)

    def ready(self) -> bool:
        return len(self._features) >= self.window_size

    def predict(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("Fusion V2 window is not ready")
        window = np.stack(list(self._features), axis=0).astype(np.float32)
        risk = predict_window(self.model, self.payload, window, device=self.device)
        return risk.reshape(1, 2)


class FusionV2RealtimeInferenceFactory:
    """Creates per-worker V2 inference buffers sharing one model instance."""

    def __init__(
        self,
        *,
        model: TemporalRiskPredictor,
        payload: dict,
        device: str,
        window_size: int | None,
    ):
        self.model = model
        self.payload = payload
        self.device = device
        self.window_size = window_size

    def create(self) -> FusionV2RealtimeInference:
        return FusionV2RealtimeInference(
            self.model,
            self.payload,
            device=self.device,
            window_size=self.window_size,
        )
