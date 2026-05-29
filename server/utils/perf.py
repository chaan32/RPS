"""Structured performance instrumentation helpers.

Runtime code should call these helpers instead of printing ad-hoc timings.
The raw pipeline metrics can keep their existing names, while these helpers add
stable `perf.*` keys that are easy to compare across benchmark runs.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator


Number = int | float


STANDARD_FIELD_DESCRIPTIONS: dict[str, str] = {
    "perf.loop.total_ms": "One complete realtime loop, including read, inference, fusion, output dispatch, and optional visualization.",
    "perf.loop.total_without_visual_ms": "Realtime loop until publish dispatch, excluding UI visualization cost.",
    "perf.io.camera_read_ms": "Time to read the latest frames from cam1 and cam2 buffers.",
    "perf.pipeline.extract_pair_wall_ms": "Wall-clock time spent waiting for all enabled camera/model extraction tasks to finish.",
    "perf.pipeline.extract.cam1_ms": "Wall time for full cam1 detection extraction.",
    "perf.pipeline.extract.cam2_ms": "Wall time for full cam2 detection extraction.",
    "perf.model.pose.cam1_ms": "YOLO pose tracking time for cam1 worker detection.",
    "perf.model.pose.cam2_ms": "YOLO pose tracking time for cam2 worker detection.",
    "perf.model.custom_yolo.cam1_ms": "Custom YOLO time for cam1 forklift/box/dropzone detection.",
    "perf.model.custom_yolo.cam2_ms": "Custom YOLO time for cam2 forklift/box/dropzone detection.",
    "perf.vision.aruco.cam1_ms": "ArUco marker detection time for cam1.",
    "perf.vision.aruco.cam2_ms": "ArUco marker detection time for cam2.",
    "perf.post.worker_collect.cam1_ms": "Worker keypoint/bbox parsing and world-coordinate conversion for cam1.",
    "perf.post.worker_collect.cam2_ms": "Worker keypoint/bbox parsing and world-coordinate conversion for cam2.",
    "perf.post.custom_yolo.cam1_ms": "Custom YOLO postprocess and world-coordinate conversion for cam1.",
    "perf.post.custom_yolo.cam2_ms": "Custom YOLO postprocess and world-coordinate conversion for cam2.",
    "perf.pipeline.cross_camera_ms": "Cross-camera worker ID propagation.",
    "perf.pipeline.refine_ms": "Detection refinement after cross-camera propagation.",
    "perf.pipeline.pick_positions_ms": "Select worker, forklift, and dropzone positions for fusion input.",
    "perf.pipeline.global_track_ms": "Global tracker update and outlier handling.",
    "perf.pipeline.motion_audio_dz_ms": "Motion history, audio score, and dropzone smoothing update.",
    "perf.pipeline.tracker_push_ms": "Push current positions into per-worker realtime fusion buffers.",
    "perf.fusion.forward_ms": "Fusion model forward inference for ready worker tracks.",
    "perf.fusion.early_warning_ms": "TTC/closest-approach early-warning calculation.",
    "perf.output.publish_dispatch_ms": "Alert dispatch scheduling, including cooldown checks and background thread creation.",
    "perf.ui.console_ms": "Console logging cost.",
    "perf.ui.visualize_ms": "BEV/camera overlay rendering and cv2 window update cost.",
}

STANDARD_SUMMARY_FIELDS: list[str] = list(STANDARD_FIELD_DESCRIPTIONS)

_CAMERA_TIMING_MAP = {
    "extract_total_ms": "perf.pipeline.extract.{cam}_ms",
    "pose_track_ms": "perf.model.pose.{cam}_ms",
    "custom_yolo_ms": "perf.model.custom_yolo.{cam}_ms",
    "aruco_detect_ms": "perf.vision.aruco.{cam}_ms",
    "worker_collect_ms": "perf.post.worker_collect.{cam}_ms",
    "custom_postprocess_ms": "perf.post.custom_yolo.{cam}_ms",
}


def _round_ms(value: Number, precision: int = 3) -> float:
    return round(float(value), precision)


def add_duration_ms(
    row: MutableMapping[str, Any],
    field: str,
    start: float,
    end: float,
    *,
    precision: int = 3,
) -> None:
    """Append one standardized duration field to a metrics row."""
    row[field] = _round_ms((end - start) * 1000.0, precision)


def add_value_ms(
    row: MutableMapping[str, Any],
    field: str,
    value: Number,
    *,
    precision: int = 3,
) -> None:
    """Append one already-computed millisecond value to a metrics row."""
    row[field] = _round_ms(value, precision)


def add_camera_timings(
    row: MutableMapping[str, Any],
    cam_id: str,
    timing: Mapping[str, Any],
    *,
    precision: int = 3,
) -> None:
    """Map DetectionPipeline timing keys onto stable `perf.*` fields."""
    for source_key, target_template in _CAMERA_TIMING_MAP.items():
        value = timing.get(source_key)
        if isinstance(value, int | float):
            row[target_template.format(cam=cam_id)] = _round_ms(value, precision)


@dataclass
class StageTimer:
    """Context-manager based timer for new code paths.

    Example:
        timer = StageTimer()
        with timer.stage("perf.model.pose.cam1_ms"):
            run_pose()
        metrics.update(timer.snapshot())
    """

    precision: int = 3
    _values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, field: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self._values[field] = _round_ms(
                (perf_counter() - started) * 1000.0,
                self.precision,
            )

    def snapshot(self) -> dict[str, float]:
        return dict(self._values)
