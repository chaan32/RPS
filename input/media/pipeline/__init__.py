"""Media pipeline 패키지.

엔드-투-엔드 흐름:
  RTSP frame → DetectionPipeline.extract → cross_camera_propagate → JSON

Public API:
    from input.media.pipeline import DetectionPipeline, ensure_calibration
    from input.media.pipeline import draw_annotated, run_image, run_live
"""

from .calibration_runtime import (
    CALIBRATION_DIR,
    PROJECT_ROOT,
    capture_rtsp_snapshot,
    ensure_calibration,
)
from .engine import DetectionPipeline
from .runner import build_default_pipeline, main, run_image, run_live
from .visualization import draw_annotated

__all__ = [
    "DetectionPipeline",
    "build_default_pipeline",
    "ensure_calibration",
    "capture_rtsp_snapshot",
    "draw_annotated",
    "run_image",
    "run_live",
    "main",
    "PROJECT_ROOT",
    "CALIBRATION_DIR",
]
