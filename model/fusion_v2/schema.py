"""Shared schema for Fusion V2 coordinate-sequence data."""

from __future__ import annotations

SAFE_THRESHOLD = 0.4
DANGER_THRESHOLD = 0.8

THREAT_NAMES = ("forklift", "dropzone")

# These columns are generated from V1 diagnostic CSVs.  Labels deliberately do
# not include V1 risk columns as features; V1 risk/early-warning output is only
# used as the teacher target while bootstrapping V2.
FEATURE_COLUMNS = (
    "worker_x",
    "worker_y",
    "forklift_x",
    "forklift_y",
    "forklift_hazard_x",
    "forklift_hazard_y",
    "dropzone_x",
    "dropzone_y",
    "worker_forklift_dist",
    "worker_forklift_hazard_dist",
    "worker_dropzone_dist",
    "worker_vx",
    "worker_vy",
    "forklift_vx",
    "forklift_vy",
    "forklift_hazard_vx",
    "forklift_hazard_vy",
    "dropzone_vx",
    "dropzone_vy",
    "has_forklift",
    "has_dropzone",
    "worker_tracker_outlier",
    "forklift_tracker_outlier",
)

LABEL_COLUMNS = (
    "forklift_target",
    "dropzone_target",
)


def risk_to_class(value: float) -> int:
    """Convert a risk score to SAFE/WARNING/DANGER class index."""
    if value >= DANGER_THRESHOLD:
        return 2
    if value >= SAFE_THRESHOLD:
        return 1
    return 0
