"""Evaluate Box1 world-coordinate strategies for elevated/lifted boxes.

The existing homography path maps an image point to the ground plane. That is
not valid for a suspended box, so this tool compares it against camera-ray
methods that estimate the box center in Unity 3D space.

Typical usage after running check_blindspot_recording.py:

    python input/media/tools/test/evaluate_box_center_methods.py \
      --detections /private/tmp/fusion_input_precheck/blindspot_world_coords.csv \
      --ground-truth simulation/Recordings/blindspot_ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Must match WorkplaceSceneBuilder.cs after the cam2 zoom update.
CAMERAS = {
    "cam1": {
        "position": (-12.8, 6.4, 0.8),
        "target": (-6.3, 0.08, 6.0),
        "vfov_deg": 72.0,
    },
    "cam2": {
        "position": (0.6, 5.6, 10.8),
        "target": (-4.7, 0.45, 7.5),
        "vfov_deg": 56.0,
    },
}


@dataclass(frozen=True)
class Estimate:
    method: str
    frame: int
    world_x: float
    world_y: float
    unity_y: float | None = None


def unity_to_world(unity_x: float, unity_z: float) -> tuple[float, float]:
    """Match WorkplaceSceneBuilder.ArucoWorldFromUnity."""
    return (-(unity_x + 8.5), unity_z - 3.5)


def camera_basis(cam_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cfg = CAMERAS[cam_id]
    origin = np.asarray(cfg["position"], dtype=np.float64)
    target = np.asarray(cfg["target"], dtype=np.float64)
    forward = target - origin
    forward /= np.linalg.norm(forward)

    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return origin, right, up, forward, float(cfg["vfov_deg"])


def pixel_ray(
    cam_id: str,
    px: float,
    py: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    origin, right, up, forward, vfov_deg = camera_basis(cam_id)
    aspect = width / height
    vfov = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = (height / 2.0) / math.tan(vfov / 2.0)

    x_cam = (px - width / 2.0) / fx
    y_cam = -(py - height / 2.0) / fy
    direction = forward + x_cam * right + y_cam * up
    direction /= np.linalg.norm(direction)
    return origin, direction


def intersect_y_plane(origin: np.ndarray, direction: np.ndarray, unity_y: float) -> np.ndarray | None:
    if abs(direction[1]) < 1e-9:
        return None
    t = (unity_y - origin[1]) / direction[1]
    if t <= 0:
        return None
    return origin + t * direction


def closest_point_between_rays(
    o1: np.ndarray,
    d1: np.ndarray,
    o2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray | None:
    # Solve o1 + s*d1 ~= o2 + t*d2 in least squares form.
    a = np.column_stack([d1, -d2])
    b = o2 - o1
    try:
        st, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    p1 = o1 + st[0] * d1
    p2 = o2 + st[1] * d2
    if st[0] <= 0 or st[1] <= 0:
        return None
    return (p1 + p2) / 2.0


def bbox_point(row: dict[str, str], x_alpha: float, y_alpha: float) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    return (x1 + (x2 - x1) * x_alpha, y1 + (y2 - y1) * y_alpha)


def load_ground_truth(path: Path, default_box_center_y: float) -> dict[int, tuple[float, float, float]]:
    truth: dict[int, tuple[float, float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["object"] != "box_1":
                continue
            unity_y = float(row.get("unity_y") or default_box_center_y)
            truth[int(row["frame"])] = (float(row["world_x"]), float(row["world_y"]), unity_y)
    return truth


def load_box_detections(path: Path) -> dict[int, dict[str, dict[str, str]]]:
    by_frame: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["type"] == "box_1":
                by_frame[int(row["frame"])][row["cam"]] = row
    return by_frame


def build_estimates(
    detections: dict[int, dict[str, dict[str, str]]],
    box_center_y: float,
    width: int,
    height: int,
) -> list[Estimate]:
    estimates: list[Estimate] = []
    for frame, cams in detections.items():
        for cam_id, row in cams.items():
            # Current pipeline estimate. Older CSVs have no coord_source, which means
            # bbox bottom-center -> ground homography.
            coord_source = row.get("coord_source") or "homography_bottom_center"
            estimates.append(
                Estimate(
                    method=f"{cam_id}:pipeline_{coord_source}",
                    frame=frame,
                    world_x=float(row["world_x"]),
                    world_y=float(row["world_y"]),
                )
            )

            # Project bbox center ray onto the known/scheduled box-center height plane.
            px, py = bbox_point(row, 0.5, 0.5)
            origin, direction = pixel_ray(cam_id, px, py, width, height)
            point = intersect_y_plane(origin, direction, box_center_y)
            if point is not None:
                wx, wy = unity_to_world(float(point[0]), float(point[2]))
                estimates.append(
                    Estimate(
                        method=f"{cam_id}:ray_plane_bbox_center",
                        frame=frame,
                        world_x=wx,
                        world_y=wy,
                        unity_y=float(point[1]),
                    )
                )

        if "cam1" in cams and "cam2" in cams:
            for x_alpha, y_alpha, label in (
                (0.5, 0.5, "bbox_center"),
                (0.5, 1.0, "bbox_bottom_center"),
            ):
                p1 = bbox_point(cams["cam1"], x_alpha, y_alpha)
                p2 = bbox_point(cams["cam2"], x_alpha, y_alpha)
                o1, d1 = pixel_ray("cam1", *p1, width, height)
                o2, d2 = pixel_ray("cam2", *p2, width, height)
                point = closest_point_between_rays(o1, d1, o2, d2)
                if point is None:
                    continue
                wx, wy = unity_to_world(float(point[0]), float(point[2]))
                estimates.append(
                    Estimate(
                        method=f"multiview:triangulate_{label}",
                        frame=frame,
                        world_x=wx,
                        world_y=wy,
                        unity_y=float(point[1]),
                    )
                )

            # Average both cameras' known-height estimates.
            plane_points = []
            for cam_id in ("cam1", "cam2"):
                px, py = bbox_point(cams[cam_id], 0.5, 0.5)
                origin, direction = pixel_ray(cam_id, px, py, width, height)
                point = intersect_y_plane(origin, direction, box_center_y)
                if point is not None:
                    plane_points.append(point)
            if len(plane_points) == 2:
                point = np.mean(plane_points, axis=0)
                wx, wy = unity_to_world(float(point[0]), float(point[2]))
                estimates.append(
                    Estimate(
                        method="multiview:avg_ray_plane_bbox_center",
                        frame=frame,
                        world_x=wx,
                        world_y=wy,
                        unity_y=float(point[1]),
                    )
                )
    return estimates


def summarize(estimates: list[Estimate], truth: dict[int, tuple[float, float, float]]) -> None:
    grouped: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    for est in estimates:
        if est.frame not in truth:
            continue
        tx, ty, truth_y = truth[est.frame]
        err_xy = math.hypot(est.world_x - tx, est.world_y - ty)
        err_y = abs(est.unity_y - truth_y) if est.unity_y is not None else None
        grouped[est.method].append((err_xy, err_y))

    print("method,n,median_xy_m,p90_xy_m,max_xy_m,median_height_m")
    rows = []
    for method, values in grouped.items():
        xy = sorted(v[0] for v in values)
        height = sorted(v[1] for v in values if v[1] is not None)
        if not xy:
            continue
        p90_idx = int(0.9 * (len(xy) - 1))
        rows.append(
            (
                float(np.median(xy)),
                method,
                len(xy),
                float(xy[p90_idx]),
                float(max(xy)),
                float(np.median(height)) if height else None,
            )
        )
    for med_xy, method, n, p90_xy, max_xy, med_h in sorted(rows):
        med_h_text = "" if med_h is None else f"{med_h:.3f}"
        print(f"{method},{n},{med_xy:.3f},{p90_xy:.3f},{max_xy:.3f},{med_h_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--box-center-y", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = parser.parse_args()

    truth = load_ground_truth(args.ground_truth, args.box_center_y)
    detections = load_box_detections(args.detections)
    estimates = build_estimates(detections, args.box_center_y, args.width, args.height)
    summarize(estimates, truth)


if __name__ == "__main__":
    main()
