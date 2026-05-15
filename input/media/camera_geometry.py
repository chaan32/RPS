"""Unity camera geometry helpers for image-pixel to world-coordinate mapping.

Homography is still the right tool for points on the ground plane. For lifted
objects such as the crane box/dropzone, use the camera ray and intersect it with
the known Unity height plane instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnityCameraConfig:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    vfov_deg: float


# Must match simulation/Assets/Scripts/Editor/WorkplaceSceneBuilder.cs.
UNITY_CAMERA_CONFIGS: dict[str, UnityCameraConfig] = {
    "cam1": UnityCameraConfig(
        position=(-12.8, 6.4, 0.8),
        target=(-6.3, 0.08, 6.0),
        vfov_deg=72.0,
    ),
    "cam2": UnityCameraConfig(
        position=(0.6, 5.6, 10.8),
        target=(-4.7, 0.45, 7.5),
        vfov_deg=56.0,
    ),
}


def unity_to_world(unity_x: float, unity_z: float) -> tuple[float, float]:
    """Convert Unity ground coordinates to the ArUco/world plane coordinates."""
    return (-(unity_x + 8.5), unity_z - 3.5)


def _camera_basis(cam_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cfg = UNITY_CAMERA_CONFIGS[cam_id]
    origin = np.asarray(cfg.position, dtype=np.float64)
    target = np.asarray(cfg.target, dtype=np.float64)
    forward = target - origin
    forward /= np.linalg.norm(forward)

    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return origin, right, up, forward, float(cfg.vfov_deg)


def pixel_ray(
    cam_id: str,
    px: float,
    py: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Unity-space camera origin and unit direction for one image pixel."""
    origin, right, up, forward, vfov_deg = _camera_basis(cam_id)
    aspect = image_width / image_height
    vfov = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)
    fx = (image_width / 2.0) / math.tan(hfov / 2.0)
    fy = (image_height / 2.0) / math.tan(vfov / 2.0)

    x_cam = (px - image_width / 2.0) / fx
    y_cam = -(py - image_height / 2.0) / fy
    direction = forward + x_cam * right + y_cam * up
    direction /= np.linalg.norm(direction)
    return origin, direction


def pixel_to_world_on_unity_y_plane(
    cam_id: str,
    px: float,
    py: float,
    unity_y: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float] | None:
    """Map one pixel to world x/y by intersecting its camera ray with a Unity y plane."""
    origin, direction = pixel_ray(cam_id, px, py, image_width, image_height)
    if abs(direction[1]) < 1e-9:
        return None

    t = (unity_y - origin[1]) / direction[1]
    if t <= 0:
        return None

    point = origin + t * direction
    return unity_to_world(float(point[0]), float(point[2]))


def closest_point_between_rays(
    origin1: np.ndarray,
    direction1: np.ndarray,
    origin2: np.ndarray,
    direction2: np.ndarray,
) -> np.ndarray | None:
    """Return the midpoint of the shortest segment between two forward rays."""
    a = np.column_stack([direction1, -direction2])
    b = origin2 - origin1
    try:
        st, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if st[0] <= 0 or st[1] <= 0:
        return None

    p1 = origin1 + st[0] * direction1
    p2 = origin2 + st[1] * direction2
    return (p1 + p2) / 2.0


def triangulate_pixels_to_world(
    cam1_px: tuple[float, float],
    cam2_px: tuple[float, float],
    cam1_size: tuple[int, int],
    cam2_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Triangulate matching cam1/cam2 pixels and return world x/y."""
    o1, d1 = pixel_ray("cam1", cam1_px[0], cam1_px[1], cam1_size[0], cam1_size[1])
    o2, d2 = pixel_ray("cam2", cam2_px[0], cam2_px[1], cam2_size[0], cam2_size[1])
    point = closest_point_between_rays(o1, d1, o2, d2)
    if point is None:
        return None
    return unity_to_world(float(point[0]), float(point[2]))
