"""카메라 픽셀 좌표 → 월드 절대 좌표(m) 변환.

Homography 행렬(H)을 calibration/{cam_id}_homography.json 에서 로드하여
pixel_to_world() 함수로 변환을 수행한다.

H는 calibrate_homography.py 로 사전 1회 생성해야 한다.
"""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

_H_CACHE: dict[str, np.ndarray] = {}


def load_homography(cam_id: str) -> np.ndarray:
    """캘리브레이션 파일에서 H(3x3) 로드. 프로세스당 1회 캐시."""
    if cam_id in _H_CACHE:
        return _H_CACHE[cam_id]

    path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{cam_id} 캘리브레이션 없음: {path}\n"
            f"먼저 실행: python input/media/calibrate_homography.py "
            f"--cam {cam_id} --image <snapshot>"
        )

    with open(path) as f:
        data = json.load(f)
    H = np.asarray(data["homography"], dtype=np.float64)
    _H_CACHE[cam_id] = H
    return H


def pixel_to_world(px: float, py: float, cam_id: str) -> tuple[float, float]:
    """카메라 픽셀 좌표 → 월드 좌표 (미터).

    변환 대상 점이 **바닥 평면 위에 있을 때만** 정확하다.
    (공중에 있는 점을 넣으면 바닥으로 수직 투영한 잘못된 값이 반환됨)

    Args:
        px, py: 카메라 이미지상의 픽셀 좌표.
        cam_id: "cam1", "cam2" 등.

    Returns:
        (world_x, world_y): 월드 좌표계 미터.
    """
    H = load_homography(cam_id)
    p = H @ np.array([px, py, 1.0], dtype=np.float64)
    return (float(p[0] / p[2]), float(p[1] / p[2]))
