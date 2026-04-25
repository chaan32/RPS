"""ArUco 마커 ID를 이미지 위에 크게 표시하는 유틸.

어떤 마커가 어느 코너에 있는지 눈으로 확인할 때 사용.

CLI 사용법:
    python input/media/identify_markers.py --image calibration/test_cam1.jpg

API 사용법 (server/main.py):
    from input.media.identify_markers import annotate_markers
    annotated, ids = annotate_markers(bgr_image)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


def annotate_markers(image: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """ArUco 마커 박스 + ID 텍스트를 이미지에 그려 넣는다.

    Args:
        image: BGR ndarray (cv2.imread 반환 형식). 함수 내부에서 직접 수정됨.

    Returns:
        (annotated_image, detected_ids) — 감지된 ID 정렬 리스트.
    """
    corners, ids, _ = _DETECTOR.detectMarkers(image)
    if ids is None:
        return image, []

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    scale = image.shape[0] / 1000.0
    font_scale = max(1.5, 2.5 * scale)
    thickness = max(3, int(6 * scale))

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]
        cx, cy = pts.mean(axis=0).astype(int)
        text = f"ID {int(marker_id)}"
        cv2.putText(
            image, text, (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4,
        )
        cv2.putText(
            image, text, (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness,
        )

    return image, sorted(int(i) for i in ids.flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    annotated, ids = annotate_markers(image)
    if not ids:
        print("⚠ 마커 감지 안 됨")
        return

    out_path = image_path.with_name(image_path.stem + "_ids.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"✓ 저장: {out_path}")
    print(f"  감지 ID: {ids}")


if __name__ == "__main__":
    main()
