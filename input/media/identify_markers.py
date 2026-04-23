"""ArUco 마커 ID를 이미지 위에 크게 표시하는 유틸.

어떤 마커가 어느 코너에 있는지 눈으로 확인할 때 사용.

사용법:
    python input/media/identify_markers.py --image calibration/test_cam1.jpg

출력:
    입력 이미지 옆에 "{원본명}_ids.jpg" 저장 (예: test_cam1_ids.jpg)
"""

import argparse
from pathlib import Path

import cv2


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        print("⚠ 마커 감지 안 됨")
        return

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

    out_path = image_path.with_name(image_path.stem + "_ids.jpg")
    cv2.imwrite(str(out_path), image)
    print(f"✓ 저장: {out_path}")
    print(f"  감지 ID: {sorted([int(i) for i in ids.flatten()])}")


if __name__ == "__main__":
    main()
