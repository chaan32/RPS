"""Homography 캘리브레이션 결과를 시각적으로 검증.

calibrate_homography.py 실행 후, 생성된 H 행렬이 실제 월드-픽셀 매핑을
올바르게 수행하는지 눈으로 확인하는 도구.

사용법:
    python input/media/verify_homography.py --cam cam1 --image snapshot.jpg
    python input/media/verify_homography.py --cam cam1 --image snapshot.jpg --interactive

기본 출력:
    calibration/{cam_id}_verify.jpg — 원본에 다음 오버레이:
      · 감지된 ArUco 마커 경계 + 월드 좌표 라벨
      · 월드 평면 격자선 (기본 0.5m 간격)
      · 월드 원점(0,0)에서 X축(빨강)/Y축(초록) 화살표

--interactive: 창에서 마우스 클릭 시 해당 픽셀의 월드 좌표를 출력

검증 기준:
    바닥에 실제로 직각인 기준선(벽, 타일, 테이프)과
    격자가 눈으로 봤을 때 정렬돼야 H가 올바름.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def load_calibration(cam_id: str) -> np.ndarray:
    path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 calibrate_homography.py 를 실행하세요."
        )
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data["homography"], dtype=np.float64)


def pixel_to_world(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    p = H @ np.array([px, py, 1.0], dtype=np.float64)
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def world_to_pixel(wx: float, wy: float, H_inv: np.ndarray) -> tuple[int, int]:
    p = H_inv @ np.array([wx, wy, 1.0], dtype=np.float64)
    return (int(round(p[0] / p[2])), int(round(p[1] / p[2])))


def draw_overlay(
    image: np.ndarray,
    H: np.ndarray,
    grid_step: float,
    grid_extent: float,
) -> np.ndarray:
    out = image.copy()
    H_inv = np.linalg.inv(H)

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(out)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners[0]
            cx, cy = pts.mean(axis=0).astype(int)
            wx, wy = pixel_to_world(cx, cy, H)
            label = f"ID{marker_id}: ({wx:.2f}, {wy:.2f})m"
            cv2.putText(
                out, label, (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

    grid_vals = np.arange(-grid_extent, grid_extent + grid_step / 2, grid_step)
    grid_color = (255, 80, 0)
    thickness = 2
    for x in grid_vals:
        pts_px = [world_to_pixel(x, y, H_inv) for y in grid_vals]
        for i in range(len(pts_px) - 1):
            cv2.line(out, pts_px[i], pts_px[i + 1], grid_color, thickness, cv2.LINE_AA)
    for y in grid_vals:
        pts_px = [world_to_pixel(x, y, H_inv) for x in grid_vals]
        for i in range(len(pts_px) - 1):
            cv2.line(out, pts_px[i], pts_px[i + 1], grid_color, thickness, cv2.LINE_AA)

    origin_px = world_to_pixel(0, 0, H_inv)
    x_axis_end = world_to_pixel(grid_step * 2, 0, H_inv)
    y_axis_end = world_to_pixel(0, grid_step * 2, H_inv)
    cv2.arrowedLine(out, origin_px, x_axis_end, (0, 0, 255), 3, tipLength=0.2)
    cv2.arrowedLine(out, origin_px, y_axis_end, (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(out, "X", x_axis_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, "Y", y_axis_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.circle(out, origin_px, 5, (255, 255, 255), -1)

    return out


def run_interactive(image: np.ndarray, H: np.ndarray) -> None:
    display = image.copy()
    window = "verify (click: px->world, q: quit)"

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        wx, wy = pixel_to_world(x, y, H)
        nonlocal display
        display = image.copy()
        cv2.circle(display, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(
            display,
            f"px({x},{y}) -> world({wx:.3f}, {wy:.3f})m",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        print(f"px ({x}, {y}) → world ({wx:.3f}, {wy:.3f}) m")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, display)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--grid-step", type=float, default=0.2,
                        help="격자 간격 (미터). 기본 0.2")
    parser.add_argument("--grid-extent", type=float, default=3.0,
                        help="격자 최대 범위 (미터, 원점 ±). 기본 3.0")
    parser.add_argument("--interactive", action="store_true",
                        help="클릭으로 픽셀→월드 변환 확인")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {args.image}")

    H = load_calibration(args.cam)
    overlay = draw_overlay(image, H, args.grid_step, args.grid_extent)

    CALIBRATION_DIR.mkdir(exist_ok=True)
    out_path = CALIBRATION_DIR / f"{args.cam}_verify.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"✓ 저장: {out_path}")
    print("  열어서 격자가 바닥/탁상의 직각 기준선과 정렬됐는지 확인.")

    if args.interactive:
        run_interactive(overlay, H)


if __name__ == "__main__":
    main()
