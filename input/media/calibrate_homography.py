"""카메라 스냅샷 + 바닥 ArUco 마커 → Homography(H) 행렬 계산 → JSON 저장.

사용법:
    python input/media/calibrate_homography.py --cam cam1 --image snapshot_cam1.jpg
    python input/media/calibrate_homography.py --cam cam2 --image snapshot_cam2.jpg

준비물:
    1. calibration/world_markers.json 에 마커 실측 좌표 기입
    2. 각 카메라로 바닥 마커가 4개 이상 보이는 스냅샷 촬영

출력:
    calibration/{cam_id}_homography.json
    (world_mapper.pixel_to_world() 가 이 파일을 읽어 변환을 수행)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def load_world_markers() -> dict[int, tuple[float, float]]:
    """아루코 코드 간의 거리를 사람이 직접 기입한 걸 가져오는 것"""
    path = CALIBRATION_DIR / "world_markers.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 바닥 마커의 월드 좌표를 먼저 기입하세요."
        )
    with open(path) as f:
        data = json.load(f)
    return {m["id"]: (m["world_x"], m["world_y"]) for m in data["markers"]}


def detect_aruco_centers(image: np.ndarray) -> dict[int, np.ndarray]:
    
    # 사전의 정의한 아루코 코드 설정 값들로 ArucoDetector 객체 생성
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    # detector 객체로 이미지에서 아루코 코드 감지 
    """
    corners = [
            array([[[312, 470], [322, 470], [322, 490], [312, 490]]]),  # 마커 1 의 4코너
            array([[...]]),                                               # 마커 2
            array([[...]]),                                               # 마커 3
            array([[...]])
        ]                                               # 마커 4
    ids     = array([[38], [27], [24], [22]])
    """
    corners, ids, _ = detector.detectMarkers(image)
    
    if ids is None:
        return {}
    result = {}
    # 네개의 코너의 중앙 구하기! 
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]  # shape (4, 2)
        center = pts.mean(axis=0)
        result[int(marker_id)] = center
    """
    result = {
    38: array([317.25, 480.30]),   # 마커 38 의 중심 픽셀 (x, y)
    27: array([193.50, 471.20]),   # 마커 27 의 중심 픽셀
    24: array([401.75, 215.80]),
    22: array([ 76.50, 218.70]),
    """
    return result


def compute_homography(
    pixel_centers: dict[int, np.ndarray],
    world_centers: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, list[int], dict]:
    """픽셀 ↔ 월드 매칭 → H(3x3) + 재투영 오차 통계."""
    common_ids = sorted(set(pixel_centers) & set(world_centers))
    if len(common_ids) < 4:
        raise RuntimeError(
            f"호모그래피에 최소 4개 마커가 필요. "
            f"공통 감지 마커: {len(common_ids)}개 ({common_ids})"
        )

    src = np.array([pixel_centers[i] for i in common_ids], dtype=np.float32)
    dst = np.array([world_centers[i] for i in common_ids], dtype=np.float32)

    H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("cv2.findHomography 실패")

    errors = {}
    for mid in common_ids:
        px, py = pixel_centers[mid]
        p = H @ np.array([px, py, 1.0])
        wx, wy = p[0] / p[2], p[1] / p[2]
        tx, ty = world_centers[mid]
        errors[mid] = float(np.hypot(wx - tx, wy - ty))

    err_stats = {
        "mean_m": float(np.mean(list(errors.values()))),
        "max_m": float(max(errors.values())),
        "per_marker_m": errors,
    }
    return H, common_ids, err_stats


def run_calibration(cam_id: str, image_path: Path) -> Path:
    """
    캠 ID + 스냅샷 이미지 경로를 받아서 H 계산하고 JSON 저장함

    반환: 저장된 JSON 경로
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    world_centers = load_world_markers()
    ''' world_centers는 이런 식으로 리턴 된다 
    {
    38: (0.00, 0.00),
    27: (-2.22, 0.00),
    24: (0.00, 2.34),
    22: (-2.22, 2.34),
    }
    '''
    pixel_centers = detect_aruco_centers(image)
    '''
    result = {
    38: array([317.25, 480.30]),   # 마커 38 의 중심 픽셀 (x, y)
    27: array([193.50, 471.20]),   # 마커 27 의 중심 픽셀
    24: array([401.75, 215.80]),
    22: array([ 76.50, 218.70]),
    '''
    print(f"\n[{cam_id}] 이미지: {image_path.name}")
    print(f"  월드 정의 마커: {len(world_centers)}개 (IDs: {sorted(world_centers)})")
    print(f"  감지된 마커:    {len(pixel_centers)}개 (IDs: {sorted(pixel_centers)})")

    H, used_ids, err = compute_homography(pixel_centers, world_centers)

    print(f"  사용 마커:      {used_ids}")
    print(
        f"  재투영 오차:    평균 {err['mean_m']*100:.2f} cm, "
        f"최대 {err['max_m']*100:.2f} cm"
    )
    if err["max_m"] > 0.10:
        print("  ⚠ 최대 오차 10cm 초과 — 마커 실측값 또는 이미지 품질 재확인 권장")

    CALIBRATION_DIR.mkdir(exist_ok=True)
    output = {
        "cam_id": cam_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": image_path.name,
        "markers_used": used_ids,
        "reprojection_error": err,
        "homography": H.tolist(),
    }
    out_path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 저장: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", required=True, help="cam1, cam2 등 카메라 식별자")
    parser.add_argument("--image", required=True, help="바닥 마커가 보이는 스냅샷 경로")
    args = parser.parse_args()
    run_calibration(args.cam, Path(args.image))


if __name__ == "__main__":
    main()
