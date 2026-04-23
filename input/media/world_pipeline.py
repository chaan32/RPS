"""엔드-투-엔드 파이프라인: 프레임 → YOLO → 발 픽셀 → 월드 좌표 → JSON.

사용자가 정의한 5단계 흐름을 한 스크립트로 엮은 것:
  1. 캘리브레이션 자동 체크 (없으면 RTSP에서 스냅샷 캡처 + calibrate 호출)
  2. ArUco id + 실측거리 + 스냅샷 → H 행렬 (기존 calibrate_homography.py 재사용)
  3. YOLO 탐지 → bbox 밑면 중점 (= 발 픽셀) 계산
  4. 발 픽셀을 H 행렬로 월드 좌표 변환
  5. cam1/cam2 관측을 하나의 JSON 페이로드로 직렬화

모드:
  # 단일 이미지 (테스트)
  python input/media/world_pipeline.py --cam cam1 --image calibration/test_cam1.jpg

  # 라이브 RTSP (실전)
  python input/media/world_pipeline.py --live
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO  # noqa: E402

from calibrate_homography import run_calibration  # noqa: E402
from world_mapper import pixel_to_world  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
CALIBRATION_DIR = PROJECT_ROOT / "calibration"


# ── 모델 로드 (모듈 로드 시 1회) ────────────────────────────────────────
pose_model = YOLO("yolo11n-pose.pt")

forklift_model = None
_best_path = os.getenv("BEST_MODEL_PATH", "")
if _best_path and not os.path.isabs(_best_path):
    _best_path = str(PROJECT_ROOT / _best_path)
if _best_path and os.path.exists(_best_path):
    forklift_model = YOLO(_best_path)
    print(f"[init] forklift 모델 로드: {_best_path}")
else:
    print("[init] forklift 모델 경로 없음 — person 감지만 수행")


# ── Step 3~4: YOLO + 발 픽셀 + 월드 변환 ────────────────────────────────
def extract_detections_with_world(frame, cam_id: str) -> list[dict]:
    """프레임 → YOLO → 발 픽셀 추출 → pixel_to_world → detection 리스트."""
    detections = []

    # (a) 사람 포즈 트래킹
    pose_results = pose_model.track(
        frame, conf=0.25, persist=True, verbose=False, classes=[0]
    )
    if pose_results[0].boxes is not None:
        boxes = pose_results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids_t = boxes.id
        ids_arr = (
            ids_t.cpu().numpy().astype(int).tolist()
            if ids_t is not None else [None] * len(xyxy)
        )
        for box, tid in zip(xyxy, ids_arr):
            x1, y1, x2, y2 = [float(v) for v in box]
            foot_x = (x1 + x2) / 2
            foot_y = y2
            wx, wy = pixel_to_world(foot_x, foot_y, cam_id)
            detections.append({
                "type": "worker",
                "track_id": tid,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [foot_x, foot_y],
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })

    # (b) 지게차 감지 (모델 있는 경우만)
    if forklift_model is not None:
        fl_results = forklift_model(frame, conf=0.5, verbose=False)
        for box in fl_results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            foot_x = (x1 + x2) / 2
            foot_y = y2
            wx, wy = pixel_to_world(foot_x, foot_y, cam_id)
            detections.append({
                "type": "forklift",
                "track_id": None,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [foot_x, foot_y],
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })

    return detections


def draw_annotated(frame, detections: list[dict]):
    """bbox + 발 점 + 월드 좌표 라벨 오버레이."""
    out = frame.copy()
    colors = {"worker": (0, 255, 255), "forklift": (0, 0, 255)}
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_px"]]
        fx, fy = [int(v) for v in d["foot_px"]]
        c = colors.get(d["type"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.circle(out, (fx, fy), 10, (0, 0, 255), -1)
        cv2.circle(out, (fx, fy), 14, (255, 255, 255), 2)
        wx, wy = d["world"]["x"], d["world"]["y"]
        label = f'{d["type"]} ({wx:.2f}, {wy:.2f})m'
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return out


# ── Step 1~2: 캘리브레이션 자동 확보 ────────────────────────────────────
def _capture_rtsp_snapshot(cam_id: str, rtsp_url: str) -> Path:
    """RTSP에서 1프레임 캡처하여 calibration/test_{cam_id}.jpg 저장.

    스레드 기반 VideoStream 대신 cv2.VideoCapture를 직접 사용.
    macOS + FFmpeg에서 스레드 VideoStream 재생성 시 Bus error가 자주 나서
    원샷 캡처는 단순 경로로 처리.
    """
    import gc
    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    print(f"[{cam_id}] 캘리브레이션 스냅샷 없음")
    input(f"  → {cam_id} 시야에 ArUco 마커 4개를 모두 배치하고 Enter: ")

    # TCP 강제 (UDP는 macOS + FFmpeg에서 bus error 빈발)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(40):  # 최대 ~4초 워밍업
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
        time.sleep(0.1)
    cap.release()
    del cap
    gc.collect()
    time.sleep(0.5)  # FFmpeg 리소스 정리 시간

    if frame is None:
        raise RuntimeError(f"[{cam_id}] RTSP 프레임 캡처 실패: {rtsp_url}")

    CALIBRATION_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(snap), frame)
    print(f"  ✓ 캡처: {snap}")
    return snap


def ensure_calibration(cam_id: str, rtsp_url: str | None = None) -> None:
    """캘리브레이션 JSON이 없으면 스냅샷 확보 → calibrate_homography.py 실행."""
    H_path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    if H_path.exists():
        print(f"[{cam_id}] ✓ 캘리브레이션 존재 ({H_path.name})")
        return

    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    if not snap.exists():
        if rtsp_url is None:
            raise FileNotFoundError(
                f"{snap} 없음. 다음 중 하나로 해결:\n"
                f"  · --live 모드로 RTSP에서 자동 캡처\n"
                f"  · calibration/ 에 직접 이미지 저장 후 재실행"
            )
        _capture_rtsp_snapshot(cam_id, rtsp_url)

    print(f"[{cam_id}] 캘리브레이션 실행...")
    run_calibration(cam_id, snap)


# ── 모드 1: 단일 이미지 테스트 ──────────────────────────────────────────
def run_image(cam_id: str, image_path: Path) -> dict:
    ensure_calibration(cam_id)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    detections = extract_detections_with_world(frame, cam_id)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "source": image_path.name,
        cam_id: detections,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # annotated 이미지 저장
    annotated = draw_annotated(frame, detections)
    out_path = CALIBRATION_DIR / f"{cam_id}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n✓ annotated: {out_path}")
    return payload


# ── 모드 2: 라이브 RTSP ─────────────────────────────────────────────────
def run_live(show: bool = True) -> None:
    from camera import VideoStream

    rtsp_1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp_2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp_1 or not rtsp_2:
        raise RuntimeError(".env의 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2 필요")

    # Phase 1: 두 카메라 스냅샷을 연속으로 먼저 확보
    # (캠1 캘리브레이션 후 캠2 캡처 시 FFmpeg 상태 오염 → bus error 방지)
    snap1 = CALIBRATION_DIR / "test_cam1.jpg"
    snap2 = CALIBRATION_DIR / "test_cam2.jpg"
    h1 = CALIBRATION_DIR / "cam1_homography.json"
    h2 = CALIBRATION_DIR / "cam2_homography.json"

    if not h1.exists() and not snap1.exists():
        _capture_rtsp_snapshot("cam1", rtsp_1)
    if not h2.exists() and not snap2.exists():
        _capture_rtsp_snapshot("cam2", rtsp_2)

    # Phase 2: 이미지 파일만으로 캘리브레이션 (RTSP 미사용)
    if not h1.exists():
        print(f"[cam1] 캘리브레이션 실행...")
        run_calibration("cam1", snap1)
    else:
        print(f"[cam1] ✓ 캘리브레이션 존재 ({h1.name})")

    if not h2.exists():
        print(f"[cam2] 캘리브레이션 실행...")
        run_calibration("cam2", snap2)
    else:
        print(f"[cam2] ✓ 캘리브레이션 존재 ({h2.name})")

    # Phase 3: 라이브 스트림 시작
    print("\n[live] 파이프라인 시작 (ESC 종료)...")
    cam1 = VideoStream(rtsp_1).start()
    cam2 = VideoStream(rtsp_2).start()
    time.sleep(2)

    try:
        while True:
            ret1, f1 = cam1.read()
            ret2, f2 = cam2.read()

            d1 = extract_detections_with_world(f1, "cam1") if ret1 and f1 is not None else []
            d2 = extract_detections_with_world(f2, "cam2") if ret2 and f2 is not None else []

            payload = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "cam1": d1,
                "cam2": d2,
            }
            print(json.dumps(payload, ensure_ascii=False))

            if show:
                if ret1 and f1 is not None:
                    cv2.imshow("cam1", cv2.resize(draw_annotated(f1, d1), (800, 600)))
                if ret2 and f2 is not None:
                    cv2.imshow("cam2", cv2.resize(draw_annotated(f2, d2), (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cam1.stop()
        cam2.stop()
        cv2.destroyAllWindows()


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--cam", help="cam1 | cam2 (이미지 모드)")
    parser.add_argument("--image", help="처리할 이미지 경로 (이미지 모드)")
    parser.add_argument("--live", action="store_true", help="RTSP 라이브 모드")
    parser.add_argument("--no-show", action="store_true", help="라이브 창 숨김")
    args = parser.parse_args()

    if args.live:
        run_live(show=not args.no_show)
    elif args.cam and args.image:
        run_image(args.cam, Path(args.image))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
