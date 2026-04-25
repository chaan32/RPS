"""작업자 ArUco 식별 단독 테스트.

목적: world_pipeline 의 worker_id 매칭 로직(마커 5/10/15 → W01/W02/W03)을
캘리브레이션/오디오/리스크모델 없이 격리해서 검증한다.

화면에 다음을 표시:
  - YOLO person bbox + 매칭된 worker_id (W01/W02/W03 또는 '?' = 미식별)
  - ArUco 마커 박스 + ID

콘솔에 매 프레임:
  - 검출된 ArUco IDs 전체
  - 매칭된 (마커 → bbox#, 거리 px)
  - 매칭 실패한 마커

사용법:
    python input/media/test_worker_id.py             # cam1 + cam2 동시
    python input/media/test_worker_id.py --cam cam1  # cam1 만
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
)

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO  # noqa: E402

from camera import VideoStream  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

WORKER_ARUCO_MAP = {5: "W01", 10: "W02", 15: "W03"}

_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_aruco_params.adaptiveThreshWinSizeMin = 3
_aruco_params.adaptiveThreshWinSizeMax = 23
_aruco_params.adaptiveThreshWinSizeStep = 10
_aruco_params.minMarkerPerimeterRate = 0.01
_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

pose_model = YOLO("yolo11n-pose.pt")


def process_frame(frame, cam_id: str):
    """프레임 → (annotated, worker_id_summary)."""
    out = frame.copy()

    # 1) ArUco 검출
    corners, ids, _ = _aruco_detector.detectMarkers(frame)
    all_ids = []
    worker_centers: dict[int, tuple[float, float]] = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for corner, mid in zip(corners, ids.flatten()):
            mid_int = int(mid)
            all_ids.append(mid_int)
            if mid_int in WORKER_ARUCO_MAP:
                pts = corner[0]
                worker_centers[mid_int] = (
                    float(pts[:, 0].mean()), float(pts[:, 1].mean())
                )

    # 2) YOLO person bbox
    results = pose_model.track(frame, conf=0.25, persist=True, verbose=False, classes=[0])
    bboxes: list[tuple[float, float, float, float]] = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [float(v) for v in box]
            bboxes.append((x1, y1, x2, y2))

    # 3) 마커 → 가장 가까운 미매칭 bbox 매칭 (greedy 1:1)
    bbox_to_wid: dict[int, str] = {}
    claimed = set()
    match_logs = []
    for mid_int in sorted(worker_centers.keys()):
        cx, cy = worker_centers[mid_int]
        best_idx = None
        best_dist = float("inf")
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            if idx in claimed:
                continue
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            max_dist = max(x2 - x1, y2 - y1)
            d = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
            if d <= max_dist and d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is not None:
            bbox_to_wid[best_idx] = WORKER_ARUCO_MAP[mid_int]
            claimed.add(best_idx)
            match_logs.append(f"{mid_int}→bbox#{best_idx}({best_dist:.0f}px)")
        else:
            match_logs.append(f"{mid_int}→FAIL")

    # 4) bbox + 라벨 그리기
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        wid = bbox_to_wid.get(idx, "?")
        color = (0, 255, 0) if wid != "?" else (0, 0, 255)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{wid}"
        cv2.putText(out, label, (int(x1), max(30, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(out, label, (int(x1), max(30, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    summary = (
        f"[{cam_id}] aruco_all={sorted(all_ids)}  "
        f"workers_seen={sorted(worker_centers.keys())}  "
        f"yolo_persons={len(bboxes)}  "
        f"matches=[{', '.join(match_logs) if match_logs else 'none'}]"
    )
    return out, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", choices=["cam1", "cam2"],
                        help="단일 카메라만 테스트 (생략 시 둘 다)")
    args = parser.parse_args()

    rtsp_1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp_2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp_1 or not rtsp_2:
        print("ERR: .env 의 CAMERA_RTSP_URL_1/2 필요")
        sys.exit(1)

    streams: dict[str, VideoStream] = {}
    if args.cam in (None, "cam1"):
        streams["cam1"] = VideoStream(rtsp_1).start()
    if args.cam in (None, "cam2"):
        streams["cam2"] = VideoStream(rtsp_2).start()
    print(f"카메라 연결: {list(streams)}  ([q] 종료)")

    try:
        while True:
            for cam_id, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue
                annotated, summary = process_frame(frame, cam_id)
                print(summary)
                cv2.imshow(cam_id, cv2.resize(annotated, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for s in streams.values():
            s.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
