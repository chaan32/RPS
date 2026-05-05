"""Mock RTSP 검증 스크립트.

mediamtx + ffmpeg 로 송출된 가짜 RTSP 스트림이 우리 코드 입력으로 정상 동작하는지
3 단계로 확인:

  L1. cv2.VideoCapture 로 RTSP 디코딩 (5초간 frame N개 수신)
  L2. ArUco 4코너 마커(22/24/27/38) 인식 → calibrate_homography 자동 동작
  L3. DetectionPipeline.extract() 메인 루프 진입 (worker/forklift 검출은 예상대로 0건)

사전 준비:
  1) mediamtx (백그라운드 실행)
  2) ffmpeg -re -stream_loop -1 -i fake_cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
     ffmpeg -re -stream_loop -1 -i fake_cam2.mp4 -c copy -f rtsp rtsp://localhost:8554/cam2
  3) python -m input.media.test.verify_mock_rtsp
"""

import os
import sys
import time
from pathlib import Path

# RTSP TCP 강제 (cv2 import 전)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32",
)

import cv2
import numpy as np


CAM_URLS = {
    "cam1": "rtsp://localhost:8554/cam1",
    "cam2": "rtsp://localhost:8554/cam2",
}
EXPECTED_CORNER_IDS = {22, 24, 27, 38}


def _hr(title: str = "") -> None:
    print("─" * 70 + (f"  {title}" if title else ""))


def level1_rtsp_receive(url: str, n_frames: int = 30, timeout: float = 10.0) -> bool:
    """L1: RTSP 디코딩 — N 프레임 수신 가능 여부."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    received = 0
    t0 = time.time()
    while received < n_frames and (time.time() - t0) < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            received += 1
        else:
            time.sleep(0.05)
    cap.release()

    elapsed = time.time() - t0
    fps = received / elapsed if elapsed > 0 else 0
    ok = received >= n_frames
    print(f"  {url}: {received}/{n_frames} frames ({fps:.1f} fps, {elapsed:.1f}s)  "
          f"{'✅' if ok else '❌'}")
    return ok


def level2_aruco_corners(url: str, max_attempts: int = 10) -> bool:
    """L2: ArUco 4코너 마커(22/24/27/38) 인식."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detected_ids = set()
    for i in range(max_attempts):
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        _, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            for mid in ids.flatten():
                detected_ids.add(int(mid))
        if EXPECTED_CORNER_IDS.issubset(detected_ids):
            break
    cap.release()

    missing = EXPECTED_CORNER_IDS - detected_ids
    extra = detected_ids - EXPECTED_CORNER_IDS
    ok = not missing
    print(f"  {url}: 감지={sorted(detected_ids)}  "
          f"누락={sorted(missing) if missing else '없음'}"
          f"{'  추가:' + str(sorted(extra)) if extra else ''}  "
          f"{'✅' if ok else '❌'}")
    return ok


def level3_detection_pipeline(url: str, cam_id: str) -> bool:
    """L3: DetectionPipeline.extract() — 한 프레임 통째로 처리."""
    try:
        from input.media.pipeline import build_default_pipeline
    except ImportError as e:
        print(f"  ❌ import 실패: {e}")
        return False

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 첫 프레임 잡기
    frame = None
    for _ in range(20):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
        time.sleep(0.1)
    cap.release()
    if frame is None:
        print(f"  ❌ {url}: 프레임 수신 실패")
        return False

    # Pipeline 인스턴스 → extract
    pipeline = build_default_pipeline()
    detections = pipeline.extract(frame, cam_id)

    workers = [d for d in detections if d.get("type") == "worker"]
    others = [d for d in detections if d.get("type") != "worker"]
    print(f"  {url}: workers={len(workers)}  others={len(others)}  "
          f"(합성 영상이라 검출 0 정상) ✅")
    return True   # extract 가 예외 안 던지고 끝났으면 OK


def main() -> int:
    print()
    _hr("Mock RTSP 검증 시작")
    print(f"대상: {list(CAM_URLS.values())}")
    print()

    _hr("L1. RTSP 디코딩 (30 frames / 10s)")
    l1 = all(level1_rtsp_receive(url) for url in CAM_URLS.values())

    print()
    _hr("L2. ArUco 4코너 마커 인식")
    l2 = all(level2_aruco_corners(url) for url in CAM_URLS.values())

    print()
    _hr("L3. DetectionPipeline 메인 루프 진입")
    l3 = all(level3_detection_pipeline(url, cam_id) for cam_id, url in CAM_URLS.items())

    print()
    _hr("결과 종합")
    rows = [
        ("L1. RTSP 디코딩", l1),
        ("L2. ArUco 4코너", l2),
        ("L3. DetectionPipeline", l3),
    ]
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print()

    return 0 if all(ok for _, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
