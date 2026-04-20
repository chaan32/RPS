"""YOLO + ArUco 카메라 모니터 실행 스크립트.

input/media/camera.py (카메라 연결) + input/media/detector.py (AI 추론)를
조합하여 실행한다. 서버(main.py)에서 subprocess로 이 파일을 실행한다.
"""

import os
import sys
import time

# 같은 디렉토리의 모듈을 import할 수 있도록 경로 추가
# (subprocess로 실행 시 작업 디렉토리가 프로젝트 루트이므로 필요)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from dotenv import load_dotenv

# .env 로드
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 작업 디렉토리를 프로젝트 루트로 변경 (모델 상대 경로 해결용)
os.chdir(PROJECT_ROOT)

from camera import VideoStream
from detector import process_frame

RTSP_1 = os.getenv("CAMERA_RTSP_URL_1", "0")
RTSP_2 = os.getenv("CAMERA_RTSP_URL_2", "")

# ── 카메라 연결 ────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("카메라 연결 시작")
print(f"{'='*50}")

# CAM 1
print(f"\n[CAM 1] URL: {RTSP_1}")
print("[CAM 1] 연결 시도 중...")
cam1 = VideoStream(RTSP_1).start()
if not cam1.stream.isOpened():
    print("[CAM 1] FAIL — stream.isOpened() = False")
else:
    ret, frame = cam1.read()
    if not ret or frame is None:
        print(f"[CAM 1] FAIL — 연결됐지만 프레임을 읽을 수 없음 (ret={ret}, frame={'None' if frame is None else 'OK'})")
    else:
        print(f"[CAM 1] OK — 해상도: {frame.shape[1]}x{frame.shape[0]}")

time.sleep(2)

# CAM 2
cam2 = None
if RTSP_2:
    print(f"\n[CAM 2] URL: {RTSP_2}")
    print("[CAM 2] 연결 시도 중...")
    cam2 = VideoStream(RTSP_2).start()
    if not cam2.stream.isOpened():
        print("[CAM 2] FAIL — stream.isOpened() = False")
        cam2 = None
    else:
        ret, frame = cam2.read()
        if not ret or frame is None:
            print(f"[CAM 2] FAIL — 연결됐지만 프레임을 읽을 수 없음 (ret={ret}, frame={'None' if frame is None else 'OK'})")
            cam2 = None
        else:
            print(f"[CAM 2] OK — 해상도: {frame.shape[1]}x{frame.shape[0]}")
else:
    print("\n[CAM 2] URL이 비어있어 건너뜁니다")

print(f"\n{'='*50}")
active = []
if cam1.stream.isOpened():
    active.append("CAM 1")
if cam2:
    active.append("CAM 2")
if not active:
    print("연결된 카메라가 없습니다. 종료합니다.")
    sys.exit(1)
print(f"활성 카메라: {', '.join(active)} (ESC로 종료)")
print(f"{'='*50}\n")

# ── 메인 루프 ──────────────────────────────────────────────────────────
track_to_worker_1 = {}
track_to_worker_2 = {}
cam1_fail_count = 0
cam2_fail_count = 0

while True:
    ret1, frame1 = cam1.read()
    if ret1 and frame1 is not None:
        cam1_fail_count = 0
        annotated1 = process_frame(frame1, track_to_worker_1)
        cv2.imshow("CAM 1 - YOLO + ArUco", annotated1)
    else:
        cam1_fail_count += 1
        if cam1_fail_count % 100 == 1:
            print(f"[CAM 1] 프레임 수신 실패 (연속 {cam1_fail_count}회)")

    if cam2:
        ret2, frame2 = cam2.read()
        if ret2 and frame2 is not None:
            cam2_fail_count = 0
            annotated2 = process_frame(frame2, track_to_worker_2)
            cv2.imshow("CAM 2 - YOLO + ArUco", annotated2)
        else:
            cam2_fail_count += 1
            if cam2_fail_count % 100 == 1:
                print(f"[CAM 2] 프레임 수신 실패 (연속 {cam2_fail_count}회)")

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cam1.stop()
if cam2:
    cam2.stop()
cv2.destroyAllWindows()
