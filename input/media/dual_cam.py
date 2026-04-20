"""두 대의 RTSP 카메라를 YOLO 없이 하나의 창에 나란히 표시하는 스크립트."""

import os
import sys

# 같은 디렉토리의 모듈을 import할 수 있도록 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from dotenv import load_dotenv

from camera import VideoStream

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

RTSP_1 = os.getenv("CAMERA_RTSP_URL_1")
RTSP_2 = os.getenv("CAMERA_RTSP_URL_2")

if not RTSP_1 or not RTSP_2:
    print("ERR: .env에 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2를 설정해주세요.")
    sys.exit(1)

# 카메라 연결
print("카메라 2대 연결 중...")

cam1 = VideoStream(RTSP_1).start()
cam2 = VideoStream(RTSP_2).start()

if not cam1.stream.isOpened():
    print(f"CAM 1 연결 실패: {RTSP_1}")
    sys.exit(1)
if not cam2.stream.isOpened():
    print(f"CAM 2 연결 실패: {RTSP_2}")
    sys.exit(1)

print("두 카메라 모두 연결 성공! (q 키로 종료)")

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or frame1 is None:
        frame1 = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame1, "CAM 1 - No Signal", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        frame1 = cv2.resize(frame1, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.putText(frame1, "CAM 1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if not ret2 or frame2 is None:
        frame2 = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame2, "CAM 2 - No Signal", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        frame2 = cv2.resize(frame2, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.putText(frame2, "CAM 2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    combined = np.hstack((frame1, frame2))
    cv2.imshow("Dual Camera View", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam1.stop()
cam2.stop()
cv2.destroyAllWindows()
