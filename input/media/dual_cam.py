"""두 대의 RTSP 카메라를 하나의 창에 나란히 표시하는 스크립트."""

import os
import sys
import threading

import cv2
import numpy as np
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32",
)

RTSP_1 = os.getenv("CAMERA_RTSP_URL_1")
RTSP_2 = os.getenv("CAMERA_RTSP_URL_2")

if not RTSP_1 or not RTSP_2:
    print("❌ .env에 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2를 설정해주세요.")
    sys.exit(1)


class VideoStream:
    def __init__(self, name, url):
        self.name = name
        self.stream = cv2.VideoCapture(url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self._stopped = False
        self._lock = threading.Lock()

    def start(self):
        self.ret, self.frame = self.stream.read()
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self._stopped:
            ret, frame = self.stream.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, self.frame

    def stop(self):
        self._stopped = True
        self.stream.release()


# 카메라 연결
print("카메라 2대 연결 중...")

cam1 = VideoStream("CAM 1", RTSP_1).start()
cam2 = VideoStream("CAM 2", RTSP_2).start()

if not cam1.stream.isOpened():
    print(f"❌ CAM 1 연결 실패: {RTSP_1}")
    sys.exit(1)
if not cam2.stream.isOpened():
    print(f"❌ CAM 2 연결 실패: {RTSP_2}")
    sys.exit(1)

print("✅ 두 카메라 모두 연결 성공! (q 키로 종료)")

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # 프레임이 아직 안 들어왔으면 검은 화면 표시
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

    # 두 프레임을 가로로 이어붙이기
    combined = np.hstack((frame1, frame2))

    cv2.imshow("Dual Camera View", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam1.stop()
cam2.stop()
cv2.destroyAllWindows()
