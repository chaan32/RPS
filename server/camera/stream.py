"""RTSP 카메라 스트림 관리 모듈.

서버 시작 시 .env에 등록된 카메라들을 백그라운드 스레드로 연결하고,
최신 프레임을 언제든 가져올 수 있도록 관리한다.
"""

import logging
import os
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    """단일 RTSP 카메라 스트림. 별도 스레드에서 프레임을 계속 갱신한다."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.stream: cv2.VideoCapture | None = None
        self.ret = False
        self.frame = None
        self._stopped = True
        self._lock = threading.Lock()

    def start(self):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32",
        )
        self.stream = cv2.VideoCapture(self.url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            logger.error("[%s] 연결 실패: %s", self.name, self.url)
            return False

        self.ret, self.frame = self.stream.read()
        self._stopped = False
        threading.Thread(target=self._update, daemon=True).start()
        logger.info("[%s] 스트림 시작됨", self.name)
        return True

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
        if self.stream:
            self.stream.release()
        logger.info("[%s] 스트림 종료됨", self.name)


class CameraManager:
    """여러 카메라를 관리하는 매니저."""

    def __init__(self):
        self.streams: dict[str, VideoStream] = {}
        self._display_stopped = True

    def start_all(self):
        """환경변수에서 CAMERA_RTSP_URL_* 패턴의 카메라를 모두 시작한다."""
        idx = 1
        while True:
            url = os.getenv(f"CAMERA_RTSP_URL_{idx}")
            if not url:
                break
            name = f"cam_{idx}"
            vs = VideoStream(name, url)
            if vs.start():
                self.streams[name] = vs
            idx += 1

        logger.info("카메라 %d대 연결 완료", len(self.streams))

    def start_display(self):
        """두 카메라를 하나의 창에 나란히 보여주는 스레드를 시작한다."""
        if not self.streams:
            return
        self._display_stopped = False
        threading.Thread(target=self._display_loop, daemon=True).start()
        logger.info("카메라 디스플레이 창 시작됨")

    def _display_loop(self):
        DISPLAY_WIDTH = 640
        DISPLAY_HEIGHT = 480

        while not self._display_stopped:
            frames = []
            for name in sorted(self.streams.keys()):
                ret, frame = self.get_frame(name)
                if not ret or frame is None:
                    f = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(f, f"{name} - No Signal", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    f = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    cv2.putText(f, name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                frames.append(f)

            combined = np.hstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow("Camera Monitor", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def stop_all(self):
        self._display_stopped = True
        for vs in self.streams.values():
            vs.stop()
        self.streams.clear()

    def get_frame(self, name: str):
        vs = self.streams.get(name)
        if not vs:
            return False, None
        return vs.read()

    def list_cameras(self) -> list[str]:
        return list(self.streams.keys())


camera_manager = CameraManager()
