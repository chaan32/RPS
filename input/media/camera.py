"""카메라 RTSP 연결 및 프레임 수집 모듈.

VideoStream: 백그라운드 스레드에서 프레임을 계속 갱신하는 스트림 클래스.
여러 파일(run_yolo.py, dual_cam.py 등)에서 공용으로 사용한다.
"""

import os
import threading
import time

import cv2

# 버퍼링 하지 않고 최대한 실시간으로 전송받기 위한 설정
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32",
)


class VideoStream:
    """백그라운드 스레드에서 프레임을 계속 갱신하는 스트림.

    __init__("rtsp://...")     <- 카메라 파이프 연결
        |
    start()                    <- 첫 프레임 대기 + 백그라운드 스레드 시작
        |
        +-- _update() [스레드]  <- 무한 반복: 프레임 계속 갱신
        |
        +-- read() [메인]       <- YOLO 돌리기 전에 최신 프레임 가져감
        |
    stop()                     <- 스레드 종료 + 연결 해제
    """

    def __init__(self, src: str):
        self.src = src
        self.stream = self._open(src)
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False

    def _open(self, src: str):
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        """첫 프레임을 최대 5초간 기다린 뒤, 백그라운드 스레드를 시작한다."""
        for _ in range(50):
            try:
                self.ret, self.frame = self.stream.read()
            except cv2.error:
                self.ret, self.frame = False, None
            if self.ret and self.frame is not None:
                break
            time.sleep(0.1)
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        """무한 반복하며 카메라에서 프레임을 계속 읽어 self.frame을 갱신.

        RTSP는 패킷 손실/디코더 오류로 cv2.read() 가 C++ 예외를 던질 수 있다.
        그대로 두면 워커 스레드가 죽고 프레임 갱신이 영구히 멎으므로,
        예외/연속 실패를 잡아 짧은 sleep 후 재시도하고, 오래 끌면 재오픈한다.
        """
        consecutive_fail = 0
        REOPEN_AFTER = 50  # 약 5초 (sleep 0.1)
        while not self._stopped:
            try:
                ret, frame = self.stream.read()
            except cv2.error as e:
                ret, frame = False, None
                if consecutive_fail % 20 == 0:
                    print(f"[VideoStream] cv2.error on read ({consecutive_fail}회): {e}")
            except Exception as e:
                ret, frame = False, None
                if consecutive_fail % 20 == 0:
                    print(f"[VideoStream] read error ({consecutive_fail}회): {e}")

            with self._lock:
                self.ret = ret
                self.frame = frame

            if not ret or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= REOPEN_AFTER:
                    print(f"[VideoStream] {self.src} 재오픈 시도")
                    try:
                        self.stream.release()
                    except Exception:
                        pass
                    try:
                        self.stream = self._open(self.src)
                    except Exception as e:
                        print(f"[VideoStream] 재오픈 실패: {e}")
                    consecutive_fail = 0
                time.sleep(0.1)
            else:
                consecutive_fail = 0

    def read(self):
        """메인 스레드에서 최신 프레임을 가져간다."""
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        """스레드 종료 + 카메라 연결 해제."""
        self._stopped = True
        self.stream.release()


# ── API용 카메라 매니저 (현재 웹에서 미사용 — 필요 시 활성화) ─────────

import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """여러 카메라를 관리하는 매니저. 서버 API에서 스냅샷/스트리밍에 사용."""

    def __init__(self):
        self.streams: dict[str, VideoStream] = {}

    def start_all(self):
        """환경변수에서 CAMERA_RTSP_URL_* 패턴의 카메라를 모두 시작한다."""
        idx = 1
        while True:
            url = os.getenv(f"CAMERA_RTSP_URL_{idx}")
            if not url:
                break
            name = f"cam_{idx}"
            vs = VideoStream(url)
            vs.start()
            if vs.stream.isOpened():
                self.streams[name] = vs
            else:
                logger.error("[%s] 연결 실패: %s", name, url)
            idx += 1
        logger.info("카메라 %d대 연결 완료", len(self.streams))

    def stop_all(self):
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
