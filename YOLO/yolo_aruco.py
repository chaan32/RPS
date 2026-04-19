"""YOLO Pose + Forklift Detection + ArUco 작업자 식별 — 듀얼 캠 지원.

.env의 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2 를 입력으로 사용한다.
각 카메라를 별도 스레드에서 읽고, 메인 스레드에서 YOLO 추론 + 화면 표시를 수행한다.
"""

import cv2
import os
import threading
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── 모델 로드 ──────────────────────────────────────────────────────────
best_model_path = os.getenv("BEST_MODEL_PATH", os.getenv("best_model_path", ""))
pose_model = YOLO("yolo11n-pose.pt")
forklift_model = YOLO(best_model_path)

# ── ArUco 설정 ─────────────────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

worker_map = {
    0: "Worker1",
    22: "Worker2",
    24: "Worker3",
    27: "Worker4",
    38: "Worker5",
}

# ── RTSP 카메라 스트림 ─────────────────────────────────────────────────
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32",
)

RTSP_1 = os.getenv("CAMERA_RTSP_URL_1", "0")
RTSP_2 = os.getenv("CAMERA_RTSP_URL_2", "")


class VideoStream:
    """백그라운드 스레드에서 프레임을 계속 갱신하는 스트림."""
    def __init__(self, src):
        if src.isdigit():
            self.stream = cv2.VideoCapture(int(src))
        else:
            self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.stream.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.stream.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False

    def start(self):
        # 첫 프레임을 최대 5초간 재시도
        import time as _time
        for _ in range(50):
            self.ret, self.frame = self.stream.read()
            if self.ret and self.frame is not None:
                break
            _time.sleep(0.1)
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
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self._stopped = True
        self.stream.release()


def process_frame(frame, track_to_worker):
    """한 프레임에 대해 YOLO + ArUco 처리 후 annotated 프레임 반환."""
    # 1. 사람 포즈 트래킹
    pose_results = pose_model.track(
        frame, conf=0.25, persist=True, verbose=False, classes=[0]
    )

    # 2. 지게차 감지
    forklift_results = forklift_model(frame, conf=0.5, verbose=False)

    # 3. 사람 박스 추출
    person_boxes = []
    if (pose_results[0].boxes is not None
            and pose_results[0].boxes.id is not None):
        xyxy = pose_results[0].boxes.xyxy.cpu().numpy()
        ids_t = pose_results[0].boxes.id.cpu().numpy().astype(int)
        for box, tid in zip(xyxy, ids_t):
            x1, y1, x2, y2 = box.astype(int)
            person_boxes.append((tid, x1, y1, x2, y2))

    # 4. ArUco 마커 감지 → 작업자 매핑
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in worker_map:
                continue
            pts = marker_corners[0]
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            for tid, x1, y1, x2, y2 in person_boxes:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    track_to_worker[tid] = worker_map[marker_id]
                    break

    # 5. 시각화
    annotated = pose_results[0].plot()

    for box in forklift_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"forklift {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for tid, x1, y1, x2, y2 in person_boxes:
        name = track_to_worker.get(tid)
        if name is None:
            display_text = "Unknown"
            color = (128, 128, 128)
        else:
            display_text = name
            color = (0, 255, 255)
        cv2.putText(annotated, display_text, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    return annotated


# ── 메인 루프 ──────────────────────────────────────────────────────────
import time

print(f"\n{'='*50}")
print("카메라 연결 시작")
print(f"{'='*50}")

# CAM 1
print(f"\n[CAM 1] URL: {RTSP_1}")
print("[CAM 1] 연결 시도 중...")
cam1 = VideoStream(RTSP_1).start()
if not cam1.stream.isOpened():
    print(f"[CAM 1] FAIL — stream.isOpened() = False")
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
        print(f"[CAM 2] FAIL — stream.isOpened() = False")
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
    exit(1)
print(f"활성 카메라: {', '.join(active)} (ESC로 종료)")
print(f"{'='*50}\n")

# 각 카메라별 트래킹 상태를 분리
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
