"""YOLO + ArUco 추론 모듈.

모델 로드와 프레임 분석(process_frame)을 담당한다.
카메라 연결과 화면 표시에는 관여하지 않는다.
"""

import os

import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

# .env 로드
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── 모델 로드 ──────────────────────────────────────────────────────────
# BEST_MODEL_PATH가 상대 경로이면 프로젝트 루트 기준으로 변환
best_model_path = os.getenv("BEST_MODEL_PATH", os.getenv("best_model_path", ""))
if best_model_path and not os.path.isabs(best_model_path):
    best_model_path = os.path.join(PROJECT_ROOT, best_model_path)

pose_model = YOLO("yolo11n-pose.pt")       # 사람 포즈(관절) 감지
forklift_model = YOLO(best_model_path)      # 지게차 감지 (커스텀 학습 모델)

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


def process_frame(frame, track_to_worker: dict):
    """한 프레임에 대해 YOLO + ArUco 처리 후 annotated 프레임 반환.

    Args:
        frame: 카메라에서 읽은 numpy 배열 (H x W x 3)
        track_to_worker: 트래킹 ID -> 작업자 이름 매핑 딕셔너리 (호출마다 누적됨)

    Returns:
        annotated: 박스, 스켈레톤, 이름이 그려진 프레임
    """
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

    # 4. ArUco 마커 감지 -> 작업자 매핑
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
