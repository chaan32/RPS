import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
best_model_path = os.getenv("best_model_path")

pose_model = YOLO("yolo11n-pose.pt")
forklift_model = YOLO(best_model_path)

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

track_to_worker = {}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pose_results = pose_model.track(
        frame, conf=0.25, persist=True, verbose=False, classes=[0]
    )

    forklift_results = forklift_model(frame, conf=0.5, verbose=False)

    person_boxes = []
    if (pose_results[0].boxes is not None
            and pose_results[0].boxes.id is not None):
        xyxy = pose_results[0].boxes.xyxy.cpu().numpy()
        ids_t = pose_results[0].boxes.id.cpu().numpy().astype(int)
        for box, tid in zip(xyxy, ids_t):
            x1, y1, x2, y2 = box.astype(int)
            person_boxes.append((tid, x1, y1, x2, y2))

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

    annotated_frame = pose_results[0].plot()

    for box in forklift_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"forklift {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for tid, x1, y1, x2, y2 in person_boxes:
        name = track_to_worker.get(tid)
        if name is None:
            display_text = "Unknown"
            color = (128, 128, 128)
        else:
            display_text = name
            color = (0, 255, 255)

        cv2.putText(annotated_frame, display_text, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

    cv2.imshow("Pose + Forklift + ArUco", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()