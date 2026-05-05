"""Detection 결과 시각화.

bbox + 발 점 + 월드 좌표 라벨을 frame 위에 그린다.
"""

import cv2


# Type 별 BGR 색상.
_TYPE_COLORS = {
    "worker":   (0, 255, 255),  # 노랑
    "forklift": (0, 0, 255),    # 빨강
    "box_1":    (0, 200, 0),    # 초록
    "box_2":    (255, 0, 200),  # 자주
}

# Worker ID 가 어디서 부여됐는지 라벨에 표기 (디버깅용).
_ID_SOURCE_TAG = {
    "aruco": "*A",
    "track_persistence": "*T",
    "cross_camera": "*X",
}


def draw_annotated(frame, detections: list[dict]):
    """Detection 결과를 frame 위에 오버레이한 새 BGR 이미지 반환.

    Args:
        frame:       입력 BGR ndarray (원본 미변경)
        detections:  DetectionPipeline.extract() 출력 형식

    Returns:
        annotated:   bbox + 발 점 + 라벨이 그려진 BGR 이미지
    """
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_px"]]
        fx, fy = [int(v) for v in d["foot_px"]]
        c = _TYPE_COLORS.get(d["type"], (200, 200, 200))

        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.circle(out, (fx, fy), 10, (0, 0, 255), -1)
        cv2.circle(out, (fx, fy), 14, (255, 255, 255), 2)

        wx, wy = d["world"]["x"], d["world"]["y"]
        if d.get("type") == "worker":
            wid = d.get("worker_id") or "??"
            tag = _ID_SOURCE_TAG.get(d.get("id_source"), "")
            label = f"{wid}{tag} ({wx:.2f}, {wy:.2f})m"
        else:
            label = f'{d["type"]} ({wx:.2f}, {wy:.2f})m'

        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return out
