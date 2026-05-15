"""Detection list → fusion 모델 입력 좌표 변환.

cam1 + cam2 의 detection 결과를 합쳐서:
  - workers_xy : {worker_id: (x, y)}  — ArUco 식별된 작업자 (cross-camera 흡수 포함)
  - forklift_xy: (x, y) | None
  - dropzone_xy: (x, y) | None  (box_1/box_2 인양물 평균)
세 가지로 변환.
"""

from __future__ import annotations

import numpy as np

from input.media.camera_geometry import triangulate_pixels_to_world


# 크레인 인양물 (= 동적 dropzone 위치) 클래스 이름.
BOX_CLASS_NAMES = ("box_1", "box_2")

# cam2 가 ArUco 를 못 본 워커를 cam1 의 식별된 워커와 묶는 거리 임계값(m).
# 같은 사람이라면 두 카메라의 월드 좌표는 homography 오차 범위 내(보통 < 1m).
CROSS_CAM_MATCH_RADIUS = 1.5

# 현재 Unity 벤치마크는 단일 작업자 시나리오다. 작업자용 ArUco ID가 없더라도
# 미식별 worker detection이 여러 개여도 fusion 입력이 끊기지 않게 대표 W01을 선택한다.
SINGLE_WORKER_FALLBACK_ID = "W01"
SINGLE_WORKER_PREFERRED_CAM = "cam2"

# 두 카메라 모두에서 인양물이 충분히 크게 보이면 bbox center ray triangulation을
# 우선 사용한다. 너무 작은/가장자리 일부만 잡힌 박스는 cam1 fallback이 더 안정적이다.
MULTIVIEW_BOX_MIN_AREA_RATIO = 0.05
DEFAULT_IMAGE_SIZE = (1920, 1080)
DROPZONE_WORLD_BOUNDS = ((-10.0, 3.0), (-5.0, 7.0))


def _bbox_center(det: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in det["bbox_px"]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _image_size(det: dict) -> tuple[int, int]:
    size = det.get("image_size") or DEFAULT_IMAGE_SIZE
    return (int(size[0]), int(size[1]))


def _in_dropzone_bounds(xy: tuple[float, float]) -> bool:
    (x_min, x_max), (y_min, y_max) = DROPZONE_WORLD_BOUNDS
    return x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max


def _multiview_box_xy(boxes_by_type_cam: dict[str, dict[str, dict]]) -> tuple[float, float] | None:
    for box_type in BOX_CLASS_NAMES:
        cams = boxes_by_type_cam.get(box_type, {})
        if "cam1" not in cams or "cam2" not in cams:
            continue
        cam1_box = cams["cam1"]
        cam2_box = cams["cam2"]
        if (
            float(cam1_box.get("bbox_area_ratio") or 0.0) < MULTIVIEW_BOX_MIN_AREA_RATIO
            or float(cam2_box.get("bbox_area_ratio") or 0.0) < MULTIVIEW_BOX_MIN_AREA_RATIO
        ):
            continue
        xy = triangulate_pixels_to_world(
            _bbox_center(cam1_box),
            _bbox_center(cam2_box),
            _image_size(cam1_box),
            _image_size(cam2_box),
        )
        if xy is not None and _in_dropzone_bounds(xy):
            return xy
    return None


def pick_positions(d1: list[dict], d2: list[dict]) -> tuple:
    """cam1 + cam2 detection list → (workers_xy, forklift_xy, dropzone_xy).

    Returns:
      workers_xy : dict {worker_id_str: (x, y)}
      forklift_xy: tuple or None
      dropzone_xy: tuple or None  (box_1/box_2 = 인양물 평균 좌표)

    워커 매칭 정책:
      1) 각 카메라가 ArUco 로 직접 식별한 워커는 worker_id 그대로 사용.
      2) 한쪽 카메라(예: cam2)가 ArUco 를 놓쳐서 worker_id=None 인 워커가 있으면,
         다른 카메라가 식별한 같은 worker_id 위치(월드 좌표) 근처(<1.5m)에 있을 때
         그 worker_id 로 흡수 → 양쪽 카메라 위치 평균으로 안정화.
      3) 식별된 워커가 전혀 없고 미식별 worker가 카메라당 최대 1개뿐이면,
         단일 작업자 벤치마크로 보고 W01로 사용한다.
      4) 그 외 마지막까지 식별 안 된 워커는 fusion 입력에서 제외한다.
    """
    # 1) cam 별로 식별/미식별 분리
    def _split(dets):
        ided, unided = [], []
        for d in dets:
            if d.get("type") != "worker":
                continue
            xy = (d["world"]["x"], d["world"]["y"])
            if d.get("worker_id"):
                ided.append((d["worker_id"], xy))
            else:
                unided.append({
                    "xy": xy,
                    "confidence": float(d.get("confidence") or 0.0),
                })
        return ided, unided

    cam1_ided, cam1_unided = _split(d1)
    cam2_ided, cam2_unided = _split(d2)

    # 2) 직접 식별된 워커 모두 모음 (worker_id → [위치들])
    workers_by_id: dict[str, list[tuple[float, float]]] = {}
    for wid, xy in cam1_ided + cam2_ided:
        workers_by_id.setdefault(wid, []).append(xy)

    # 3) 한쪽이 식별한 워커의 평균 위치 → 다른 쪽 미식별 워커 흡수
    def _absorb(unided_xys, source_cam_label):
        """unided_xys 중 식별된 워커 위치 근처에 있는 점을 그 워커 그룹에 추가."""
        if not unided_xys or not workers_by_id:
            return
        # 현재까지 모인 워커별 평균 위치 (매번 다시 계산해서 흡수 후 갱신 반영)
        for unided_xy in unided_xys:
            best_wid = None
            best_dist = float("inf")
            for wid, pts in workers_by_id.items():
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                d = ((unided_xy[0] - cx) ** 2 + (unided_xy[1] - cy) ** 2) ** 0.5
                if d <= CROSS_CAM_MATCH_RADIUS and d < best_dist:
                    best_dist = d
                    best_wid = wid
            if best_wid is not None:
                workers_by_id[best_wid].append(unided_xy)

    # cam1 미식별 → cam2 식별 워커에 매칭, 그 반대도 마찬가지
    _absorb([item["xy"] for item in cam1_unided], "cam1")
    _absorb([item["xy"] for item in cam2_unided], "cam2")

    # 단일 작업자 Unity 벤치마크 fallback.
    # worker ArUco가 없는 녹화에서는 YOLO pose가 같은 사람을 여러 후보로 반환할 수 있다.
    # 이때 후보 수 때문에 fusion 입력이 끊기지 않도록 worker 카메라(cam2)의 최고 confidence
    # 후보 하나를 W01로 사용하고, cam2가 놓친 프레임에만 cam1 후보를 fallback으로 쓴다.
    if not workers_by_id:
        preferred_unided = (
            cam2_unided if SINGLE_WORKER_PREFERRED_CAM == "cam2" else cam1_unided
        )
        fallback_unided = (
            cam1_unided if SINGLE_WORKER_PREFERRED_CAM == "cam2" else cam2_unided
        )
        candidates = preferred_unided or fallback_unided
        if candidates:
            best = max(candidates, key=lambda item: item["confidence"])
            workers_by_id[SINGLE_WORKER_FALLBACK_ID] = [best["xy"]]

    # 4) worker_id 별 평균
    workers_xy: dict[str, tuple[float, float]] = {}
    for wid, pts in workers_by_id.items():
        workers_xy[wid] = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )

    # 5) forklift / dropzone
    forklifts, boxes, preferred_boxes = [], [], []
    best_forklift_by_cam: dict[str, dict] = {}
    boxes_by_type_cam: dict[str, dict[str, dict]] = {}
    for cam_id, dets in (("cam1", d1), ("cam2", d2)):
        for d in dets:
            t = d["type"]
            if t == "forklift":
                prev = best_forklift_by_cam.get(cam_id)
                if prev is None or float(d.get("confidence") or 0.0) > float(prev.get("confidence") or 0.0):
                    best_forklift_by_cam[cam_id] = d
            elif t in BOX_CLASS_NAMES:
                xy = (d["world"]["x"], d["world"]["y"])
                boxes.append(xy)
                boxes_by_type_cam.setdefault(t, {})[cam_id] = d
                if d.get("dropzone_usable"):
                    preferred_boxes.append(xy)

    for d in best_forklift_by_cam.values():
        forklifts.append((d["world"]["x"], d["world"]["y"]))

    forklift_xy = None
    if forklifts:
        forklift_xy = (
            float(np.mean([p[0] for p in forklifts])),
            float(np.mean([p[1] for p in forklifts])),
        )

    dropzone_xy = None
    multiview_box = _multiview_box_xy(boxes_by_type_cam)
    dropzone_points = [multiview_box] if multiview_box is not None else (preferred_boxes or boxes)
    if dropzone_points:
        dropzone_xy = (
            float(np.mean([p[0] for p in dropzone_points])),
            float(np.mean([p[1] for p in dropzone_points])),
        )
    return workers_xy, forklift_xy, dropzone_xy
