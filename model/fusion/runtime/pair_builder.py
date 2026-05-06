"""Detection list → fusion 모델 입력 좌표 변환.

cam1 + cam2 의 detection 결과를 합쳐서:
  - workers_xy : {worker_id: (x, y)}  — ArUco 식별된 작업자 (cross-camera 흡수 포함)
  - forklift_xy: (x, y) | None
  - dropzone_xy: (x, y) | None  (box_1/box_2 인양물 평균)
세 가지로 변환.
"""

from __future__ import annotations

import numpy as np


# 크레인 인양물 (= 동적 dropzone 위치) 클래스 이름.
BOX_CLASS_NAMES = ("box_1", "box_2")

# cam2 가 ArUco 를 못 본 워커를 cam1 의 식별된 워커와 묶는 거리 임계값(m).
# 같은 사람이라면 두 카메라의 월드 좌표는 homography 오차 범위 내(보통 < 1m).
CROSS_CAM_MATCH_RADIUS = 1.5


def pick_positions(d1: list[dict], d2: list[dict]) -> tuple:
    """cam1 + cam2 detection list → (workers_xy, forklift_xy, dropzone_xy).

    Returns:
      workers_xy : dict {worker_id_str: (x, y)}  — ArUco 식별된 작업자만
      forklift_xy: tuple or None
      dropzone_xy: tuple or None  (box_1/box_2 = 인양물 평균 좌표)

    워커 매칭 정책:
      1) 각 카메라가 ArUco 로 직접 식별한 워커는 worker_id 그대로 사용.
      2) 한쪽 카메라(예: cam2)가 ArUco 를 놓쳐서 worker_id=None 인 워커가 있으면,
         다른 카메라가 식별한 같은 worker_id 위치(월드 좌표) 근처(<1.5m)에 있을 때
         그 worker_id 로 흡수 → 양쪽 카메라 위치 평균으로 안정화.
      3) 마지막까지 식별 안 된 워커는 fusion 입력에서 제외 (track_id 없이는 위험 평가
         단위가 흔들리기 때문).
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
                unided.append(xy)
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
    _absorb(cam1_unided, "cam1")
    _absorb(cam2_unided, "cam2")

    # 4) worker_id 별 평균
    workers_xy: dict[str, tuple[float, float]] = {}
    for wid, pts in workers_by_id.items():
        workers_xy[wid] = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )

    # 5) forklift / dropzone (기존 로직)
    forklifts, boxes = [], []
    for d in d1 + d2:
        t = d["type"]
        if t == "forklift":
            forklifts.append((d["world"]["x"], d["world"]["y"]))
        elif t in BOX_CLASS_NAMES:
            boxes.append((d["world"]["x"], d["world"]["y"]))

    forklift_xy = None
    if forklifts:
        forklift_xy = (
            float(np.mean([p[0] for p in forklifts])),
            float(np.mean([p[1] for p in forklifts])),
        )

    dropzone_xy = None
    if boxes:
        dropzone_xy = (
            float(np.mean([p[0] for p in boxes])),
            float(np.mean([p[1] for p in boxes])),
        )
    return workers_xy, forklift_xy, dropzone_xy
