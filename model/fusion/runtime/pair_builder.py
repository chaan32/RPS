"""Detection list → fusion 모델 입력 좌표 변환.

cam1 + cam2 의 detection 결과를 합쳐서:
  - workers_xy : {worker_id: (x, y)}  — ArUco 식별된 작업자 (cross-camera 흡수 포함)
  - forklift_xy: (x, y) | None
  - dropzone_xy: (x, y) | None  (box_1/box_2 인양물 평균)
세 가지로 변환.
"""

from __future__ import annotations

import os
import time

import numpy as np

from input.media.camera_geometry import triangulate_pixels_to_world


# 크레인 인양물 (= 동적 dropzone 위치) 클래스 이름.
BOX_CLASS_NAMES = ("box_1", "box_2")

# cam2 가 ArUco 를 못 본 워커를 cam1 의 식별된 워커와 묶는 거리 임계값(m).
# 같은 사람이라면 두 카메라의 월드 좌표는 homography 오차 범위 내(보통 < 1m).
CROSS_CAM_MATCH_RADIUS = 1.5

# 작업자용 ArUco ID가 없는 Unity 시나리오에서는 pose 검출 좌표를 클러스터링하고
# 짧은 시간 동안 유지되는 익명 worker_id(W01/W02)를 부여한다.
ANON_WORKER_MAX = int(os.getenv("ANON_WORKER_MAX", "2"))
ANON_CLUSTER_RADIUS_M = float(os.getenv("ANON_WORKER_CLUSTER_RADIUS_M", "1.15"))
ANON_ID_MATCH_RADIUS_M = float(os.getenv("ANON_WORKER_ID_MATCH_RADIUS_M", "1.80"))
ANON_TRACK_TTL_S = float(os.getenv("ANON_WORKER_TRACK_TTL_S", "1.50"))

# 두 카메라 모두에서 인양물이 충분히 크게 보이면 bbox center ray triangulation을
# 우선 사용한다. 너무 작은/가장자리 일부만 잡힌 박스는 cam1 fallback이 더 안정적이다.
MULTIVIEW_BOX_MIN_AREA_RATIO = 0.05
DEFAULT_IMAGE_SIZE = (1920, 1080)
DROPZONE_WORLD_BOUNDS = ((-10.0, 3.0), (-5.0, 7.0))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _mean_xy(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        float(np.mean([p[0] for p in points])),
        float(np.mean([p[1] for p in points])),
    )


class AnonymousWorkerAssigner:
    """Assign stable W01/W02 IDs to workers without body-mounted ArUco markers."""

    def __init__(
        self,
        max_workers: int = ANON_WORKER_MAX,
        match_radius_m: float = ANON_ID_MATCH_RADIUS_M,
        ttl_s: float = ANON_TRACK_TTL_S,
    ):
        self.max_workers = max_workers
        self.match_radius_m = match_radius_m
        self.ttl_s = ttl_s
        self.tracks: dict[str, dict[str, object]] = {}

    def observe_named(self, workers_xy: dict[str, tuple[float, float]]) -> None:
        """Keep anonymous ID state aligned when an ID was resolved upstream."""
        now = time.time()
        for worker_id, xy in workers_xy.items():
            if worker_id.startswith("W"):
                self.tracks[worker_id] = {"xy": xy, "ts": now}

    def assign(
        self,
        candidates: list[dict],
        reserved_ids: set[str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return stable IDs for anonymous worker candidate centroids."""
        now = time.time()
        reserved_ids = reserved_ids or set()
        self._prune(now)

        assigned: dict[str, tuple[float, float]] = {}
        used_tracks: set[str] = set()
        candidates = sorted(
            candidates,
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )[: self.max_workers]

        for candidate in candidates:
            xy = candidate["xy"]
            best_id = None
            best_dist = float("inf")
            for worker_id, track in self.tracks.items():
                if worker_id in reserved_ids or worker_id in used_tracks:
                    continue
                track_xy = track["xy"]
                dist = _dist(xy, track_xy)
                if dist <= self.match_radius_m and dist < best_dist:
                    best_dist = dist
                    best_id = worker_id

            if best_id is None:
                # A stale or fast-moving track must not block a currently visible
                # worker.  Reuse an unmatched anonymous ID if every existing track
                # is outside the match radius; missing a visible worker is worse
                # than a temporary W01/W02 identity swap in the Unity benchmark.
                best_id = self._next_available_id(
                    reserved_ids | used_tracks | set(assigned)
                )
            if best_id is None:
                continue

            assigned[best_id] = xy
            used_tracks.add(best_id)
            self.tracks[best_id] = {"xy": xy, "ts": now}

        return assigned

    def _next_available_id(self, blocked: set[str]) -> str | None:
        for idx in range(1, self.max_workers + 1):
            worker_id = f"W{idx:02d}"
            if worker_id not in blocked:
                return worker_id
        return None

    def _prune(self, now: float) -> None:
        stale = [
            worker_id
            for worker_id, track in self.tracks.items()
            if now - float(track.get("ts") or 0.0) > self.ttl_s
        ]
        for worker_id in stale:
            self.tracks.pop(worker_id, None)


_ANON_ASSIGNER = AnonymousWorkerAssigner()


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


def _cluster_anonymous_workers(candidates: list[dict]) -> list[dict]:
    """Merge cam1/cam2 anonymous detections that refer to the same person."""
    clusters: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        xy = candidate["xy"]
        best_cluster = None
        best_dist = float("inf")
        for cluster in clusters:
            dist = _dist(xy, cluster["xy"])
            if dist <= ANON_CLUSTER_RADIUS_M and dist < best_dist:
                best_dist = dist
                best_cluster = cluster

        if best_cluster is None:
            clusters.append({
                "points": [xy],
                "xy": xy,
                "confidence": candidate["confidence"],
            })
            continue

        best_cluster["points"].append(xy)
        best_cluster["xy"] = _mean_xy(best_cluster["points"])
        best_cluster["confidence"] = max(best_cluster["confidence"], candidate["confidence"])

    return clusters


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
      3) 끝까지 식별 안 된 워커는 cam1/cam2 좌표를 클러스터링한 뒤 W01/W02 익명 ID를
         부여한다. 이 ID는 짧은 시간 동안 유지되어 BEV/fusion/DB 기록이 끊기지 않는다.
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
    def _absorb(unided_items):
        """식별된 워커 위치 근처의 미식별 점은 같은 worker_id로 흡수한다."""
        leftovers = []
        unided_xys = [item["xy"] for item in unided_items]
        if not unided_xys or not workers_by_id:
            return list(unided_items)
        # 현재까지 모인 워커별 평균 위치 (매번 다시 계산해서 흡수 후 갱신 반영)
        for item in unided_items:
            unided_xy = item["xy"]
            best_wid = None
            best_dist = float("inf")
            for wid, pts in workers_by_id.items():
                centroid = _mean_xy(pts)
                d = _dist(unided_xy, centroid)
                if d <= CROSS_CAM_MATCH_RADIUS and d < best_dist:
                    best_dist = d
                    best_wid = wid
            if best_wid is not None:
                workers_by_id[best_wid].append(unided_xy)
            else:
                leftovers.append(item)
        return leftovers

    # cam1 미식별 → cam2 식별 워커에 매칭, 그 반대도 마찬가지
    anonymous_candidates = _absorb(cam1_unided) + _absorb(cam2_unided)
    if anonymous_candidates:
        clusters = _cluster_anonymous_workers(anonymous_candidates)
        assigned = _ANON_ASSIGNER.assign(clusters, reserved_ids=set(workers_by_id))
        for wid, xy in assigned.items():
            workers_by_id.setdefault(wid, []).append(xy)

    # 4) worker_id 별 평균
    workers_xy: dict[str, tuple[float, float]] = {}
    for wid, pts in workers_by_id.items():
        workers_xy[wid] = _mean_xy(pts)
    _ANON_ASSIGNER.observe_named(workers_xy)

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
