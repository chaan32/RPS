"""DetectionPipeline — 프레임 → detection list 변환 엔진.

이 클래스 한 개에 다음을 캡슐화한다:
  · YOLO 모델 (pose + custom)
  · ArUco detector
  · Worker ID persistence 상태 (시간축 + 카메라축)
  · extract / cross_camera_propagate / reset_state 메서드

기존 world_pipeline.py 의 모듈 전역 글로벌 변수들을 모두 인스턴스 멤버로 옮겼으므로
한 프로세스에서 여러 인스턴스를 띄울 수도 있고, 단위 테스트에서 mock 도 쉽다.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from pathlib import Path

import cv2
from ultralytics import YOLO

from ..camera_geometry import pixel_to_world_on_unity_y_plane
from ..world_mapper import pixel_to_world
from .constants import (
    CUSTOM_OBJECT_CLASS_NAMES,
    FORKLIFT_REF_X_RATIO,
    FORKLIFT_REF_Y_RATIO,
    KPT_CONF_THRESHOLD,
    LEFT_ANKLE,
    LIFTED_BOX_CENTER_UNITY_Y,
    LIFTED_BOX_CLASS_NAMES,
    LIFTED_BOX_PRIMARY_CAM_ID,
    POSE_CONF_THRESHOLD,
    RIGHT_ANKLE,
    WORKER_ARUCO_MAP,
    WORKER_STATE_TTL_S,
    WORKER_WORLD_BOUNDS,
    WORLD_MATCH_RADIUS_M,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _parse_worker_world_bounds(
    value: str | None,
) -> tuple[float, float, float, float] | None:
    """Parse worker world-bounds override from env.

    The original Unity benchmark used a hard-coded worker lane. New scenes can
    put workers outside that lane, so diagnostics/runtime can disable the filter
    with WORKER_WORLD_BOUNDS=none or replace it with x_min,x_max,y_min,y_max.
    """
    if value is None or value.strip() == "":
        return WORKER_WORLD_BOUNDS

    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "WORKER_WORLD_BOUNDS must be 'none' or 'x_min,x_max,y_min,y_max'"
        )
    try:
        min_x, max_x, min_y, max_y = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "WORKER_WORLD_BOUNDS values must be numeric: x_min,x_max,y_min,y_max"
        ) from exc
    return min_x, max_x, min_y, max_y


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment setting for YOLO image size."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class DetectionPipeline:
    """프레임 → detection list 엔진.

    Args:
        pose_model_path:    YOLO11n-pose 가중치 (사람 + 17 keypoint)
        custom_model_path:  forklift / box_1 / box_2 검출 가중치 (None 이면 person 만)
        worker_aruco_map:   ArUco ID → worker_id 매핑 (default WORKER_ARUCO_MAP)
        world_match_radius_m / worker_state_ttl_s:
                            cross-camera 전파 파라미터
        debug_aruco:        매 프레임 ArUco 감지 ID 콘솔 출력
    """

    def __init__(
        self,
        pose_model_path: str = "yolo11n-pose.pt",
        custom_model_path: str | None = None,
        worker_aruco_map: dict[int, str] | None = None,
        world_match_radius_m: float = WORLD_MATCH_RADIUS_M,
        worker_state_ttl_s: float = WORKER_STATE_TTL_S,
        debug_aruco: bool = False,
        load_pose: bool = True,
        load_custom: bool = True,
    ):
        # ── 모델 로드 ──
        self.pose_model = YOLO(pose_model_path) if load_pose else None

        self.custom_model = None
        self.custom_names: dict[int, str] = {}
        if custom_model_path and load_custom:
            path = custom_model_path
            if not os.path.isabs(path):
                path = str(PROJECT_ROOT / path)
            if os.path.exists(path):
                self.custom_model = YOLO(path)
                self.custom_names = self.custom_model.names
                print(f"[init] 커스텀 모델 로드: {path}")
                print(f"[init] 클래스: {self.custom_names}")
            else:
                print(f"[init] 커스텀 모델 경로 없음: {path}")
        else:
            print("[init] 커스텀 모델 미지정 — person 감지만 수행")

        # ── ArUco detector ──
        self.aruco_detector = self._build_aruco_detector()

        # ── 정책 파라미터 ──
        self.worker_aruco_map = worker_aruco_map or WORKER_ARUCO_MAP
        self.world_match_radius = world_match_radius_m
        self.worker_state_ttl = worker_state_ttl_s
        self.debug_aruco = debug_aruco
        self.worker_world_bounds = _parse_worker_world_bounds(
            os.getenv("WORKER_WORLD_BOUNDS")
        )
        default_imgsz = _parse_positive_int_env("YOLO_IMGSZ", 640)
        self.pose_imgsz = _parse_positive_int_env("POSE_IMGSZ", default_imgsz)
        self.custom_imgsz = _parse_positive_int_env("CUSTOM_IMGSZ", default_imgsz)
        self.pose_every_n_frames = _parse_positive_int_env("POSE_EVERY_N_FRAMES", 1)
        self.pose_skip_max_extrapolate_s = float(
            os.getenv("POSE_SKIP_MAX_EXTRAPOLATE_S", "0.5")
        )

        # ── State (인스턴스 캡슐화) ──
        # {cam_id: {track_id: worker_id}}
        self.cam_track_to_worker: dict[str, dict[int, str]] = {}
        # {worker_id: {"world": (x, y), "ts": float, "by_cam": str}}
        self.worker_world_state: dict[str, dict] = {}
        # {cam_id: frame sequence}. Used only when pose inference is skipped.
        self.pose_frame_seq_by_cam: dict[str, int] = {}
        # {cam_id: {"current": entries, "previous": entries, "ts": perf_counter}}
        self.pose_cache_by_cam: dict[str, dict] = {}
        # {cam_id: {stage_name: milliseconds_or_count}}
        # realtime benchmark code reads this after each extract() call.
        self.last_timings_by_cam: dict[str, dict[str, float | int | str]] = {}

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    def get_last_timing(self, cam_id: str) -> dict[str, float | int | str]:
        """Return the most recent per-stage timing captured by extract()."""
        return dict(self.last_timings_by_cam.get(cam_id, {}))

    # ── ArUco detector 빌드 (작은 마커 감지 강화) ──
    @staticmethod
    def _build_aruco_detector():
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 10
        p.minMarkerPerimeterRate = 0.01  # default 0.03 — 멀리 있는 작은 마커도 감지
        return cv2.aruco.ArucoDetector(d, p)

    # ── 메인 API: 1프레임 → detection list ──
    def extract(self, frame, cam_id: str) -> list[dict]:
        """프레임 → YOLO + ArUco + worker ID 매칭 + 월드 좌표 변환."""
        extract_started = time.perf_counter()
        worker_entries, timing = self.extract_workers(frame, cam_id)
        custom_entries, custom_timing = self.extract_custom(frame, cam_id)
        for key, value in custom_timing.items():
            if key != "cam_id":
                timing[key] = value

        detections = worker_entries + custom_entries
        timing["detections_total"] = len(detections)
        timing["extract_total_ms"] = self._elapsed_ms(extract_started)
        self.last_timings_by_cam[cam_id] = timing
        return detections

    def extract_workers(
        self,
        frame,
        cam_id: str,
    ) -> tuple[list[dict], dict[str, float | int | str]]:
        """Worker pose + worker ArUco만 추출한다.

        4-thread benchmark에서는 custom YOLO와 독립적으로 이 메서드를 실행한다.
        """
        if self.pose_model is None:
            return [], {
                "cam_id": cam_id,
                "pose_model_enabled": 0,
                "pose_track_ms": 0.0,
                "aruco_detect_ms": 0.0,
                "worker_collect_ms": 0.0,
                "worker_entries": 0,
            }

        timing: dict[str, float | int | str] = {
            "cam_id": cam_id,
            "pose_model_enabled": 1,
            "pose_imgsz": self.pose_imgsz,
            "pose_every_n_frames": self.pose_every_n_frames,
        }
        should_run_pose, pose_sequence = self._should_run_pose(cam_id)
        timing["pose_sequence"] = pose_sequence
        if not should_run_pose:
            stage_started = time.perf_counter()
            worker_entries, cache_age_s = self._predict_cached_worker_entries(
                cam_id,
                now_perf=stage_started,
            )
            timing.update({
                "pose_inference_skipped": 1,
                "pose_cache_age_ms": round(cache_age_s * 1000.0, 3),
                "pose_track_ms": 0.0,
                "aruco_detect_ms": 0.0,
                "aruco_ids_total": 0,
                "worker_aruco_ids": 0,
                "worker_collect_ms": self._elapsed_ms(stage_started),
                "worker_entries": len(worker_entries),
                "track_persistence_ms": 0.0,
                "worker_aruco_match_ms": 0.0,
            })
            return worker_entries, timing

        timing["pose_inference_skipped"] = 0
        # ── (a) 사람 포즈 트래킹 ──
        # 양 발목(15,16) 중점을 발 픽셀로 사용 (bbox 중심보다 카메라 각도에 robust).
        stage_started = time.perf_counter()
        pose_results = self.pose_model.track(
            frame,
            conf=POSE_CONF_THRESHOLD,
            persist=True,
            verbose=False,
            classes=[0],
            imgsz=self.pose_imgsz,
        )
        timing["pose_track_ms"] = self._elapsed_ms(stage_started)

        # ── (a-1) 작업자 ArUco ──
        stage_started = time.perf_counter()
        aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(frame)
        timing["aruco_detect_ms"] = self._elapsed_ms(stage_started)
        worker_aruco_centers: dict[int, tuple[float, float]] = {}
        all_detected_ids: list[int] = []
        if aruco_ids is not None:
            for corner, mid in zip(aruco_corners, aruco_ids.flatten()):
                mid_int = int(mid)
                all_detected_ids.append(mid_int)
                if mid_int in self.worker_aruco_map:
                    pts = corner[0]
                    worker_aruco_centers[mid_int] = (
                        float(pts[:, 0].mean()), float(pts[:, 1].mean()),
                    )
        timing["aruco_ids_total"] = len(all_detected_ids)
        timing["worker_aruco_ids"] = len(worker_aruco_centers)

        if self.debug_aruco:
            print(
                f"[aruco/{cam_id}] all={sorted(all_detected_ids)}  "
                f"worker_ids={sorted(worker_aruco_centers.keys())}  "
                f"(map={list(self.worker_aruco_map)})"
            )

        # ── 1) Worker bbox + 발 픽셀 + 월드 좌표 (worker_id 미정 상태) ──
        stage_started = time.perf_counter()
        worker_entries = self._collect_worker_entries(pose_results, cam_id)
        timing["worker_collect_ms"] = self._elapsed_ms(stage_started)
        timing["worker_entries"] = len(worker_entries)

        # ── 1.5) 시간축 persistence: track_id → worker_id ──
        stage_started = time.perf_counter()
        cam_track_map = self.cam_track_to_worker.setdefault(cam_id, {})
        for entry in worker_entries:
            tid = entry["track_id"]
            if tid is not None and tid in cam_track_map:
                entry["worker_id"] = cam_track_map[tid]
                entry["id_source"] = "track_persistence"
        timing["track_persistence_ms"] = self._elapsed_ms(stage_started)

        # ── 2) ArUco 마커 → 가장 가까운 bbox 매칭 (greedy 1:1) ──
        stage_started = time.perf_counter()
        self._match_aruco_to_bbox(
            worker_aruco_centers, worker_entries, cam_track_map, cam_id,
        )
        timing["worker_aruco_match_ms"] = self._elapsed_ms(stage_started)
        timing["pose_cache_age_ms"] = 0.0
        self._update_pose_cache(cam_id, worker_entries, now_perf=time.perf_counter())

        return worker_entries, timing

    def extract_custom(
        self,
        frame,
        cam_id: str,
    ) -> tuple[list[dict], dict[str, float | int | str]]:
        """Custom YOLO 객체만 추출한다."""
        timing: dict[str, float | int | str] = {"cam_id": cam_id}
        custom_entries = self._extract_custom_objects(frame, cam_id, timing)
        return custom_entries, timing

    # ── 내부: pose 결과 → worker_entries ──
    def _collect_worker_entries(self, pose_results, cam_id: str) -> list[dict]:
        entries: list[dict] = []
        if pose_results[0].boxes is None:
            return entries

        boxes = pose_results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf_arr = (
            boxes.conf.cpu().numpy().tolist()
            if boxes.conf is not None else [None] * len(xyxy)
        )
        ids_t = boxes.id
        ids_arr = (
            ids_t.cpu().numpy().astype(int).tolist()
            if ids_t is not None else [None] * len(xyxy)
        )
        orig_h, orig_w = pose_results[0].orig_shape

        kpts_xy = None
        kpts_conf = None
        if pose_results[0].keypoints is not None:
            kpts_xy = pose_results[0].keypoints.xy.cpu().numpy()      # (N, 17, 2)
            if pose_results[0].keypoints.conf is not None:
                kpts_conf = pose_results[0].keypoints.conf.cpu().numpy()

        for i, (box, tid, conf) in enumerate(zip(xyxy, ids_arr, conf_arr)):
            x1, y1, x2, y2 = [float(v) for v in box]

            foot_x, foot_y, foot_source = self._pick_foot_point(
                x1, y1, x2, y2, kpts_xy, kpts_conf, i,
            )
            wx, wy = pixel_to_world(foot_x, foot_y, cam_id)
            if self.worker_world_bounds is not None:
                min_x, max_x, min_y, max_y = self.worker_world_bounds
                if not (min_x <= wx <= max_x and min_y <= wy <= max_y):
                    continue

            bbox_area_ratio = (
                max(0.0, x2 - x1) * max(0.0, y2 - y1)
                / float(orig_w * orig_h)
            )

            entries.append({
                "type": "worker",
                "worker_id": None,
                "track_id": tid,
                "id_source": None,
                "confidence": round(float(conf), 4) if conf is not None else None,
                "bbox_px": [x1, y1, x2, y2],
                "bbox_area_ratio": round(bbox_area_ratio, 4),
                "foot_px": [float(foot_x), float(foot_y)],
                "foot_source": foot_source,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })
        return entries

    def _should_run_pose(self, cam_id: str) -> tuple[bool, int]:
        """Return whether the current frame should run pose inference.

        POSE_EVERY_N_FRAMES=1 keeps the original behavior. Larger values run
        YOLO pose once, then reuse a short temporal prediction for intermediate
        frames so fusion receives continuous worker coordinates.
        """
        seq = self.pose_frame_seq_by_cam.get(cam_id, 0) + 1
        self.pose_frame_seq_by_cam[cam_id] = seq

        if self.pose_every_n_frames <= 1:
            return True, seq

        cache = self.pose_cache_by_cam.get(cam_id)
        if not cache or not cache.get("current"):
            return True, seq

        return (seq - 1) % self.pose_every_n_frames == 0, seq

    def _update_pose_cache(
        self,
        cam_id: str,
        worker_entries: list[dict],
        *,
        now_perf: float,
    ) -> None:
        """Store the last two real pose outputs for short extrapolation."""
        previous_cache = self.pose_cache_by_cam.get(cam_id, {})
        self.pose_cache_by_cam[cam_id] = {
            "previous": deepcopy(previous_cache.get("current", [])),
            "previous_ts": previous_cache.get("ts"),
            "current": deepcopy(worker_entries),
            "ts": now_perf,
        }

    def _predict_cached_worker_entries(
        self,
        cam_id: str,
        *,
        now_perf: float,
    ) -> tuple[list[dict], float]:
        """Reuse cached worker entries and extrapolate world coordinates briefly."""
        cache = self.pose_cache_by_cam.get(cam_id, {})
        current = deepcopy(cache.get("current", []))
        current_ts = float(cache.get("ts") or now_perf)
        cache_age_s = max(0.0, now_perf - current_ts)

        previous = cache.get("previous") or []
        previous_ts = cache.get("previous_ts")
        can_extrapolate = (
            previous
            and previous_ts is not None
            and current_ts > float(previous_ts)
            and cache_age_s <= self.pose_skip_max_extrapolate_s
        )
        previous_by_key = {
            self._worker_entry_key(entry, idx): entry
            for idx, entry in enumerate(previous)
        }
        dt = current_ts - float(previous_ts) if previous_ts is not None else 0.0

        for idx, entry in enumerate(current):
            entry["pose_inference_skipped"] = True
            entry["pose_source"] = "temporal_hold"
            entry["pose_cache_age_ms"] = round(cache_age_s * 1000.0, 3)
            if not can_extrapolate or dt <= 1e-6:
                continue

            prev = previous_by_key.get(self._worker_entry_key(entry, idx))
            if not prev:
                continue
            try:
                cur_world = entry["world"]
                prev_world = prev["world"]
                vx = (float(cur_world["x"]) - float(prev_world["x"])) / dt
                vy = (float(cur_world["y"]) - float(prev_world["y"])) / dt
                entry["world"] = {
                    "x": round(float(cur_world["x"]) + vx * cache_age_s, 3),
                    "y": round(float(cur_world["y"]) + vy * cache_age_s, 3),
                }
                entry["pose_source"] = "temporal_extrapolation"
            except (KeyError, TypeError, ValueError):
                continue

        return current, cache_age_s

    @staticmethod
    def _worker_entry_key(entry: dict, index: int) -> tuple[str, object]:
        """Stable key for matching cached worker entries across real pose frames."""
        track_id = entry.get("track_id")
        if track_id is not None:
            return "track", track_id
        worker_id = entry.get("worker_id")
        if worker_id is not None:
            return "worker", worker_id
        return "index", index

    # ── 내부: 발 픽셀 결정 (양발목 평균 → 한쪽 발목 → bbox 하단 폴백) ──
    @staticmethod
    def _pick_foot_point(x1, y1, x2, y2, kpts_xy, kpts_conf, i):
        foot_x = foot_y = None
        foot_source = "bbox_bottom"
        if kpts_xy is not None and i < len(kpts_xy):
            la = kpts_xy[i, LEFT_ANKLE]
            ra = kpts_xy[i, RIGHT_ANKLE]
            la_ok = la[0] > 0 and la[1] > 0 and (
                kpts_conf is None or kpts_conf[i, LEFT_ANKLE] >= KPT_CONF_THRESHOLD
            )
            ra_ok = ra[0] > 0 and ra[1] > 0 and (
                kpts_conf is None or kpts_conf[i, RIGHT_ANKLE] >= KPT_CONF_THRESHOLD
            )
            if la_ok and ra_ok:
                foot_x = (la[0] + ra[0]) / 2
                foot_y = (la[1] + ra[1]) / 2
                foot_source = "ankles_mid"
            elif la_ok:
                foot_x, foot_y = float(la[0]), float(la[1])
                foot_source = "left_ankle"
            elif ra_ok:
                foot_x, foot_y = float(ra[0]), float(ra[1])
                foot_source = "right_ankle"

        if foot_x is None:
            foot_x = (x1 + x2) / 2
            foot_y = y2
        return foot_x, foot_y, foot_source

    # ── 내부: ArUco 마커 → 가장 가까운 미매칭 bbox (greedy 1:1) ──
    def _match_aruco_to_bbox(
        self,
        worker_aruco_centers,
        worker_entries,
        cam_track_map,
        cam_id,
    ):
        claimed: set[int] = set()
        for mid_int in sorted(worker_aruco_centers.keys()):
            cx, cy = worker_aruco_centers[mid_int]
            best_idx, best_dist = None, float("inf")
            for idx, entry in enumerate(worker_entries):
                if idx in claimed:
                    continue
                x1, y1, x2, y2 = entry["bbox_px"]
                bcx = (x1 + x2) / 2
                bcy = (y1 + y2) / 2
                max_dist = max(x2 - x1, y2 - y1)
                dist = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
                if dist <= max_dist and dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx is not None:
                wid = self.worker_aruco_map[mid_int]
                worker_entries[best_idx]["worker_id"] = wid
                worker_entries[best_idx]["id_source"] = "aruco"
                claimed.add(best_idx)
                tid = worker_entries[best_idx]["track_id"]
                if tid is not None:
                    cam_track_map[tid] = wid
                if self.debug_aruco:
                    print(
                        f"[aruco/{cam_id}] marker {mid_int} → "
                        f"bbox#{best_idx} (dist={best_dist:.0f}px)"
                    )
            elif self.debug_aruco:
                print(
                    f"[aruco/{cam_id}] marker {mid_int} 매칭 실패 "
                    f"(가장 가까운 bbox 가 거리 임계 초과 또는 bbox 0개)"
                )

    # ── 내부: custom_model 검출 (forklift / box_1 / box_2) ──
    def _extract_custom_objects(
        self,
        frame,
        cam_id: str,
        timing: dict[str, float | int | str] | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        if self.custom_model is None:
            if timing is not None:
                timing["custom_model_enabled"] = 0
                timing["custom_yolo_ms"] = 0.0
                timing["custom_postprocess_ms"] = 0.0
                timing["custom_objects"] = 0
            return out

        if timing is not None:
            timing["custom_model_enabled"] = 1
            timing["custom_imgsz"] = self.custom_imgsz
        stage_started = time.perf_counter()
        results = self.custom_model(
            frame,
            conf=0.5,
            verbose=False,
            imgsz=self.custom_imgsz,
        )
        if timing is not None:
            timing["custom_yolo_ms"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            cls_name = self.custom_names.get(cls_id, f"cls_{cls_id}")
            conf = float(box.conf[0]) if box.conf is not None else None

            if cls_name not in CUSTOM_OBJECT_CLASS_NAMES:
                continue

            ref_x, ref_y = (x1 + x2) / 2, y2
            ref_source = "bbox_bottom_center"
            coord_source = "homography_ground"
            dropzone_usable = cls_name not in LIFTED_BOX_CLASS_NAMES
            bbox_area_ratio = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(frame_w * frame_h)

            if cls_name == "forklift":
                ref_x = x1 + (x2 - x1) * FORKLIFT_REF_X_RATIO
                ref_y = y1 + (y2 - y1) * FORKLIFT_REF_Y_RATIO
                ref_source = "forklift_front_axle"
                wx, wy = pixel_to_world(ref_x, ref_y, cam_id)
            elif cls_name in LIFTED_BOX_CLASS_NAMES and cam_id == LIFTED_BOX_PRIMARY_CAM_ID:
                ref_x, ref_y = (x1 + x2) / 2, (y1 + y2) / 2
                ray_world = pixel_to_world_on_unity_y_plane(
                    cam_id=cam_id,
                    px=ref_x,
                    py=ref_y,
                    unity_y=LIFTED_BOX_CENTER_UNITY_Y,
                    image_width=frame_w,
                    image_height=frame_h,
                )
                if ray_world is not None:
                    wx, wy = ray_world
                    ref_source = "bbox_center"
                    coord_source = f"ray_unity_y_{LIFTED_BOX_CENTER_UNITY_Y:g}"
                    dropzone_usable = True
                else:
                    wx, wy = pixel_to_world(ref_x, ref_y, cam_id)
            else:
                wx, wy = pixel_to_world(ref_x, ref_y, cam_id)
            out.append({
                "type": cls_name,
                "track_id": None,
                "confidence": round(conf, 4) if conf is not None else None,
                "bbox_px": [x1, y1, x2, y2],
                "bbox_area_ratio": round(bbox_area_ratio, 4),
                "image_size": [frame_w, frame_h],
                "foot_px": [ref_x, ref_y],
                "ref_source": ref_source,
                "coord_source": coord_source,
                "dropzone_usable": dropzone_usable,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })
        if timing is not None:
            timing["custom_postprocess_ms"] = self._elapsed_ms(stage_started)
            timing["custom_objects"] = len(out)
        return out

    # ── 메인 API: 두 카메라 detection 사이 worker_id 교차 전파 ──
    def cross_camera_propagate(
        self,
        detections_by_cam: dict[str, list[dict]],
        now_ts: float | None = None,
    ) -> None:
        """한쪽 카메라가 식별한 worker_id 를 다른 카메라의 미식별 person 에 전파.

        in-place 로 entry["worker_id"] / ["id_source"] = "cross_camera" 채움.
        """
        if now_ts is None:
            now_ts = time.time()

        # 1) 식별된 worker 로 state 갱신
        for cam_id, dets in detections_by_cam.items():
            for entry in dets:
                if entry.get("type") != "worker":
                    continue
                wid = entry.get("worker_id")
                if wid is None:
                    continue
                wx, wy = entry["world"]["x"], entry["world"]["y"]
                self.worker_world_state[wid] = {
                    "world": (wx, wy),
                    "ts": now_ts,
                    "by_cam": cam_id,
                }

        # 2) TTL 초과 항목 정리
        stale = [
            w for w, st in self.worker_world_state.items()
            if now_ts - st["ts"] > self.worker_state_ttl
        ]
        for w in stale:
            self.worker_world_state.pop(w, None)

        # 3) 미식별 entry 에 전파
        for cam_id, dets in detections_by_cam.items():
            cam_track_map = self.cam_track_to_worker.setdefault(cam_id, {})
            used_in_cam = {
                e["worker_id"] for e in dets
                if e.get("type") == "worker" and e.get("worker_id") is not None
            }
            for entry in dets:
                if (entry.get("type") != "worker"
                        or entry.get("worker_id") is not None):
                    continue
                wx, wy = entry["world"]["x"], entry["world"]["y"]
                best_wid, best_dist = None, float("inf")
                for wid, st in self.worker_world_state.items():
                    if wid in used_in_cam:
                        continue
                    sx, sy = st["world"]
                    dist = ((wx - sx) ** 2 + (wy - sy) ** 2) ** 0.5
                    if dist < best_dist and dist <= self.world_match_radius:
                        best_dist, best_wid = dist, wid
                if best_wid is not None:
                    entry["worker_id"] = best_wid
                    entry["id_source"] = "cross_camera"
                    used_in_cam.add(best_wid)
                    tid = entry.get("track_id")
                    if tid is not None:
                        cam_track_map[tid] = best_wid
                    if self.debug_aruco:
                        print(
                            f"[cross/{cam_id}] {best_wid} 전파 "
                            f"(dist={best_dist:.2f}m)"
                        )

    # ── 메인 API: state 초기화 (테스트 / 새 세션) ──
    def reset_state(self) -> None:
        """worker ID persistence state 모두 비움."""
        self.cam_track_to_worker.clear()
        self.worker_world_state.clear()
        self.pose_frame_seq_by_cam.clear()
        self.pose_cache_by_cam.clear()
        self.last_timings_by_cam.clear()
