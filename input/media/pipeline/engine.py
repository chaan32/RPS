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
from pathlib import Path

import cv2
from ultralytics import YOLO

from ..world_mapper import pixel_to_world
from .constants import (
    BOX_CLASS_NAMES,
    KPT_CONF_THRESHOLD,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    WORKER_ARUCO_MAP,
    WORKER_STATE_TTL_S,
    WORLD_MATCH_RADIUS_M,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
    ):
        # ── 모델 로드 ──
        self.pose_model = YOLO(pose_model_path)

        self.custom_model = None
        self.custom_names: dict[int, str] = {}
        if custom_model_path:
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

        # ── State (인스턴스 캡슐화) ──
        # {cam_id: {track_id: worker_id}}
        self.cam_track_to_worker: dict[str, dict[int, str]] = {}
        # {worker_id: {"world": (x, y), "ts": float, "by_cam": str}}
        self.worker_world_state: dict[str, dict] = {}

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
        detections: list[dict] = []

        # ── (a) 사람 포즈 트래킹 ──
        # 양 발목(15,16) 중점을 발 픽셀로 사용 (bbox 중심보다 카메라 각도에 robust).
        pose_results = self.pose_model.track(
            frame, conf=0.25, persist=True, verbose=False, classes=[0],
        )

        # ── (a-1) 작업자 ArUco ──
        aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(frame)
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

        if self.debug_aruco:
            print(
                f"[aruco/{cam_id}] all={sorted(all_detected_ids)}  "
                f"worker_ids={sorted(worker_aruco_centers.keys())}  "
                f"(map={list(self.worker_aruco_map)})"
            )

        # ── 1) Worker bbox + 발 픽셀 + 월드 좌표 (worker_id 미정 상태) ──
        worker_entries = self._collect_worker_entries(pose_results, cam_id)

        # ── 1.5) 시간축 persistence: track_id → worker_id ──
        cam_track_map = self.cam_track_to_worker.setdefault(cam_id, {})
        for entry in worker_entries:
            tid = entry["track_id"]
            if tid is not None and tid in cam_track_map:
                entry["worker_id"] = cam_track_map[tid]
                entry["id_source"] = "track_persistence"

        # ── 2) ArUco 마커 → 가장 가까운 bbox 매칭 (greedy 1:1) ──
        self._match_aruco_to_bbox(
            worker_aruco_centers, worker_entries, cam_track_map, cam_id,
        )

        detections.extend(worker_entries)

        # ── (b) custom_model: forklift / box_1 / box_2 ──
        detections.extend(self._extract_custom_objects(frame, cam_id))

        return detections

    # ── 내부: pose 결과 → worker_entries ──
    def _collect_worker_entries(self, pose_results, cam_id: str) -> list[dict]:
        entries: list[dict] = []
        if pose_results[0].boxes is None:
            return entries

        boxes = pose_results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids_t = boxes.id
        ids_arr = (
            ids_t.cpu().numpy().astype(int).tolist()
            if ids_t is not None else [None] * len(xyxy)
        )

        kpts_xy = None
        kpts_conf = None
        if pose_results[0].keypoints is not None:
            kpts_xy = pose_results[0].keypoints.xy.cpu().numpy()      # (N, 17, 2)
            if pose_results[0].keypoints.conf is not None:
                kpts_conf = pose_results[0].keypoints.conf.cpu().numpy()

        for i, (box, tid) in enumerate(zip(xyxy, ids_arr)):
            x1, y1, x2, y2 = [float(v) for v in box]

            foot_x, foot_y, foot_source = self._pick_foot_point(
                x1, y1, x2, y2, kpts_xy, kpts_conf, i,
            )
            wx, wy = pixel_to_world(foot_x, foot_y, cam_id)

            entries.append({
                "type": "worker",
                "worker_id": None,
                "track_id": tid,
                "id_source": None,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [float(foot_x), float(foot_y)],
                "foot_source": foot_source,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })
        return entries

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

    # ── 내부: custom_model 검출 (forklift, box_1, box_2) ──
    def _extract_custom_objects(self, frame, cam_id: str) -> list[dict]:
        out: list[dict] = []
        if self.custom_model is None:
            return out

        results = self.custom_model(frame, conf=0.5, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            cls_name = self.custom_names.get(cls_id, f"cls_{cls_id}")

            if cls_name in BOX_CLASS_NAMES:
                # 공중 인양물: 밑면 중심 (지면 projection 오차 최소화)
                ref_x, ref_y = (x1 + x2) / 2, y2
                ref_source = "bbox_bottom_center_airborne"
            elif cls_name == "forklift":
                # 지면 객체: 밑면 중심
                ref_x, ref_y = (x1 + x2) / 2, y2
                ref_source = "bbox_bottom_center"
            else:
                ref_x, ref_y = (x1 + x2) / 2, (y1 + y2) / 2
                ref_source = "bbox_center"

            wx, wy = pixel_to_world(ref_x, ref_y, cam_id)
            out.append({
                "type": cls_name,
                "track_id": None,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [ref_x, ref_y],
                "ref_source": ref_source,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })
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
