"""Detection refinement between raw YOLO output and fusion input.

The detector is intentionally permissive for the Unity CCTV benchmark because
the worker can be small or partially occluded.  This module removes obvious
single-frame false positives before BEV/fusion consume the detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


WORKER_TYPE = "worker"
THREAT_TYPES = ("forklift", "box_1", "box_2")


@dataclass(frozen=True)
class DetectionRefinerConfig:
    """Thresholds for the Unity CCTV benchmark refiner."""

    min_unidentified_worker_confidence: float = 0.05
    low_conf_overlap_confidence: float = 0.08
    low_conf_overlap_iou: float = 0.10
    low_conf_overlap_worker_coverage: float = 0.45
    forklift_attached_pose_confidence: float = 0.35
    forklift_attached_pose_iou: float = 0.30
    forklift_attached_pose_worker_coverage: float = 0.65
    fallback_overlap_confidence: float = 0.16
    fallback_block_iou: float = 0.08
    fallback_block_worker_coverage: float = 0.35
    max_unidentified_workers_per_camera: int = 2
    max_worker_bbox_area_ratio: float = 0.20
    resolve_single_worker_cross_camera_disagreement: bool = False
    worker_cross_camera_disagreement_m: float = 1.25
    worker_score_tie_margin: float = 0.10
    preferred_worker_cam: str = "cam2"


class DetectionRefiner:
    """Filter raw per-camera detections into fusion-ready detections.

    Current benchmark assumption:
      - one or two workers can be active in the Unity collision scenarios,
      - one forklift is active,
      - at most one box/dropzone object of each class is useful per camera.

    The class is intentionally stateless for now.  It can later be extended with
    a Kalman/global-track state without changing call sites.
    """

    def __init__(self, config: DetectionRefinerConfig | None = None):
        self.config = config or DetectionRefinerConfig()

    def refine(self, detections_by_cam: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """Return refined copies keyed by camera id."""
        refined = {
            cam_id: self.refine_camera(cam_id, detections)
            for cam_id, detections in detections_by_cam.items()
        }
        self._resolve_cross_camera_worker_disagreement(refined)
        return refined

    def refine_camera(self, cam_id: str, detections: list[dict]) -> list[dict]:
        threats = self._best_threats(detections)
        workers = [d for d in detections if d.get("type") == WORKER_TYPE]
        refined_workers = self._refine_workers(workers, threats.values())
        return refined_workers + list(threats.values())

    def _best_threats(self, detections: list[dict]) -> dict[str, dict]:
        best: dict[str, dict] = {}
        for det in detections:
            det_type = det.get("type")
            if det_type not in THREAT_TYPES:
                continue
            prev = best.get(det_type)
            if prev is None or _confidence(det) > _confidence(prev):
                copied = dict(det)
                copied["refined"] = True
                copied["refine_reason"] = "best_threat_by_class"
                best[det_type] = copied
        return best

    def _refine_workers(self, workers: list[dict], threats: Iterable[dict]) -> list[dict]:
        threats = list(threats)
        accepted: list[dict] = []
        rejected: list[tuple[dict, str]] = []
        for worker in workers:
            reject_reason = self._reject_worker_reason(worker, threats)
            if reject_reason is not None:
                rejected.append((worker, reject_reason))
                continue
            copied = dict(worker)
            copied["refine_score"] = round(self._worker_score(copied, threats), 4)
            copied["refined"] = True
            accepted.append(copied)

        if not accepted and workers:
            fallback_original = max(workers, key=_confidence)
            fallback = dict(fallback_original)
            fallback_reason = next(
                (reason for worker, reason in rejected if worker is fallback_original),
                "",
            )
            if not self._allow_rejected_worker_fallback(fallback, fallback_reason, threats):
                return []
            fallback["refine_score"] = round(self._worker_score(fallback, threats), 4)
            fallback["refined"] = True
            fallback["refine_reason"] = (
                "best_worker_fallback"
                if not fallback_reason
                else f"best_worker_fallback_after_{fallback_reason}"
            )
            return [fallback]

        identified_by_id: dict[str, dict] = {}
        unidentified: list[dict] = []
        for worker in accepted:
            worker_id = worker.get("worker_id")
            if worker_id:
                prev = identified_by_id.get(worker_id)
                if prev is None or self._worker_score(worker, threats) > self._worker_score(prev, threats):
                    worker["refine_reason"] = "best_identified_worker"
                    identified_by_id[worker_id] = worker
            else:
                unidentified.append(worker)

        if identified_by_id:
            return sorted(identified_by_id.values(), key=lambda d: str(d.get("worker_id")))

        unidentified.sort(
            key=lambda worker: self._worker_score(worker, threats),
            reverse=True,
        )
        kept = unidentified[: self.config.max_unidentified_workers_per_camera]
        for worker in kept:
            worker["refine_reason"] = "best_unidentified_worker"
        return kept

    def _reject_worker_reason(self, worker: dict, threats: Iterable[dict]) -> str | None:
        if worker.get("worker_id"):
            return None

        conf = _confidence(worker)
        if conf < self.config.min_unidentified_worker_confidence:
            return "low_confidence"

        if float(worker.get("bbox_area_ratio") or 0.0) > self.config.max_worker_bbox_area_ratio:
            return "oversized_bbox"

        worker_box = worker.get("bbox_px")
        if worker_box is None:
            return None

        for threat in threats:
            if threat.get("type") != "forklift":
                continue
            iou, worker_coverage = _overlap_metrics(worker_box, threat.get("bbox_px"))
            if (
                conf < self.config.forklift_attached_pose_confidence
                and (
                    iou >= self.config.forklift_attached_pose_iou
                    or worker_coverage >= self.config.forklift_attached_pose_worker_coverage
                )
            ):
                return "forklift_attached_pose"
            if (
                conf < self.config.low_conf_overlap_confidence
                and (
                    iou >= self.config.low_conf_overlap_iou
                    or worker_coverage >= self.config.low_conf_overlap_worker_coverage
                )
            ):
                return "low_conf_forklift_overlap"
        return None

    def _allow_rejected_worker_fallback(
        self,
        worker: dict,
        reject_reason: str,
        threats: Iterable[dict],
    ) -> bool:
        """Only revive rejected workers when they are not forklift-attached poses.

        The Unity forklift can produce low-confidence person poses around the
        cage/seat area.  Reviving those detections creates believable-looking
        but completely wrong world coordinates.  Missing one frame is safer
        than injecting a false worker into BEV/fusion.
        """
        if reject_reason in {
            "low_confidence",
            "oversized_bbox",
            "forklift_attached_pose",
        }:
            return False

        worker_box = worker.get("bbox_px")
        if worker_box is None:
            return True

        conf = _confidence(worker)
        for threat in threats:
            if threat.get("type") != "forklift":
                continue
            iou, worker_coverage = _overlap_metrics(worker_box, threat.get("bbox_px"))
            if (
                conf < self.config.fallback_overlap_confidence
                and (
                    iou >= self.config.fallback_block_iou
                    or worker_coverage >= self.config.fallback_block_worker_coverage
                )
            ):
                return False
        return True

    def _worker_score(self, worker: dict, threats: Iterable[dict]) -> float:
        score = _confidence(worker)
        if worker.get("worker_id"):
            score += 2.0
        if worker.get("id_source") == "aruco":
            score += 0.5
        if worker.get("foot_source") in {"ankles_mid", "left_ankle", "right_ankle"}:
            score += 0.05

        worker_box = worker.get("bbox_px")
        if worker_box is not None:
            for threat in threats:
                if threat.get("type") != "forklift":
                    continue
                iou, worker_coverage = _overlap_metrics(worker_box, threat.get("bbox_px"))
                score -= 0.35 * iou
                score -= 0.15 * worker_coverage
        return score

    def _resolve_cross_camera_worker_disagreement(self, refined: dict[str, list[dict]]) -> None:
        """For the single-worker benchmark, keep one view if cam coords disagree.

        Homography itself is stable, but worker pose foot points can be wrong
        under occlusion.  If two cameras produce worker coordinates that are too
        far apart to be the same person, keeping both makes the overlay and
        fallback fusion input misleading.
        """
        if not self.config.resolve_single_worker_cross_camera_disagreement:
            return

        per_cam: dict[str, dict] = {}
        for cam_id, detections in refined.items():
            workers = [d for d in detections if d.get("type") == WORKER_TYPE]
            if workers:
                per_cam[cam_id] = max(
                    workers,
                    key=lambda d: float(d.get("refine_score") or _confidence(d)),
                )

        if len(per_cam) < 2:
            return

        cams = sorted(per_cam.keys())
        best_pair: tuple[str, str] | None = None
        best_dist = -1.0
        for i, cam_a in enumerate(cams):
            for cam_b in cams[i + 1:]:
                dist = _world_distance(per_cam[cam_a], per_cam[cam_b])
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (cam_a, cam_b)

        if (
            best_pair is None
            or best_dist <= self.config.worker_cross_camera_disagreement_m
        ):
            return

        keep_cam = self._choose_worker_cam(per_cam)
        for cam_id, detections in refined.items():
            if cam_id == keep_cam:
                for det in detections:
                    if det.get("type") == WORKER_TYPE:
                        det["cross_camera_disagreement_m"] = round(best_dist, 3)
                        det["refine_reason"] = (
                            f"{det.get('refine_reason', 'worker')}|cross_camera_selected"
                        )
                continue
            refined[cam_id] = [
                det for det in detections
                if det.get("type") != WORKER_TYPE
            ]

    def _choose_worker_cam(self, per_cam: dict[str, dict]) -> str:
        scored = sorted(
            per_cam.items(),
            key=lambda item: float(item[1].get("refine_score") or _confidence(item[1])),
            reverse=True,
        )
        best_cam, best_det = scored[0]
        if len(scored) == 1:
            return best_cam
        second_cam, second_det = scored[1]
        best_score = float(best_det.get("refine_score") or _confidence(best_det))
        second_score = float(second_det.get("refine_score") or _confidence(second_det))
        if (
            abs(best_score - second_score) <= self.config.worker_score_tie_margin
            and self.config.preferred_worker_cam in per_cam
        ):
            return self.config.preferred_worker_cam
        return best_cam


def _confidence(det: dict) -> float:
    value = det.get("confidence")
    return float(value) if value is not None else 0.0


def _bbox_area(box: list[float] | tuple[float, ...] | None) -> float:
    if box is None or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_metrics(
    a: list[float] | tuple[float, ...] | None,
    b: list[float] | tuple[float, ...] | None,
) -> tuple[float, float]:
    """Return (IoU, intersection / area(a))."""
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0, 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    coverage_a = inter / area_a if area_a > 0 else 0.0
    return iou, coverage_a


def _world_distance(a: dict, b: dict) -> float:
    aw, bw = a.get("world"), b.get("world")
    if not aw or not bw:
        return 0.0
    ax, ay = float(aw["x"]), float(aw["y"])
    bx, by = float(bw["x"]), float(bw["y"])
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
