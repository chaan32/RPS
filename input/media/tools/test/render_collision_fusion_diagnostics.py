"""Run recorded collision scenarios through YOLO, BEV, and fusion risk.

The recorded Unity clips are 12 FPS, while the fusion model was trained at 5 Hz.
This tool renders every video frame, but feeds the model at 5 Hz using the frame
timestamps, then carries the latest risk value into the visual overlay.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from input.media.pipeline import DetectionRefiner, build_default_pipeline  # noqa: E402
from input.media.tools.test.check_blindspot_recording import FrameSource, make_writer  # noqa: E402
from input.media.tools.test.render_blindspot_bev import (  # noqa: E402
    add_title,
    apply_visual_worker_ids,
    pad_to_height,
    resize_for_panel,
)
from model.fusion.inference import DEFAULT_THRESHOLD, RealtimeInference, load_dual_model  # noqa: E402
from model.fusion.runtime.kinematics import (  # noqa: E402
    DROPZONE_ALERT_RADIUS,
    WorkerKinematics,
    avg_speed,
    forklift_hazard_point,
)
from model.fusion.runtime.early_warning import (  # noqa: E402
    EarlyWarning,
    MotionHistory,
    evaluate_worker_forklift,
)
from model.fusion.runtime.global_tracker import GlobalTrackManager  # noqa: E402
from model.fusion.runtime.pair_builder import pick_positions  # noqa: E402
from model.fusion.runtime.viz import draw_camera_overlay, render_bev  # noqa: E402


SCENARIO_DROPZONE_ALERT_RADII = {
    # Lifted box coordinates are less stable than ground-plane objects.  For the
    # portfolio scenario, treat the moving box as a conservative danger zone
    # rather than a point-accurate collision target.
    "scenario_04_box_dropzone": 2.0,
}


def scenario_dirs(root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        return [Path(item) if "/" in item else root / item for item in selected]
    return sorted(path for path in root.glob("scenario_*") if path.is_dir())


def read_recording_fps(scenario_dir: Path, fallback: float) -> float:
    info = scenario_dir / "recording_info.txt"
    if not info.exists():
        return fallback
    for line in info.read_text().splitlines():
        if line.startswith("fps="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return fallback
    return fallback


def dropzone_alert_radius_for(scenario_name: str) -> float:
    return SCENARIO_DROPZONE_ALERT_RADII.get(scenario_name, DROPZONE_ALERT_RADIUS)


def make_composite(cam1: np.ndarray, cam2: np.ndarray, bev: np.ndarray) -> np.ndarray:
    cam_w = 640
    cam1_small = add_title(resize_for_panel(cam1, cam_w), "cam1 + fusion risk")
    cam2_small = add_title(resize_for_panel(cam2, cam_w), "cam2 + fusion risk")
    cams = np.vstack([cam1_small, cam2_small])
    max_h = max(cams.shape[0], bev.shape[0])
    cams = pad_to_height(cams, max_h)
    bev = pad_to_height(add_title(bev, "BEV fusion risk"), max_h)
    return np.hstack([cams, bev])


def force_dropzone_alerts(
    risks: dict[str, np.ndarray],
    workers_xy: dict[str, tuple[float, float]],
    dropzone_xy: tuple[float, float] | None,
    alert_radius_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    if dropzone_xy is None:
        return risks, {wid: False for wid in risks}

    out: dict[str, np.ndarray] = {}
    forced: dict[str, bool] = {}
    for wid, risk in risks.items():
        wxy = workers_xy.get(wid)
        dz_force = False
        risk_out = risk
        if wxy is not None:
            d_wd = math.hypot(wxy[0] - dropzone_xy[0], wxy[1] - dropzone_xy[1])
            if d_wd <= alert_radius_m:
                dz_force = True
                risk_out = risk.copy()
                risk_out[0, 1] = max(float(risk_out[0, 1]), 1.0)
        out[wid] = risk_out
        forced[wid] = dz_force
    return out, forced


def alert_label(level: str | None) -> str:
    if level == "warning":
        return "WARNING"
    if level == "danger":
        return "DANGER"
    if level == "safe":
        return "SAFE"
    return ""


def write_risk_row(
    writer: csv.DictWriter,
    frame_idx: int,
    time_s: float,
    model_tick: bool,
    workers_xy: dict[str, tuple[float, float]],
    forklift_xy: tuple[float, float] | None,
    forklift_hazard_xy: tuple[float, float] | None,
    dropzone_xy: tuple[float, float] | None,
    raw_workers_xy: dict[str, tuple[float, float]],
    raw_forklift_xy: tuple[float, float] | None,
    raw_dropzone_xy: tuple[float, float] | None,
    tracker: GlobalTrackManager,
    risks: dict[str, np.ndarray],
    dz_forced: dict[str, bool],
    early_warnings: dict[str, EarlyWarning],
) -> None:
    for wid, wxy in sorted(workers_xy.items()):
        risk = risks.get(wid)
        warning = early_warnings.get(wid)
        f_risk = float(risk[0, 0]) if risk is not None else ""
        d_risk = float(risk[0, 1]) if risk is not None else ""
        wf_dist = (
            math.hypot(wxy[0] - forklift_xy[0], wxy[1] - forklift_xy[1])
            if forklift_xy is not None else ""
        )
        wh_dist = (
            math.hypot(wxy[0] - forklift_hazard_xy[0], wxy[1] - forklift_hazard_xy[1])
            if forklift_hazard_xy is not None else ""
        )
        wd_dist = (
            math.hypot(wxy[0] - dropzone_xy[0], wxy[1] - dropzone_xy[1])
            if dropzone_xy is not None else ""
        )
        raw_wxy = raw_workers_xy.get(wid)
        worker_update = tracker.update_for(f"worker:{wid}")
        forklift_update = tracker.update_for("forklift")
        writer.writerow({
            "frame": frame_idx,
            "time_s": round(float(time_s), 3),
            "model_tick": int(model_tick),
            "worker_id": wid,
            "raw_worker_x": "" if raw_wxy is None else round(float(raw_wxy[0]), 3),
            "raw_worker_y": "" if raw_wxy is None else round(float(raw_wxy[1]), 3),
            "worker_x": round(float(wxy[0]), 3),
            "worker_y": round(float(wxy[1]), 3),
            "raw_forklift_x": "" if raw_forklift_xy is None else round(float(raw_forklift_xy[0]), 3),
            "raw_forklift_y": "" if raw_forklift_xy is None else round(float(raw_forklift_xy[1]), 3),
            "forklift_x": "" if forklift_xy is None else round(float(forklift_xy[0]), 3),
            "forklift_y": "" if forklift_xy is None else round(float(forklift_xy[1]), 3),
            "forklift_hazard_x": (
                "" if forklift_hazard_xy is None else round(float(forklift_hazard_xy[0]), 3)
            ),
            "forklift_hazard_y": (
                "" if forklift_hazard_xy is None else round(float(forklift_hazard_xy[1]), 3)
            ),
            "raw_dropzone_x": "" if raw_dropzone_xy is None else round(float(raw_dropzone_xy[0]), 3),
            "raw_dropzone_y": "" if raw_dropzone_xy is None else round(float(raw_dropzone_xy[1]), 3),
            "dropzone_x": "" if dropzone_xy is None else round(float(dropzone_xy[0]), 3),
            "dropzone_y": "" if dropzone_xy is None else round(float(dropzone_xy[1]), 3),
            "worker_tracker_residual_m": (
                "" if worker_update is None else round(float(worker_update.residual_m), 3)
            ),
            "worker_tracker_outlier": int(worker_update.outlier) if worker_update is not None else "",
            "forklift_tracker_residual_m": (
                "" if forklift_update is None else round(float(forklift_update.residual_m), 3)
            ),
            "forklift_tracker_outlier": int(forklift_update.outlier) if forklift_update is not None else "",
            "worker_forklift_dist": "" if wf_dist == "" else round(float(wf_dist), 3),
            "worker_forklift_hazard_dist": "" if wh_dist == "" else round(float(wh_dist), 3),
            "worker_dropzone_dist": "" if wd_dist == "" else round(float(wd_dist), 3),
            "forklift_risk": "" if f_risk == "" else round(float(f_risk), 4),
            "dropzone_risk": "" if d_risk == "" else round(float(d_risk), 4),
            "dropzone_forced": int(dz_forced.get(wid, False)),
            "early_level": "" if warning is None else alert_label(warning.level),
            "early_reason": "" if warning is None else warning.reason,
            "early_ttc_s": (
                "" if warning is None or warning.ttc_s is None
                else round(float(warning.ttc_s), 3)
            ),
            "early_closest_distance_m": (
                "" if warning is None or warning.closest_distance_m is None
                else round(float(warning.closest_distance_m), 3)
            ),
            "early_current_distance_m": (
                "" if warning is None or warning.current_distance_m is None
                else round(float(warning.current_distance_m), 3)
            ),
        })


def render_scenario(
    scenario_dir: Path,
    model,
    threshold: float,
    recording_fps_fallback: float,
    model_fps: float,
    audio_score: float,
    crane_active: int,
) -> dict:
    fps = read_recording_fps(scenario_dir, recording_fps_fallback)
    dropzone_alert_radius = dropzone_alert_radius_for(scenario_dir.name)
    cam1_src = FrameSource(scenario_dir / "cam1_frames", fps)
    cam2_src = FrameSource(scenario_dir / "cam2_frames", fps)
    out_dir = scenario_dir / "diagnostics" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_default_pipeline()
    refiner = DetectionRefiner()
    global_tracker = GlobalTrackManager()
    trackers: dict[str, RealtimeInference] = {}
    kinematics: dict[str, WorkerKinematics] = {}
    forklift_history: deque[tuple[float, float]] = deque(maxlen=5)
    forklift_hazard_motion = MotionHistory()
    worker_motion: dict[str, MotionHistory] = {}
    dz_history: deque[tuple[float, float]] = deque(maxlen=5)
    last_risks: dict[str, np.ndarray] = {}
    next_model_time = 0.0

    bev_writer = None
    composite_writer = None
    risk_csv = out_dir / "fusion_risk.csv"
    summary = {
        "scenario": scenario_dir.name,
        "frames": 0,
        "worker_frames": 0,
        "forklift_frames": 0,
        "both_frames": 0,
        "prediction_frames": 0,
        "max_forklift_risk": 0.0,
        "max_dropzone_risk": 0.0,
        "min_worker_forklift_dist": float("inf"),
        "min_worker_forklift_hazard_dist": float("inf"),
        "min_worker_dropzone_dist": float("inf"),
        "warning_frames": 0,
        "danger_frames": 0,
        "first_warning_frame": None,
        "first_danger_frame": None,
        "dropzone_alert_radius_m": dropzone_alert_radius,
        "dropzone_forced_frames": 0,
        "worker_tracker_outlier_frames": 0,
        "forklift_tracker_outlier_frames": 0,
        "processing_ms_per_frame": 0.0,
    }

    fieldnames = [
        "frame", "time_s", "model_tick", "worker_id",
        "raw_worker_x", "raw_worker_y", "worker_x", "worker_y",
        "raw_forklift_x", "raw_forklift_y", "forklift_x", "forklift_y",
        "forklift_hazard_x", "forklift_hazard_y",
        "raw_dropzone_x", "raw_dropzone_y", "dropzone_x", "dropzone_y",
        "worker_tracker_residual_m", "worker_tracker_outlier",
        "forklift_tracker_residual_m", "forklift_tracker_outlier",
        "worker_forklift_dist", "worker_forklift_hazard_dist", "worker_dropzone_dist",
        "forklift_risk", "dropzone_risk", "dropzone_forced",
        "early_level", "early_reason", "early_ttc_s",
        "early_closest_distance_m", "early_current_distance_m",
    ]

    started_at = time.perf_counter()
    try:
        with risk_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            frame_idx = 0
            while True:
                ok1, frame1 = cam1_src.read()
                ok2, frame2 = cam2_src.read()
                if not ok1 or not ok2 or frame1 is None or frame2 is None:
                    break

                time_s = frame_idx / fps
                d1 = pipeline.extract(frame1, "cam1")
                d2 = pipeline.extract(frame2, "cam2")
                pipeline.cross_camera_propagate({"cam1": d1, "cam2": d2})
                refined = refiner.refine({"cam1": d1, "cam2": d2})
                d1, d2 = refined["cam1"], refined["cam2"]
                raw_workers_xy, raw_forklift_xy, raw_dropzone_xy = pick_positions(d1, d2)
                apply_visual_worker_ids(d1 + d2, raw_workers_xy)
                workers_xy, forklift_xy, tracked_dropzone_xy = global_tracker.update(
                    time_s,
                    raw_workers_xy,
                    raw_forklift_xy,
                    raw_dropzone_xy,
                )

                if any(
                    (global_tracker.update_for(f"worker:{wid}") or None)
                    and global_tracker.update_for(f"worker:{wid}").outlier
                    for wid in workers_xy
                ):
                    summary["worker_tracker_outlier_frames"] += 1
                forklift_update = global_tracker.update_for("forklift")
                if forklift_update is not None and forklift_update.outlier:
                    summary["forklift_tracker_outlier_frames"] += 1

                for wid, xy in workers_xy.items():
                    kinematics.setdefault(wid, WorkerKinematics()).update(xy)
                headings = {wid: kin.heading for wid, kin in kinematics.items()}

                if forklift_xy is not None:
                    forklift_history.append(forklift_xy)
                forklift_hazard_xy = forklift_hazard_point(forklift_xy, forklift_history)
                if forklift_hazard_xy is not None:
                    forklift_hazard_motion.update(time_s, forklift_hazard_xy)

                smoothed_dz = tracked_dropzone_xy
                if tracked_dropzone_xy is not None:
                    dz_history.append(tracked_dropzone_xy)
                    xs = sorted(p[0] for p in dz_history)
                    ys = sorted(p[1] for p in dz_history)
                    mid = len(xs) // 2
                    smoothed_dz = (xs[mid], ys[mid])

                model_tick = time_s + 1e-9 >= next_model_time
                if model_tick:
                    next_model_time += 1.0 / model_fps
                    for wid, wxy in workers_xy.items():
                        if wid not in trackers:
                            trackers[wid] = RealtimeInference(model, device="cpu")
                        if smoothed_dz is not None:
                            trackers[wid].update_dropzone(center=smoothed_dz)
                        trackers[wid].push(forklift_hazard_xy, wxy, audio_score, crane_active)
                        if trackers[wid].ready():
                            last_risks[wid] = trackers[wid].predict()

                risks_for_frame = {
                    wid: risk for wid, risk in last_risks.items()
                    if wid in workers_xy
                }
                risks_for_frame, dz_forced = force_dropzone_alerts(
                    risks_for_frame, workers_xy, smoothed_dz, dropzone_alert_radius,
                )
                dropzone_danger = any(dz_forced.values())

                early_warnings: dict[str, EarlyWarning] = {}
                for wid, wxy in workers_xy.items():
                    hist = worker_motion.setdefault(wid, MotionHistory())
                    hist.update(time_s, wxy)
                    risk = risks_for_frame.get(wid)
                    fusion_risk = float(risk[0, 0]) if risk is not None else None
                    early_warnings[wid] = evaluate_worker_forklift(
                        worker_xy=wxy,
                        forklift_xy=forklift_hazard_xy,
                        worker_history=hist,
                        forklift_history=forklift_hazard_motion,
                        fusion_risk=fusion_risk,
                        fusion_threshold=threshold,
                    )

                if any(ew.level == "warning" for ew in early_warnings.values()):
                    summary["warning_frames"] += 1
                    if summary["first_warning_frame"] is None:
                        summary["first_warning_frame"] = frame_idx
                if dropzone_danger:
                    summary["dropzone_forced_frames"] += 1
                if any(ew.level == "danger" for ew in early_warnings.values()) or dropzone_danger:
                    summary["danger_frames"] += 1
                    if summary["first_danger_frame"] is None:
                        summary["first_danger_frame"] = frame_idx

                if risks_for_frame:
                    summary["prediction_frames"] += 1
                    summary["max_forklift_risk"] = max(
                        summary["max_forklift_risk"],
                        max(float(r[0, 0]) for r in risks_for_frame.values()),
                    )
                    summary["max_dropzone_risk"] = max(
                        summary["max_dropzone_risk"],
                        max(float(r[0, 1]) for r in risks_for_frame.values()),
                    )

                for wxy in workers_xy.values():
                    if forklift_xy is not None:
                        summary["min_worker_forklift_dist"] = min(
                            summary["min_worker_forklift_dist"],
                            math.hypot(wxy[0] - forklift_xy[0], wxy[1] - forklift_xy[1]),
                        )
                    if forklift_hazard_xy is not None:
                        summary["min_worker_forklift_hazard_dist"] = min(
                            summary["min_worker_forklift_hazard_dist"],
                            math.hypot(
                                wxy[0] - forklift_hazard_xy[0],
                                wxy[1] - forklift_hazard_xy[1],
                            ),
                        )
                    if smoothed_dz is not None:
                        summary["min_worker_dropzone_dist"] = min(
                            summary["min_worker_dropzone_dist"],
                            math.hypot(wxy[0] - smoothed_dz[0], wxy[1] - smoothed_dz[1]),
                        )

                bev = render_bev(
                    workers_xy,
                    forklift_xy,
                    audio_score=audio_score,
                    risks_per_worker=risks_for_frame,
                    threshold=threshold,
                    dropzone_xy=smoothed_dz,
                    dropzone_radius=dropzone_alert_radius,
                    forklift_hazard_xy=forklift_hazard_xy,
                    worker_headings=headings,
                    early_warnings=early_warnings,
                )
                overlay1 = draw_camera_overlay(frame1, d1, risks_for_frame, threshold)
                overlay2 = draw_camera_overlay(frame2, d2, risks_for_frame, threshold)
                composite = make_composite(overlay1, overlay2, bev)

                if bev_writer is None:
                    bev_writer = make_writer(out_dir / "fusion_bev.mp4", fps, bev.shape)
                    composite_writer = make_writer(
                        out_dir / "fusion_cameras_bev.mp4", fps, composite.shape,
                    )
                bev_writer.write(bev)
                composite_writer.write(composite)

                write_risk_row(
                    writer,
                    frame_idx,
                    time_s,
                    model_tick,
                    workers_xy,
                    forklift_xy,
                    forklift_hazard_xy,
                    smoothed_dz,
                    raw_workers_xy,
                    raw_forklift_xy,
                    raw_dropzone_xy,
                    global_tracker,
                    risks_for_frame,
                    dz_forced,
                    early_warnings,
                )

                summary["frames"] += 1
                if workers_xy:
                    summary["worker_frames"] += 1
                if forklift_xy is not None:
                    summary["forklift_frames"] += 1
                if workers_xy and forklift_xy is not None:
                    summary["both_frames"] += 1

                if frame_idx % 20 == 0:
                    print(
                        f"[{scenario_dir.name} frame {frame_idx}] "
                        f"workers={workers_xy} forklift={forklift_xy} "
                        f"hazard={forklift_hazard_xy} "
                        f"dz={smoothed_dz} risks={risks_for_frame} "
                        f"early={early_warnings}"
                    )
                frame_idx += 1
    finally:
        cam1_src.close()
        cam2_src.close()
        if bev_writer is not None:
            bev_writer.release()
        if composite_writer is not None:
            composite_writer.release()

    if summary["min_worker_forklift_dist"] == float("inf"):
        summary["min_worker_forklift_dist"] = None
    if summary["min_worker_forklift_hazard_dist"] == float("inf"):
        summary["min_worker_forklift_hazard_dist"] = None
    if summary["min_worker_dropzone_dist"] == float("inf"):
        summary["min_worker_dropzone_dist"] = None
    if summary["frames"]:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        summary["processing_ms_per_frame"] = elapsed_ms / summary["frames"]

    print(f"[saved] {out_dir / 'fusion_bev.mp4'}")
    print(f"[saved] {out_dir / 'fusion_cameras_bev.mp4'}")
    print(f"[saved] {risk_csv}")
    print(
        f"[summary/{scenario_dir.name}] frames={summary['frames']} "
        f"both={summary['both_frames']} predictions={summary['prediction_frames']} "
        f"max_fork={summary['max_forklift_risk']:.3f} "
        f"max_dz={summary['max_dropzone_risk']:.3f}"
    )
    return summary


def _load_truth(scenario_dir: Path) -> dict[int, dict[str, tuple[float, float]]]:
    truth_path = scenario_dir / "ground_truth.csv"
    truth: dict[int, dict[str, tuple[float, float]]] = {}
    if not truth_path.exists():
        return truth
    with truth_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            truth.setdefault(frame, {})[row["object"]] = (
                float(row["world_x"]),
                float(row["world_y"]),
            )
    return truth


def _maybe_xy(row: dict[str, str], x_key: str, y_key: str) -> tuple[float, float] | None:
    if not row.get(x_key) or not row.get(y_key):
        return None
    return float(row[x_key]), float(row[y_key])


def _err(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _stats(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"mean": "", "median": "", "p95": "", "max": ""}
    vals = sorted(values)
    n = len(vals)
    p95_idx = min(n - 1, int(round((n - 1) * 0.95)))
    return {
        "mean": sum(vals) / n,
        "median": vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0,
        "p95": vals[p95_idx],
        "max": vals[-1],
    }


def _round_stat(value):
    return "" if value == "" else round(float(value), 3)


def write_coordinate_evaluations(root: Path, summaries: list[dict]) -> Path:
    summary_by_name = {row["scenario"]: row for row in summaries}
    rows_out: list[dict] = []
    for scenario_dir in scenario_dirs(root, None):
        truth = _load_truth(scenario_dir)
        risk_path = scenario_dir / "diagnostics" / "fusion" / "fusion_risk.csv"
        if not truth or not risk_path.exists():
            continue

        frame_rows: list[dict] = []
        raw_worker_errs: list[float] = []
        worker_errs: list[float] = []
        raw_forklift_errs: list[float] = []
        forklift_errs: list[float] = []
        first_warning_frame: int | None = None
        first_danger_frame: int | None = None
        closest_frame: int | None = None
        closest_distance = float("inf")
        closest_time_s = None

        with risk_path.open(newline="") as f:
            for row in csv.DictReader(f):
                frame = int(row["frame"])
                frame_truth = truth.get(frame, {})
                worker_truth = frame_truth.get("worker")
                forklift_truth = frame_truth.get("forklift")

                raw_worker = _maybe_xy(row, "raw_worker_x", "raw_worker_y")
                worker = _maybe_xy(row, "worker_x", "worker_y")
                raw_forklift = _maybe_xy(row, "raw_forklift_x", "raw_forklift_y")
                forklift = _maybe_xy(row, "forklift_x", "forklift_y")

                raw_worker_err = _err(raw_worker, worker_truth)
                worker_err = _err(worker, worker_truth)
                raw_forklift_err = _err(raw_forklift, forklift_truth)
                forklift_err = _err(forklift, forklift_truth)
                if raw_worker_err is not None:
                    raw_worker_errs.append(raw_worker_err)
                if worker_err is not None:
                    worker_errs.append(worker_err)
                if raw_forklift_err is not None:
                    raw_forklift_errs.append(raw_forklift_err)
                if forklift_err is not None:
                    forklift_errs.append(forklift_err)

                if worker_truth is not None and forklift_truth is not None:
                    gt_dist = math.hypot(
                        worker_truth[0] - forklift_truth[0],
                        worker_truth[1] - forklift_truth[1],
                    )
                    if gt_dist < closest_distance:
                        closest_distance = gt_dist
                        closest_frame = frame
                        closest_time_s = float(row["time_s"])

                if row.get("early_level") == "WARNING" and first_warning_frame is None:
                    first_warning_frame = frame
                if row.get("early_level") == "DANGER" and first_danger_frame is None:
                    first_danger_frame = frame

                frame_rows.append({
                    "frame": frame,
                    "time_s": row["time_s"],
                    "raw_worker_err_m": "" if raw_worker_err is None else round(raw_worker_err, 3),
                    "worker_err_m": "" if worker_err is None else round(worker_err, 3),
                    "raw_forklift_err_m": "" if raw_forklift_err is None else round(raw_forklift_err, 3),
                    "forklift_err_m": "" if forklift_err is None else round(forklift_err, 3),
                    "early_level": row.get("early_level", ""),
                    "worker_tracker_outlier": row.get("worker_tracker_outlier", ""),
                    "forklift_tracker_outlier": row.get("forklift_tracker_outlier", ""),
                })

        eval_path = scenario_dir / "diagnostics" / "fusion" / "coordinate_eval.csv"
        with eval_path.open("w", newline="") as f:
            fieldnames = [
                "frame", "time_s",
                "raw_worker_err_m", "worker_err_m",
                "raw_forklift_err_m", "forklift_err_m",
                "early_level", "worker_tracker_outlier", "forklift_tracker_outlier",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows)

        rw = _stats(raw_worker_errs)
        sw = _stats(worker_errs)
        rf = _stats(raw_forklift_errs)
        sf = _stats(forklift_errs)
        summary = summary_by_name.get(scenario_dir.name, {})
        warning_lead = (
            "" if first_warning_frame is None or closest_frame is None
            else round((closest_frame - first_warning_frame) * read_recording_fps(scenario_dir, 12.0) ** -1, 3)
        )
        danger_lead = (
            "" if first_danger_frame is None or closest_frame is None
            else round((closest_frame - first_danger_frame) * read_recording_fps(scenario_dir, 12.0) ** -1, 3)
        )
        rows_out.append({
            "scenario": scenario_dir.name,
            "frames": summary.get("frames", ""),
            "processing_ms_per_frame": _round_stat(summary.get("processing_ms_per_frame", "")),
            "raw_worker_mean_err_m": _round_stat(rw["mean"]),
            "worker_mean_err_m": _round_stat(sw["mean"]),
            "raw_worker_p95_err_m": _round_stat(rw["p95"]),
            "worker_p95_err_m": _round_stat(sw["p95"]),
            "raw_worker_max_err_m": _round_stat(rw["max"]),
            "worker_max_err_m": _round_stat(sw["max"]),
            "raw_forklift_mean_err_m": _round_stat(rf["mean"]),
            "forklift_mean_err_m": _round_stat(sf["mean"]),
            "raw_forklift_p95_err_m": _round_stat(rf["p95"]),
            "forklift_p95_err_m": _round_stat(sf["p95"]),
            "closest_frame": "" if closest_frame is None else closest_frame,
            "closest_time_s": "" if closest_time_s is None else round(float(closest_time_s), 3),
            "closest_gt_distance_m": "" if closest_distance == float("inf") else round(closest_distance, 3),
            "first_warning_frame": "" if first_warning_frame is None else first_warning_frame,
            "first_danger_frame": "" if first_danger_frame is None else first_danger_frame,
            "warning_lead_s": warning_lead,
            "danger_lead_s": danger_lead,
        })

    out_path = root / "coordinate_eval_summary.csv"
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "scenario", "frames", "processing_ms_per_frame",
            "raw_worker_mean_err_m", "worker_mean_err_m",
            "raw_worker_p95_err_m", "worker_p95_err_m",
            "raw_worker_max_err_m", "worker_max_err_m",
            "raw_forklift_mean_err_m", "forklift_mean_err_m",
            "raw_forklift_p95_err_m", "forklift_p95_err_m",
            "closest_frame", "closest_time_s", "closest_gt_distance_m",
            "first_warning_frame", "first_danger_frame",
            "warning_lead_s", "danger_lead_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "simulation" / "Recordings" / "collision_scenarios",
    )
    parser.add_argument("--scenario", action="append", help="Scenario folder name or path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--model-fps", type=float, default=5.0)
    parser.add_argument("--audio-score", type=float, default=0.05)
    parser.add_argument("--crane-active", type=int, default=1)
    args = parser.parse_args()

    ckpt_dir = PROJECT_ROOT / "model" / "fusion" / "checkpoints"
    model = load_dual_model(str(ckpt_dir), device="cpu")
    pipeline_summaries = []
    for scenario_dir in scenario_dirs(args.root, args.scenario):
        if not scenario_dir.exists():
            raise FileNotFoundError(f"scenario dir not found: {scenario_dir}")
        pipeline_summaries.append(
            render_scenario(
                scenario_dir=scenario_dir,
                model=model,
                threshold=args.threshold,
                recording_fps_fallback=args.fps,
                model_fps=args.model_fps,
                audio_score=args.audio_score,
                crane_active=args.crane_active,
            )
        )

    summary_path = args.root / "fusion_summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "scenario", "frames", "worker_frames", "forklift_frames", "both_frames",
            "prediction_frames", "max_forklift_risk", "max_dropzone_risk",
            "min_worker_forklift_dist", "min_worker_forklift_hazard_dist",
            "min_worker_dropzone_dist",
            "warning_frames", "danger_frames",
            "first_warning_frame", "first_danger_frame",
            "dropzone_alert_radius_m", "dropzone_forced_frames",
            "worker_tracker_outlier_frames", "forklift_tracker_outlier_frames",
            "processing_ms_per_frame",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pipeline_summaries:
            writer.writerow(row)
    print(f"[saved] {summary_path}")
    eval_summary_path = write_coordinate_evaluations(args.root, pipeline_summaries)
    print(f"[saved] {eval_summary_path}")


if __name__ == "__main__":
    main()
