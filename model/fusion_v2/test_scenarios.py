"""Run Fusion V2 on recorded scenario diagnostics and render review videos.

V2 does not consume raw camera frames directly.  It consumes the BEV/world
coordinate traces produced by the existing V1 video pipeline, then predicts the
final risk from a recent coordinate window.  This script tests that layer on
selected scenario folders and writes a compact video for visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .dataset import (
    DEFAULT_DROPZONE_DANGER_M,
    DEFAULT_DROPZONE_WARNING_M,
    DEFAULT_FORKLIFT_DANGER_M,
    DEFAULT_FORKLIFT_WARNING_M,
    DEFAULT_FUTURE_HORIZON_FRAMES,
    LABEL_MODES,
    _feature_table,
    _int,
    _labels_for_mode,
    _read_csv,
    _windowize,
)
from .inference import load_checkpoint
from .schema import DANGER_THRESHOLD, SAFE_THRESHOLD, THREAT_NAMES


LEVELS = ("SAFE", "WARNING", "DANGER")
LEVEL_COLORS = {
    "SAFE": (80, 180, 80),
    "WARNING": (0, 165, 255),
    "DANGER": (40, 40, 230),
    "MISS": (150, 150, 150),
}


def _level(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return "MISS"
    if score >= DANGER_THRESHOLD:
        return "DANGER"
    if score >= SAFE_THRESHOLD:
        return "WARNING"
    return "SAFE"


def _score_from_level(level: str | None) -> float:
    if level == "DANGER":
        return 1.0
    if level == "WARNING":
        return 0.5
    return 0.0


def _class_id(score: float) -> int:
    if score >= DANGER_THRESHOLD:
        return 2
    if score >= SAFE_THRESHOLD:
        return 1
    return 0


def _metrics(rows: list[dict[str, object]], threat: str) -> dict[str, object]:
    v1_key = f"reference_{threat}_target"
    v2_key = f"v2_{threat}_prob"
    y_true = np.array([float(r[v1_key]) for r in rows], dtype=np.float32)
    y_pred = np.array([float(r[v2_key]) for r in rows], dtype=np.float32)
    true_class = np.array([_class_id(float(v)) for v in y_true], dtype=np.int64)
    pred_class = np.array([_class_id(float(v)) for v in y_pred], dtype=np.int64)

    pred_danger = y_pred >= DANGER_THRESHOLD
    true_danger = y_true >= DANGER_THRESHOLD
    tp = int(np.logical_and(pred_danger, true_danger).sum())
    fp = int(np.logical_and(pred_danger, ~true_danger).sum())
    fn = int(np.logical_and(~pred_danger, true_danger).sum())
    tn = int(np.logical_and(~pred_danger, ~true_danger).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "class_accuracy": float((true_class == pred_class).mean()) if len(rows) else 0.0,
        "danger_accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
        "danger_precision": float(precision),
        "danger_recall": float(recall),
        "danger_f1": float(f1),
        "support_danger": int(true_danger.sum()),
        "predicted_danger": int(pred_danger.sum()),
    }


def _first_frame(rows: list[dict[str, object]], key: str, threshold: float) -> int | None:
    frames = [int(r["frame"]) for r in rows if float(r[key]) >= threshold]
    return min(frames) if frames else None


def _frame_paths(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.jpg"))


def _read_frame(path: Path, fallback_shape: tuple[int, int, int] = (360, 640, 3)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    return image


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    height = max(1, int(h * width / w))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((42, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([header, image])


def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.vstack([image, pad])


def _make_side_panel(
    scenario: str,
    frame: int,
    latest_by_worker: dict[str, dict[str, object]],
    width: int = 420,
    height: int = 760,
) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Fusion V2 Scenario Test", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2)
    cv2.putText(panel, scenario, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1)
    cv2.putText(panel, f"frame={frame}", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1)

    y = 140
    for worker_id in sorted(latest_by_worker):
        row = latest_by_worker[worker_id]
        cv2.putText(panel, worker_id, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (30, 30, 30), 2)
        y += 30
        for threat in THREAT_NAMES:
            v1 = float(row[f"reference_{threat}_target"])
            v2 = float(row[f"v2_{threat}_prob"])
            v1_level = _level(v1)
            v2_level = _level(v2)
            color = LEVEL_COLORS[v2_level]
            cv2.rectangle(panel, (18, y - 18), (34, y - 2), color, -1)
            cv2.putText(
                panel,
                f"{threat:8s} REF {v1_level:<7s} {v1:0.2f}",
                (42, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (45, 45, 45),
                1,
            )
            y += 23
            cv2.putText(
                panel,
                f"         V2 {v2_level:<7s} {v2:0.2f}",
                (42, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
            )
            y += 30
        y += 14
    return panel


def _write_review_video(
    *,
    scenario_dir: Path,
    scenario: str,
    predictions: list[dict[str, object]],
    out_path: Path,
    fps: float = 12.0,
) -> None:
    cam1_paths = _frame_paths(scenario_dir / "cam1_frames")
    cam2_paths = _frame_paths(scenario_dir / "cam2_frames")
    if not cam1_paths or not cam2_paths:
        return

    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        by_frame[int(row["frame"])].append(row)

    latest_by_worker: dict[str, dict[str, object]] = {}
    writer = None
    frame_count = min(len(cam1_paths), len(cam2_paths))
    try:
        for idx in range(frame_count):
            for row in by_frame.get(idx, []):
                latest_by_worker[str(row["worker_id"])] = row

            cam1 = _add_title(_resize_width(_read_frame(cam1_paths[idx]), 640), "cam1")
            cam2 = _add_title(_resize_width(_read_frame(cam2_paths[idx]), 640), "cam2")
            cams_h = max(cam1.shape[0], cam2.shape[0])
            cams = np.vstack([
                _pad_height(cam1, cams_h),
                _pad_height(cam2, cams_h),
            ])
            panel = _make_side_panel(scenario, idx, latest_by_worker, height=cams.shape[0])
            composite = np.hstack([cams, panel])

            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (composite.shape[1], composite.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"VideoWriter open failed: {out_path}")
            writer.write(composite)
    finally:
        if writer is not None:
            writer.release()


def _predict_scenario(
    *,
    scenario_dir: Path,
    scenario: str,
    model: torch.nn.Module,
    payload: dict,
    device: str,
    window_size: int,
    label_mode: str,
    future_horizon_frames: int,
    forklift_danger_m: float,
    forklift_warning_m: float,
    dropzone_danger_m: float,
    dropzone_warning_m: float,
) -> list[dict[str, object]]:
    rows = _read_csv(scenario_dir / "diagnostics" / "fusion" / "fusion_risk.csv")
    by_worker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_worker[row.get("worker_id") or "W00"].append(row)

    mean = payload["mean"].astype(np.float32)
    std = payload["std"].astype(np.float32)
    predictions: list[dict[str, object]] = []
    model.eval()

    for worker_id, worker_rows in sorted(by_worker.items()):
        worker_rows.sort(key=lambda r: _int(r.get("frame"), 0))
        features, teacher_labels, _ = _feature_table(worker_rows)
        reference_labels = _labels_for_mode(
            worker_rows,
            features,
            label_mode=label_mode,
            future_horizon_frames=future_horizon_frames,
            forklift_danger_m=forklift_danger_m,
            forklift_warning_m=forklift_warning_m,
            dropzone_danger_m=dropzone_danger_m,
            dropzone_warning_m=dropzone_warning_m,
        )
        x, y, end_idx = _windowize(features, reference_labels, window_size=window_size, stride=1)
        _, teacher_y, _ = _windowize(features, teacher_labels, window_size=window_size, stride=1)
        if len(x) == 0:
            continue
        x_norm = ((x.astype(np.float32) - mean) / std).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(x_norm).float().to(device)).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        for i, end in enumerate(end_idx):
            source_row = worker_rows[int(end)]
            predictions.append({
                "scenario": scenario,
                "worker_id": worker_id,
                "frame": _int(source_row.get("frame"), int(end)),
                "time_s": float(source_row.get("time_s") or 0.0),
                "reference_forklift_target": float(y[i, 0]),
                "v1_teacher_forklift_target": float(teacher_y[i, 0]),
                "v2_forklift_prob": float(probs[i, 0]),
                "reference_forklift_level": _level(float(y[i, 0])),
                "v1_teacher_forklift_level": _level(float(teacher_y[i, 0])),
                "v2_forklift_level": _level(float(probs[i, 0])),
                "reference_dropzone_target": float(y[i, 1]),
                "v1_teacher_dropzone_target": float(teacher_y[i, 1]),
                "v2_dropzone_prob": float(probs[i, 1]),
                "reference_dropzone_level": _level(float(y[i, 1])),
                "v1_teacher_dropzone_level": _level(float(teacher_y[i, 1])),
                "v2_dropzone_level": _level(float(probs[i, 1])),
            })
    predictions.sort(key=lambda r: (int(r["frame"]), str(r["worker_id"])))
    return predictions


def _summarize_scenario(predictions: list[dict[str, object]]) -> dict[str, object]:
    by_worker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        by_worker[str(row["worker_id"])].append(row)

    summary: dict[str, object] = {
        "windows": len(predictions),
        "workers": sorted(by_worker),
        "forklift": _metrics(predictions, "forklift") if predictions else {},
        "dropzone": _metrics(predictions, "dropzone") if predictions else {},
        "first_events": {},
    }

    first_events = {}
    for worker_id, rows in sorted(by_worker.items()):
        first_events[worker_id] = {}
        for threat in THREAT_NAMES:
            first_events[worker_id][threat] = {
                "v1_warning_frame": _first_frame(rows, f"v1_teacher_{threat}_target", SAFE_THRESHOLD),
                "reference_warning_frame": _first_frame(rows, f"reference_{threat}_target", SAFE_THRESHOLD),
                "v2_warning_frame": _first_frame(rows, f"v2_{threat}_prob", SAFE_THRESHOLD),
                "v1_danger_frame": _first_frame(rows, f"v1_teacher_{threat}_target", DANGER_THRESHOLD),
                "reference_danger_frame": _first_frame(rows, f"reference_{threat}_target", DANGER_THRESHOLD),
                "v2_danger_frame": _first_frame(rows, f"v2_{threat}_prob", DANGER_THRESHOLD),
                "v1_max": max(float(r[f"v1_teacher_{threat}_target"]) for r in rows),
                "reference_max": max(float(r[f"reference_{threat}_target"]) for r in rows),
                "v2_max": max(float(r[f"v2_{threat}_prob"]) for r in rows),
            }
    summary["first_events"] = first_events
    return summary


def _write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not predictions:
        return
    fields = list(predictions[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def run(
    *,
    root: Path,
    scenarios: Iterable[str],
    checkpoint: Path,
    output_dir: Path,
    device: str,
    window_size: int,
    render_video: bool,
    label_mode: str,
    future_horizon_frames: int,
    forklift_danger_m: float,
    forklift_warning_m: float,
    dropzone_danger_m: float,
    dropzone_warning_m: float,
) -> dict[str, object]:
    model, payload = load_checkpoint(checkpoint, device=device)
    all_summary: dict[str, object] = {}
    for scenario in scenarios:
        scenario_dir = root / scenario
        csv_path = scenario_dir / "diagnostics" / "fusion" / "fusion_risk.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        predictions = _predict_scenario(
            scenario_dir=scenario_dir,
            scenario=scenario,
            model=model,
            payload=payload,
            device=device,
            window_size=window_size,
            label_mode=label_mode,
            future_horizon_frames=future_horizon_frames,
            forklift_danger_m=forklift_danger_m,
            forklift_warning_m=forklift_warning_m,
            dropzone_danger_m=dropzone_danger_m,
            dropzone_warning_m=dropzone_warning_m,
        )
        scenario_out = output_dir / scenario
        _write_predictions(scenario_out / "v2_predictions.csv", predictions)
        summary = _summarize_scenario(predictions)
        if render_video:
            video_path = scenario_out / f"{scenario}_v2_test.mp4"
            _write_review_video(
                scenario_dir=scenario_dir,
                scenario=scenario,
                predictions=predictions,
                out_path=video_path,
            )
            summary["video_path"] = str(video_path)
        with (scenario_out / "v2_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        all_summary[scenario] = summary

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenario_v2_summary.json").open("w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    return all_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Fusion V2 on selected recorded scenarios")
    parser.add_argument("--root", type=Path, default=Path("simulation/Recordings/collision_scenarios"))
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario folder name. Can be repeated.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/fusion_v2/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/fusion_v2/reports/scenario_tests"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="teacher")
    parser.add_argument("--future-horizon-frames", type=int, default=DEFAULT_FUTURE_HORIZON_FRAMES)
    parser.add_argument("--forklift-danger-m", type=float, default=DEFAULT_FORKLIFT_DANGER_M)
    parser.add_argument("--forklift-warning-m", type=float, default=DEFAULT_FORKLIFT_WARNING_M)
    parser.add_argument("--dropzone-danger-m", type=float, default=DEFAULT_DROPZONE_DANGER_M)
    parser.add_argument("--dropzone-warning-m", type=float, default=DEFAULT_DROPZONE_WARNING_M)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    scenarios = args.scenarios or [
        "scenario_01_user_current",
        "scenario_02_swapped_positions",
        "scenario_03_opposite_worker",
        "scenario_04_box_dropzone",
    ]
    summary = run(
        root=args.root,
        scenarios=scenarios,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        window_size=args.window_size,
        render_video=not args.no_video,
        label_mode=args.label_mode,
        future_horizon_frames=args.future_horizon_frames,
        forklift_danger_m=args.forklift_danger_m,
        forklift_warning_m=args.forklift_warning_m,
        dropzone_danger_m=args.dropzone_danger_m,
        dropzone_warning_m=args.dropzone_warning_m,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
