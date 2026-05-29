"""Evaluate detection coverage across YOLO input image sizes.

This is a test-only helper. It reads recorded Unity scenario frames directly,
runs the same DetectionPipeline + DetectionRefiner used by realtime_camera, and
writes per-size detection coverage to CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SCENARIOS = [
    "scenario_01_user_current",
    "scenario_02_swapped_positions",
    "scenario_03_opposite_worker",
]


def mean(values: list[float]) -> float:
    """Return mean for a possibly empty list."""
    return statistics.fmean(values) if values else 0.0


def scenario_path(name: str) -> Path:
    """Resolve a collision scenario directory."""
    root = PROJECT_ROOT / "simulation" / "Recordings" / "collision_scenarios"
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"scenario not found: {path}")
    return path


def read_frame_pairs(path: Path, stride: int) -> list[tuple[Path, Path]]:
    """Return paired cam1/cam2 frame paths for one scenario."""
    cam1 = sorted((path / "cam1_frames").glob("frame_*.jpg"))
    cam2 = sorted((path / "cam2_frames").glob("frame_*.jpg"))
    pairs = list(zip(cam1, cam2))
    return pairs[::stride]


def count_type(detections: list[dict[str, Any]], det_type: str) -> bool:
    """Return whether a detection type exists in this frame."""
    return any(det.get("type") == det_type for det in detections)


def evaluate_size(args: argparse.Namespace, imgsz: int) -> list[dict[str, Any]]:
    """Evaluate one image-size setting over all requested scenarios."""
    pose_imgsz = args.pose_imgsz or imgsz
    custom_imgsz = args.custom_imgsz or imgsz
    os.environ["POSE_IMGSZ"] = str(pose_imgsz)
    os.environ["CUSTOM_IMGSZ"] = str(custom_imgsz)

    from input.media.pipeline.refinement import DetectionRefiner
    from input.media.pipeline.runner import build_default_pipeline

    pipeline = build_default_pipeline()
    refiner = DetectionRefiner()
    rows: list[dict[str, Any]] = []

    for scenario in args.scenarios:
        path = scenario_path(scenario)
        pairs = read_frame_pairs(path, args.frame_stride)
        pipeline.reset_state()

        worker_frames = 0
        cam1_worker_frames = 0
        cam2_worker_frames = 0
        forklift_frames = 0
        cam1_forklift_frames = 0
        cam2_forklift_frames = 0
        cam1_pose_ms: list[float] = []
        cam2_pose_ms: list[float] = []
        cam1_custom_ms: list[float] = []
        cam2_custom_ms: list[float] = []
        loop_ms: list[float] = []

        started = time.perf_counter()
        for cam1_path, cam2_path in pairs:
            loop_started = time.perf_counter()
            f1 = cv2.imread(str(cam1_path))
            f2 = cv2.imread(str(cam2_path))
            if f1 is None or f2 is None:
                continue

            d1 = pipeline.extract(f1, "cam1")
            t1 = pipeline.get_last_timing("cam1")
            d2 = pipeline.extract(f2, "cam2")
            t2 = pipeline.get_last_timing("cam2")
            pipeline.cross_camera_propagate({"cam1": d1, "cam2": d2})
            refined = refiner.refine({"cam1": d1, "cam2": d2})
            r1, r2 = refined["cam1"], refined["cam2"]

            c1_worker = count_type(r1, "worker")
            c2_worker = count_type(r2, "worker")
            c1_forklift = count_type(r1, "forklift")
            c2_forklift = count_type(r2, "forklift")
            worker_frames += int(c1_worker or c2_worker)
            cam1_worker_frames += int(c1_worker)
            cam2_worker_frames += int(c2_worker)
            forklift_frames += int(c1_forklift or c2_forklift)
            cam1_forklift_frames += int(c1_forklift)
            cam2_forklift_frames += int(c2_forklift)

            cam1_pose_ms.append(float(t1.get("pose_track_ms", 0.0) or 0.0))
            cam2_pose_ms.append(float(t2.get("pose_track_ms", 0.0) or 0.0))
            cam1_custom_ms.append(float(t1.get("custom_yolo_ms", 0.0) or 0.0))
            cam2_custom_ms.append(float(t2.get("custom_yolo_ms", 0.0) or 0.0))
            loop_ms.append((time.perf_counter() - loop_started) * 1000.0)

        frames = len(pairs)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "imgsz": imgsz,
                "pose_imgsz": pose_imgsz,
                "custom_imgsz": custom_imgsz,
                "scenario": scenario,
                "frames": frames,
                "elapsed_s": round(elapsed, 3),
                "offline_fps": round(frames / elapsed, 3) if elapsed > 0 else 0.0,
                "loop_mean_ms": round(mean(loop_ms), 3),
                "worker_detected_rate": round(worker_frames / frames, 4) if frames else 0.0,
                "cam1_worker_rate": round(cam1_worker_frames / frames, 4) if frames else 0.0,
                "cam2_worker_rate": round(cam2_worker_frames / frames, 4) if frames else 0.0,
                "forklift_detected_rate": round(forklift_frames / frames, 4) if frames else 0.0,
                "cam1_forklift_rate": round(cam1_forklift_frames / frames, 4) if frames else 0.0,
                "cam2_forklift_rate": round(cam2_forklift_frames / frames, 4) if frames else 0.0,
                "cam1_pose_mean_ms": round(mean(cam1_pose_ms), 3),
                "cam2_pose_mean_ms": round(mean(cam2_pose_ms), 3),
                "cam1_custom_mean_ms": round(mean(cam1_custom_ms), 3),
                "cam2_custom_mean_ms": round(mean(cam2_custom_ms), 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-size scenario rows."""
    out: list[dict[str, Any]] = []
    keys = sorted(
        {
            (int(row["pose_imgsz"]), int(row["custom_imgsz"]), int(row["imgsz"]))
            for row in rows
        },
        reverse=True,
    )
    for pose_imgsz, custom_imgsz, imgsz in keys:
        group = [
            row
            for row in rows
            if int(row["pose_imgsz"]) == pose_imgsz
            and int(row["custom_imgsz"]) == custom_imgsz
            and int(row["imgsz"]) == imgsz
        ]
        out.append(
            {
                "imgsz": imgsz,
                "pose_imgsz": pose_imgsz,
                "custom_imgsz": custom_imgsz,
                "scenarios": len(group),
                "worker_rate_min": min(float(row["worker_detected_rate"]) for row in group),
                "worker_rate_mean": round(mean([float(row["worker_detected_rate"]) for row in group]), 4),
                "forklift_rate_min": min(float(row["forklift_detected_rate"]) for row in group),
                "forklift_rate_mean": round(mean([float(row["forklift_detected_rate"]) for row in group]), 4),
                "loop_mean_ms": round(mean([float(row["loop_mean_ms"]) for row in group]), 3),
                "offline_fps_mean": round(mean([float(row["offline_fps"]) for row in group]), 3),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 512, 416, 384, 320])
    parser.add_argument(
        "--pose-imgsz",
        type=int,
        default=0,
        help="Keep pose inference at this size while iterating custom sizes.",
    )
    parser.add_argument(
        "--custom-imgsz",
        type=int,
        default=0,
        help="Keep custom inference at this size while iterating pose sizes.",
    )
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "metrics/imgsz_detection_eval_20260520",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    rows: list[dict[str, Any]] = []
    print(
        f"[imgsz-eval] sizes={args.sizes} scenarios={args.scenarios} "
        f"stride={args.frame_stride}"
    )
    print(f"[imgsz-eval] out_dir={args.out_dir}")
    for imgsz in args.sizes:
        print(f"[imgsz-eval] evaluating imgsz={imgsz}")
        rows.extend(evaluate_size(args, imgsz))
        write_csv(args.out_dir / "results_by_scenario.csv", rows)
        write_csv(args.out_dir / "aggregate_by_imgsz.csv", aggregate(rows))

    print(f"[imgsz-eval] results={args.out_dir / 'results_by_scenario.csv'}")
    print(f"[imgsz-eval] aggregate={args.out_dir / 'aggregate_by_imgsz.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
