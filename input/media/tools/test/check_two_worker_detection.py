"""Check whether two Unity workers survive detection, refinement, and BEV input.

This diagnostic is intentionally separate from the runtime pipeline.  It helps
validate freshly recorded Unity scenarios before running the full RTSP/fusion
flow:

  1. raw YOLO-pose worker bboxes per camera
  2. worker bboxes after DetectionRefiner
  3. worker IDs produced for BEV/fusion by pick_positions()

Example:
    POSE_MODEL_PATH=model/yolo/yolo11s-pose.pt \
    BEST_MODEL_PATH=model/yolo/best_forklift_box_colab.pt \
    python input/media/tools/test/check_two_worker_detection.py \
      --scenario scenario_01_user_current \
      --scenario scenario_02_swapped_positions \
      --scenario scenario_03_opposite_worker
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from input.media.pipeline import DetectionRefiner, build_default_pipeline  # noqa: E402
from model.fusion.runtime.pair_builder import pick_positions  # noqa: E402


def _scenario_path(root: Path, name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    return root / name_or_path


def _frame_path(scenario_dir: Path, cam_id: str, frame_idx: int) -> Path:
    return scenario_dir / f"{cam_id}_frames" / f"frame_{frame_idx:04d}.jpg"


def _draw_worker_boxes(frame, title: str, detections: list[dict], color) -> object:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    workers = [det for det in detections if det.get("type") == "worker"]
    if not workers:
        cv2.putText(
            image,
            "NO WORKER BBOX",
            (24, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        return image

    for idx, det in enumerate(workers, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in det["bbox_px"]]
        world = det.get("world") or {}
        label = (
            f"#{idx} conf={det.get('confidence')} "
            f"xy=({world.get('x')},{world.get('y')}) "
            f"{det.get('refine_reason') or ''}"
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            image,
            label,
            (max(4, x1), max(86, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def _count_worker_truth(scenario_dir: Path) -> int:
    truth_path = scenario_dir / "ground_truth.csv"
    if not truth_path.exists():
        return 0
    worker_names: set[str] = set()
    with truth_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("object") or ""
            if name.startswith("worker"):
                worker_names.add(name)
    return len(worker_names)


def check_scenario(
    scenario_dir: Path,
    frame_indices: list[int],
    output_dir: Path,
) -> list[dict[str, object]]:
    pipeline = build_default_pipeline()
    refiner = DetectionRefiner()
    rows: list[dict[str, object]] = []
    contact_rows = []

    for frame_idx in frame_indices:
        raw_by_cam: dict[str, list[dict]] = {}
        refined_by_cam: dict[str, list[dict]] = {}
        panels = []

        for cam_id in ("cam1", "cam2"):
            frame = cv2.imread(str(_frame_path(scenario_dir, cam_id, frame_idx)))
            if frame is None:
                raise FileNotFoundError(_frame_path(scenario_dir, cam_id, frame_idx))

            raw = pipeline.extract(frame, cam_id)
            refined = refiner.refine_camera(cam_id, raw)
            raw_by_cam[cam_id] = raw
            refined_by_cam[cam_id] = refined

            raw_panel = _draw_worker_boxes(
                frame,
                f"{scenario_dir.name} f{frame_idx:04d} {cam_id} RAW",
                raw,
                (0, 255, 255),
            )
            refined_panel = _draw_worker_boxes(
                frame,
                f"{scenario_dir.name} f{frame_idx:04d} {cam_id} REFINED",
                refined,
                (0, 255, 0),
            )
            panels.extend([
                cv2.resize(raw_panel, (480, 270)),
                cv2.resize(refined_panel, (480, 270)),
            ])

        refined_by_cam = refiner.refine(raw_by_cam)
        workers_xy, forklift_xy, _ = pick_positions(
            refined_by_cam["cam1"],
            refined_by_cam["cam2"],
        )
        row = {
            "scenario": scenario_dir.name,
            "frame": frame_idx,
            "truth_worker_count": _count_worker_truth(scenario_dir),
            "cam1_raw_workers": sum(1 for det in raw_by_cam["cam1"] if det.get("type") == "worker"),
            "cam2_raw_workers": sum(1 for det in raw_by_cam["cam2"] if det.get("type") == "worker"),
            "cam1_refined_workers": sum(
                1 for det in refined_by_cam["cam1"] if det.get("type") == "worker"
            ),
            "cam2_refined_workers": sum(
                1 for det in refined_by_cam["cam2"] if det.get("type") == "worker"
            ),
            "bev_worker_count": len(workers_xy),
            "bev_worker_ids": " ".join(sorted(workers_xy)),
            "forklift_detected": int(forklift_xy is not None),
        }
        rows.append(row)
        contact_rows.append(cv2.hconcat(panels))

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = cv2.vconcat(contact_rows)
    image_path = output_dir / f"{scenario_dir.name}_two_worker_check.jpg"
    cv2.imwrite(str(image_path), contact_sheet)
    print(f"[saved] {image_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "simulation" / "Recordings" / "collision_scenarios",
    )
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument(
        "--frames",
        default="0,60,119",
        help="Comma-separated frame indices to render into the contact sheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "simulation"
        / "Recordings"
        / "collision_scenarios"
        / "diagnostics_worker_bbox",
    )
    args = parser.parse_args()

    frame_indices = [int(item.strip()) for item in args.frames.split(",") if item.strip()]
    all_rows: list[dict[str, object]] = []
    for scenario in args.scenario:
        all_rows.extend(
            check_scenario(_scenario_path(args.root, scenario), frame_indices, args.output_dir)
        )

    summary_path = args.output_dir / "two_worker_detection_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "scenario",
            "frame",
            "truth_worker_count",
            "cam1_raw_workers",
            "cam2_raw_workers",
            "cam1_refined_workers",
            "cam2_refined_workers",
            "bev_worker_count",
            "bev_worker_ids",
            "forklift_detected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[saved] {summary_path}")
    by_scenario: dict[str, Counter] = {}
    for row in all_rows:
        by_scenario.setdefault(str(row["scenario"]), Counter())
        by_scenario[str(row["scenario"])][int(row["bev_worker_count"])] += 1
    for scenario, counts in by_scenario.items():
        print(f"[summary] {scenario}: BEV worker count frames {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
