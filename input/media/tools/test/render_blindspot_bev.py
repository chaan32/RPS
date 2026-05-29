"""Render BEV diagnostics from recorded cam1/cam2 blindspot frames.

This uses the same coordinate path as the live fusion loop:
DetectionPipeline.extract -> cross_camera_propagate -> pick_positions -> render_bev.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from input.media.pipeline import DetectionRefiner, build_default_pipeline, draw_annotated  # noqa: E402
from input.media.tools.test.check_blindspot_recording import FrameSource, make_writer  # noqa: E402
from model.fusion.runtime.global_tracker import GlobalTrackManager  # noqa: E402
from model.fusion.runtime.kinematics import WorkerKinematics  # noqa: E402
from model.fusion.runtime.pair_builder import pick_positions  # noqa: E402
from model.fusion.runtime.viz import render_bev  # noqa: E402


def resize_for_panel(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    out = np.full((height, w, 3), 245, dtype=np.uint8)
    y = max(0, (height - h) // 2)
    out[y:y + h, :w] = frame
    return out


def add_title(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_composite(cam1: np.ndarray, cam2: np.ndarray, bev: np.ndarray) -> np.ndarray:
    cam_w = 640
    cam1_small = add_title(resize_for_panel(cam1, cam_w), "cam1 detections")
    cam2_small = add_title(resize_for_panel(cam2, cam_w), "cam2 detections")
    cams = np.vstack([cam1_small, cam2_small])
    max_h = max(cams.shape[0], bev.shape[0])
    cams = pad_to_height(cams, max_h)
    bev = pad_to_height(add_title(bev, "BEV fusion input coordinates"), max_h)
    return np.hstack([cams, bev])


def write_position_rows(
    writer: csv.DictWriter,
    frame_idx: int,
    workers_xy: dict[str, tuple[float, float]],
    forklift_xy: tuple[float, float] | None,
    dropzone_xy: tuple[float, float] | None,
) -> None:
    for wid, xy in sorted(workers_xy.items()):
        writer.writerow({
            "frame": frame_idx,
            "type": "worker",
            "id": wid,
            "world_x": round(float(xy[0]), 3),
            "world_y": round(float(xy[1]), 3),
        })
    if forklift_xy is not None:
        writer.writerow({
            "frame": frame_idx,
            "type": "forklift",
            "id": "",
            "world_x": round(float(forklift_xy[0]), 3),
            "world_y": round(float(forklift_xy[1]), 3),
        })
    if dropzone_xy is not None:
        writer.writerow({
            "frame": frame_idx,
            "type": "dropzone",
            "id": "",
            "world_x": round(float(dropzone_xy[0]), 3),
            "world_y": round(float(dropzone_xy[1]), 3),
        })


def apply_visual_worker_ids(
    detections: list[dict],
    workers_xy: dict[str, tuple[float, float]],
) -> None:
    """Mirror the single-worker fusion fallback ID into camera overlays only."""
    if len(workers_xy) != 1:
        return
    wid = next(iter(workers_xy))
    unlabelled = [
        det for det in detections
        if det.get("type") == "worker" and not det.get("worker_id")
    ]
    if len(unlabelled) == 1:
        unlabelled[0]["worker_id"] = wid
        unlabelled[0]["id_source"] = "fusion_input_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam1", required=True, help="cam1 mp4 또는 JPG 프레임 디렉터리")
    parser.add_argument("--cam2", required=True, help="cam2 mp4 또는 JPG 프레임 디렉터리")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지 처리")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "simulation" / "Recordings" / "diagnostics" / "bev"),
        help="결과 저장 디렉터리",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam1_src = FrameSource(Path(args.cam1), args.fps)
    cam2_src = FrameSource(Path(args.cam2), args.fps)
    pipeline = build_default_pipeline()
    refiner = DetectionRefiner()
    global_tracker = GlobalTrackManager()
    kinematics: dict[str, WorkerKinematics] = {}

    bev_writer = None
    composite_writer = None
    csv_path = out_dir / "bev_positions.csv"
    counts = {"frames": 0, "worker": 0, "forklift": 0, "both": 0}

    try:
        with csv_path.open("w", newline="") as f:
            fieldnames = ["frame", "type", "id", "world_x", "world_y"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            frame_idx = 0
            while True:
                ok1, frame1 = cam1_src.read()
                ok2, frame2 = cam2_src.read()
                if not ok1 or not ok2 or frame1 is None or frame2 is None:
                    break
                if args.max_frames and frame_idx >= args.max_frames:
                    break

                d1 = pipeline.extract(frame1, "cam1")
                d2 = pipeline.extract(frame2, "cam2")
                pipeline.cross_camera_propagate({"cam1": d1, "cam2": d2})
                refined = refiner.refine({"cam1": d1, "cam2": d2})
                d1, d2 = refined["cam1"], refined["cam2"]
                raw_workers_xy, raw_forklift_xy, raw_dropzone_xy = pick_positions(d1, d2)
                apply_visual_worker_ids(d1 + d2, raw_workers_xy)
                time_s = frame_idx / cam1_src.fps
                workers_xy, forklift_xy, dropzone_xy = global_tracker.update(
                    time_s,
                    raw_workers_xy,
                    raw_forklift_xy,
                    raw_dropzone_xy,
                )

                for wid, xy in workers_xy.items():
                    kinematics.setdefault(wid, WorkerKinematics()).update(xy)
                headings = {wid: kin.heading for wid, kin in kinematics.items()}
                risks = {
                    wid: np.asarray([[0.0, 0.0]], dtype=np.float32)
                    for wid in workers_xy
                }
                bev = render_bev(
                    workers_xy,
                    forklift_xy,
                    audio_score=0.05,
                    risks_per_worker=risks,
                    dropzone_xy=dropzone_xy,
                    worker_headings=headings,
                )

                ann1 = draw_annotated(frame1, d1)
                ann2 = draw_annotated(frame2, d2)
                composite = make_composite(ann1, ann2, bev)

                if bev_writer is None:
                    bev_writer = make_writer(out_dir / "blindspot_bev.mp4", cam1_src.fps, bev.shape)
                    composite_writer = make_writer(
                        out_dir / "blindspot_bev_with_cameras.mp4",
                        cam1_src.fps,
                        composite.shape,
                    )
                bev_writer.write(bev)
                composite_writer.write(composite)

                write_position_rows(writer, frame_idx, workers_xy, forklift_xy, dropzone_xy)
                counts["frames"] += 1
                if workers_xy:
                    counts["worker"] += 1
                if forklift_xy is not None:
                    counts["forklift"] += 1
                if workers_xy and forklift_xy is not None:
                    counts["both"] += 1

                if frame_idx % 10 == 0:
                    print(
                        f"[frame {frame_idx}] workers={workers_xy} "
                        f"forklift={forklift_xy} dropzone={dropzone_xy}"
                    )
                frame_idx += 1
    finally:
        cam1_src.close()
        cam2_src.close()
        if bev_writer is not None:
            bev_writer.release()
        if composite_writer is not None:
            composite_writer.release()

    print(f"[saved] {out_dir / 'blindspot_bev.mp4'}")
    print(f"[saved] {out_dir / 'blindspot_bev_with_cameras.mp4'}")
    print(f"[saved] {csv_path}")
    print(
        "[summary] "
        f"frames={counts['frames']} worker={counts['worker']} "
        f"forklift={counts['forklift']} both={counts['both']}"
    )


if __name__ == "__main__":
    main()
