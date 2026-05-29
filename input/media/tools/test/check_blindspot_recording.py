"""cam1/cam2 녹화물에서 YOLO + Homography 월드 좌표를 검증한다.

입력은 mp4 파일 또는 Unity가 저장한 JPG 프레임 디렉터리 둘 다 가능하다.

예:
    python input/media/tools/test/check_blindspot_recording.py \
      --cam1 simulation/Recordings/blindspot_cam1_frames \
      --cam2 simulation/Recordings/blindspot_cam2_frames
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from input.media.pipeline import DetectionRefiner, build_default_pipeline, draw_annotated  # noqa: E402


def calibration_size(cam_id: str) -> tuple[int, int] | None:
    img_path = PROJECT_ROOT / "calibration" / f"test_{cam_id}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def resize_to_calibration(frame: np.ndarray, cam_id: str) -> np.ndarray:
    size = calibration_size(cam_id)
    if size is None:
        return frame
    w, h = size
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


@dataclass
class FrameSource:
    path: Path
    fps: float
    _frames: list[Path] | None = None
    _cap: cv2.VideoCapture | None = None
    _idx: int = 0

    def __post_init__(self) -> None:
        if self.path.is_dir():
            frames = sorted(self.path.glob("*.jpg"))
            canonical = [
                frame for frame in frames
                if re.fullmatch(r"frame_\d{4}\.jpg", frame.name)
            ]
            self._frames = canonical or frames
            if not self._frames:
                raise FileNotFoundError(f"JPG 프레임 없음: {self.path}")
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"영상 열기 실패: {self.path}")
            src_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 0:
                self.fps = float(src_fps)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames is not None:
            if self._idx >= len(self._frames):
                return False, None
            frame = cv2.imread(str(self._frames[self._idx]))
            self._idx += 1
            return frame is not None, frame

        assert self._cap is not None
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def make_writer(path: Path, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    h, w = frame_shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 열기 실패: {path}")
    return writer


def add_header(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = min(left.shape[0], right.shape[0])
    lw = int(left.shape[1] * h / left.shape[0])
    rw = int(right.shape[1] * h / right.shape[0])
    l = cv2.resize(left, (lw, h))
    r = cv2.resize(right, (rw, h))
    gap = np.full((h, 12, 3), 255, dtype=np.uint8)
    return np.hstack([l, gap, r])


def update_summary(summary: dict, cam_id: str, detections: list[dict]) -> None:
    cam_summary = summary.setdefault(cam_id, {})
    seen_types = set()
    for det in detections:
        typ = det["type"]
        seen_types.add(typ)
        item = cam_summary.setdefault(
            typ,
            {"frames": 0, "boxes": 0, "xs": [], "ys": []},
        )
        item["boxes"] += 1
        item["xs"].append(float(det["world"]["x"]))
        item["ys"].append(float(det["world"]["y"]))

    for typ in seen_types:
        cam_summary[typ]["frames"] += 1


def write_detection_rows(
    writer: csv.DictWriter,
    frame_idx: int,
    cam_id: str,
    detections: list[dict],
) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox_px"]
        fx, fy = det["foot_px"]
        writer.writerow({
            "frame": frame_idx,
            "cam": cam_id,
            "type": det["type"],
            "worker_id": det.get("worker_id") or "",
            "confidence": det.get("confidence") if det.get("confidence") is not None else "",
            "world_x": det["world"]["x"],
            "world_y": det["world"]["y"],
            "foot_x": round(float(fx), 2),
            "foot_y": round(float(fy), 2),
            "point_source": det.get("ref_source") or det.get("foot_source") or "",
            "coord_source": det.get("coord_source") or "",
            "dropzone_usable": det.get("dropzone_usable", ""),
            "bbox_x1": round(float(x1), 2),
            "bbox_y1": round(float(y1), 2),
            "bbox_x2": round(float(x2), 2),
            "bbox_y2": round(float(y2), 2),
            "bbox_area_ratio": det.get("bbox_area_ratio", ""),
        })


def print_summary(summary: dict) -> None:
    print("\n[summary]")
    for cam_id in sorted(summary):
        print(f"  {cam_id}")
        for typ in sorted(summary[cam_id]):
            item = summary[cam_id][typ]
            xs = np.asarray(item["xs"], dtype=np.float32)
            ys = np.asarray(item["ys"], dtype=np.float32)
            print(
                f"    {typ}: frames={item['frames']} boxes={item['boxes']} "
                f"world_x median={np.median(xs):+.2f} range=[{xs.min():+.2f},{xs.max():+.2f}] "
                f"world_y median={np.median(ys):+.2f} range=[{ys.min():+.2f},{ys.max():+.2f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam1", required=True, help="cam1 mp4 또는 JPG 프레임 디렉터리")
    parser.add_argument("--cam2", required=True, help="cam2 mp4 또는 JPG 프레임 디렉터리")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지 처리")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "simulation" / "Recordings"),
        help="결과 저장 디렉터리",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam1_src = FrameSource(Path(args.cam1), args.fps)
    cam2_src = FrameSource(Path(args.cam2), args.fps)

    pipeline = build_default_pipeline()
    refiner = DetectionRefiner()

    csv_path = out_dir / "blindspot_world_coords.csv"
    raw1_writer = raw2_writer = ann1_writer = ann2_writer = side_writer = None
    summary: dict = {}

    try:
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "frame", "cam", "type", "worker_id", "confidence",
                "world_x", "world_y", "foot_x", "foot_y", "point_source",
                "coord_source", "dropzone_usable",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_area_ratio",
            ]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()

            frame_idx = 0
            while True:
                ok1, frame1 = cam1_src.read()
                ok2, frame2 = cam2_src.read()
                if not ok1 or not ok2 or frame1 is None or frame2 is None:
                    break
                if args.max_frames and frame_idx >= args.max_frames:
                    break

                frame1 = resize_to_calibration(frame1, "cam1")
                frame2 = resize_to_calibration(frame2, "cam2")

                d1 = pipeline.extract(frame1, "cam1")
                d2 = pipeline.extract(frame2, "cam2")
                pipeline.cross_camera_propagate({"cam1": d1, "cam2": d2})
                refined = refiner.refine({"cam1": d1, "cam2": d2})
                d1, d2 = refined["cam1"], refined["cam2"]

                update_summary(summary, "cam1", d1)
                update_summary(summary, "cam2", d2)
                write_detection_rows(csv_writer, frame_idx, "cam1", d1)
                write_detection_rows(csv_writer, frame_idx, "cam2", d2)

                ann1 = add_header(draw_annotated(frame1, d1), f"cam1 frame={frame_idx}")
                ann2 = add_header(draw_annotated(frame2, d2), f"cam2 frame={frame_idx}")
                side = side_by_side(ann1, ann2)

                if raw1_writer is None:
                    raw1_writer = make_writer(out_dir / "blindspot_cam1.mp4", cam1_src.fps, frame1.shape)
                    raw2_writer = make_writer(out_dir / "blindspot_cam2.mp4", cam2_src.fps, frame2.shape)
                    ann1_writer = make_writer(out_dir / "blindspot_world_check_cam1.mp4", cam1_src.fps, ann1.shape)
                    ann2_writer = make_writer(out_dir / "blindspot_world_check_cam2.mp4", cam2_src.fps, ann2.shape)
                    side_writer = make_writer(out_dir / "blindspot_world_check_side_by_side.mp4", cam1_src.fps, side.shape)

                raw1_writer.write(frame1)
                raw2_writer.write(frame2)
                ann1_writer.write(ann1)
                ann2_writer.write(ann2)
                side_writer.write(side)

                if frame_idx % 10 == 0:
                    print(f"[frame {frame_idx}] cam1 det={len(d1)} cam2 det={len(d2)}")
                frame_idx += 1
    finally:
        cam1_src.close()
        cam2_src.close()
        for writer in [raw1_writer, raw2_writer, ann1_writer, ann2_writer, side_writer]:
            if writer is not None:
                writer.release()

    print(f"\n[saved] {csv_path}")
    print(f"[saved] {out_dir / 'blindspot_cam1.mp4'}")
    print(f"[saved] {out_dir / 'blindspot_cam2.mp4'}")
    print(f"[saved] {out_dir / 'blindspot_world_check_side_by_side.mp4'}")
    print_summary(summary)


if __name__ == "__main__":
    main()
