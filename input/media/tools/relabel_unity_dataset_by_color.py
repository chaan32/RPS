"""Relabel the Unity forklift/box dataset from rendered colors.

The Unity scene uses a bright yellow forklift and a magenta Box1.  For this
controlled synthetic dataset, pixel-color segmentation produces tighter labels
than projecting Unity renderer bounds through the camera.

Class order:
  0 forklift
  1 box_1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASS_FORKLIFT = 0
CLASS_BOX1 = 1


def _largest_component_bbox(mask: np.ndarray, min_area: int) -> list[int] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[int, int, int, int, int] | None = None
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area:
            continue
        item = (x, y, x + w, y + h, area)
        if best is None or item[4] > best[4]:
            best = item
    if best is None:
        return None
    return [best[0], best[1], best[2], best[3]]


def _expand_bbox(box: list[int], width: int, height: int, x_ratio: float, y_ratio: float, base: int) -> list[int]:
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * x_ratio) + base
    pad_y = int((y2 - y1) * y_ratio) + base
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]


def detect_boxes(image: np.ndarray) -> tuple[list[int] | None, list[int] | None]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # OpenCV hue range is 0..179.
    forklift_mask = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([42, 255, 255]))
    box_mask = cv2.inRange(hsv, np.array([135, 70, 80]), np.array([175, 255, 255]))

    kernel = np.ones((5, 5), np.uint8)
    forklift_mask = cv2.morphologyEx(forklift_mask, cv2.MORPH_OPEN, kernel)
    forklift_mask = cv2.morphologyEx(forklift_mask, cv2.MORPH_CLOSE, kernel)
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_OPEN, kernel)
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kernel)

    forklift_box = _largest_component_bbox(forklift_mask, min_area=100)
    box1_box = _largest_component_bbox(box_mask, min_area=200)

    # Yellow segmentation mainly captures the forklift body. Expand to include
    # forks, mast, wheels, and small dark parts close to the yellow body.
    if forklift_box is not None:
        forklift_box = _expand_bbox(forklift_box, width, height, x_ratio=0.35, y_ratio=0.25, base=12)
    if box1_box is not None:
        box1_box = _expand_bbox(box1_box, width, height, x_ratio=0.02, y_ratio=0.02, base=3)

    return forklift_box, box1_box


def to_yolo_line(class_id: int, box: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def backup_labels(dataset: Path) -> None:
    src = dataset / "labels"
    dst = dataset / "labels_3d_auto_backup"
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)


def relabel_split(dataset: Path, split: str, drop_failures: bool) -> tuple[int, list[str]]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = dataset / "images_color_failures" / split

    for label_path in label_dir.glob("*.txt"):
        label_path.unlink()

    written = 0
    failures: list[str] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append(f"{split}/{image_path.name}: image_read_failed")
            continue
        height, width = image.shape[:2]
        forklift_box, box1_box = detect_boxes(image)
        if forklift_box is None or box1_box is None:
            failures.append(
                f"{split}/{image_path.name}: "
                f"forklift={'ok' if forklift_box else 'missing'} "
                f"box_1={'ok' if box1_box else 'missing'}"
            )
            if drop_failures:
                failure_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(failure_dir / image_path.name))
            continue

        lines = [
            to_yolo_line(CLASS_FORKLIFT, forklift_box, width, height),
            to_yolo_line(CLASS_BOX1, box1_box, width, height),
        ]
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
    return written, failures


def write_data_yaml(dataset: Path) -> None:
    yaml = (
        "# YOLO dataset relabeled from Unity colors\n"
        f"path: {dataset}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: forklift\n"
        "  1: box_1\n"
    )
    (dataset / "data.yaml").write_text(yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="simulation/Assets/Dataset")
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Keep images where color relabeling fails. Default moves them out of images/{split}.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    backup_labels(dataset)
    total_written = 0
    all_failures: list[str] = []
    for split in ("train", "val"):
        written, failures = relabel_split(dataset, split, drop_failures=not args.keep_failures)
        total_written += written
        all_failures.extend(failures)
        print(f"{split}: labels_written={written}, failures={len(failures)}")

    write_data_yaml(dataset)
    print(f"dataset={dataset}")
    print(f"total_labels_written={total_written}")
    if all_failures:
        print("failures:")
        for failure in all_failures[:30]:
            print(f"  {failure}")
        if len(all_failures) > 30:
            print(f"  ... {len(all_failures) - 30} more")


if __name__ == "__main__":
    main()
