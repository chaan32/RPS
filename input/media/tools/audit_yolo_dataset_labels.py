"""Audit YOLO label geometry for Unity-generated datasets.

This catches the failure mode where synthetic labels cover most of the image,
which trains a detector to classify the whole scene as the object.

Example:
    python input/media/tools/audit_yolo_dataset_labels.py \
      --dataset simulation/Assets/Dataset \
      --out simulation/Recordings/diagnostics/dataset_label_audit.jpg
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Label:
    split: str
    label_path: Path
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


def read_names(dataset: Path) -> dict[int, str]:
    yaml_path = dataset / "data.yaml"
    if not yaml_path.exists():
        return {}
    names: dict[int, str] = {}
    in_names = False
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            m = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if not m:
                if not raw.startswith(" "):
                    in_names = False
                continue
            names[int(m.group(1))] = m.group(2).strip().strip("'\"")
    return names


def iter_labels(dataset: Path, split: str) -> tuple[list[Label], int, int]:
    label_dir = dataset / "labels" / split
    labels: list[Label] = []
    empty_files = 0
    invalid_lines = 0
    if not label_dir.exists():
        return labels, empty_files, invalid_lines

    for path in sorted(label_dir.glob("*.txt")):
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_files += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                invalid_lines += 1
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                invalid_lines += 1
                continue
            labels.append(Label(split, path, cls_id, cx, cy, w, h))
    return labels, empty_files, invalid_lines


def find_image(dataset: Path, label: Label) -> Path | None:
    image_dir = dataset / "images" / label.split
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{label.label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def draw_label(dataset: Path, label: Label, names: dict[int, str]) -> np.ndarray | None:
    image_path = find_image(dataset, label)
    if image_path is None:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h_img, w_img = image.shape[:2]
    label_file = label.label_path
    for raw in label_file.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        x1 = int(round((cx - bw / 2) * w_img))
        y1 = int(round((cy - bh / 2) * h_img))
        x2 = int(round((cx + bw / 2) * w_img))
        y2 = int(round((cy + bh / 2) * h_img))
        color = (0, 0, 255) if cls_id == label.cls_id else (255, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cls_name = names.get(cls_id, str(cls_id))
        cv2.putText(
            image,
            f"{cls_name} {bw * bh:.2f}",
            (max(0, x1 + 4), max(26, y1 + 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(image, (0, 0), (w_img, 36), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"{label.split}/{label.label_path.stem}",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def save_contact_sheet(
    dataset: Path,
    labels: list[Label],
    names: dict[int, str],
    out_path: Path,
    limit: int,
) -> None:
    selected = sorted(labels, key=lambda item: item.area, reverse=True)[:limit]
    thumbs: list[np.ndarray] = []
    for label in selected:
        image = draw_label(dataset, label, names)
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return

    rows = []
    for idx in range(0, len(thumbs), 2):
        row = thumbs[idx : idx + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def print_summary(
    split: str,
    labels: list[Label],
    empty_files: int,
    invalid_lines: int,
    names: dict[int, str],
    large_area: float,
    large_dim: float,
) -> None:
    files = {label.label_path for label in labels}
    print(f"[{split}] files_with_labels={len(files)} labels={len(labels)} empty={empty_files} invalid={invalid_lines}")
    by_cls: dict[int, list[Label]] = {}
    for label in labels:
        by_cls.setdefault(label.cls_id, []).append(label)
    for cls_id in sorted(by_cls):
        cls_labels = by_cls[cls_id]
        areas = [label.area for label in cls_labels]
        widths = [label.w for label in cls_labels]
        heights = [label.h for label in cls_labels]
        too_large = [
            label for label in cls_labels
            if label.area > large_area or label.w > large_dim or label.h > large_dim
        ]
        cls_name = names.get(cls_id, str(cls_id))
        print(
            f"  cls={cls_id} {cls_name}: count={len(cls_labels)} "
            f"area median={median(areas):.3f} min={min(areas):.3f} max={max(areas):.3f} "
            f"w_med={median(widths):.3f} h_med={median(heights):.3f} "
            f"too_large={len(too_large)}/{len(cls_labels)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="simulation/Assets/Dataset")
    parser.add_argument("--out", default="simulation/Recordings/diagnostics/dataset_label_audit.jpg")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--large-area", type=float, default=0.55)
    parser.add_argument("--large-dim", type=float, default=0.92)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    out_path = Path(args.out).resolve()
    names = read_names(dataset)
    all_labels: list[Label] = []

    print(f"dataset={dataset}")
    print(f"names={names}")
    for split in ("train", "val"):
        labels, empty_files, invalid_lines = iter_labels(dataset, split)
        all_labels.extend(labels)
        print_summary(split, labels, empty_files, invalid_lines, names, args.large_area, args.large_dim)

    save_contact_sheet(dataset, all_labels, names, out_path, args.limit)
    if out_path.exists():
        print(f"contact_sheet={out_path}")


if __name__ == "__main__":
    main()
