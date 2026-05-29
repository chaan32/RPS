"""Build and load Fusion V2 sequence datasets.

Input source:
  simulation/Recordings/collision_scenarios/*/diagnostics/fusion/fusion_risk.csv

The CSVs are produced by the working V1 pipeline.  V2 uses their BEV/world
coordinate traces as input.  Labels can be built either from the V1 teacher
outputs or independently from future geometric overlap in absolute coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import FEATURE_COLUMNS, LABEL_COLUMNS


EARLY_LEVEL_SCORE = {
    "SAFE": 0.0,
    "WARNING": 0.5,
    "DANGER": 1.0,
    "safe": 0.0,
    "warning": 0.5,
    "danger": 1.0,
    "": 0.0,
}

LABEL_MODES = ("teacher", "geometry_future")
DEFAULT_FUTURE_HORIZON_FRAMES = 12
DEFAULT_FORKLIFT_DANGER_M = 1.25
DEFAULT_FORKLIFT_WARNING_M = 2.4
DEFAULT_DROPZONE_DANGER_M = 2.0
DEFAULT_DROPZONE_WARNING_M = 2.8


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    n_windows: int
    n_scenarios: int
    feature_dim: int
    window_size: int


def _float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _ffill_bfill(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    if out.ndim == 1:
        mask = np.isfinite(out)
        if not mask.any():
            out[:] = fill_value
            return out
        first = np.argmax(mask)
        out[:first] = out[first]
        for i in range(first + 1, len(out)):
            if not np.isfinite(out[i]):
                out[i] = out[i - 1]
        return out

    for col in range(out.shape[1]):
        out[:, col] = _ffill_bfill(out[:, col], fill_value)
    return out


def _velocity(xy: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy, dtype=np.float32)
    if len(xy) < 2:
        return vel
    dt = np.diff(time_s).astype(np.float32)
    dt = np.where(dt <= 1e-6, 1.0, dt)
    vel[1:] = (xy[1:] - xy[:-1]) / dt[:, None]
    vel[0] = vel[1]
    return vel


def _label_from_row(row: dict[str, str]) -> tuple[float, float]:
    forklift_risk = _float(row.get("forklift_risk"), 0.0)
    dropzone_risk = _float(row.get("dropzone_risk"), 0.0)
    early_score = EARLY_LEVEL_SCORE.get(row.get("early_level", ""), 0.0)
    dropzone_forced = 1.0 if _int(row.get("dropzone_forced"), 0) else 0.0
    return (
        float(max(forklift_risk, early_score)),
        float(max(dropzone_risk, dropzone_forced)),
    )


def _geometry_instant_labels(
    features: np.ndarray,
    *,
    forklift_danger_m: float,
    forklift_warning_m: float,
    dropzone_danger_m: float,
    dropzone_warning_m: float,
) -> np.ndarray:
    """Build labels only from absolute coordinates, not from V1 risk columns.

    Forklift risk uses the smaller distance to the forklift center and the
    forklift front-hazard point.  DropZone risk uses the distance to the live
    box/dropzone center.  These are instant scores; future labels are generated
    by taking the max score over the next N frames.
    """
    worker_xy = features[:, 0:2]
    forklift_xy = features[:, 2:4]
    forklift_hazard_xy = features[:, 4:6]
    dropzone_xy = features[:, 6:8]
    has_forklift = features[:, 19] > 0.5
    has_dropzone = features[:, 20] > 0.5

    forklift_center_dist = np.linalg.norm(worker_xy - forklift_xy, axis=1)
    forklift_hazard_dist = np.linalg.norm(worker_xy - forklift_hazard_xy, axis=1)
    forklift_dist = np.minimum(forklift_center_dist, forklift_hazard_dist)
    dropzone_dist = np.linalg.norm(worker_xy - dropzone_xy, axis=1)

    forklift_score = np.zeros((len(features),), dtype=np.float32)
    forklift_score[np.logical_and(has_forklift, forklift_dist <= forklift_warning_m)] = 0.5
    forklift_score[np.logical_and(has_forklift, forklift_dist <= forklift_danger_m)] = 1.0

    dropzone_score = np.zeros((len(features),), dtype=np.float32)
    dropzone_score[np.logical_and(has_dropzone, dropzone_dist <= dropzone_warning_m)] = 0.5
    dropzone_score[np.logical_and(has_dropzone, dropzone_dist <= dropzone_danger_m)] = 1.0

    return np.column_stack([forklift_score, dropzone_score]).astype(np.float32)


def _future_max_labels(labels: np.ndarray, horizon_frames: int) -> np.ndarray:
    """Promote each frame label when danger appears within the future horizon."""
    if horizon_frames <= 0 or len(labels) == 0:
        return labels.astype(np.float32, copy=True)
    out = np.zeros_like(labels, dtype=np.float32)
    for i in range(len(labels)):
        end = min(len(labels), i + horizon_frames + 1)
        out[i] = labels[i:end].max(axis=0)
    return out


def _labels_for_mode(
    rows: list[dict[str, str]],
    features: np.ndarray,
    *,
    label_mode: str,
    future_horizon_frames: int,
    forklift_danger_m: float,
    forklift_warning_m: float,
    dropzone_danger_m: float,
    dropzone_warning_m: float,
) -> np.ndarray:
    if label_mode == "teacher":
        return np.array([_label_from_row(r) for r in rows], dtype=np.float32)
    if label_mode == "geometry_future":
        instant = _geometry_instant_labels(
            features,
            forklift_danger_m=forklift_danger_m,
            forklift_warning_m=forklift_warning_m,
            dropzone_danger_m=dropzone_danger_m,
            dropzone_warning_m=dropzone_warning_m,
        )
        return _future_max_labels(instant, future_horizon_frames)
    raise ValueError(f"unknown label_mode: {label_mode}")


def _feature_table(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.array([_float(r.get("time_s"), 0.0) for r in rows], dtype=np.float32)

    worker_xy = _ffill_bfill(np.array([
        [_float(r.get("worker_x")), _float(r.get("worker_y"))]
        for r in rows
    ], dtype=np.float32))
    forklift_xy_raw = np.array([
        [_float(r.get("forklift_x")), _float(r.get("forklift_y"))]
        for r in rows
    ], dtype=np.float32)
    forklift_hazard_xy_raw = np.array([
        [_float(r.get("forklift_hazard_x")), _float(r.get("forklift_hazard_y"))]
        for r in rows
    ], dtype=np.float32)
    dropzone_xy_raw = np.array([
        [_float(r.get("dropzone_x")), _float(r.get("dropzone_y"))]
        for r in rows
    ], dtype=np.float32)

    has_forklift = np.isfinite(forklift_xy_raw).all(axis=1).astype(np.float32)
    has_dropzone = np.isfinite(dropzone_xy_raw).all(axis=1).astype(np.float32)

    forklift_xy = _ffill_bfill(forklift_xy_raw)
    forklift_hazard_xy = _ffill_bfill(forklift_hazard_xy_raw)
    dropzone_xy = _ffill_bfill(dropzone_xy_raw)

    worker_v = _velocity(worker_xy, time_s)
    forklift_v = _velocity(forklift_xy, time_s)
    forklift_hazard_v = _velocity(forklift_hazard_xy, time_s)
    dropzone_v = _velocity(dropzone_xy, time_s)

    distances = np.array([
        [
            _float(r.get("worker_forklift_dist"), 20.0),
            _float(r.get("worker_forklift_hazard_dist"), 20.0),
            _float(r.get("worker_dropzone_dist"), 20.0),
        ]
        for r in rows
    ], dtype=np.float32)
    distances = _ffill_bfill(distances, fill_value=20.0)

    outliers = np.array([
        [
            _float(r.get("worker_tracker_outlier"), 0.0),
            _float(r.get("forklift_tracker_outlier"), 0.0),
        ]
        for r in rows
    ], dtype=np.float32)

    features = np.column_stack([
        worker_xy,
        forklift_xy,
        forklift_hazard_xy,
        dropzone_xy,
        distances,
        worker_v,
        forklift_v,
        forklift_hazard_v,
        dropzone_v,
        has_forklift[:, None],
        has_dropzone[:, None],
        outliers,
    ]).astype(np.float32)

    labels = np.array([_label_from_row(r) for r in rows], dtype=np.float32)
    return features, labels, time_s


def _windowize(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ends = [], [], []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        xs.append(features[start:end])
        ys.append(labels[end - 1])
        ends.append(end - 1)
    if not xs:
        return (
            np.zeros((0, window_size, features.shape[1]), dtype=np.float32),
            np.zeros((0, len(LABEL_COLUMNS)), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return np.stack(xs), np.stack(ys), np.array(ends, dtype=np.int32)


def _augment_windows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_aug: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_aug <= 0 or len(x) == 0:
        return x, y, np.zeros((len(x),), dtype=np.int8)

    batches = [x]
    labels = [y]
    aug_flags = [np.zeros((len(x),), dtype=np.int8)]

    # Coordinate, distance, and velocity columns are continuous.  Presence and
    # outlier flags are kept untouched.
    flag_start = len(FEATURE_COLUMNS) - 4
    for _ in range(n_aug):
        jitter = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        jitter[:, :, flag_start:] = 0.0
        batches.append(x + jitter)
        labels.append(y.copy())
        aug_flags.append(np.ones((len(x),), dtype=np.int8))

    return np.concatenate(batches), np.concatenate(labels), np.concatenate(aug_flags)


def _find_csv_paths(input_roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in input_roots:
        paths.extend(sorted(root.glob("*/diagnostics/fusion/fusion_risk.csv")))
    return sorted(set(paths))


def build_dataset(
    *,
    input_root: Path | list[Path],
    output_path: Path,
    window_size: int = 24,
    stride: int = 2,
    augment: int = 0,
    noise_std: float = 0.03,
    seed: int = 42,
    label_mode: str = "teacher",
    future_horizon_frames: int = DEFAULT_FUTURE_HORIZON_FRAMES,
    forklift_danger_m: float = DEFAULT_FORKLIFT_DANGER_M,
    forklift_warning_m: float = DEFAULT_FORKLIFT_WARNING_M,
    dropzone_danger_m: float = DEFAULT_DROPZONE_DANGER_M,
    dropzone_warning_m: float = DEFAULT_DROPZONE_WARNING_M,
) -> BuildResult:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"--label-mode must be one of {LABEL_MODES}, got {label_mode}")
    input_roots = [input_root] if isinstance(input_root, Path) else list(input_root)
    csv_paths = _find_csv_paths(input_roots)
    if not csv_paths:
        raise FileNotFoundError(f"no fusion_risk.csv files found under {input_roots}")

    rng = np.random.default_rng(seed)
    x_parts, y_parts = [], []
    scenario_parts, worker_parts, end_frame_parts, aug_parts = [], [], [], []

    for csv_path in csv_paths:
        scenario = csv_path.parents[2].name
        rows = _read_csv(csv_path)
        by_worker: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_worker.setdefault(row.get("worker_id") or "W00", []).append(row)

        for worker_id, worker_rows in sorted(by_worker.items()):
            worker_rows.sort(key=lambda r: _int(r.get("frame"), 0))
            features, _, _ = _feature_table(worker_rows)
            labels = _labels_for_mode(
                worker_rows,
                features,
                label_mode=label_mode,
                future_horizon_frames=future_horizon_frames,
                forklift_danger_m=forklift_danger_m,
                forklift_warning_m=forklift_warning_m,
                dropzone_danger_m=dropzone_danger_m,
                dropzone_warning_m=dropzone_warning_m,
            )
            x, y, end_idx = _windowize(
                features, labels, window_size=window_size, stride=stride
            )
            if len(x) == 0:
                continue
            x, y, aug_flags = _augment_windows(
                x, y, n_aug=augment, noise_std=noise_std, rng=rng
            )

            frame_values = np.array([
                _int(worker_rows[min(int(i), len(worker_rows) - 1)].get("frame"), int(i))
                for i in np.resize(end_idx, len(x))
            ], dtype=np.int32)

            x_parts.append(x)
            y_parts.append(y)
            scenario_parts.append(np.array([scenario] * len(x), dtype=object))
            worker_parts.append(np.array([worker_id] * len(x), dtype=object))
            end_frame_parts.append(frame_values)
            aug_parts.append(aug_flags)

    if not x_parts:
        raise RuntimeError("no windows were created; reduce --window-size")

    x_all = np.concatenate(x_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.float32)
    scenarios = np.concatenate(scenario_parts)
    workers = np.concatenate(worker_parts)
    end_frames = np.concatenate(end_frame_parts)
    aug_flags = np.concatenate(aug_parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_columns": list(FEATURE_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "window_size": window_size,
        "stride": stride,
        "augment": augment,
        "noise_std": noise_std,
        "label_mode": label_mode,
        "future_horizon_frames": future_horizon_frames,
        "forklift_danger_m": forklift_danger_m,
        "forklift_warning_m": forklift_warning_m,
        "dropzone_danger_m": dropzone_danger_m,
        "dropzone_warning_m": dropzone_warning_m,
        "source": [str(root) for root in input_roots],
    }
    np.savez_compressed(
        output_path,
        x=x_all,
        y=y_all,
        scenario=scenarios,
        worker_id=workers,
        end_frame=end_frames,
        augmented=aug_flags,
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return BuildResult(
        output_path=output_path,
        n_windows=len(x_all),
        n_scenarios=len(set(scenarios.tolist())),
        feature_dim=x_all.shape[-1],
        window_size=window_size,
    )


class FusionV2Dataset(Dataset):
    """Torch Dataset for Fusion V2 windows."""

    def __init__(
        self,
        npz_path: Path,
        indices: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.scenario = data["scenario"]
        self.worker_id = data["worker_id"]
        self.end_frame = data["end_frame"]
        self.indices = (
            np.arange(len(self.x), dtype=np.int64)
            if indices is None
            else indices.astype(np.int64)
        )
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        x = self.x[real_idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x).float(), torch.from_numpy(self.y[real_idx]).float()


def load_metadata(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return json.loads(str(data["meta"].item()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fusion V2 sequence dataset")
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        default=Path("simulation/Recordings/collision_scenarios"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/fusion_v2/data/fusion_v2_dataset.npz"),
    )
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--augment", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="teacher")
    parser.add_argument("--future-horizon-frames", type=int, default=DEFAULT_FUTURE_HORIZON_FRAMES)
    parser.add_argument("--forklift-danger-m", type=float, default=DEFAULT_FORKLIFT_DANGER_M)
    parser.add_argument("--forklift-warning-m", type=float, default=DEFAULT_FORKLIFT_WARNING_M)
    parser.add_argument("--dropzone-danger-m", type=float, default=DEFAULT_DROPZONE_DANGER_M)
    parser.add_argument("--dropzone-warning-m", type=float, default=DEFAULT_DROPZONE_WARNING_M)
    args = parser.parse_args()

    result = build_dataset(
        input_root=args.input_root if isinstance(args.input_root, list) else [args.input_root],
        output_path=args.output,
        window_size=args.window_size,
        stride=args.stride,
        augment=args.augment,
        noise_std=args.noise_std,
        seed=args.seed,
        label_mode=args.label_mode,
        future_horizon_frames=args.future_horizon_frames,
        forklift_danger_m=args.forklift_danger_m,
        forklift_warning_m=args.forklift_warning_m,
        dropzone_danger_m=args.dropzone_danger_m,
        dropzone_warning_m=args.dropzone_warning_m,
    )
    print(f"[fusion-v2] saved: {result.output_path}")
    print(
        f"[fusion-v2] windows={result.n_windows} scenarios={result.n_scenarios} "
        f"window={result.window_size} feature_dim={result.feature_dim}"
    )


if __name__ == "__main__":
    main()
