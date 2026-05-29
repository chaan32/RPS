"""Generate balanced coordinate-level scenarios for Fusion V2.

This does not touch V1 runtime code.  It creates V1-compatible
``fusion_risk.csv`` files under a separate root so the V2 dataset builder can
train on many SAFE/WARNING/DANGER coordinate sequences.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FPS = 12.0
FRAMES = 120
DROPZONE_ALERT_RADIUS_M = 2.0
WORKSPACE_X = (-12.0, 2.0)
WORKSPACE_Y = (-3.0, 12.0)

CSV_FIELDS = [
    "frame", "time_s", "model_tick", "worker_id",
    "raw_worker_x", "raw_worker_y", "worker_x", "worker_y",
    "raw_forklift_x", "raw_forklift_y", "forklift_x", "forklift_y",
    "forklift_hazard_x", "forklift_hazard_y",
    "raw_dropzone_x", "raw_dropzone_y", "dropzone_x", "dropzone_y",
    "worker_tracker_residual_m", "worker_tracker_outlier",
    "forklift_tracker_residual_m", "forklift_tracker_outlier",
    "worker_forklift_dist", "worker_forklift_hazard_dist", "worker_dropzone_dist",
    "forklift_risk", "dropzone_risk", "dropzone_forced",
    "early_level", "early_reason", "early_ttc_s", "early_closest_distance_m",
    "early_current_distance_m",
]


@dataclass(frozen=True)
class Trajectory:
    xy: np.ndarray


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a[None, :] + (b - a)[None, :] * t[:, None]


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _clip(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[:, 0] = np.clip(out[:, 0], *WORKSPACE_X)
    out[:, 1] = np.clip(out[:, 1], *WORKSPACE_Y)
    return out


def _line(start: tuple[float, float], end: tuple[float, float], *, smooth: bool = True) -> Trajectory:
    t = np.linspace(0.0, 1.0, FRAMES, dtype=np.float32)
    if smooth:
        t = _smooth(t)
    return Trajectory(_clip(_lerp(np.array(start), np.array(end), t)).astype(np.float32))


def _static(xy: tuple[float, float]) -> Trajectory:
    return Trajectory(np.tile(np.array(xy, dtype=np.float32), (FRAMES, 1)))


def _velocity(xy: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy, dtype=np.float32)
    vel[1:] = (xy[1:] - xy[:-1]) * FPS
    vel[0] = vel[1]
    return vel


def _hazard_point(forklift: np.ndarray) -> np.ndarray:
    vel = _velocity(forklift)
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    unit = np.divide(vel, np.maximum(speed, 1e-6))
    return forklift + unit * 1.0


def _closest_approach(
    worker_xy: np.ndarray,
    worker_v: np.ndarray,
    threat_xy: np.ndarray,
    threat_v: np.ndarray,
    horizon_s: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = worker_xy - threat_xy
    rel_v = worker_v - threat_v
    rel_speed_sq = (rel_v * rel_v).sum(axis=1)
    dot = (rel * rel_v).sum(axis=1)
    approaching = dot < 0.0
    ttc = np.zeros((len(worker_xy),), dtype=np.float32)
    active = rel_speed_sq > 1e-9
    ttc[active] = np.clip(-dot[active] / rel_speed_sq[active], 0.0, horizon_s)
    closest = rel + rel_v * ttc[:, None]
    closest_dist = np.linalg.norm(closest, axis=1).astype(np.float32)
    return ttc, closest_dist, approaching


def _forklift_risk(worker: np.ndarray, hazard: np.ndarray, forklift: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    wv = _velocity(worker)
    hv = _velocity(hazard)
    ttc, closest, approaching = _closest_approach(worker, wv, hazard, hv)
    current = np.linalg.norm(worker - hazard, axis=1).astype(np.float32)
    score = np.zeros((FRAMES,), dtype=np.float32)
    levels: list[str] = []
    for i in range(FRAMES):
        level = "SAFE"
        if approaching[i] and ttc[i] <= 2.5 and closest[i] <= 1.0:
            score[i] = 1.0
            level = "DANGER"
        elif approaching[i] and ttc[i] <= 7.0 and closest[i] <= 1.2:
            score[i] = 0.5
            level = "WARNING"
        # Physical overlap should still become danger even if velocity estimate
        # is momentarily weak.
        if np.linalg.norm(worker[i] - forklift[i]) <= 0.65:
            score[i] = 1.0
            level = "DANGER"
        levels.append(level)
    return score, levels, ttc, closest, current


def _dropzone_risk(worker: np.ndarray, dropzone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.linalg.norm(worker - dropzone, axis=1).astype(np.float32)
    score = np.zeros((FRAMES,), dtype=np.float32)
    score[(dist > DROPZONE_ALERT_RADIUS_M) & (dist <= DROPZONE_ALERT_RADIUS_M + 0.8)] = 0.5
    score[dist <= DROPZONE_ALERT_RADIUS_M] = 1.0
    return score, dist


def _round(value: float | int | str) -> str | int:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    return round(float(value), 4)


def _write_scenario(
    out_root: Path,
    name: str,
    workers: dict[str, Trajectory],
    forklift: Trajectory,
    dropzone: Trajectory,
) -> None:
    scenario_dir = out_root / name / "diagnostics" / "fusion"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    risk_csv = scenario_dir / "fusion_risk.csv"

    forklift_xy = forklift.xy
    hazard_xy = _hazard_point(forklift_xy)
    dropzone_xy = dropzone.xy
    model_period = max(1, int(round(FPS / 5.0)))

    with risk_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame in range(FRAMES):
            for worker_id, trajectory in workers.items():
                worker_xy = trajectory.xy
                fork_score, levels, ttc, closest, current = _forklift_risk(
                    worker_xy, hazard_xy, forklift_xy
                )
                dz_score, dz_dist = _dropzone_risk(worker_xy, dropzone_xy)
                wf_dist = float(np.linalg.norm(worker_xy[frame] - forklift_xy[frame]))
                wh_dist = float(np.linalg.norm(worker_xy[frame] - hazard_xy[frame]))
                wd_dist = float(dz_dist[frame])
                writer.writerow({
                    "frame": frame,
                    "time_s": _round(frame / FPS),
                    "model_tick": int(frame % model_period == 0),
                    "worker_id": worker_id,
                    "raw_worker_x": _round(worker_xy[frame, 0]),
                    "raw_worker_y": _round(worker_xy[frame, 1]),
                    "worker_x": _round(worker_xy[frame, 0]),
                    "worker_y": _round(worker_xy[frame, 1]),
                    "raw_forklift_x": _round(forklift_xy[frame, 0]),
                    "raw_forklift_y": _round(forklift_xy[frame, 1]),
                    "forklift_x": _round(forklift_xy[frame, 0]),
                    "forklift_y": _round(forklift_xy[frame, 1]),
                    "forklift_hazard_x": _round(hazard_xy[frame, 0]),
                    "forklift_hazard_y": _round(hazard_xy[frame, 1]),
                    "raw_dropzone_x": _round(dropzone_xy[frame, 0]),
                    "raw_dropzone_y": _round(dropzone_xy[frame, 1]),
                    "dropzone_x": _round(dropzone_xy[frame, 0]),
                    "dropzone_y": _round(dropzone_xy[frame, 1]),
                    "worker_tracker_residual_m": 0.0,
                    "worker_tracker_outlier": 0,
                    "forklift_tracker_residual_m": 0.0,
                    "forklift_tracker_outlier": 0,
                    "worker_forklift_dist": _round(wf_dist),
                    "worker_forklift_hazard_dist": _round(wh_dist),
                    "worker_dropzone_dist": _round(wd_dist),
                    "forklift_risk": _round(fork_score[frame]),
                    "dropzone_risk": _round(dz_score[frame]),
                    "dropzone_forced": int(dz_score[frame] >= 1.0),
                    "early_level": levels[frame],
                    "early_reason": levels[frame].lower(),
                    "early_ttc_s": "" if levels[frame] == "SAFE" else _round(ttc[frame]),
                    "early_closest_distance_m": "" if levels[frame] == "SAFE" else _round(closest[frame]),
                    "early_current_distance_m": _round(current[frame]),
                })

    # Minimal ground truth for traceability.
    gt_path = out_root / name / "ground_truth.csv"
    with gt_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "object", "world_x", "world_y"])
        for frame in range(FRAMES):
            t = frame / FPS
            writer.writerow([frame, round(t, 4), "forklift", *np.round(forklift_xy[frame], 4)])
            writer.writerow([frame, round(t, 4), "dropzone", *np.round(dropzone_xy[frame], 4)])
            for worker_id, trajectory in workers.items():
                writer.writerow([frame, round(t, 4), worker_id, *np.round(trajectory.xy[frame], 4)])


def _scenario(kind: str, rng: np.random.Generator, idx: int) -> tuple[str, dict[str, Trajectory], Trajectory, Trajectory]:
    jitter = lambda s=0.25: rng.normal(0.0, s, size=2)
    dz_center = np.array([-8.0, 2.0], dtype=np.float32) + jitter(0.6)
    distractor = _line((-10.5, 9.8), (-9.0, 9.0), smooth=True)

    if kind == "forklift_danger":
        impact = np.array([-5.5, 4.8], dtype=np.float32) + jitter(0.45)
        worker = _line(tuple(impact + np.array([-3.6, 2.4]) + jitter(0.25)), tuple(impact), smooth=True)
        forklift = _line(tuple(impact + np.array([4.4, -4.0]) + jitter(0.25)), tuple(impact), smooth=True)
        dropzone = _static(tuple(dz_center))
    elif kind == "forklift_warning":
        center = np.array([-5.5, 4.8], dtype=np.float32) + jitter(0.45)
        worker = _line(tuple(center + np.array([-3.4, 2.6]) + jitter(0.2)), tuple(center + np.array([-0.8, 0.9])), smooth=True)
        forklift = _line(tuple(center + np.array([4.4, -3.8]) + jitter(0.2)), tuple(center + np.array([0.7, -0.7])), smooth=True)
        dropzone = _static(tuple(dz_center))
    elif kind == "dropzone_danger":
        dropzone = _line(tuple(dz_center + np.array([2.5, 0.6]) + jitter(0.2)), tuple(dz_center), smooth=True)
        worker = _static(tuple(dz_center + np.array([1.2, 0.4]) + jitter(0.2)))
        forklift = _line((-0.5, -2.2), (-0.2, -0.8), smooth=True)
    elif kind == "dropzone_warning":
        dropzone = _line(tuple(dz_center + np.array([3.2, 0.7]) + jitter(0.2)), tuple(dz_center), smooth=True)
        worker = _static(tuple(dz_center + np.array([2.45, 0.3]) + jitter(0.15)))
        forklift = _line((-0.5, -2.2), (-0.2, -0.8), smooth=True)
    elif kind == "combined_danger":
        impact = np.array([-6.8, 4.2], dtype=np.float32) + jitter(0.3)
        worker = _line(tuple(impact + np.array([-2.7, 2.2]) + jitter(0.2)), tuple(impact), smooth=True)
        forklift = _line(tuple(impact + np.array([4.2, -3.0]) + jitter(0.2)), tuple(impact), smooth=True)
        dropzone = _line(tuple(impact + np.array([-2.1, -0.5]) + jitter(0.15)), tuple(impact + np.array([-0.6, 0.2])), smooth=True)
    elif kind == "forklift_opposite_danger":
        # Hard case: the worker approaches the blind corner from the opposite
        # side, similar to the scenario that made V2 react late.
        impact = np.array([-1.1, 4.3], dtype=np.float32) + jitter(0.35)
        worker = _line(tuple(impact + np.array([-4.6, 0.3]) + jitter(0.25)), tuple(impact + np.array([0.2, 0.15])), smooth=True)
        forklift = _line(tuple(impact + np.array([0.8, -6.0]) + jitter(0.25)), tuple(impact + np.array([-0.1, 0.05])), smooth=True)
        dropzone = _static(tuple(dz_center))
    elif kind == "forklift_short_danger":
        # Hard case: the actual danger interval is short, so the future-label
        # horizon should teach the model to raise risk before the overlap.
        impact = np.array([-4.8, 5.1], dtype=np.float32) + jitter(0.25)
        worker = _line(tuple(impact + np.array([-5.0, 1.2]) + jitter(0.2)), tuple(impact + np.array([1.5, -0.2])), smooth=True)
        forklift = _line(tuple(impact + np.array([3.8, -4.2]) + jitter(0.2)), tuple(impact + np.array([-1.2, 1.0])), smooth=True)
        dropzone = _static(tuple(dz_center))
    elif kind == "dropzone_short_danger":
        # Hard case: the box/dropzone briefly crosses the worker radius.
        center = np.array([-8.4, 4.1], dtype=np.float32) + jitter(0.25)
        dropzone = _line(tuple(center + np.array([4.3, -0.7]) + jitter(0.15)), tuple(center + np.array([-1.8, 0.6])), smooth=True)
        worker = _static(tuple(center + np.array([0.2, 0.1]) + jitter(0.12)))
        forklift = _line((-0.6, -2.2), (-0.4, -0.5), smooth=True)
    elif kind == "dropzone_edge_danger":
        # Hard case: the target stays near the danger/warning boundary.  This
        # helps reduce the "stuck at warning" problem.
        center = np.array([-8.2, 3.2], dtype=np.float32) + jitter(0.25)
        dropzone = _line(tuple(center + np.array([3.0, 0.0]) + jitter(0.1)), tuple(center + np.array([-0.1, 0.0])), smooth=True)
        worker = _static(tuple(center + np.array([1.55, 0.25]) + jitter(0.08)))
        forklift = _line((-0.6, -2.2), (-0.4, -0.5), smooth=True)
    else:
        worker = _line(tuple(np.array([-10.5, 8.0]) + jitter(0.5)), tuple(np.array([-9.0, 6.8]) + jitter(0.5)), smooth=True)
        forklift = _line(tuple(np.array([-0.3, -2.4]) + jitter(0.2)), tuple(np.array([-0.1, -1.2]) + jitter(0.2)), smooth=True)
        dropzone = _static(tuple(dz_center + np.array([0.0, -3.0]) + jitter(0.4)))

    name = f"v2_{kind}_{idx:03d}"
    if kind == "dropzone_short_danger":
        workers = {
            "W01": worker,
            "W02": _static(tuple(worker.xy[0] + np.array([2.3, -0.1], dtype=np.float32))),
        }
    elif kind == "dropzone_edge_danger":
        workers = {
            "W01": worker,
            "W02": _static(tuple(worker.xy[0] + np.array([-2.2, 0.2], dtype=np.float32))),
        }
    else:
        workers = {"W01": worker, "W02": distractor}
    return name, workers, forklift, dropzone


def generate(output_root: Path, count_per_kind: int, seed: int, clean: bool) -> int:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kinds = [
        "safe",
        "forklift_warning",
        "forklift_danger",
        "dropzone_warning",
        "dropzone_danger",
        "combined_danger",
        "forklift_opposite_danger",
        "forklift_short_danger",
        "dropzone_short_danger",
        "dropzone_edge_danger",
    ]
    total = 0
    for kind in kinds:
        for idx in range(count_per_kind):
            name, workers, forklift, dropzone = _scenario(kind, rng, idx)
            _write_scenario(output_root, name, workers, forklift, dropzone)
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fusion V2 coordinate scenarios")
    parser.add_argument("--output-root", type=Path, default=Path("model/fusion_v2/generated_scenarios"))
    parser.add_argument("--count-per-kind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    total = generate(args.output_root, args.count_per_kind, args.seed, args.clean)
    print(f"[fusion-v2] generated {total} scenarios under {args.output_root}")


if __name__ == "__main__":
    main()
