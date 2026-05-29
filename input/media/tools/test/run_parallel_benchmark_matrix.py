"""Run repeated realtime-camera benchmarks across scenarios and extraction modes.

This is a test-only harness for comparing:
  - serial extraction
  - camera-level 2-thread extraction
  - model-level 4-thread extraction

It starts MediaMTX, publishes one recorded Unity scenario as RTSP, runs the
realtime loop, parses the JSONL metrics, then repeats for every requested
scenario/mode/repetition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENARIOS = [
    "scenario_01_user_current",
    "scenario_02_swapped_positions",
    "scenario_03_opposite_worker",
]
DEFAULT_MODES = ["serial", "camera_parallel", "model_parallel"]


@dataclass(frozen=True)
class RunSpec:
    mode: str
    scenario: str
    repeat_index: int

    @property
    def run_label(self) -> str:
        return f"{self.mode}__{self.scenario}__r{self.repeat_index:02d}"


def percentile(values: list[float], pct: float) -> float:
    """Return an interpolated percentile for a metric list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    """Average a numeric metric field across JSONL rows."""
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), int | float)
    ]
    return statistics.fmean(values) if values else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read realtime_camera JSONL metrics, skipping malformed lines."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def duration_from_rows(rows: list[dict[str, Any]]) -> float:
    """Estimate benchmark duration from elapsed_s, falling back to ts."""
    elapsed = [
        float(row["elapsed_s"])
        for row in rows
        if isinstance(row.get("elapsed_s"), int | float)
    ]
    if len(elapsed) >= 2:
        return max(elapsed) - min(elapsed)

    ts_values = [
        float(row["ts"])
        for row in rows
        if isinstance(row.get("ts"), int | float)
    ]
    if len(ts_values) >= 2:
        return max(ts_values) - min(ts_values)
    return 0.0


def summarize_run(
    spec: RunSpec,
    metrics_path: Path,
    returncode: int | None,
) -> dict[str, Any]:
    """Convert one benchmark JSONL file into one CSV row."""
    rows = read_jsonl(metrics_path)
    duration_s = duration_from_rows(rows)
    fps = len(rows) / duration_s if duration_s > 0 else 0.0
    loop_values = [
        float(row["perf.loop.total_ms"])
        for row in rows
        if isinstance(row.get("perf.loop.total_ms"), int | float)
    ]

    worker_rows = sum(1 for row in rows if int(row.get("n_workers", 0) or 0) > 0)
    prediction_rows = sum(1 for row in rows if int(row.get("n_predictions", 0) or 0) > 0)
    forklift_rows = sum(
        1
        for row in rows
        if int(row.get("has_raw_forklift", 0) or 0)
        or int(row.get("has_tracked_forklift", 0) or 0)
    )

    issues: list[str] = []
    if not rows:
        issues.append("no_metric_rows")
    if rows and forklift_rows / len(rows) < 0.8:
        issues.append("forklift_low_detection")
    if rows and worker_rows == 0:
        issues.append("worker_not_detected")
    if returncode not in (0, None):
        issues.append(f"returncode_{returncode}")

    return {
        "mode": spec.mode,
        "scenario": spec.scenario,
        "repeat": spec.repeat_index,
        "run_label": spec.run_label,
        "returncode": returncode if returncode is not None else "",
        "rows": len(rows),
        "duration_s": round(duration_s, 3),
        "effective_fps": round(fps, 3),
        "loop_mean_ms": round(mean_field(rows, "perf.loop.total_ms"), 3),
        "loop_p95_ms": round(percentile(loop_values, 0.95), 3),
        "extract_pair_wall_mean_ms": round(
            mean_field(rows, "perf.pipeline.extract_pair_wall_ms"),
            3,
        ),
        "cam1_extract_mean_ms": round(
            mean_field(rows, "perf.pipeline.extract.cam1_ms"),
            3,
        ),
        "cam2_extract_mean_ms": round(
            mean_field(rows, "perf.pipeline.extract.cam2_ms"),
            3,
        ),
        "cam1_pose_mean_ms": round(mean_field(rows, "perf.model.pose.cam1_ms"), 3),
        "cam2_pose_mean_ms": round(mean_field(rows, "perf.model.pose.cam2_ms"), 3),
        "cam1_custom_mean_ms": round(
            mean_field(rows, "perf.model.custom_yolo.cam1_ms"),
            3,
        ),
        "cam2_custom_mean_ms": round(
            mean_field(rows, "perf.model.custom_yolo.cam2_ms"),
            3,
        ),
        "cam1_pose_skipped_rate": round(
            mean_field(rows, "cam1_pose_inference_skipped"),
            4,
        ),
        "cam2_pose_skipped_rate": round(
            mean_field(rows, "cam2_pose_inference_skipped"),
            4,
        ),
        "worker_detected_rows": worker_rows,
        "worker_detected_rate": round(worker_rows / len(rows), 4) if rows else 0.0,
        "forklift_detected_rows": forklift_rows,
        "forklift_detected_rate": round(forklift_rows / len(rows), 4) if rows else 0.0,
        "prediction_rows": prediction_rows,
        "prediction_rate": round(prediction_rows / len(rows), 4) if rows else 0.0,
        "issues": ";".join(issues),
        "metrics_path": str(metrics_path),
    }


def terminate_process(proc: subprocess.Popen[Any] | None) -> None:
    """Terminate a child process without leaving ffmpeg or MediaMTX behind."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_mediamtx(config_path: Path, log_path: Path) -> subprocess.Popen[Any]:
    """Start MediaMTX for the benchmark RTSP port."""
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["mediamtx", str(config_path)],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.0)
    if proc.poll() is not None:
        log_file.close()
        raise RuntimeError(
            f"MediaMTX exited early with code {proc.returncode}. "
            f"See {log_path}"
        )
    return proc


def start_bridge(
    spec: RunSpec,
    args: argparse.Namespace,
    log_path: Path,
) -> subprocess.Popen[Any]:
    """Publish the current scenario frames to cam1/cam2 RTSP paths."""
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "input/media/tools/stream_collision_scenario_rtsp.py",
        "--scenario",
        spec.scenario,
        "--rtsp-base",
        args.rtsp_base,
        "--duration",
        str(args.bridge_duration),
        "--loglevel",
        args.ffmpeg_loglevel,
    ]
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def run_realtime(
    spec: RunSpec,
    args: argparse.Namespace,
    metrics_path: Path,
    log_path: Path,
) -> int:
    """Run realtime_camera once and write stdout/stderr to a per-run log."""
    env = os.environ.copy()
    env.update(
        {
            "HEADLESS": "1",
            "CAMERA_RTSP_URL_1": f"{args.rtsp_base.rstrip('/')}/cam1",
            "CAMERA_RTSP_URL_2": f"{args.rtsp_base.rstrip('/')}/cam2",
            "FUSION_SERVER_URL": args.fusion_server_url,
            "MQTT_BROKER": args.mqtt_broker,
            "LOCAL_SNAPSHOT_PATH": str(PROJECT_ROOT / "snapshots"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "model.fusion.runtime.realtime_camera",
        "--no-audio",
        "--no-prompt",
        "--duration",
        str(args.duration),
        "--target-rate",
        "0",
        "--extract-mode",
        spec.mode,
        "--run-label",
        spec.run_label,
        "--metrics-path",
        str(metrics_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=args.duration + args.realtime_timeout_buffer,
            check=False,
        )
    return completed.returncode


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run benchmark rows by mode and scenario."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["scenario"]), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (mode, scenario), group in sorted(groups.items()):
        fps_values = [float(row["effective_fps"]) for row in group]
        loop_values = [float(row["loop_mean_ms"]) for row in group]
        extract_values = [float(row["extract_pair_wall_mean_ms"]) for row in group]
        pose_skip_values = [
            (
                float(row.get("cam1_pose_skipped_rate", 0.0))
                + float(row.get("cam2_pose_skipped_rate", 0.0))
            ) / 2.0
            for row in group
        ]
        issue_runs = [row["run_label"] for row in group if row["issues"]]
        aggregated.append(
            {
                "mode": mode,
                "scenario": scenario,
                "runs": len(group),
                "fps_mean": round(statistics.fmean(fps_values), 3),
                "fps_min": round(min(fps_values), 3),
                "fps_max": round(max(fps_values), 3),
                "fps_stdev": round(statistics.stdev(fps_values), 3)
                if len(fps_values) >= 2
                else 0.0,
                "loop_mean_ms": round(statistics.fmean(loop_values), 3),
                "extract_pair_wall_mean_ms": round(statistics.fmean(extract_values), 3),
                "pose_skipped_rate_mean": round(statistics.fmean(pose_skip_values), 4),
                "worker_detected_rate_mean": round(
                    statistics.fmean(float(row["worker_detected_rate"]) for row in group),
                    4,
                ),
                "forklift_detected_rate_mean": round(
                    statistics.fmean(float(row["forklift_detected_rate"]) for row in group),
                    4,
                ),
                "prediction_rate_mean": round(
                    statistics.fmean(float(row["prediction_rate"]) for row in group),
                    4,
                ),
                "issue_runs": ";".join(issue_runs),
            }
        )
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--bridge-duration", type=float, default=55.0)
    parser.add_argument("--startup-wait", type=float, default=3.0)
    parser.add_argument("--realtime-timeout-buffer", type=float, default=90.0)
    parser.add_argument("--rtsp-base", default="rtsp://localhost:8555")
    parser.add_argument(
        "--mediamtx-config",
        type=Path,
        default=PROJECT_ROOT / "input/media/tools/test/mediamtx_rtsp_8555.yml",
    )
    parser.add_argument("--ffmpeg-loglevel", default="error")
    parser.add_argument("--fusion-server-url", default="http://127.0.0.1:1122")
    parser.add_argument("--mqtt-broker", default="127.0.0.1")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "metrics/parallel_matrix_20260519",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        RunSpec(mode=mode, scenario=scenario, repeat_index=repeat)
        for mode in args.modes
        for scenario in args.scenarios
        for repeat in range(1, args.repeats + 1)
    ]

    print(
        f"[matrix] total_runs={len(specs)} "
        f"modes={','.join(args.modes)} scenarios={','.join(args.scenarios)}"
    )
    print(f"[matrix] out_dir={out_dir}")

    mediamtx_proc: subprocess.Popen[Any] | None = None
    run_rows: list[dict[str, Any]] = []
    started_at = time.time()

    def stop_children(signum: int, frame: object) -> None:  # noqa: ARG001
        terminate_process(mediamtx_proc)
        raise SystemExit(128 + signum)

    previous_sigint = signal.signal(signal.SIGINT, stop_children)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_children)

    try:
        mediamtx_proc = start_mediamtx(
            args.mediamtx_config.resolve(),
            logs_dir / "mediamtx.log",
        )

        for index, spec in enumerate(specs, start=1):
            metrics_path = out_dir / f"{spec.run_label}.jsonl"
            bridge_log = logs_dir / f"{spec.run_label}.bridge.log"
            realtime_log = logs_dir / f"{spec.run_label}.realtime.log"
            bridge_proc: subprocess.Popen[Any] | None = None
            returncode: int | None = None

            print(
                f"[matrix] {index:02d}/{len(specs):02d} "
                f"mode={spec.mode} scenario={spec.scenario} repeat={spec.repeat_index}"
            )
            try:
                bridge_proc = start_bridge(spec, args, bridge_log)
                time.sleep(args.startup_wait)
                returncode = run_realtime(spec, args, metrics_path, realtime_log)
            except subprocess.TimeoutExpired:
                returncode = 124
            finally:
                terminate_process(bridge_proc)

            row = summarize_run(spec, metrics_path, returncode)
            run_rows.append(row)
            write_csv(out_dir / "runs.csv", run_rows)
            write_csv(out_dir / "aggregate_by_mode_scenario.csv", aggregate_rows(run_rows))

            issue_text = f" issues={row['issues']}" if row["issues"] else ""
            print(
                f"[matrix] done fps={row['effective_fps']} "
                f"loop={row['loop_mean_ms']}ms "
                f"worker_rate={row['worker_detected_rate']} "
                f"forklift_rate={row['forklift_detected_rate']}{issue_text}"
            )

    finally:
        terminate_process(mediamtx_proc)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    elapsed = time.time() - started_at
    print(f"[matrix] finished in {elapsed:.1f}s")
    print(f"[matrix] runs_csv={out_dir / 'runs.csv'}")
    print(f"[matrix] aggregate_csv={out_dir / 'aggregate_by_mode_scenario.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
