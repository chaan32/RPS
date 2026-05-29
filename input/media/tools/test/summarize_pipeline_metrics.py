"""Summarize realtime_camera JSONL benchmark metrics.

Use this after a Mac-only or Docker-only run to compare module-level latency.

Example:
  python input/media/tools/test/summarize_pipeline_metrics.py \
      metrics/mac_only_s01.jsonl --run-label mac_only_s01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from server.utils.perf import STANDARD_FIELD_DESCRIPTIONS, STANDARD_SUMMARY_FIELDS

DEFAULT_FIELDS = STANDARD_SUMMARY_FIELDS


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_rows(path: Path, run_label: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_label and row.get("run_label") != run_label:
                continue
            rows.append(row)
    return rows


def summarize_field(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | str]:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), int | float)
    ]
    if not values:
        return {
            "field": field,
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "description": STANDARD_FIELD_DESCRIPTIONS.get(field, ""),
        }
    return {
        "field": field,
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "description": STANDARD_FIELD_DESCRIPTIONS.get(field, ""),
    }


def format_row(row: dict[str, float | int | str]) -> str:
    return (
        f"{row['field']:<32} "
        f"{int(row['count']):>6d} "
        f"{float(row['mean_ms']):>10.2f} "
        f"{float(row['p50_ms']):>10.2f} "
        f"{float(row['p95_ms']):>10.2f} "
        f"{float(row['max_ms']):>10.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--run-label")
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument(
        "--field",
        action="append",
        help="Metric field to summarize. Repeatable. Defaults to key pipeline fields.",
    )
    args = parser.parse_args()

    rows = read_rows(args.path, args.run_label)
    if not rows:
        raise SystemExit("No matching metric rows found.")

    fields = args.field or DEFAULT_FIELDS
    summaries = [summarize_field(rows, field) for field in fields]

    elapsed_values = [
        float(row["elapsed_s"])
        for row in rows
        if isinstance(row.get("elapsed_s"), int | float)
    ]
    if len(elapsed_values) >= 2:
        duration = max(elapsed_values) - min(elapsed_values)
    else:
        ts_values = [
            float(row["ts"])
            for row in rows
            if isinstance(row.get("ts"), int | float)
        ]
        duration = max(ts_values) - min(ts_values) if len(ts_values) >= 2 else 0.0
    fps = len(rows) / duration if duration > 0 else 0.0

    print(f"rows={len(rows)} duration_s={duration:.2f} effective_fps={fps:.2f}")
    print(f"{'field':<32} {'count':>6} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'max_ms':>10}")
    print("-" * 86)
    for summary in summaries:
        print(format_row(summary))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "field",
                    "count",
                    "mean_ms",
                    "p50_ms",
                    "p95_ms",
                    "max_ms",
                    "description",
                ],
            )
            writer.writeheader()
            writer.writerows(summaries)
        print(f"saved={args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
