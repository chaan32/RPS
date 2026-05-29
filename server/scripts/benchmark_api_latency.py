"""Measure API latency for selected FastAPI endpoints.

This script is intentionally small and external to the API implementation so
load-test instrumentation does not change the production request handlers.
It records client-observed latency for each endpoint and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Endpoint:
    name: str
    path: str


@dataclass
class EndpointStats:
    latencies_ms: list[float] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def add(self, latency_ms: float, status_code: int | None) -> None:
        self.latencies_ms.append(latency_ms)
        key = str(status_code) if status_code is not None else "error"
        self.status_counts[key] = self.status_counts.get(key, 0) + 1
        if status_code is None or status_code >= 500:
            self.errors += 1


def percentile(values: list[float], pct: float) -> float | None:
    """Return nearest-rank percentile for a non-empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def summarize(stats: EndpointStats) -> dict[str, Any]:
    values = stats.latencies_ms
    if not values:
        return {
            "count": 0,
            "status_counts": stats.status_counts,
            "errors": stats.errors,
        }

    return {
        "count": len(values),
        "status_counts": stats.status_counts,
        "errors": stats.errors,
        "min_ms": round(min(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(percentile(values, 0.50) or 0.0, 3),
        "p95_ms": round(percentile(values, 0.95) or 0.0, 3),
        "p99_ms": round(percentile(values, 0.99) or 0.0, 3),
        "max_ms": round(max(values), 3),
    }


async def worker(
    *,
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    stats: EndpointStats,
    stop_at: float,
    sleep_s: float,
) -> None:
    while time.perf_counter() < stop_at:
        started = time.perf_counter()
        status_code: int | None = None
        try:
            response = await client.get(endpoint.path)
            status_code = response.status_code
            response.read()
        except Exception:
            status_code = None
        finally:
            stats.add((time.perf_counter() - started) * 1000.0, status_code)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    endpoints = [
        Endpoint("workers", "/workers"),
        Endpoint(
            "incidents_postgres",
            f"/incident-logs/search/postgres?target_date={args.target_date}&limit={args.limit}",
        ),
        Endpoint("reports", "/reports"),
        Endpoint("reports_summary", "/reports/summary?limit=20"),
    ]
    stats = {endpoint.name: EndpointStats() for endpoint in endpoints}
    stop_at = time.perf_counter() + args.duration

    limits = httpx.Limits(
        max_connections=max(10, args.concurrency * len(endpoints) + 2),
        max_keepalive_connections=max(10, args.concurrency * len(endpoints) + 2),
    )
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=httpx.Timeout(args.timeout),
        limits=limits,
    ) as client:
        tasks = [
            asyncio.create_task(
                worker(
                    client=client,
                    endpoint=endpoint,
                    stats=stats[endpoint.name],
                    stop_at=stop_at,
                    sleep_s=args.sleep,
                )
            )
            for endpoint in endpoints
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)

    return {
        "label": args.label,
        "base_url": args.base_url,
        "target_date": args.target_date,
        "duration_s": args.duration,
        "concurrency_per_endpoint": args.concurrency,
        "total_requests": sum(len(stat.latencies_ms) for stat in stats.values()),
        "endpoints": {name: summarize(stat) for name, stat in stats.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1122")
    parser.add_argument("--target-date", default="2026-05-26")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--label", default="api_latency")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
