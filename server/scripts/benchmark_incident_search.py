"""Benchmark PostgreSQL and Elasticsearch incident-log search latency.

This script measures the same search conditions against both backends. It can
also hit FastAPI endpoints if the server is running, but the direct backend
measurement is enough to isolate database/search-engine performance.
"""

import argparse
import asyncio
import json
import os
import statistics
import time

from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import asyncpg
import httpx
from dotenv import load_dotenv

from ..search.incident_elasticsearch import ELASTICSEARCH_URL, INCIDENT_INDEX


def _asyncpg_url() -> str:
    load_dotenv()
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    idx95 = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "count": len(values),
        "avg_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[idx95], 3),
        "max_ms": round(max(values), 3),
    }


async def _measure_postgres(iterations: int, target_date: date | None, limit: int, q: str | None) -> dict:
    conn = await asyncpg.connect(_asyncpg_url())
    timings: list[float] = []
    rows_count = 0
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            if target_date is not None and q:
                rows = await conn.fetch(
                    """
                    SELECT id, worker_id, incident_type, snapshot_path, status, date, created_at
                    FROM incident_logs
                    WHERE date = $1
                      AND incident_type = $2
                      AND worker_id = $3
                      AND snapshot_path ILIKE $4
                    ORDER BY created_at DESC, id DESC
                    LIMIT $5
                    """,
                    target_date,
                    "Danger",
                    1,
                    f"%{q}%",
                    limit,
                )
            elif target_date is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, worker_id, incident_type, snapshot_path, status, date, created_at
                    FROM incident_logs
                    WHERE date = $1 AND incident_type = $2 AND worker_id = $3
                    ORDER BY created_at DESC, id DESC
                    LIMIT $4
                    """,
                    target_date,
                    "Danger",
                    1,
                    limit,
                )
            elif q:
                rows = await conn.fetch(
                    """
                    SELECT id, worker_id, incident_type, snapshot_path, status, date, created_at
                    FROM incident_logs
                    WHERE incident_type = $1
                      AND worker_id = $2
                      AND snapshot_path ILIKE $3
                    ORDER BY created_at DESC, id DESC
                    LIMIT $4
                    """,
                    "Danger",
                    1,
                    f"%{q}%",
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, worker_id, incident_type, snapshot_path, status, date, created_at
                    FROM incident_logs
                    WHERE incident_type = $1 AND worker_id = $2
                    ORDER BY created_at DESC, id DESC
                    LIMIT $3
                    """,
                    "Danger",
                    1,
                    limit,
                )
            timings.append((time.perf_counter() - started) * 1000.0)
            rows_count = len(rows)
    finally:
        await conn.close()
    return {"backend": "postgres_direct", "query": q or "", "rows": rows_count, **_summary(timings)}


async def _measure_elasticsearch(iterations: int, target_date: date | None, limit: int, q: str | None) -> dict:
    must = []
    if q:
        must.append({"wildcard": {"snapshot_path": {"value": f"*{q}*", "case_insensitive": True}}})
    filters = [
        {"term": {"incident_type": "Danger"}},
        {"term": {"worker_id": 1}},
    ]
    if target_date is not None:
        filters.insert(0, {"term": {"date": target_date.isoformat()}})
    body = {
        "query": {
            "bool": {
                "must": must,
                "filter": filters,
            }
        },
        "size": limit,
        "sort": [
            {"created_at": {"order": "desc", "missing": "_last"}},
            {"id": {"order": "desc"}},
        ],
        "track_total_hits": True,
    }
    timings: list[float] = []
    rows_count = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(iterations):
            started = time.perf_counter()
            response = await client.post(
                f"{ELASTICSEARCH_URL}/{INCIDENT_INDEX}/_search",
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            timings.append((time.perf_counter() - started) * 1000.0)
            rows_count = len(data.get("hits", {}).get("hits", []))
    return {"backend": "elasticsearch_direct", "query": q or "", "rows": rows_count, **_summary(timings)}


async def _measure_api(
    iterations: int,
    api_base: str,
    endpoint: str,
    target_date: date | None,
    limit: int,
    q: str | None,
) -> dict:
    params = {
        "incident_type": "Danger",
        "worker_id": 1,
        "limit": limit,
    }
    if target_date is not None:
        params["target_date"] = target_date.isoformat()
    if q:
        params["q"] = q
    url = f"{api_base.rstrip('/')}{endpoint}?{urlencode(params)}"
    timings: list[float] = []
    rows_count = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(iterations):
            started = time.perf_counter()
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            timings.append((time.perf_counter() - started) * 1000.0)
            rows_count = len(data.get("items", []))
    return {"backend": endpoint.strip("/").replace("/", "_"), "rows": rows_count, **_summary(timings)}


async def run(args: argparse.Namespace) -> dict:
    target_date = None if args.no_date_filter else date.fromisoformat(args.date)
    result = {
        "date": target_date.isoformat() if target_date else None,
        "date_filter": target_date is not None,
        "iterations": args.iterations,
        "limit": args.limit,
        "measurements": [
            await _measure_postgres(args.iterations, target_date, args.limit, args.q),
            await _measure_elasticsearch(args.iterations, target_date, args.limit, args.q),
        ],
    }
    if args.api_base:
        result["measurements"].append(
            await _measure_api(
                args.iterations,
                args.api_base,
                "/incident-logs/search/postgres",
                target_date,
                args.limit,
                args.q,
            )
        )
        result["measurements"].append(
            await _measure_api(
                args.iterations,
                args.api_base,
                "/incident-logs/search/elasticsearch",
                target_date,
                args.limit,
                args.q,
            )
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved={output}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--no-date-filter", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--q", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--output", default="metrics/incident_search_benchmark.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
