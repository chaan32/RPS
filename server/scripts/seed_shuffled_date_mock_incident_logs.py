"""Seed shuffled production-shaped IncidentLog rows across many dates.

Usage:
    python -m server.scripts.seed_shuffled_date_mock_incident_logs \
      --start-date 2024-01-01 --date-count 1000 --per-date 300 \
      --delete-after-id 391

The generated rows use the realtime snapshot path shape:
`<project>/snapshots/YYYY-MM-DD/realtime_forklift_YYYYMMDD_HHMMSS_micro.jpg`.
Rows are shuffled before bulk insert so physical insert order does not match
date order.
"""

import argparse
import asyncio
import os
import random

from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


def _asyncpg_url() -> str:
    load_dotenv()
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _snapshot_path(project_root: Path, created_at: datetime) -> str:
    date_dir = created_at.strftime("%Y-%m-%d")
    stamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    return str(project_root / "snapshots" / date_dir / f"realtime_forklift_{stamp}.jpg")


def _build_records(
    *,
    project_root: Path,
    start_date: date,
    date_count: int,
    per_date: int,
    seed: int,
) -> list[tuple]:
    random.seed(seed)
    workers = [1, 2]
    incident_types = ["Warning", "Danger"]
    statuses = ["success", "fail"]
    records = []

    step_us = max(1, (24 * 60 * 60 * 1_000_000) // per_date)
    for day_index in range(date_count):
        target_date = start_date + timedelta(days=day_index)
        base = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        for i in range(per_date):
            worker_id = random.choice(workers)
            incident_type = random.choices(incident_types, weights=[0.7, 0.3], k=1)[0]
            status = random.choices(statuses, weights=[0.96, 0.04], k=1)[0]
            created_at = base + timedelta(microseconds=i * step_us)
            records.append((
                worker_id,
                incident_type,
                _snapshot_path(project_root, created_at),
                status,
                target_date,
                created_at,
            ))

    random.shuffle(records)
    return records


async def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    start_date = date.fromisoformat(args.start_date)
    records = _build_records(
        project_root=project_root,
        start_date=start_date,
        date_count=args.date_count,
        per_date=args.per_date,
        seed=args.seed,
    )

    conn = await asyncpg.connect(_asyncpg_url())
    try:
        if args.delete_after_id is not None:
            result = await conn.execute(
                "DELETE FROM incident_logs WHERE id > $1",
                args.delete_after_id,
            )
            print(f"deleted_existing_mock={result}")

        await conn.copy_records_to_table(
            "incident_logs",
            records=records,
            columns=["worker_id", "incident_type", "snapshot_path", "status", "date", "created_at"],
        )

        total = await conn.fetchval("SELECT count(*) FROM incident_logs")
        generated_date_rows = await conn.fetchval(
            """
            SELECT count(*)
            FROM incident_logs
            WHERE date >= $1 AND date < $2
              AND snapshot_path LIKE $3
            """,
            start_date,
            start_date + timedelta(days=args.date_count),
            f"{project_root / 'snapshots'}%",
        )
        date_buckets = await conn.fetchval(
            """
            SELECT count(DISTINCT date)
            FROM incident_logs
            WHERE date >= $1 AND date < $2
              AND snapshot_path LIKE $3
            """,
            start_date,
            start_date + timedelta(days=args.date_count),
            f"{project_root / 'snapshots'}%",
        )
        print(f"inserted={len(records)}")
        print(f"generated_date_rows={generated_date_rows}")
        print(f"generated_date_buckets={date_buckets}")
        print(f"all_incident_logs={total}")
    finally:
        await conn.close()


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--date-count", type=int, default=1000)
    parser.add_argument("--per-date", type=int, default=300)
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delete-after-id", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
