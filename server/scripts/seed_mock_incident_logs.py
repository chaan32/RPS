"""Seed production-shaped mock IncidentLog rows.

Usage:
    python -m server.scripts.seed_mock_incident_logs --count 100000 --date 2026-05-24

The generated snapshot_path follows the same shape as realtime logs:
`<project>/snapshots/YYYY-MM-DD/realtime_forklift_YYYYMMDD_HHMMSS_micro.jpg`.
"""

import argparse
import asyncio
import os
import random

from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


BENCHMARK_PREFIX = "/benchmark/es_benchmark/"


def _asyncpg_url() -> str:
    load_dotenv()
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _snapshot_path(project_root: Path, created_at: datetime) -> str:
    date_dir = created_at.strftime("%Y-%m-%d")
    stamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    return str(project_root / "snapshots" / date_dir / f"realtime_forklift_{stamp}.jpg")


async def seed(count: int, target_date: date, project_root: Path, clear_date: bool) -> None:
    conn = await asyncpg.connect(_asyncpg_url())
    try:
        if clear_date:
            deleted = await conn.execute(
                """
                DELETE FROM incident_logs
                WHERE date = $1
                  AND snapshot_path LIKE $2
                """,
                target_date,
                f"{project_root / 'snapshots' / target_date.isoformat()}%",
            )
            print(f"cleared_existing_date_rows={deleted}")

        workers = [1, 2]
        incident_types = ["Warning", "Danger"]
        statuses = ["success", "fail"]
        base = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        step_us = max(1, (24 * 60 * 60 * 1_000_000) // count)
        records = []

        for i in range(count):
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

        await conn.copy_records_to_table(
            "incident_logs",
            records=records,
            columns=["worker_id", "incident_type", "snapshot_path", "status", "date", "created_at"],
        )
        inserted_date_count = await conn.fetchval(
            "SELECT count(*) FROM incident_logs WHERE date = $1",
            target_date,
        )
        benchmark_count = await conn.fetchval(
            "SELECT count(*) FROM incident_logs WHERE snapshot_path LIKE $1",
            f"{BENCHMARK_PREFIX}%",
        )
        total_count = await conn.fetchval("SELECT count(*) FROM incident_logs")
        print(f"inserted={count}")
        print(f"date_rows={inserted_date_count}")
        print(f"benchmark_rows={benchmark_count}")
        print(f"all_incident_logs={total_count}")
    finally:
        await conn.close()


async def main_async(args: argparse.Namespace) -> None:
    if args.delete_benchmark:
        conn = await asyncpg.connect(_asyncpg_url())
        try:
            result = await conn.execute(
                "DELETE FROM incident_logs WHERE snapshot_path LIKE $1",
                f"{BENCHMARK_PREFIX}%",
            )
            print(f"deleted_benchmark_rows={result}")
        finally:
            await conn.close()

    await seed(
        count=args.count,
        target_date=date.fromisoformat(args.date),
        project_root=Path(args.project_root).resolve(),
        clear_date=args.clear_date,
    )


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--delete-benchmark", action="store_true")
    parser.add_argument("--clear-date", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
