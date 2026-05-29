"""Create large benchmark IncidentLog rows.

Usage:
    python -m server.scripts.seed_benchmark_incident_logs --count 50000

Rows are marked by the snapshot_path prefix `/benchmark/es_benchmark/` so they
can be counted or deleted deliberately during benchmark cleanup.
"""

import argparse
import asyncio
import os
import random

from datetime import date, datetime, timedelta

import asyncpg
from dotenv import load_dotenv


BENCHMARK_PREFIX = "/benchmark/es_benchmark/"


def _asyncpg_url() -> str:
    load_dotenv()
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def seed(count: int, target_date: date) -> None:
    conn = await asyncpg.connect(_asyncpg_url())
    try:
        workers = [1, 2]
        incident_types = ["Warning", "Danger"]
        statuses = ["success", "fail"]
        base = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        records = []
        for i in range(count):
            worker_id = random.choice(workers)
            incident_type = random.choices(incident_types, weights=[0.7, 0.3], k=1)[0]
            status = random.choices(statuses, weights=[0.96, 0.04], k=1)[0]
            created_at = base + timedelta(seconds=random.randint(0, 86_399))
            records.append((
                worker_id,
                incident_type,
                f"{BENCHMARK_PREFIX}{target_date.isoformat()}/worker-{worker_id}/snapshot-{i:06d}.jpg",
                status,
                target_date,
                created_at,
            ))

        await conn.copy_records_to_table(
            "incident_logs",
            records=records,
            columns=["worker_id", "incident_type", "snapshot_path", "status", "date", "created_at"],
        )
        benchmark_count = await conn.fetchval(
            "SELECT count(*) FROM incident_logs WHERE snapshot_path LIKE $1",
            f"{BENCHMARK_PREFIX}%",
        )
        total_count = await conn.fetchval("SELECT count(*) FROM incident_logs")
        print(f"inserted={count}")
        print(f"benchmark_rows={benchmark_count}")
        print(f"all_incident_logs={total_count}")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    asyncio.run(seed(args.count, date.fromisoformat(args.date)))


if __name__ == "__main__":
    main()
