"""Build the Elasticsearch read index from PostgreSQL IncidentLog rows.

Usage:
    python -m server.scripts.index_incident_logs_to_elasticsearch --reset
"""

import argparse
import asyncio
import os
import time

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import asyncpg
from dotenv import load_dotenv

from ..search.incident_elasticsearch import (
    bulk_index_documents,
    ensure_incident_index,
    incident_to_document,
)


@dataclass
class IncidentRow:
    id: int
    worker_id: int
    incident_type: str
    snapshot_path: str
    status: str
    date: date
    created_at: datetime


def _asyncpg_url() -> str:
    load_dotenv()
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _row_to_incident(row: asyncpg.Record) -> IncidentRow:
    return IncidentRow(
        id=row["id"],
        worker_id=row["worker_id"],
        incident_type=row["incident_type"],
        snapshot_path=row["snapshot_path"],
        status=row["status"],
        date=row["date"],
        created_at=row["created_at"],
    )


async def _fetch_batch(conn: asyncpg.Connection, last_id: int, batch_size: int) -> list[Any]:
    rows = await conn.fetch(
        """
        SELECT id, worker_id, incident_type, snapshot_path, status, date, created_at
        FROM incident_logs
        WHERE id > $1
        ORDER BY id
        LIMIT $2
        """,
        last_id,
        batch_size,
    )
    return [_row_to_incident(row) for row in rows]


async def run(reset: bool, batch_size: int) -> None:
    started = time.perf_counter()
    await ensure_incident_index(reset=reset)

    indexed = 0
    last_id = 0
    conn = await asyncpg.connect(_asyncpg_url())
    try:
        while True:
            batch = await _fetch_batch(conn, last_id, batch_size)
            if not batch:
                break
            await bulk_index_documents([incident_to_document(row) for row in batch])
            indexed += len(batch)
            last_id = batch[-1].id
            print(f"indexed={indexed} last_id={last_id}")
    finally:
        await conn.close()

    elapsed = time.perf_counter() - started
    print(f"done indexed={indexed} elapsed_sec={elapsed:.3f} docs_per_sec={indexed / elapsed:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2_000)
    args = parser.parse_args()
    asyncio.run(run(reset=args.reset, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
