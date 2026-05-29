"""Create PostgreSQL indexes used by IncidentLog list/search screens.

SQLAlchemy model metadata documents the intended indexes, but an already
created local database is not changed just by editing the model class. Run this
script once after large benchmark data is inserted, or whenever a fresh
database needs the same search indexes.
"""

import argparse
import asyncio
import os

from pathlib import Path

import asyncpg
from dotenv import load_dotenv


INDEX_STATEMENTS = (
    (
        "idx_incident_logs_date_created_id",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_created_id
        ON incident_logs (date, created_at DESC, id DESC)
        """,
    ),
    (
        "idx_incident_logs_date_type_worker_created_id",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_type_worker_created_id
        ON incident_logs (date, incident_type, worker_id, created_at DESC, id DESC)
        """,
    ),
)


def _asyncpg_url() -> str:
    """Return asyncpg-compatible DATABASE_URL from the project .env file."""
    load_dotenv(Path.cwd() / ".env")
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _show_indexes(conn: asyncpg.Connection) -> None:
    """Print IncidentLog indexes so the caller can verify DB state."""
    rows = await conn.fetch(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'incident_logs'
        ORDER BY indexname
        """
    )
    for row in rows:
        print(f"{row['indexname']}: {row['indexdef']}")


async def run(args: argparse.Namespace) -> None:
    """Create and verify indexes for date-oriented incident log search."""
    conn = await asyncpg.connect(_asyncpg_url())
    try:
        if args.show_only:
            await _show_indexes(conn)
            return

        for name, statement in INDEX_STATEMENTS:
            print(f"creating_or_reusing={name}")
            started = asyncio.get_running_loop().time()
            await conn.execute(statement)
            elapsed = asyncio.get_running_loop().time() - started
            print(f"done={name} elapsed_sec={elapsed:.3f}")

        if args.analyze:
            print("running=ANALYZE incident_logs")
            await conn.execute("ANALYZE incident_logs")

        print("indexes:")
        await _show_indexes(conn)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-only", action="store_true")
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.set_defaults(analyze=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
