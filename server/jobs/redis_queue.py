"""Minimal Redis-backed background job queue.

Celery is intentionally not used here. This project only needs a small number
of backend-owned jobs, so a Redis list + status hash keeps the architecture
measurable and easy to explain before introducing a distributed worker stack.
"""

import asyncio
import json
import os
import uuid

from datetime import date, datetime, timezone
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - handled at runtime for missing dependency
    redis = None

from fastapi import HTTPException

from ..report.service import generate_daily_report


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_KEY = os.getenv("REDIS_JOB_QUEUE_KEY", "rps:jobs:queue")
JOB_KEY_PREFIX = os.getenv("REDIS_JOB_KEY_PREFIX", "rps:jobs:")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(start: str | None, end: str | None) -> float | None:
    """Return milliseconds between two ISO timestamps when both exist."""
    if not start or not end:
        return None
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return None
    return round((end_dt - start_dt).total_seconds() * 1000.0, 3)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: str | bytes | None) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    if value == "":
        return None
    return json.loads(value)


def _client():
    if redis is None:
        raise RuntimeError("redis package is not installed. Run `pip install redis==5.0.8`.")
    return redis.from_url(REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


async def enqueue_job(job_type: str, payload: dict[str, Any]) -> dict[str, str]:
    """Push a job request to Redis and persist initial status."""
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "payload": _json_dump(payload),
        "result": "",
        "error": "",
        "created_at": _now(),
        "started_at": "",
        "finished_at": "",
    }

    try:
        client = _client()
        await client.hset(_job_key(job_id), mapping=record)
        await client.rpush(QUEUE_KEY, _json_dump({"job_id": job_id, "job_type": job_type}))
        await client.aclose()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis job enqueue failed: {exc}") from exc

    return {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "status_url": f"/jobs/{job_id}",
    }


async def get_job_status(job_id: str) -> dict[str, Any]:
    """Read a job status hash from Redis."""
    try:
        client = _client()
        data = await client.hgetall(_job_key(job_id))
        await client.aclose()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis job status read failed: {exc}") from exc

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    created_at = data.get("created_at") or None
    started_at = data.get("started_at") or None
    finished_at = data.get("finished_at") or None

    return {
        "job_id": data.get("job_id", job_id),
        "job_type": data.get("job_type", ""),
        "status": data.get("status", ""),
        "payload": _json_load(data.get("payload")),
        "result": _json_load(data.get("result")),
        "error": data.get("error") or None,
        "created_at": created_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "queued_ms": _duration_ms(created_at, started_at),
        "runtime_ms": _duration_ms(started_at, finished_at),
        "total_ms": _duration_ms(created_at, finished_at),
    }


async def _run_job(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if job_type == "daily_report":
        raw_date = payload.get("target_date")
        target = date.fromisoformat(raw_date) if raw_date else None
        report = await generate_daily_report(target)
        return {
            "report_id": report.id,
            "date": report.date.isoformat(),
            "created_at": report.created_at.isoformat() if report.created_at else None,
        }
    raise ValueError(f"Unsupported job_type: {job_type}")


async def redis_worker_loop() -> None:
    """Continuously consume Redis jobs while the FastAPI process is alive."""
    if redis is None:
        print("[redis-jobs] skipped: redis package is not installed")
        return

    client = _client()
    try:
        await client.ping()
    except Exception as exc:
        print(f"[redis-jobs] skipped: Redis is unavailable ({exc})")
        await client.aclose()
        return

    print(f"[redis-jobs] worker started queue={QUEUE_KEY} url={REDIS_URL}")
    try:
        while True:
            item = await client.blpop(QUEUE_KEY, timeout=1)
            if item is None:
                await asyncio.sleep(0)
                continue

            _, raw = item
            job = _json_load(raw)
            job_id = job["job_id"]
            job_type = job["job_type"]
            key = _job_key(job_id)
            await client.hset(key, mapping={"status": "running", "started_at": _now()})

            try:
                payload = _json_load(await client.hget(key, "payload")) or {}
                result = await _run_job(job_type, payload)
                await client.hset(key, mapping={
                    "status": "done",
                    "result": _json_dump(result),
                    "finished_at": _now(),
                })
            except Exception as exc:
                await client.hset(key, mapping={
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": _now(),
                })
    except asyncio.CancelledError:
        print("[redis-jobs] worker stopping")
        raise
    finally:
        await client.aclose()
