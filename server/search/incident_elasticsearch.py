"""Elasticsearch read model for IncidentLog search.

PostgreSQL remains the source of truth. Elasticsearch is used only as a
read-optimized index for large list/search pages and benchmark comparison.
"""

import json
import os
import time

from datetime import date as date_cls, datetime
from typing import Any

import httpx
from fastapi import HTTPException

from ..schemas import IncidentLogSearchResponse


ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200").rstrip("/")
INCIDENT_INDEX = os.getenv("ELASTICSEARCH_INCIDENT_INDEX", "incident_logs_v1")


INCIDENT_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "id": {"type": "integer"},
            "worker_id": {"type": "integer"},
            "incident_type": {"type": "keyword"},
            "status": {"type": "keyword"},
            "date": {"type": "date"},
            "created_at": {"type": "date"},
            "snapshot_path": {"type": "keyword"},
            "snapshot_text": {"type": "text"},
        }
    },
}


def incident_to_document(log: Any) -> dict[str, Any]:
    """Convert SQLAlchemy/asyncpg incident rows to an Elasticsearch document."""
    created_at = getattr(log, "created_at", None)
    log_date = getattr(log, "date", None)
    snapshot_path = getattr(log, "snapshot_path", "")
    return {
        "id": int(getattr(log, "id")),
        "worker_id": int(getattr(log, "worker_id", getattr(log, "maker_id", 1))),
        "incident_type": getattr(log, "incident_type"),
        "snapshot_path": snapshot_path,
        "snapshot_text": snapshot_path,
        "status": getattr(log, "status"),
        "date": log_date.isoformat() if hasattr(log_date, "isoformat") else log_date,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }


async def ensure_incident_index(reset: bool = False) -> None:
    """Create the incident log index and mapping if needed."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        if reset:
            await client.delete(f"{ELASTICSEARCH_URL}/{INCIDENT_INDEX}")

        exists = await client.head(f"{ELASTICSEARCH_URL}/{INCIDENT_INDEX}")
        if exists.status_code == 404:
            response = await client.put(
                f"{ELASTICSEARCH_URL}/{INCIDENT_INDEX}",
                json=INCIDENT_MAPPING,
            )
            response.raise_for_status()
        elif exists.status_code >= 400:
            exists.raise_for_status()


async def bulk_index_documents(documents: list[dict[str, Any]]) -> None:
    """Bulk index IncidentLog documents into Elasticsearch."""
    if not documents:
        return

    lines: list[str] = []
    for document in documents:
        doc_id = document["id"]
        lines.append(json.dumps({"index": {"_index": INCIDENT_INDEX, "_id": doc_id}}, separators=(",", ":")))
        lines.append(json.dumps(document, separators=(",", ":")))
    payload = "\n".join(lines) + "\n"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ELASTICSEARCH_URL}/_bulk",
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("errors"):
            first_error = next(
                (
                    item
                    for item in data.get("items", [])
                    if item.get("index", {}).get("error")
                ),
                None,
            )
            raise RuntimeError(f"Elasticsearch bulk indexing failed: {first_error}")


def _build_query(
    *,
    target_date: date_cls | None,
    incident_type: str | None,
    worker_id: int | None,
    q: str | None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    must: list[dict[str, Any]] = []

    if target_date is not None:
        filters.append({"term": {"date": target_date.isoformat()}})
    if incident_type:
        filters.append({"term": {"incident_type": incident_type}})
    if worker_id is not None:
        filters.append({"term": {"worker_id": worker_id}})
    if q:
        must.append({
            "bool": {
                "should": [
                    {"wildcard": {"snapshot_path": {"value": f"*{q}*", "case_insensitive": True}}},
                    {"term": {"incident_type": q}},
                    {"term": {"status": q}},
                ],
                "minimum_should_match": 1,
            }
        })

    if not filters and not must:
        return {"match_all": {}}
    return {"bool": {"filter": filters, "must": must}}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


async def search_incident_logs(
    *,
    target_date: date_cls | None = None,
    incident_type: str | None = None,
    worker_id: int | None = None,
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> IncidentLogSearchResponse:
    """Search IncidentLog documents in Elasticsearch and return API schema."""
    started = time.perf_counter()
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    body = {
        "query": _build_query(
            target_date=target_date,
            incident_type=incident_type,
            worker_id=worker_id,
            q=q,
        ),
        "from": offset,
        "size": limit,
        "sort": [
            {"created_at": {"order": "desc", "missing": "_last"}},
            {"id": {"order": "desc"}},
        ],
        "track_total_hits": True,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{ELASTICSEARCH_URL}/{INCIDENT_INDEX}/_search",
            json=body,
        )
    if response.status_code == 404:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Elasticsearch index '{INCIDENT_INDEX}' is missing. "
                "Run `python -m server.scripts.index_incident_logs_to_elasticsearch` first."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=503, detail=response.text)

    data = response.json()
    total_raw = data.get("hits", {}).get("total", 0)
    total = total_raw.get("value", 0) if isinstance(total_raw, dict) else int(total_raw)
    items = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit["_source"]
        items.append({
            "id": src["id"],
            "worker_id": src["worker_id"],
            "maker_id": src["worker_id"],
            "incident_type": src["incident_type"],
            "snapshot_path": src["snapshot_path"],
            "status": src["status"],
            "date": date_cls.fromisoformat(src["date"]),
            "created_at": _parse_datetime(src.get("created_at")) or datetime.min,
        })

    return IncidentLogSearchResponse(
        backend="elasticsearch",
        total=total,
        took_ms=(time.perf_counter() - started) * 1000.0,
        items=items,
    )
