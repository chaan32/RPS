"""IncidentLog (사고 기록) DB CRUD 서비스."""

import time

from datetime import date as date_cls, datetime

from fastapi import UploadFile
from sqlalchemy import case, func, or_, select, update

from ..database import AsyncSessionLocal, IncidentLog, Worker
from ..database.store import save_file
from ..schemas import IncidentLogCreate, IncidentLogSearchResponse, IncidentLogSummaryResponse
from ..search.incident_elasticsearch import search_incident_logs as search_es_incident_logs


async def create_with_snapshot(
    worker_id: int,
    incident_type: str,
    file: UploadFile,
) -> IncidentLog:
    """스냅샷 업로드 + IncidentLog INSERT 한 번에."""
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{today}/{file.filename}"
    url = save_file(contents, key, content_type=file.content_type)

    async with AsyncSessionLocal() as session:
        log = IncidentLog(
            worker_id=worker_id,
            incident_type=incident_type,
            snapshot_path=url,
            status="success",
        )
        session.add(log)
        await session.flush()
        await _increment_worker_count(session, worker_id)
        await session.commit()
        await session.refresh(log)
    return log


async def create(body: IncidentLogCreate) -> IncidentLog:
    """스냅샷 없이 IncidentLog INSERT (placeholder 경로)."""
    async with AsyncSessionLocal() as session:
        data = body.model_dump(exclude_none=True)
        log = IncidentLog(**data)
        session.add(log)
        await session.flush()
        await _increment_worker_count(session, log.worker_id)
        await session.commit()
        await session.refresh(log)
    return log


async def list_all() -> list[IncidentLog]:
    """모든 IncidentLog SELECT."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(IncidentLog))
        return result.scalars().all()


def _apply_search_filters(
    stmt,
    *,
    target_date: date_cls | None,
    incident_type: str | None,
    worker_id: int | None,
    q: str | None,
):
    """IncidentLog 검색 조건을 SQLAlchemy SELECT 문에 공통 적용한다."""
    if target_date is not None:
        stmt = stmt.where(IncidentLog.date == target_date)
    if incident_type:
        stmt = stmt.where(IncidentLog.incident_type == incident_type)
    if worker_id is not None:
        stmt = stmt.where(IncidentLog.worker_id == worker_id)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                IncidentLog.snapshot_path.ilike(like),
                IncidentLog.status.ilike(like),
                IncidentLog.incident_type.ilike(like),
            )
        )
    return stmt


async def search_postgres(
    *,
    target_date: date_cls | None = None,
    incident_type: str | None = None,
    worker_id: int | None = None,
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> IncidentLogSearchResponse:
    """PostgreSQL 기준 검색 API.

    전체 목록을 모두 내려주는 `/incident-logs`와 달리, 실제 검색 조건을 걸고
    필요한 행만 반환한다. Elasticsearch 도입 전후를 같은 조건으로 비교할 때
    기준선으로 쓴다.
    """
    started = time.perf_counter()
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    base = _apply_search_filters(
        select(IncidentLog),
        target_date=target_date,
        incident_type=incident_type,
        worker_id=worker_id,
        q=q,
    )
    count_stmt = _apply_search_filters(
        select(func.count()).select_from(IncidentLog),
        target_date=target_date,
        incident_type=incident_type,
        worker_id=worker_id,
        q=q,
    )

    async with AsyncSessionLocal() as session:
        total = int((await session.execute(count_stmt)).scalar_one())
        result = await session.execute(
            base.order_by(IncidentLog.created_at.desc(), IncidentLog.id.desc())
            .offset(offset)
            .limit(limit)
        )
        items = list(result.scalars().all())

    return IncidentLogSearchResponse(
        backend="postgres",
        total=total,
        took_ms=(time.perf_counter() - started) * 1000.0,
        items=items,
    )


async def search_elasticsearch(
    *,
    target_date: date_cls | None = None,
    incident_type: str | None = None,
    worker_id: int | None = None,
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> IncidentLogSearchResponse:
    """Elasticsearch 기준 검색 API."""
    return await search_es_incident_logs(
        target_date=target_date,
        incident_type=incident_type,
        worker_id=worker_id,
        q=q,
        limit=limit,
        offset=offset,
    )


async def summarize_by_date(target_date: date_cls) -> IncidentLogSummaryResponse:
    """Return date-scoped incident totals grouped by worker.

    `/workers` stores the lifetime cumulative count. The dashboard needs a
    separate date-scoped summary so the selected date is not confused with the
    cumulative worker counter.
    """
    warning_expr = func.sum(
        case((func.lower(IncidentLog.incident_type) == "warning", 1), else_=0)
    )
    danger_expr = func.sum(
        case((func.lower(IncidentLog.incident_type) == "danger", 1), else_=0)
    )

    stmt = (
        select(
            IncidentLog.worker_id,
            func.count().label("total"),
            warning_expr.label("warning"),
            danger_expr.label("danger"),
        )
        .where(IncidentLog.date == target_date)
        .group_by(IncidentLog.worker_id)
        .order_by(IncidentLog.worker_id)
    )

    async with AsyncSessionLocal() as session:
        rows = list((await session.execute(stmt)).all())

    workers = [
        {
            "worker_id": int(row.worker_id),
            "total": int(row.total or 0),
            "warning": int(row.warning or 0),
            "danger": int(row.danger or 0),
        }
        for row in rows
    ]

    return IncidentLogSummaryResponse(
        target_date=target_date,
        total=sum(row["total"] for row in workers),
        warning=sum(row["warning"] for row in workers),
        danger=sum(row["danger"] for row in workers),
        workers=workers,
    )


async def _increment_worker_count(session, worker_id: int) -> None:
    """Increment the per-worker alert counter after an incident log is created."""
    await session.execute(
        update(Worker)
        .where(Worker.id == worker_id)
        .values(count=Worker.count + 1)
    )
