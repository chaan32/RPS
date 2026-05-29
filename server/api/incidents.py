"""IncidentLog (사고 기록) CRUD 엔드포인트."""

from datetime import date as date_cls

from fastapi import APIRouter, File, Query, UploadFile

from ..schemas import (
    IncidentLogCreate,
    IncidentLogResponse,
    IncidentLogSearchResponse,
    IncidentLogSummaryResponse,
)
from ..service import incident_service

router = APIRouter()


@router.post("/incident-logs/with-snapshot", response_model=IncidentLogResponse)
async def create_incident_with_snapshot(
    incident_type: str,
    file: UploadFile = File(...),
    worker_id: int | None = None,
    maker_id: int | None = None,
):
    resolved_worker_id = worker_id or maker_id
    if resolved_worker_id is None:
        resolved_worker_id = 1
    return await incident_service.create_with_snapshot(resolved_worker_id, incident_type, file)


@router.post("/incident-logs", response_model=IncidentLogResponse)
async def create_incident_log(body: IncidentLogCreate):
    return await incident_service.create(body)


@router.get("/incident-logs", response_model=list[IncidentLogResponse])
async def get_incident_logs():
    return await incident_service.list_all()


@router.get("/incident-logs/search/postgres", response_model=IncidentLogSearchResponse)
async def search_incident_logs_postgres(
    target_date: date_cls | None = Query(default=None),
    incident_type: str | None = Query(default=None),
    worker_id: int | None = Query(default=None),
    maker_id: int | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """PostgreSQL 기준 검색 성능 측정용 엔드포인트."""
    return await incident_service.search_postgres(
        target_date=target_date,
        incident_type=incident_type,
        worker_id=worker_id if worker_id is not None else maker_id,
        q=q,
        limit=limit,
        offset=offset,
    )


@router.get("/incident-logs/summary", response_model=IncidentLogSummaryResponse)
async def summarize_incident_logs(target_date: date_cls = Query(...)):
    """선택 날짜 기준 사고 로그 총합과 작업자별 건수를 반환한다."""
    return await incident_service.summarize_by_date(target_date)


@router.get("/incident-logs/search/elasticsearch", response_model=IncidentLogSearchResponse)
async def search_incident_logs_elasticsearch(
    target_date: date_cls | None = Query(default=None),
    incident_type: str | None = Query(default=None),
    worker_id: int | None = Query(default=None),
    maker_id: int | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Elasticsearch 기준 검색 성능 측정용 엔드포인트."""
    return await incident_service.search_elasticsearch(
        target_date=target_date,
        incident_type=incident_type,
        worker_id=worker_id if worker_id is not None else maker_id,
        q=q,
        limit=limit,
        offset=offset,
    )
