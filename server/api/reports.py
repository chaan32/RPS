"""Daily Report 엔드포인트 (생성 / 조회 / HTML)."""

from datetime import date as date_cls

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ..jobs.redis_queue import enqueue_job
from ..schemas import JobSubmitResponse, ReportListItemResponse, ReportResponse
from ..service import report_service

router = APIRouter()


@router.post("/reports/generate", response_model=ReportResponse)
async def create_daily_report(
    target_date: date_cls | None = Query(
        default=None, description="YYYY-MM-DD (KST). 생략 시 오늘"
    ),
):
    """수동 트리거: 지정 날짜(KST)의 IncidentLog 를 LLM 으로 요약 → Report 저장."""
    return await report_service.daily_report(target_date)


@router.post("/reports/generate-async", response_model=JobSubmitResponse)
async def create_daily_report_async(
    target_date: date_cls | None = Query(
        default=None, description="YYYY-MM-DD (KST). 생략 시 오늘"
    ),
):
    """리포트 생성을 Redis background job으로 넘기고 즉시 job_id를 반환."""
    return await enqueue_job(
        "daily_report",
        {"target_date": target_date.isoformat() if target_date else None},
    )


@router.get("/reports", response_model=list[ReportResponse])
async def list_reports():
    return await report_service.list_reports()


@router.get("/reports/summary", response_model=list[ReportListItemResponse])
async def list_report_summaries(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """리포트 목록 화면용 경량 조회. 큰 HTML contents는 반환하지 않는다."""
    return await report_service.list_report_summaries(limit=limit, offset=offset)


@router.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: int):
    """리포트 상세 JSON. 목록에서는 /reports/summary를 우선 사용."""
    return await report_service.get_report(report_id)


@router.get("/reports/{report_id}/html", response_class=HTMLResponse)
async def get_report_html(report_id: int):
    """Report.contents 를 브라우저에서 바로 HTML 로 렌더링."""
    return await report_service.render_html(report_id)
