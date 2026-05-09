"""Daily Report 엔드포인트 (생성 / 조회 / HTML)."""

from datetime import date as date_cls

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ..schemas import ReportResponse
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


@router.get("/reports", response_model=list[ReportResponse])
async def list_reports():
    return await report_service.list_reports()


@router.get("/reports/{report_id}/html", response_class=HTMLResponse)
async def get_report_html(report_id: int):
    """Report.contents 를 브라우저에서 바로 HTML 로 렌더링."""
    return await report_service.render_html(report_id)
