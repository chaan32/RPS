"""Daily Report 생성 / 조회 / HTML 렌더링 서비스.

실제 LLM 요약 + DB 저장 로직은 server/report.py 의 generate_daily_report 가 담당.
이 서비스는 그것을 감싸는 thin layer + HTML 렌더링.
"""

from datetime import date as date_cls

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from ..database import AsyncSessionLocal, Report
from ..report import generate_daily_report


async def daily_report(target_date: date_cls | None) -> Report:
    """지정 날짜의 IncidentLog 를 LLM 으로 요약해 Report 저장 후 반환."""
    try:
        return await generate_daily_report(target_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


async def list_reports() -> list[Report]:
    """모든 Report 를 created_at DESC 로 SELECT."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Report).order_by(Report.created_at.desc())
        )
        return result.scalars().all()


async def render_html(report_id: int) -> HTMLResponse:
    """Report.contents 를 브라우저에 렌더링 가능한 HTML 페이지로 감싸 반환."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Report).where(Report.id == report_id))
        report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    page = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Report - {report.date}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; color: #333; }}
        h2 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
        h3 {{ color: #16213e; margin-top: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
        th {{ background: #16213e; color: #fff; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        ul {{ line-height: 1.8; }}
        p {{ line-height: 1.6; }}

        /* 스냅샷 2열 그리드 */
        .snapshot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
        .snapshot-card {{ padding: 10px; border: 1px solid #eee; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
        .snapshot-card img {{ width: 100%; height: 220px; object-fit: cover; border: 2px solid #e94560; border-radius: 6px; display: block; }}
        .snapshot-card p {{ margin: 8px 0 0; font-size: 0.9em; color: #555; }}
    </style>
</head>
<body>
{report.contents}
</body>
</html>"""
    return HTMLResponse(content=page)
