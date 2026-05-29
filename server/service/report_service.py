"""Daily Report 생성 / 조회 / HTML 렌더링 서비스.

실제 LLM 요약 + DB 저장 로직은 server/report.py 의 generate_daily_report 가 담당.
이 서비스는 그것을 감싸는 thin layer + HTML 렌더링.
"""

import html as html_lib
import re

from datetime import date as date_cls
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from ..database import AsyncSessionLocal, Report
from ..database.store.service import LOCAL_FALLBACK_PATH, USB_BASE_PATH
from ..report.service import _to_image_url, generate_daily_report


_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)
_SNAPSHOT_CARD_RE = re.compile(
    r'(<div class="snapshot-card">.*?<img\b[^>]*\bsrc=")([^"]+)(".*?</div>)',
    re.IGNORECASE | re.DOTALL,
)
_EMPTY_SNAPSHOT_GRID_RE = re.compile(
    r'<h3>위험 상황 스냅샷</h3>\s*<div class="snapshot-grid">\s*</div>',
    re.IGNORECASE,
)


def _image_src_exists(src: str) -> bool:
    """브라우저에 내려줄 이미지 URL이 실제 로컬 저장소 파일을 가리키는지 확인."""
    if src.startswith(("http://", "https://")):
        return True

    parsed = urlparse(src)
    if parsed.path != "/api/images/serve":
        return False

    path_values = parse_qs(parsed.query).get("path", [])
    if not path_values:
        return False

    stores = {
        "usb": Path(USB_BASE_PATH).resolve(),
        "local": Path(LOCAL_FALLBACK_PATH).resolve(),
    }
    prefix, sep, rest = path_values[0].partition("/")
    if not sep or prefix not in stores:
        return False

    target = (stores[prefix] / rest).resolve()
    try:
        target.relative_to(stores[prefix])
    except ValueError:
        return False
    return target.is_file()


def _normalize_report_contents(contents: str) -> str:
    """기존 리포트 HTML의 로컬 파일 src를 브라우저 접근 가능한 이미지 API URL로 보정."""
    def replace_src(match: re.Match[str]) -> str:
        prefix, raw_src, suffix = match.groups()
        normalized = _to_image_url(html_lib.unescape(raw_src)) or raw_src
        return f'{prefix}{html_lib.escape(normalized, quote=True)}{suffix}'

    normalized_contents = _IMG_SRC_RE.sub(replace_src, contents)

    def remove_missing_snapshot_card(match: re.Match[str]) -> str:
        prefix, raw_src, suffix = match.groups()
        src = html_lib.unescape(raw_src)
        if _image_src_exists(src):
            return f"{prefix}{raw_src}{suffix}"
        return ""

    normalized_contents = _SNAPSHOT_CARD_RE.sub(
        remove_missing_snapshot_card,
        normalized_contents,
    )
    return _EMPTY_SNAPSHOT_GRID_RE.sub("", normalized_contents)


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
        reports = list(result.scalars().all())
        for report in reports:
            report.contents = _normalize_report_contents(report.contents)
        return reports


async def list_report_summaries(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return lightweight report rows for list screens.

    `/reports` keeps backward compatibility and returns full HTML contents.
    This summary query avoids transferring and normalizing large report HTML
    when the caller only needs a list of available reports.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    Report.id,
                    Report.date,
                    Report.created_at,
                    func.length(Report.contents).label("contents_length"),
                )
                .order_by(Report.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()

    return [
        {
            "id": row.id,
            "date": row.date,
            "created_at": row.created_at,
            "contents_length": int(row.contents_length or 0),
        }
        for row in rows
    ]


async def get_report(report_id: int) -> Report:
    """Return one full Report row with browser-loadable image URLs."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Report).where(Report.id == report_id))
        report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.contents = _normalize_report_contents(report.contents)
    return report


async def render_html(report_id: int) -> HTMLResponse:
    """Report.contents 를 브라우저에 렌더링 가능한 HTML 페이지로 감싸 반환."""
    report = await get_report(report_id)
    contents = report.contents
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
{contents}
</body>
</html>"""
    return HTMLResponse(content=page)
