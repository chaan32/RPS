import os
import html as html_lib

from datetime import datetime, timedelta, timezone, date as date_cls
from pathlib import Path
from urllib.parse import quote
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal, IncidentLog, Report
from ..database.store.service import USB_BASE_PATH


KST = timezone(timedelta(hours=9))


REPORT_STYLE = """<style>
.safety-report { font-family: 'Segoe UI','Pretendard',Arial,sans-serif; color: #1f2937 !important; line-height: 1.65; font-size: 15px; font-weight: 400 !important; }
.safety-report h2 { font-size: 22px; color: #0f172a !important; margin: 0 0 12px; padding-bottom: 10px; border-bottom: 3px solid #ef4444; font-weight: 800 !important; }
.safety-report h3 { font-size: 18px; color: #1e293b !important; margin: 28px 0 10px; font-weight: 700 !important; }
.safety-report p { color: #334155 !important; margin: 6px 0; }
.safety-report .summary-cards { display: flex; gap: 12px; margin: 16px 0 24px; flex-wrap: wrap; }
.safety-report .summary-card { flex: 1 1 150px; padding: 14px 18px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; }
.safety-report .summary-card .label { font-size: 12px; font-weight: 700 !important; color: #64748b !important; letter-spacing: .5px; text-transform: uppercase; }
.safety-report .summary-card .value { font-size: 26px; font-weight: 800 !important; color: #0f172a !important; margin-top: 6px; }
.safety-report .summary-card.warn { background: #fffbeb; border-color: #fde68a; }
.safety-report .summary-card.warn .value { color: #d97706 !important; }
.safety-report .summary-card.danger { background: #fef2f2; border-color: #fecaca; }
.safety-report .summary-card.danger .value { color: #dc2626 !important; }
.safety-report table { width: 100%; border-collapse: collapse; margin: 10px 0 20px; font-size: 14px; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
.safety-report th { background: #1e293b !important; color: #ffffff !important; padding: 11px 12px; text-align: left; font-weight: 700 !important; border-bottom: 1px solid #0f172a; }
.safety-report td { border-bottom: 1px solid #e2e8f0; padding: 10px 12px; color: #1f2937 !important; }
.safety-report tbody tr:nth-child(even) { background: #f8fafc; }
.safety-report ul { margin: 8px 0 18px 22px; padding: 0; list-style: disc outside; }
.safety-report ul li { margin-bottom: 6px; color: #334155 !important; }
.safety-report .recommendation { padding: 14px 16px; background: #fef3c7 !important; border-left: 4px solid #f59e0b; border-radius: 6px; color: #78350f !important; font-weight: 600 !important; margin-top: 12px; }
.safety-report .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 700 !important; }
.safety-report .badge.warn { background: #fef3c7; color: #b45309 !important; }
.safety-report .badge.danger { background: #fee2e2; color: #b91c1c !important; }
.safety-report .snapshot-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 16px; margin-top: 14px; }
.safety-report .snapshot-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.safety-report .snapshot-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 6px; border: 2px solid #ef4444; display: block; }
.safety-report .snapshot-card p { margin: 8px 0 0; font-size: 13px; color: #475569 !important; line-height: 1.45; font-weight: 500 !important; }
</style>"""


def _kst_iso(dt_utc_iso: str | None) -> str:
    if not dt_utc_iso:
        return ""
    try:
        dt = datetime.fromisoformat(dt_utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_utc_iso


def _build_snapshot_gallery(serialized_logs: list[dict]) -> str:
    """Danger 등급 로그만 추려서 스냅샷 그리드 HTML을 생성한다."""
    danger_logs = [l for l in serialized_logs if (l.get("incident_type") or "").lower() == "danger"]
    if not danger_logs:
        return ""

    cards: list[str] = []
    for log in danger_logs:
        src = html_lib.escape(log.get("snapshot_path") or "", quote=True)
        maker = html_lib.escape(str(log.get("maker_id", "")))
        time_str = html_lib.escape(_kst_iso(log.get("created_at_utc")))
        alt = html_lib.escape(f"Maker {maker} - {time_str} - Danger", quote=True)
        cards.append(
            '<div class="snapshot-card">'
            f'<img src="{src}" alt="{alt}">'
            f'<p><strong>Maker ID:</strong> {maker}<br>'
            f'<strong>시각:</strong> {time_str}<br>'
            f'<strong>유형:</strong> Danger</p>'
            '</div>'
        )

    return (
        "<h3>위험 상황 스냅샷</h3>"
        '<div class="snapshot-grid">' + "".join(cards) + "</div>"
    )


def _kst_day_bounds_utc(target: date_cls) -> tuple[datetime, datetime]:
    start_kst = datetime(target.year, target.month, target.day, 0, 0, 0, tzinfo=KST)
    end_kst = start_kst + timedelta(days=1)
    return start_kst.astimezone(timezone.utc).replace(tzinfo=None), \
           end_kst.astimezone(timezone.utc).replace(tzinfo=None)


# incidentLog 테이블에서 원하는 날짜에 해당하는 행 가져오기 
async def _fetch_logs(session: AsyncSession, target: date_cls) -> list[IncidentLog]:
    start_utc, end_utc = _kst_day_bounds_utc(target)
    stmt = (
        select(IncidentLog)
        .where(IncidentLog.created_at >= start_utc)
        .where(IncidentLog.created_at < end_utc)
        .order_by(IncidentLog.created_at)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _to_image_url(snapshot_path: str | None) -> str | None:
    """DB에 저장된 USB 절대 경로를 브라우저에서 로드 가능한 URL로 변환.

    프론트엔드가 vite proxy로 `/api` → 백엔드 를 연결하므로
    `/api/images/serve?path=<key>` 형태의 URL을 반환한다.
    USB_BASE_PATH 범위를 벗어나거나 변환에 실패하면 원본을 그대로 돌려준다.
    """
    if not snapshot_path:
        return snapshot_path
    try:
        rel = Path(snapshot_path).resolve().relative_to(Path(USB_BASE_PATH).resolve())
    except (ValueError, OSError):
        return snapshot_path
    key = rel.as_posix()
    return f"/api/images/serve?path={quote(key, safe='/')}"


def _serialize(logs: list[IncidentLog]) -> list[dict]:
    return [
        {
            "id": log.id,
            "maker_id": log.maker_id,
            "incident_type": log.incident_type,
            "snapshot_path": _to_image_url(log.snapshot_path),
            "status": log.status,
            "created_at_utc": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


async def generate_daily_report(target: date_cls | None = None) -> Report:
    if target is None:
        target = datetime.now(KST).date()

    async with AsyncSessionLocal() as session:
        print("{target} report 생성 시도 함")
        logs = await _fetch_logs(session, target)
        if not logs:
            raise ValueError(f"No incident logs found for {target} (KST)")

        serialized = _serialize(logs)
        date_iso = target.isoformat()

        backend = os.getenv("LLM_BACKEND", "gemini").lower()
        if backend == "gemini":
            from .llm import summarize_logs_to_html
            llm_html = summarize_logs_to_html(date_iso, serialized)
        elif backend == "local":
            from .llm_local import summarize_logs_to_html as local_summarize
            llm_html = await local_summarize(date_iso, serialized)
        else:
            raise ValueError(
                f"잘못된 LLM_BACKEND 값: '{backend}'. "
                "'gemini' 또는 'local'만 사용 가능합니다."
            )

        # LLM이 스타일/이미지를 누락해도 문제없도록, 여기서 결정적으로 조립한다.
        # 1) 공용 스타일 주입, 2) Danger 스냅샷 갤러리는 Python에서 직접 렌더.
        gallery_html = _build_snapshot_gallery(serialized)
        html = (
            '<div class="safety-report">'
            f"{REPORT_STYLE}"
            f"{llm_html}"
            f"{gallery_html}"
            "</div>"
        )

        report = Report(contents=html, date=target)
        session.add(report)
        await session.commit()
        await session.refresh(report)
        return report
