import os

from datetime import datetime, timedelta, timezone, date as date_cls
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal, IncidentLog, Report


KST = timezone(timedelta(hours=9))


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


def _serialize(logs: list[IncidentLog]) -> list[dict]:
    return [
        {
            "id": log.id,
            "maker_id": log.maker_id,
            "incident_type": log.incident_type,
            "snapshot_path": log.snapshot_path,
            "status": log.status,
            "created_at_utc": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


async def generate_daily_report(target: date_cls | None = None) -> Report:
    if target is None:
        target = datetime.now(KST).date()

    async with AsyncSessionLocal() as session:
        logs = await _fetch_logs(session, target)
        if not logs:
            raise ValueError(f"No incident logs found for {target} (KST)")

        serialized = _serialize(logs)
        date_iso = target.isoformat()

        backend = os.getenv("LLM_BACKEND", "gemini").lower()
        if backend == "gemini":
            from .llm import summarize_logs_to_html
            html = summarize_logs_to_html(date_iso, serialized)
        elif backend == "local":
            from .llm_local import summarize_logs_to_html as local_summarize
            html = await local_summarize(date_iso, serialized)
        else:
            raise ValueError(
                f"잘못된 LLM_BACKEND 값: '{backend}'. "
                "'gemini' 또는 'local'만 사용 가능합니다."
            )

        report = Report(contents=html, date=target)
        session.add(report)
        await session.commit()
        await session.refresh(report)
        return report
