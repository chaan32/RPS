"""IncidentLog (사고 기록) DB CRUD 서비스."""

from datetime import datetime

from fastapi import UploadFile
from sqlalchemy import select

from ..database import AsyncSessionLocal, IncidentLog
from ..database.store import save_file
from ..schemas import IncidentLogCreate


async def create_with_snapshot(
    maker_id: int,
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
            maker_id=maker_id,
            incident_type=incident_type,
            snapshot_path=url,
            status="success",
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
    return log


async def create(body: IncidentLogCreate) -> IncidentLog:
    """스냅샷 없이 IncidentLog INSERT (placeholder 경로)."""
    async with AsyncSessionLocal() as session:
        log = IncidentLog(**body.model_dump())
        session.add(log)
        await session.commit()
        await session.refresh(log)
    return log


async def list_all() -> list[IncidentLog]:
    """모든 IncidentLog SELECT."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(IncidentLog))
        return result.scalars().all()
