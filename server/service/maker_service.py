"""Maker (작업자 디바이스) DB CRUD 서비스."""

from sqlalchemy import select

from ..database import AsyncSessionLocal, Maker
from ..schemas import MakerCreate


async def create_maker(body: MakerCreate) -> Maker:
    """새 Maker 1개 INSERT."""
    async with AsyncSessionLocal() as session:
        maker = Maker(**body.model_dump())
        session.add(maker)
        await session.commit()
        await session.refresh(maker)
    return maker


async def list_makers() -> list[Maker]:
    """모든 Maker SELECT."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker))
        return result.scalars().all()
