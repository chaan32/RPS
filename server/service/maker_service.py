"""Worker DB CRUD 서비스."""

from sqlalchemy import select

from ..database import AsyncSessionLocal, Worker
from ..schemas import WorkerCreate


async def create_worker(body: WorkerCreate) -> Worker:
    """새 Worker 1개 INSERT."""
    async with AsyncSessionLocal() as session:
        worker = Worker(**body.model_dump())
        session.add(worker)
        await session.commit()
        await session.refresh(worker)
    return worker


async def list_workers() -> list[Worker]:
    """모든 Worker SELECT."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Worker).order_by(Worker.id))
        return result.scalars().all()


async def create_maker(body: WorkerCreate) -> Worker:
    """Backward-compatible alias for old /makers calls."""
    return await create_worker(body)


async def list_makers() -> list[Worker]:
    """Backward-compatible alias for old /makers calls."""
    return await list_workers()
