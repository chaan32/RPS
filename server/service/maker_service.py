"""Worker DB CRUD 서비스.

maker라는 이름은 초기 설계의 legacy 명칭이다. 기존 API 호환을 위해 alias만
남기고 실제 도메인 명칭은 worker로 통일한다.
"""

from sqlalchemy import select

from ..database import AsyncSessionLocal, Worker
from ..schemas import WorkerCreate


async def create_worker(body: WorkerCreate) -> Worker:
    """작업자 레코드를 생성한다."""
    async with AsyncSessionLocal() as session:
        worker = Worker(**body.model_dump())
        session.add(worker)
        await session.commit()
        await session.refresh(worker)
    return worker


async def list_workers() -> list[Worker]:
    """작업자 목록을 id 순서로 조회한다."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Worker).order_by(Worker.id))
        return result.scalars().all()


async def create_maker(body: WorkerCreate) -> Worker:
    """기존 /makers 호출을 위한 호환 alias."""
    return await create_worker(body)


async def list_makers() -> list[Worker]:
    """기존 /makers 호출을 위한 호환 alias."""
    return await list_workers()
