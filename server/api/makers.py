"""Legacy Maker CRUD endpoints kept as aliases for /workers."""

from fastapi import APIRouter

from ..schemas import WorkerCreate, WorkerResponse
from ..service import maker_service

router = APIRouter()


@router.post("/makers", response_model=WorkerResponse)
async def create_maker(body: WorkerCreate):
    return await maker_service.create_maker(body)


@router.get("/makers", response_model=list[WorkerResponse])
async def get_makers():
    return await maker_service.list_makers()
