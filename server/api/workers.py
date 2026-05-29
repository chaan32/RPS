"""Worker CRUD 엔드포인트."""

from fastapi import APIRouter

from ..schemas import WorkerCreate, WorkerResponse
from ..service import maker_service

router = APIRouter()


@router.post("/workers", response_model=WorkerResponse)
async def create_worker(body: WorkerCreate):
    return await maker_service.create_worker(body)


@router.get("/workers", response_model=list[WorkerResponse])
async def get_workers():
    return await maker_service.list_workers()
