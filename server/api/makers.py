"""Maker (작업자 디바이스) CRUD 엔드포인트."""

from fastapi import APIRouter

from ..schemas import MakerCreate, MakerResponse
from ..service import maker_service

router = APIRouter()


@router.post("/makers", response_model=MakerResponse)
async def create_maker(body: MakerCreate):
    return await maker_service.create_maker(body)


@router.get("/makers", response_model=list[MakerResponse])
async def get_makers():
    return await maker_service.list_makers()
