"""헬스체크 엔드포인트."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "FastAPI", "Status": "Running"}
