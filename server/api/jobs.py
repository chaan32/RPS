"""Background job status API."""

from fastapi import APIRouter

from ..jobs.redis_queue import get_job_status
from ..schemas import JobStatusResponse


router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    return await get_job_status(job_id)

