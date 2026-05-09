"""IncidentLog (사고 기록) CRUD 엔드포인트."""

from fastapi import APIRouter, File, UploadFile

from ..schemas import IncidentLogCreate, IncidentLogResponse
from ..service import incident_service

router = APIRouter()


@router.post("/incident-logs/with-snapshot", response_model=IncidentLogResponse)
async def create_incident_with_snapshot(
    maker_id: int,
    incident_type: str,
    file: UploadFile = File(...),
):
    return await incident_service.create_with_snapshot(maker_id, incident_type, file)


@router.post("/incident-logs", response_model=IncidentLogResponse)
async def create_incident_log(body: IncidentLogCreate):
    return await incident_service.create(body)


@router.get("/incident-logs", response_model=list[IncidentLogResponse])
async def get_incident_logs():
    return await incident_service.list_all()
