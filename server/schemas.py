from pydantic import BaseModel
from datetime import datetime


class MakerCreate(BaseModel):
    count: int = 0


class MakerResponse(BaseModel):
    id: int
    count: int
    created_at: datetime

    class Config:
        from_attributes = True


class IncidentLogCreate(BaseModel):
    maker_id: int
    incident_type: str   # Warning / Danger
    snapshot_path: str   # S3 object URL
    status: str = "success"


class IncidentLogResponse(BaseModel):
    id: int
    maker_id: int
    incident_type: str
    snapshot_path: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
