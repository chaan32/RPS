from pydantic import BaseModel
from datetime import date as _date, datetime


class MakerCreate(BaseModel):
    count: int = 0


class MakerResponse(BaseModel):
    id: int
    count: int
    created_at: datetime

    class Config:
        from_attributes = True


class AlertSend(BaseModel):
    maker_id: int
    message: str   # Warning / Danger


class IncidentLogCreate(BaseModel):
    maker_id: int
    incident_type: str   # Warning / Danger
    snapshot_path: str   # S3 object URL
    status: str = "success"
    date: _date | None = None  # 생략 시 오늘 날짜


class IncidentLogResponse(BaseModel):
    id: int
    maker_id: int
    incident_type: str
    snapshot_path: str
    status: str
    date: _date
    created_at: datetime

    class Config:
        from_attributes = True


class ReportResponse(BaseModel):
    id: int
    contents: str
    date: _date
    created_at: datetime

    class Config:
        from_attributes = True
