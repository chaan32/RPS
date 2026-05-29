from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from datetime import date as _date, datetime
from typing import Any


class WorkerCreate(BaseModel):
    count: int = 0


class WorkerResponse(BaseModel):
    id: int
    count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


MakerCreate = WorkerCreate
MakerResponse = WorkerResponse


class AlertSend(BaseModel):
    worker_id: int
    message: str   # Warning / Danger


class IncidentLogCreate(BaseModel):
    worker_id: int = Field(validation_alias=AliasChoices("worker_id", "maker_id"))
    incident_type: str   # Warning / Danger
    snapshot_path: str   # S3 object URL
    status: str = "success"
    date: _date | None = None  # 생략 시 오늘 날짜

    model_config = ConfigDict(populate_by_name=True)


class IncidentLogResponse(BaseModel):
    id: int
    worker_id: int
    maker_id: int
    incident_type: str
    snapshot_path: str
    status: str
    date: _date
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IncidentLogSearchResponse(BaseModel):
    backend: str
    total: int
    took_ms: float
    items: list[IncidentLogResponse]


class WorkerIncidentSummary(BaseModel):
    worker_id: int
    total: int
    warning: int
    danger: int


class IncidentLogSummaryResponse(BaseModel):
    target_date: _date
    total: int
    warning: int
    danger: int
    workers: list[WorkerIncidentSummary]


class ReportResponse(BaseModel):
    id: int
    contents: str
    date: _date
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportListItemResponse(BaseModel):
    id: int
    date: _date
    created_at: datetime
    contents_length: int


class JobSubmitResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    status_url: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    payload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queued_ms: float | None = None
    runtime_ms: float | None = None
    total_ms: float | None = None
