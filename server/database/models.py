from sqlalchemy import Column, Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from .base import Base


class Worker(Base):
    __tablename__ = "workers"

    id          = Column(Integer, primary_key=True, index=True)
    count       = Column(Integer, nullable=False, default=0)
    created_at  = Column(DateTime, server_default=func.now())


Maker = Worker


class IncidentLog(Base):
    __tablename__ = "incident_logs"

    id              = Column(Integer, primary_key=True, index=True)
    worker_id       = Column(Integer, ForeignKey('workers.id'), nullable=False)
    incident_type   = Column(String(10), nullable=False)   # Warning, Danger
    snapshot_path   = Column(String(512), nullable=False)   # S3 object URL
    status          = Column(String(10), nullable=False)   # success / fail
    date            = Column(Date, nullable=False, server_default=func.current_date())
    created_at      = Column(DateTime, server_default=func.now())
    __table_args__ = (
        Index(
            "idx_incident_logs_date_created_id",
            date,
            created_at.desc(),
            id.desc(),
        ),
        Index(
            "idx_incident_logs_date_type_worker_created_id",
            date,
            incident_type,
            worker_id,
            created_at.desc(),
            id.desc(),
        ),
    )

    @property
    def maker_id(self) -> int:
        """Backward-compatible alias for older frontend/report code."""
        return self.worker_id

    @maker_id.setter
    def maker_id(self, value: int) -> None:
        self.worker_id = value

class Report(Base):
    __tablename__ = "reports"

    id              = Column(Integer, primary_key=True, index=True)
    contents        = Column(Text, nullable=False)
    date            = Column(Date, nullable=False)
    created_at      = Column(DateTime, server_default=func.now())
