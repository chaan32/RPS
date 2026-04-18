from sqlalchemy import Column, Integer, String, Text, Date, DateTime, func, ForeignKey
from .base import Base


class Maker(Base):
    __tablename__ = "makers"

    id          = Column(Integer, primary_key=True, index=True)
    count       = Column(Integer, nullable=False, default=0)
    created_at  = Column(DateTime, server_default=func.now())


class IncidentLog(Base):
    __tablename__ = "incident_logs"

    id              = Column(Integer, primary_key=True, index=True)
    maker_id        = Column(Integer, ForeignKey('makers.id'), nullable=False)  # 수정
    incident_type   = Column(String(10), nullable=False)   # Warning, Danger
    snapshot_path   = Column(String(512), nullable=False)   # S3 object URL
    status          = Column(String(10), nullable=False)   # success / fail
    date            = Column(Date, nullable=False, server_default=func.current_date())
    created_at      = Column(DateTime, server_default=func.now())

class Report(Base):
    __tablename__ = "reports"

    id              = Column(Integer, primary_key=True, index=True)
    contents        = Column(Text, nullable=False)
    date            = Column(Date, nullable=False)
    created_at      = Column(DateTime, server_default=func.now())
