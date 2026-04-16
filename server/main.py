import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from aiomqtt import Client, MqttError
from sqlalchemy import select
from datetime import date as date_cls
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse
from .database import engine, AsyncSessionLocal, Base, Maker, IncidentLog, Report
from .mqtt import MQTTHandler
from .s3 import upload_file
from .schemas import MakerCreate, MakerResponse, IncidentLogCreate, IncidentLogResponse, AlertSend, ReportResponse
from .report import generate_daily_report
import os


async def mqtt_consumer(queue: asyncio.Queue):
    while True:
        data = await queue.get()
        print(f"🛠  Consumer got: {data}")
        # TODO: 필요 시 DB 저장 등 추가 처리


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 최초 실행 시 Maker 5개 시드
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker))
        if not result.scalars().first():
            session.add_all([Maker(id=i, count=0) for i in range(1, 6)])
            await session.commit()

    # MQTT 파이프라인 시작
    queue: asyncio.Queue = asyncio.Queue()
    handler = MQTTHandler(queue)
    producer_task = asyncio.create_task(handler.run())
    consumer_task = asyncio.create_task(mqtt_consumer(queue))

    try:
        yield
    finally:
        producer_task.cancel()
        consumer_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "FastAPI", "Status": "Running"}


# ── MQTT Send ─────────────────────────────────────────────────────────

@app.post("/send-alert")
async def send_alert(maker_id : str, direction : str):
    """
    server->arduino by MQTT pipeline
    
    topic: crane/{maker_id}/vibration
    """

    broker = os.getenv("MQTT_BROKER", "127.0.0.1")
    topic  = f"crane/{maker_id}/vibration"
    if maker_id == '5' or maker_id == '4':
        topic = f"forklift/{maker_id}/vibration"
    try:
        async with Client(broker, timeout=3) as client:
            await client.publish(topic, payload=direction)
        return {"status": "success", "topic": topic, "message": direction}
    except MqttError as e:
        return {"status": "fail", "error": str(e)}


# ── Maker ──────────────────────────────────────────────────────────────

@app.post("/makers", response_model=MakerResponse)
async def create_maker(body: MakerCreate):
    async with AsyncSessionLocal() as session:
        maker = Maker(**body.model_dump())
        session.add(maker)
        await session.commit()
        await session.refresh(maker)
    return maker


@app.get("/makers", response_model=list[MakerResponse])
async def get_makers():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker))
        makers = result.scalars().all()
    return makers


# ── IncidentLog + S3 ───────────────────────────────────────────────────

@app.post("/incident-logs/with-snapshot", response_model=IncidentLogResponse)
async def create_incident_with_snapshot(
    maker_id: int,
    incident_type: str,
    file: UploadFile = File(...),
):
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"snapshots/{today}/{file.filename}"
    url = upload_file(contents, key, content_type=file.content_type)

    async with AsyncSessionLocal() as session:
        log = IncidentLog(
            maker_id=maker_id,
            incident_type=incident_type,
            snapshot_path=url,
            status="success",
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
    return log

@app.post("/incident-logs", response_model=IncidentLogResponse)
async def create_incident_log(body: IncidentLogCreate):
    async with AsyncSessionLocal() as session:
        log = IncidentLog(**body.model_dump())
        session.add(log)
        await session.commit()
        await session.refresh(log)
    return log


@app.get("/incident-logs", response_model=list[IncidentLogResponse])
async def get_incident_logs():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(IncidentLog))
        logs = result.scalars().all()
    return logs


# ── Report ─────────────────────────────────────────────────────────────

@app.post("/reports/generate", response_model=ReportResponse)
async def create_daily_report(target_date: date_cls | None = Query(default=None, description="YYYY-MM-DD (KST). 생략 시 오늘")):
    """
    수동 트리거: 지정 날짜(KST)의 IncidentLog를 모아 LLM에 요약시키고 Report로 저장.
    """
    try:
        report = await generate_daily_report(target_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return report


@app.get("/reports", response_model=list[ReportResponse])
async def list_reports():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Report).order_by(Report.created_at.desc()))
        return result.scalars().all()


@app.get("/reports/{report_id}/html", response_class=HTMLResponse)
async def get_report_html(report_id: int):
    """Report contents를 브라우저에서 바로 HTML로 렌더링."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Report).where(Report.id == report_id))
        report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    page = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Report - {report.date}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; color: #333; }}
        h2 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
        h3 {{ color: #16213e; margin-top: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
        th {{ background: #16213e; color: #fff; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        ul {{ line-height: 1.8; }}
        p {{ line-height: 1.6; }}

        /* 스냅샷 2열 그리드 */
        .snapshot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
        .snapshot-card {{ padding: 10px; border: 1px solid #eee; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
        .snapshot-card img {{ width: 100%; height: 220px; object-fit: cover; border: 2px solid #e94560; border-radius: 6px; display: block; }}
        .snapshot-card p {{ margin: 8px 0 0; font-size: 0.9em; color: #555; }}
    </style>
</head>
<body>
{report.contents}
</body>
</html>"""
    return HTMLResponse(content=page)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)
