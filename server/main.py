from contextlib import asynccontextmanager
from fastapi import FastAPI
from aiomqtt import Client, MqttError
from sqlalchemy import select
from .database import engine, AsyncSessionLocal, Base, Maker, IncidentLog
from .schemas import MakerCreate, MakerResponse, IncidentLogCreate, IncidentLogResponse, AlertSend
import os


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

    yield


app = FastAPI(lifespan=lifespan)


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


# ── IncidentLog ────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)
