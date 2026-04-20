import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from aiomqtt import Client, MqttError
from sqlalchemy import select
from datetime import date as date_cls
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from .database import engine, AsyncSessionLocal, Base, Maker, IncidentLog, Report
from .pipeline import MQTTHandler
from .database.store import save_file
from .database.store.service import USB_BASE_PATH
from .schemas import MakerCreate, MakerResponse, IncidentLogCreate, IncidentLogResponse, AlertSend, ReportResponse
from .report import generate_daily_report
# 카메라 API 필요 시 아래 주석 해제
# import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "input", "media"))
# from camera import camera_manager
import os
# import cv2  # 현재 웹에서 카메라 미사용
# from fastapi.responses import StreamingResponse  # 현재 웹에서 카메라 미사용


async def mqtt_consumer(queue: asyncio.Queue):
    while True:
        data = await queue.get()
        print(f"🛠  Consumer got: {data}")
        # TODO: 필요 시 DB 저장 등 추가 처리

''' 
서버를 실행하고 끌 때 카메라 관련 내용 정리하게 하는 함수
'''
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB가 없으면 생성하기 - 있으면 생략 
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # DB에 Maker가 1~5까지 있는지 보고 없으면 생성하기 - 있으면 생략 
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker))
        if not result.scalars().first():
            session.add_all([Maker(id=i, count=0) for i in range(1, 6)])
            await session.commit()

    # 카메라 스트림 시작 (API 스냅샷/스트리밍용) — 현재 웹에서 미사용
    # camera_manager.start_all()

    # YOLO + ArUco 카메라 모니터를 별도 프로세스로 띄우기
    import subprocess, sys
    yolo_path = os.path.join(os.path.dirname(__file__), "..", "input", "media", "run_yolo.py")
    
    # 서버가 동작하는 프로세스와 다른 별도의 프로세스 새로 생성! 
    cam_proc = subprocess.Popen([sys.executable, yolo_path])

    # MQTT 파이프라인 시작
    queue: asyncio.Queue = asyncio.Queue() # 비동기 큐로 생성
    handler = MQTTHandler(queue) # 메세지가 오면 큐에 메세지를 넣음 
    producer_task = asyncio.create_task(handler.run()) # 둘 다 백그라운드 비동기 태스크로 등록
    consumer_task = asyncio.create_task(mqtt_consumer(queue))

    # 여기까지 서버를 실행하기 전에 실행할 것들 
    try:
        yield
    finally:
        # 서버가 꺼지고 실행할 것들
        producer_task.cancel()
        consumer_task.cancel()
        cam_proc.terminate()
        # camera_manager.stop_all()  # 현재 웹에서 카메라 미사용


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


# ── Image Storage (USB) ───────────────────────────────────────────────

@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """이미지를 USB에 저장하고 저장 경로를 반환"""
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{today}/{file.filename}"
    path = save_file(contents, key, content_type=file.content_type)
    return {"status": "success", "path": path, "filename": file.filename}


@app.get("/images/serve")
async def serve_usb_image(path: str):
    """USB에 저장된 이미지를 HTTP로 서빙.

    브라우저가 `<img src="/api/images/serve?path=...">` 로 요청하면
    USB_BASE_PATH 아래의 실제 파일을 반환한다. 경로 탈출을 방지한다.
    """
    base = Path(USB_BASE_PATH).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(target)


# ── IncidentLog ───────────────────────────────────────────────────────

@app.post("/incident-logs/with-snapshot", response_model=IncidentLogResponse)
async def create_incident_with_snapshot(
    maker_id: int,
    incident_type: str,
    file: UploadFile = File(...),
):
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{today}/{file.filename}"
    url = save_file(contents, key, content_type=file.content_type)

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


# ── Camera (현재 웹에서 미사용 — 필요 시 주석 해제) ─────────────────────

# @app.get("/cameras")
# def list_cameras():
#     """연결된 카메라 목록 반환."""
#     return {"cameras": camera_manager.list_cameras()}
#
#
# def _generate_mjpeg(cam_name: str):
#     """MJPEG 스트림 제너레이터."""
#     while True:
#         ret, frame = camera_manager.get_frame(cam_name)
#         if not ret or frame is None:
#             continue
#         _, buf = cv2.imencode(".jpg", frame)
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
#         )
#
#
# @app.get("/cameras/{cam_name}/stream")
# def camera_stream(cam_name: str):
#     """브라우저에서 MJPEG 실시간 영상을 볼 수 있는 엔드포인트."""
#     if cam_name not in camera_manager.streams:
#         raise HTTPException(status_code=404, detail=f"카메라 '{cam_name}' 없음")
#     return StreamingResponse(
#         _generate_mjpeg(cam_name),
#         media_type="multipart/x-mixed-replace; boundary=frame",
#     )
#
#
# @app.get("/cameras/{cam_name}/snapshot")
# def camera_snapshot(cam_name: str):
#     """카메라의 현재 프레임을 JPEG 이미지로 반환."""
#     if cam_name not in camera_manager.streams:
#         raise HTTPException(status_code=404, detail=f"카메라 '{cam_name}' 없음")
#     ret, frame = camera_manager.get_frame(cam_name)
#     if not ret or frame is None:
#         raise HTTPException(status_code=503, detail="프레임을 가져올 수 없음")
#     _, buf = cv2.imencode(".jpg", frame)
#     return StreamingResponse(
#         iter([buf.tobytes()]),
#         media_type="image/jpeg",
#     )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)
