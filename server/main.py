"""FastAPI 앱 진입점 — 라우터 등록 + lifespan 설정만 (얇게).

도메인별 라우터: server/api/
비즈니스 로직: server/service/
앱 생명주기: server/lifespan.py
"""

import os
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from input.audio import esp32_audio_router

from .api import alerts, aruco, audio, health, images, incidents, jobs, makers, reports, workers
from .lifespan import lifespan
from .utils.metrics import JsonLinesLogger


app = FastAPI(lifespan=lifespan)


request_metrics_logger = JsonLinesLogger(
    os.getenv("SERVER_METRICS_PATH", "metrics/server_requests.jsonl")
)


@app.middleware("http")
async def log_request_metrics(request, call_next):
    """Record per-request latency for Mac/Docker benchmark comparison."""
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    status_code = 500
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["Server-Timing"] = f"app;dur={duration_ms:.2f}"
        if os.getenv("DISABLE_SERVER_METRICS", "").lower() not in ("1", "true", "yes"):
            request_metrics_logger.log({
                "ts": time.time(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            })

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ────────────────────────────────────────────────────────
# ESP32-S3-WROOM-1 오디오 입력 WebSocket  (/ws/audio)
app.include_router(esp32_audio_router)

# 도메인별 REST API
app.include_router(health.router)
app.include_router(audio.router)
app.include_router(alerts.router)
app.include_router(workers.router)
app.include_router(makers.router)
app.include_router(images.router)
app.include_router(incidents.router)
app.include_router(reports.router)
app.include_router(jobs.router)
app.include_router(aruco.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)
