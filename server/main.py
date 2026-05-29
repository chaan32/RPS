"""FastAPI 애플리케이션 진입점.

이 파일은 전역 미들웨어, CORS, API 라우터, lifespan 연결만 담당한다.
도메인별 비즈니스 로직은 server/service/ 하위 모듈에 둔다.
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
    """요청별 응답 시간을 JSONL로 기록하고 Server-Timing 헤더에 노출한다."""
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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
# ESP32 오디오 WebSocket 입력. 현재 Fusion V2에서는 선택 입력으로 남겨둔다.
app.include_router(esp32_audio_router)

# 도메인별 REST API. /makers는 기존 프론트/DB 호환용 alias다.
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
