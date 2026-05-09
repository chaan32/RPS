"""FastAPI 앱 진입점 — 라우터 등록 + lifespan 설정만 (얇게).

도메인별 라우터: server/api/
비즈니스 로직: server/service/
앱 생명주기: server/lifespan.py
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from input.audio import esp32_audio_router

from .api import alerts, aruco, audio, health, images, incidents, makers, reports
from .lifespan import lifespan


app = FastAPI(lifespan=lifespan)

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
app.include_router(makers.router)
app.include_router(images.router)
app.include_router(incidents.router)
app.include_router(reports.router)
app.include_router(aruco.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)
