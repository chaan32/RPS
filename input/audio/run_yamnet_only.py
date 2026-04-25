"""ESP32 → YAMnet 단독 테스트 서버.

목적: 카메라/Fusion/DB 없이 ESP32 오디오 WebSocket + YAMnet 만 띄워서
오디오 입력 경로를 격리 검증한다.

실행:
    python ai_project/input/audio/run_yamnet_only.py

ESP32 펌웨어가 ws://<이 PC의 IP>:1122/ws/audio 로 접속하면 YAMnet 분석 결과가
콘솔에 [audio] ... 형태로 찍힌다.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from input.audio import esp32_audio_router  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="YAMnet Only (ESP32 audio test)")
app.include_router(esp32_audio_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "ws_endpoint": "/ws/audio",
        "expected_format": "16kHz int16 LE mono PCM (WebSocket binary)",
    }


if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", 1122))
    print(f"\n=== YAMnet Only 서버 ===")
    print(f"WS endpoint: ws://<this-pc-ip>:{port}/ws/audio")
    print(f"확인 URL:    http://localhost:{port}/\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
