"""ESP32-S3-WROOM-1 → 서버 WebSocket 오디오 입력.

ESP32가 I2S 마이크로 받은 PCM을 WebSocket 바이너리 프레임으로 흘려보내면,
서버가 BUFFER_SEC만큼 모아 Yamnet 탐지기에 넘기고 결과를 돌려준다.

프로토콜
--------
URL
    ws://<server>:<port>/ws/audio

클라이언트 → 서버
    - 바이너리 프레임(권장): 16kHz / 16-bit signed little-endian / mono PCM 연속 스트림.
      한 프레임 안의 샘플 수는 자유. 서버가 알아서 누적 후 윈도우로 잘라낸다.
    - 텍스트 프레임(선택): {"type": "config", "sample_rate": 16000} — 현재는 로그용.
    - 텍스트 프레임 {"type": "reset"} 을 보내면 서버 버퍼를 비운다.

서버 → 클라이언트 (모두 JSON)
    {"type": "ready", "sample_rate": 16000, "buffer_sec": 1.92}
        — 접속 직후 1회
    {"type": "silence", "rms": 0.0012}
        — 윈도우가 조용해서 스킵한 경우
    {"type": "detection", "is_anomaly": true, "max_sim": 0.87,
     "rms": 0.04, "frames": 3, "threshold": 0.80}
    {"type": "error", "message": "..."}

ESP32 펌웨어 쪽 메모
-----------------
- 48kHz로 샘플링해서 서버에서 16kHz로 다시 샘플링하는 것보단, ESP32에서
  16kHz로 샘플링(혹은 decimate)해서 보내는 편이 대역폭/지연이 훨씬 낫다.
- int16 little-endian 이 기본 가정이다. 다른 포맷을 쓰려면 config 메시지에
  {"format": "int16" | "float32"} 를 같이 보내도록 확장하면 된다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from model.yamnet import YamnetDetector


router = APIRouter()
log = logging.getLogger(__name__)


# ── 설정 ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
BUFFER_SEC = 1.92                       # 1회 분석 윈도우 길이
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SEC)
BUFFER_BYTES_INT16 = BUFFER_SAMPLES * 2  # int16 기준
MIN_RMS = 0.003                          # 이보다 조용하면 Yamnet 스킵


# ── 탐지기 (lazy singleton) ─────────────────────────────────────────────
_detector: Optional[YamnetDetector] = None
_detector_lock = asyncio.Lock()


async def get_detector() -> YamnetDetector:
    """Yamnet 모델을 딱 한 번만 로드하고 이후엔 재사용한다."""
    global _detector
    if _detector is not None:
        return _detector
    async with _detector_lock:
        if _detector is None:
            log.info("YamnetDetector 로드 중...")
            _detector = await asyncio.to_thread(YamnetDetector)
            log.info("YamnetDetector 준비 완료 (threshold=%.3f)", _detector.threshold)
    return _detector


# ── WebSocket 엔드포인트 ────────────────────────────────────────────────
@router.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    log.info("ESP32 WS 접속: %s", client)

    try:
        detector = await get_detector()
    except Exception as e:
        log.exception("Detector 로드 실패")
        await ws.send_json({"type": "error", "message": f"detector load failed: {e}"})
        await ws.close()
        return

    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "buffer_sec": BUFFER_SEC,
        "threshold": detector.threshold,
    })

    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if (text := msg.get("text")) is not None:
                await _handle_text(ws, text, buf)
                continue

            data = msg.get("bytes")
            if not data:
                continue

            buf.extend(data)
            while len(buf) >= BUFFER_BYTES_INT16:
                window = bytes(buf[:BUFFER_BYTES_INT16])
                del buf[:BUFFER_BYTES_INT16]
                await _process_window(ws, window, detector)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("WS 처리 중 오류")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        log.info("ESP32 WS 종료: %s", client)


async def _handle_text(ws: WebSocket, text: str, buf: bytearray) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return

    kind = payload.get("type")
    if kind == "reset":
        buf.clear()
    elif kind == "config":
        log.info("ESP32 config: %s", payload)


async def _process_window(ws: WebSocket, pcm_bytes: bytes, detector: YamnetDetector) -> None:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(pcm ** 2)))

    if rms < MIN_RMS:
        await ws.send_json({"type": "silence", "rms": rms})
        return

    max_sim, is_anomaly, n_frames = await asyncio.to_thread(detector.predict, pcm)
    await ws.send_json({
        "type": "detection",
        "is_anomaly": bool(is_anomaly),
        "max_sim": max_sim,
        "rms": rms,
        "frames": n_frames,
        "threshold": detector.threshold,
    })
