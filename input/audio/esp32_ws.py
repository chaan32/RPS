"""ESP32-S3-WROOM-1 → 서버 WebSocket 오디오 입력.

ESP32가 I2S 마이크로 받은 PCM을 WebSocket 바이너리 프레임으로 흘려보내면,
서버가 BUFFER_SEC만큼 모아 Yamnet 탐지기에 넘기고 결과를 돌려준다.

프로토콜
--------
URL - ws://<server>:<port>/ws/audio

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
- 48kHz로 샘플링해서 서버에서 16kHz로 다시 샘플링하는 것보단, ESP32에서 16kHz로 샘플링(혹은 decimate)해서 보내는 편이 대역폭/지연이 훨씬 낫다.
- int16 little-endian 이 기본 가정이다. 다른 포맷을 쓰려면 config 메시지에
{"format": "int16" | "float32"} 를 같이 보내도록 확장하면 된다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
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
MIN_RMS = 0.003                          # rms 임계치. 이보다 조용하면 Yamnet 스킵



# ── 최신 score 공유 (다른 프로세스의 fusion 파이프라인이 HTTP로 읽어감) ──
# realtime_camera.py(별도 subprocess)가 sounddevice 대신 ESP32 audio를 쓰도록
# server.main 의 /audio/score 엔드포인트에서 이 값을 노출한다.

# latest_score를 쓰는 와중에 가져가면 안되니까 크리티컬 섹션에 진입해서 가져갈 수 있도록 구현함
_latest = {
    "score": 0.0,         # max_sim ∈ [0, 1] (silence 시 0.0)
    "is_anomaly": False,
    "rms": 0.0,
    "ts": 0.0,            # epoch sec — fresh 여부 판단용
}
# _latest_lock은 좌물쇠 이름인 것 
_latest_lock = threading.Lock()


def get_latest_score() -> dict:
    """가장 최근 ESP32 yamnet 결과 스냅샷."""
    with _latest_lock: # 크리티컬 섹션 진입하기 
        return dict(_latest)


def _update_latest(score: float, is_anomaly: bool, rms: float) -> None:
    with _latest_lock: # 크리티컬 섹션 진입하기 
        # 값들 저장하기 
        _latest["score"] = float(np.clip(score, 0.0, 1.0))
        _latest["is_anomaly"] = bool(is_anomaly)
        _latest["rms"] = float(rms)
        _latest["ts"] = time.time()


# ── 탐지기 (lazy singleton) ─────────────────────────────────────────────
_detector: Optional[YamnetDetector] = None
_detector_lock = asyncio.Lock()

# async def : 코루틴 함수이다 ㅇㅇ 
async def get_detector() -> YamnetDetector:
    """Yamnet 모델을 딱 한 번만 로드하고 이후엔 재사용한다."""
    global _detector
    if _detector is not None:
        return _detector
    async with _detector_lock:
        if _detector is None:
            log.info("YamnetDetector 로드 중...")
            # _detector에 YamnetDetector 객체를 얻고자 하기 위해서 기다리는 것 -> 이거 좀 오래 걸려서 쩔 수 없음
            _detector = await asyncio.to_thread(YamnetDetector)
            log.info("YamnetDetector 준비 완료 (threshold=%.3f)", _detector.threshold)
    return _detector


# ── WebSocket 엔드포인트 ────────────────────────────────────────────────
@router.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    # 받은 웹소켓 객체로, 웹소켓 핸드쉐이크ㅇㅇ
    await ws.accept()
    # client 
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    log.info("ESP32 WS 접속: %s", client)

    try:
        # YamnetDetector 객체 받아오기 
        detector = await get_detector()
    except Exception as e:
        log.exception("Detector 로드 실패")
        await ws.send_json({"type": "error", "message": f"detector load failed: {e}"})
        await ws.close()
        return

    # OS 네트워크가 송신 버터를 비우는 이벤트를 감지하고 일어남 
    # 제대로 연결이 됐다고 알리는 메세지
    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "buffer_sec": BUFFER_SEC,
        "threshold": detector.threshold,
    })

    buf = bytearray()
    try:
        while True:
            # OS 네트워크가 클라이언트가 메세지를 보냄을 감지 함 
            msg = await ws.receive()

            # 연결 안됨 
            if msg["type"] == "websocket.disconnect":
                break
            

            if (text := msg.get("text")) is not None:
                await _handle_text(ws, text, buf)
                continue

            data = msg.get("bytes")
            if not data:
                continue

            buf.extend(data)
            while len(buf) >= BUFFER_BYTES_INT16: # 버퍼 사이즈가 16kHz * 1.92(초) * 2 = 61440 바이트 
                window = bytes(buf[:BUFFER_BYTES_INT16])
                del buf[:BUFFER_BYTES_INT16]
                # 복사본으로 값을 넘기는게 데이터의 오전송을 막을 수 있음 
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
    '''
1.92초 분량의 PCM 바이트를 받아서 (from audio_ws 함수) YAMnet 분석 가능한 형태로 변환하고, 
RMS(에너지) 체크 후 조용하면 skip / 시끄러우면 YAMnet 추론 → 결과 클라이언트에 송신.
    '''

    # PCM 변환 -> float32 형태로 바꿈 왜냐면 np에서 빠르게 계산하기 위해서 ?
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Root Mean Square로 오디오 신호의 크기를 평가 함 
    rms = float(np.sqrt(np.mean(pcm ** 2)))

    # silence 임계는 0.003 -> 무음이라고 봄 
    if rms < MIN_RMS:
        _update_latest(score=0.0, is_anomaly=False, rms=rms)
        print(f"[audio] silence rms={rms:.4f}", flush=True)
        # 조용하다. 라고 ws로 메세지 전송 함 
        await ws.send_json({"type": "silence", "rms": rms})
        return

    # TensorFlow로 추론하는데 시간이 50ms 정도 걸리니까 지연이 걸리는 문제 발생 ㅇㅇ 
    # max_sim : anomamly centroid와 코사인 유사도
    # is_anomaly : 이상음 판단 
    # n_frames : 분석한 내부 프레임의 수 
    max_sim, is_anomaly, n_frames = await asyncio.to_thread(detector.predict, pcm)

    # 크리티컬 섹션에 진입해서 저장하는 임무 
    _update_latest(score=max_sim, is_anomaly=is_anomaly, rms=rms)
    mark = "★ANOMALY" if is_anomaly else "  normal "
    print(f"[audio] {mark} max_sim={max_sim:.3f} rms={rms:.4f} frames={n_frames}", flush=True)
    await ws.send_json({
        "type": "detection",
        "is_anomaly": bool(is_anomaly),
        "max_sim": max_sim,
        "rms": rms,
        "frames": n_frames,
        "threshold": detector.threshold,
    })
