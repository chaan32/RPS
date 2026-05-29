"""
Fusion 알림 → 서버 /incident-logs 비동기 저장.

realtime_camera는 server/main.py와 다른 프로세스로 띄워지므로 (subprocess),
서버의 SQLAlchemy 세션을 공유할 수 없다.
대신 같은 머신의 FastAPI 서버에 HTTP POST 로 incident_logs 행을 등록한다.

사용:
  from db_logger import log_pair_sync
  log_pair_sync(pair_risk)               # 동기 (cv2 루프에서 직접 호출)
  await log_pair(pair_risk)              # async (서버 안에서 사용)
"""

from __future__ import annotations

import os
import asyncio
from datetime import datetime

import httpx

from ..risk_output import PairRisk, RiskLevel
from .worker_ids import worker_label_to_int


# ── 설정 ────────────────────────────────────────────────
DEFAULT_SERVER_URL = os.getenv("FUSION_SERVER_URL", "http://127.0.0.1:1122")

def _incident_type_for(level: RiskLevel) -> str:
    """RiskLevel → IncidentLog.incident_type ('Warning' or 'Danger')."""
    return "Danger" if level is RiskLevel.DANGER else "Warning"


def _placeholder_snapshot_path(pair: PairRisk) -> str:
    """스냅샷 미업로드 시 사용하는 placeholder 경로.

    schema 가 NOT NULL 이라 빈 문자열은 안 됨. 추후 실제 캡처 업로드 가능.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"realtime/{pair.threat_type.value}/{ts}.jpg"


# ── 서버 API 호출 ──────────────────────────────────────
async def log_pair(
    pair: PairRisk,
    server_url: str = DEFAULT_SERVER_URL,
    snapshot_path: str | None = None,
    timeout: float = 5.0,
) -> dict:
    """
    단일 PairRisk → POST /incident-logs.

    Returns:
      서버 응답 dict, 또는 {"status": "fail", "error": ...}
    """
    body = {
        "worker_id": worker_label_to_int(pair.worker_id),
        "incident_type": _incident_type_for(pair.level),
        "snapshot_path": snapshot_path or _placeholder_snapshot_path(pair),
        "status": "success",
    }
    url = f"{server_url}/incident-logs"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return {"status": "ok", "id": resp.json().get("id"), "body": body}
    except httpx.HTTPStatusError as e:
        return {
            "status": "fail",
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "body": body,
        }
    except Exception as e:
        return {"status": "fail", "error": str(e), "body": body}


def log_pair_sync(pair: PairRisk, **kwargs) -> dict:
    """동기 컨텍스트(cv2 루프)에서 호출 가능."""
    return asyncio.run(log_pair(pair, **kwargs))


# ── 스냅샷과 함께 저장 ──────────────────────────────────
async def log_pair_with_snapshot(
    pair: PairRisk,
    frame_jpeg: bytes,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 10.0,
) -> dict:
    """
    PairRisk + 카메라 프레임 → POST /incident-logs/with-snapshot.

    이 엔드포인트는 multipart upload 받아 USB 저장 + DB 행 추가까지 한 번에 처리.

    Args:
      frame_jpeg : cv2.imencode('.jpg', frame)[1].tobytes() 결과 바이트
    """
    worker_id = worker_label_to_int(pair.worker_id)
    incident_type = _incident_type_for(pair.level)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"realtime_{pair.threat_type.value}_{ts}.jpg"

    url = f"{server_url}/incident-logs/with-snapshot"
    params = {"worker_id": worker_id, "incident_type": incident_type}
    files = {"file": (filename, frame_jpeg, "image/jpeg")}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, params=params, files=files)
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "ok",
                "id": data.get("id"),
                "snapshot_path": data.get("snapshot_path"),
            }
    except httpx.HTTPStatusError as e:
        return {
            "status": "fail",
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
        }
    except Exception as e:
        return {"status": "fail", "error": str(e)}


def log_pair_with_snapshot_sync(pair: PairRisk, frame_jpeg: bytes, **kwargs) -> dict:
    return asyncio.run(log_pair_with_snapshot(pair, frame_jpeg, **kwargs))


# ── Sanity check ───────────────────────────────────────
def _sanity_check():
    import numpy as np
    from risk_output import FusionPrediction

    fake_matrix = np.array([[0.92, 0.85]], dtype=np.float32)
    pred = FusionPrediction.from_model_output(fake_matrix)

    print(f"server URL: {DEFAULT_SERVER_URL}\n")
    for p in pred.triggered():
        print(f"[POST /incident-logs] {p.threat_type.value} prob={p.prob:.2f}")
        result = log_pair_sync(p)
        print(f"  result: {result}\n")


if __name__ == "__main__":
    _sanity_check()
