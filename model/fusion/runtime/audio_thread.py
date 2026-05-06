"""ESP32 audio score 폴링 스레드.

서버(/audio/score) ← esp32_ws.py 가 ESP32 → YAMnet 결과를 보관 중.
realtime_camera 는 별도 subprocess 라 메모리 공유가 안 되므로 HTTP 로 가져온다.

사용:
    import threading
    from .audio_thread import audio_worker, get_score, stop

    threading.Thread(target=audio_worker, daemon=True).start()
    score = get_score()
    ...
    stop()    # 종료 시
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np


AUDIO_SCORE_URL = os.getenv(
    "AUDIO_SCORE_URL",
    f"http://127.0.0.1:{os.getenv('SERVER_PORT', '1122')}/audio/score",
)
AUDIO_POLL_SEC = 0.5            # 0.5s 마다 폴링 (yamnet 1.92s 윈도우 대비 충분)
AUDIO_STALE_SEC = 5.0           # 마지막 갱신 후 이만큼 지나면 score=0


# ── 공유 상태 ────────────────────────────────────────────
_state: dict = {"score": 0.05, "ts": 0.0}
_lock = threading.Lock()
_running = True


def get_score() -> float:
    """현재 audio score 를 thread-safe 하게 반환."""
    with _lock:
        return _state["score"]


def stop() -> None:
    """audio_worker 루프 종료 (메인이 종료할 때 호출)."""
    global _running
    _running = False


def audio_worker(verbose: bool = False) -> None:
    """서버 /audio/score 폴링 → ESP32 yamnet score 를 _state 에 반영."""
    import urllib.request
    import urllib.error

    print(f"[audio] ESP32 score 폴링 시작: {AUDIO_SCORE_URL}")
    fail_count = 0
    while _running:
        try:
            with urllib.request.urlopen(AUDIO_SCORE_URL, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            score = float(data.get("score", 0.0))
            ts = float(data.get("ts", 0.0))
            # 너무 오래된 값이면 무시 (ESP32 끊김)
            if ts > 0 and (time.time() - ts) > AUDIO_STALE_SEC:
                score = 0.0
            with _lock:
                _state["score"] = float(np.clip(score, 0.0, 1.0))
                _state["ts"] = ts
            fail_count = 0
            if verbose:
                print(f"[audio] score={score:.3f} ts_age={time.time()-ts:.1f}s")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            fail_count += 1
            if fail_count == 1 or fail_count % 20 == 0:
                print(f"[audio] /audio/score 연결 실패 ({fail_count}회): {e}")
        except Exception as e:
            print(f"[audio] 폴링 오류: {e}")

        time.sleep(AUDIO_POLL_SEC)
