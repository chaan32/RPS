"""Ollama 기반 로컬 LLM 모듈.

환경변수 LOCAL_LLM_MODEL 로 모델을 선택한다 (기본값: qwen3:8b).
Qwen3 모델은 thinking mode를 끄기 위해 유저 메시지 끝에 /no_think 를 추가한다.
"""

import logging
import os
import time

import httpx

from .prompts import SYSTEM_PROMPT, build_user_message, strip_code_fence

logger = logging.getLogger(__name__)

async def summarize_logs_to_html(date_str: str, logs: list[dict]) -> str:
    """Ollama /api/chat 엔드포인트를 호출하여 HTML 리포트를 생성한다."""
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("LOCAL_LLM_MODEL", "qwen3:8b")
    url = f"{ollama_host}/api/chat"
    print(f"요청을 해봄 :{url}")
    user_message = build_user_message(date_str, logs)

    # Qwen3 계열은 thinking mode 비활성화
    if model.startswith("qwen3"):
        user_message += "\n/no_think"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
        },
    }

    logger.info("Ollama 호출 시작 — 모델: %s, 로그 건수: %d", model, len(logs))
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(800.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Ollama 서버에 연결할 수 없습니다 ({ollama_host}). "
            "ollama serve 가 실행 중인지 확인하세요."
        )
    except httpx.TimeoutException:
        raise RuntimeError( 
            f"Local LLM 응답 타임아웃 (300초 초과). 모델: {model}"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Local LLM 응답 HTTP 에러 {e.response.status_code}: {e.response.text}"
        )

    elapsed = time.perf_counter() - start
    logger.info("Ollama 응답 완료 — 모델: %s, 소요 시간: %.1f초", model, elapsed)

    data = resp.json()
    content = data["message"]["content"]
    return strip_code_fence(content)
