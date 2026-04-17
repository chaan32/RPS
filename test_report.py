"""목업 데이터로 로컬 LLM 리포트 비교 테스트.

사용법:
    python test_report.py qwen3:8b
    python test_report.py gemma3:4b
    python test_report.py all        # 둘 다 실행
"""

import asyncio
import os
import sys
import time

# .env 로드
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from server.report.prompts import SYSTEM_PROMPT, build_user_message, strip_code_fence

MOCK_LOGS = [
    {"id": 1, "maker_id": "CRANE-01", "incident_type": "Warning",
     "snapshot_path": "https://placehold.co/320x240/orange/white?text=CRANE-01+Warning",
     "status": "resolved", "created_at_utc": "2026-04-17T02:30:00"},
    {"id": 2, "maker_id": "CRANE-01", "incident_type": "Danger",
     "snapshot_path": "https://placehold.co/320x240/red/white?text=CRANE-01+Danger",
     "status": "open", "created_at_utc": "2026-04-17T04:15:00"},
    {"id": 3, "maker_id": "CRANE-01", "incident_type": "Danger",
     "snapshot_path": "https://placehold.co/320x240/red/white?text=CRANE-01+Danger2",
     "status": "open", "created_at_utc": "2026-04-17T04:22:00"},
    {"id": 4, "maker_id": "FORKLIFT-04", "incident_type": "Warning",
     "snapshot_path": "https://placehold.co/320x240/orange/white?text=FORK-04+Warning",
     "status": "resolved", "created_at_utc": "2026-04-17T06:00:00"},
    {"id": 5, "maker_id": "FORKLIFT-04", "incident_type": "Danger",
     "snapshot_path": "https://placehold.co/320x240/red/white?text=FORK-04+Danger",
     "status": "open", "created_at_utc": "2026-04-17T06:05:00"},
    {"id": 6, "maker_id": "CONVEYOR-02", "incident_type": "Warning",
     "snapshot_path": "https://placehold.co/320x240/orange/white?text=CONV-02+Warning",
     "status": "resolved", "created_at_utc": "2026-04-17T09:10:00"},
    {"id": 7, "maker_id": "CONVEYOR-02", "incident_type": "Warning",
     "snapshot_path": "https://placehold.co/320x240/orange/white?text=CONV-02+Warning2",
     "status": "resolved", "created_at_utc": "2026-04-17T09:45:00"},
    {"id": 8, "maker_id": "CRANE-01", "incident_type": "Danger",
     "snapshot_path": "https://placehold.co/320x240/red/white?text=CRANE-01+Danger3",
     "status": "open", "created_at_utc": "2026-04-17T14:30:00"},
]

DATE_STR = "2026-04-17"

HTML_WRAPPER = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{model} 리포트 테스트</title>
<style>
  body {{ font-family: 'Malgun Gothic', sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 1rem; }}
  .snapshot-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem; }}
  .snapshot-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 0.5rem; text-align: center; }}
  .snapshot-card img {{ max-width: 100%; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<div class="meta">
  <strong>모델:</strong> {model} | <strong>소요 시간:</strong> {elapsed:.1f}초 | <strong>HTML 길이:</strong> {length}자
</div>
{content}
</body>
</html>"""


async def run_ollama(model: str) -> tuple[str, float]:
    import httpx

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    user_message = build_user_message(DATE_STR, MOCK_LOGS)
    if model.startswith("qwen3"):
        user_message += "\n/no_think"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    print(f"[{model}] Ollama 호출 중...")
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        resp = await client.post(f"{host}/api/chat", json=payload)
        resp.raise_for_status()
    elapsed = time.perf_counter() - start

    data = resp.json()
    html = strip_code_fence(data["message"]["content"])
    print(f"[{model}] 완료 - {elapsed:.1f}초, {len(html)}자")
    return html, elapsed


async def test_model(model: str):
    html, elapsed = await run_ollama(model)
    safe_name = model.replace(":", "_")
    filename = f"result_{safe_name}.html"

    full_html = HTML_WRAPPER.format(
        model=model, elapsed=elapsed, length=len(html), content=html
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[{model}] 저장 완료 -> {filename}")


async def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "all":
        for model in ["qwen3:8b", "gemma3:4b"]:
            try:
                await test_model(model)
            except Exception as e:
                print(f"[{model}] 실패: {e}")
    else:
        await test_model(arg)

    print("\n브라우저에서 result_*.html 파일을 열어 비교하세요.")


if __name__ == "__main__":
    asyncio.run(main())
