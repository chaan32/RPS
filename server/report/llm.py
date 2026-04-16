import os
import re
import google.generativeai as genai


_FENCE_RE = re.compile(r"^```(?:html)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


SYSTEM_PROMPT = """당신은 산업 현장 안전 관제 리포트를 작성하는 전문가입니다.
주어진 하루치 사고/경고 로그를 분석하여, 다음 조건을 만족하는 HTML 요약 리포트를 작성하세요.

요구사항:
- 순수 HTML string만 반환 (설명, 마크다운, 코드펜스 금지).
- <html>, <head>, <body> 래퍼 없이 <div> 루트 하나로 시작.
- 상단에 날짜 제목, 총 이벤트 수, Warning/Danger 건수 요약.
- maker(설비)별 발생 현황을 표(<table>)로 정리.
- 주목할 패턴(특정 시간대 집중, 특정 maker 반복 등)을 <ul>로 bullet 요약.
- 마지막에 간단한 권고(<p>) 1~2문장.
- 인라인 스타일 최소 사용 가능 (table border 정도).

위험 이미지 첨부 규칙:
- 각 로그에는 snapshot_path(이미지 URL)가 포함되어 있음.
- Danger 등급 로그의 이미지를 리포트 하단에 "위험 상황 스냅샷" 갤러리 섹션(<h3>)으로 추가.
- 반드시 아래 HTML 구조를 사용할 것:
  <div class="snapshot-grid">
    <div class="snapshot-card">
      <img src="snapshot_path" alt="Maker X - 시각 - Danger">
      <p>Maker ID: X, 시각: YYYY-MM-DD HH:MM:SS, 유형: Danger</p>
    </div>
    <!-- Danger 로그마다 snapshot-card 반복 -->
  </div>
- img 태그에 인라인 스타일을 넣지 말 것. class="snapshot-grid"과 class="snapshot-card"만 사용.
- Warning 등급 이미지는 포함하지 않음.
"""


def summarize_logs_to_html(date_str: str, logs: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    user_content = (
        f"[리포트 대상 날짜] {date_str} (Asia/Seoul)\n"
        f"[총 로그 건수] {len(logs)}\n"
        f"[로그 목록(JSON)]\n{logs}"
    )

    response = model.generate_content(user_content)
    return _strip_code_fence(response.text)
