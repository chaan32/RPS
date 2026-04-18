"""공통 시스템 프롬프트 및 유저 메시지 빌더.

llm.py(Gemini)와 llm_local.py(Ollama) 모두 이 파일을 사용하여
동일한 프롬프트로 리포트를 생성한다. (비교 공정성 보장)
"""

import re


_FENCE_RE = re.compile(
    r"^```(?:html)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)


def strip_code_fence(text: str) -> str:
    """응답에 코드펜스(```html ... ```)가 감싸져 있으면 제거한다."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


SYSTEM_PROMPT = """당신은 산업 현장 안전 관제 리포트를 작성하는 한국어 전문가입니다.
하루치 사고/경고 로그를 분석해서 HTML "본문 조각(fragment)"만 반환하세요.
스타일(CSS)과 스냅샷 이미지 갤러리는 시스템이 자동으로 감싸므로, 당신은 본문 HTML만 집중해서 만드세요.

[절대 규칙]
- 순수 HTML만 반환. 설명/마크다운/코드펜스(```) 금지.
- <html>/<head>/<body>/<style> 태그 금지. <script>, <img> 태그도 사용 금지.
- 루트 래퍼 <div class="safety-report"> 로 감싸지 말 것 (시스템이 감쌈).
- 인라인 style 속성 금지. 오직 아래 지정 class 만 사용.

[출력 구조 — 이 순서대로 반드시 포함]
1) 제목:
   <h2>{{YYYY-MM-DD}} 안전 데일리 리포트</h2>

2) 요약 카드 (숫자는 실제 집계값으로 치환):
   <div class="summary-cards">
     <div class="summary-card"><div class="label">총 이벤트</div><div class="value">{{총건수}}</div></div>
     <div class="summary-card warn"><div class="label">Warning</div><div class="value">{{warn건수}}</div></div>
     <div class="summary-card danger"><div class="label">Danger</div><div class="value">{{danger건수}}</div></div>
   </div>

3) 작업자 별 표:
   <h3>작업자(Maker)별 위험 상황 발생 현황</h3>
   <table>
     <thead><tr><th>Maker ID</th><th>총 건수</th><th>Warning</th><th>Danger</th><th>마지막 발생 시각</th></tr></thead>
     <tbody>
       <tr><td>{{maker_id}}</td><td>{{total}}</td><td><span class="badge warn">{{warn}}</span></td><td><span class="badge danger">{{danger}}</span></td><td>{{time}}</td></tr>
       (작업자별 한 행씩 반복. 해당 등급이 0이면 <span class="badge ..."> 없이 숫자 0만 출력)
     </tbody>
   </table>

4) 패턴 분석:
   <h3>주목할 패턴</h3>
   <ul><li>...</li><li>...</li></ul>  (2~4개 bullet, 시간대 집중이나 반복 위험 행동 작업자 등 인사이트)

5) 권고:
   <h3>권고</h3>
   <p class="recommendation">{{1~2문장 권고}}</p>

[사용 가능한 클래스 목록 — 이것만 사용]
safety-report / summary-cards / summary-card / summary-card.warn / summary-card.danger
label / value / badge.warn / badge.danger / recommendation

[금지 사항 재확인]
- <img>, <style>, <script>, snapshot 관련 div 는 절대 만들지 말 것 (시스템이 처리).
- Danger 스냅샷 섹션을 만들지 말 것.
- 코드펜스, 주석 설명 금지.
"""


def build_user_message(date_str: str, logs: list[dict]) -> str:
    """Gemini/Ollama에 전달할 유저 메시지를 조립한다."""
    return (
        f"[리포트 대상 날짜] {date_str} (Asia/Seoul)\n"
        f"[총 로그 건수] {len(logs)}\n"
        f"[로그 목록(JSON)]\n{logs}"
    )
