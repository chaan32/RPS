"""공통 시스템 프롬프트 및 유저 메시지 빌더.

llm.py(Gemini)와 llm_local.py(Ollama) 모두 이 파일을 사용하여
동일한 프롬프트로 리포트를 생성한다. (비교 공정성 보장)
"""

import re
import json
from collections import Counter, defaultdict


_FENCE_RE = re.compile(
    r"^```(?:html)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)


def strip_code_fence(text: str) -> str:
    """응답에 코드펜스(```html ... ```)가 감싸져 있으면 제거한다."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _build_report_facts(logs: list[dict]) -> dict:
    """Compress raw incident rows into facts that a local LLM can follow.

    Local Ollama models are much more stable when they receive pre-aggregated
    facts instead of hundreds of raw rows. The raw rows are still used by the
    Python service for the snapshot gallery, but the LLM only needs these facts
    to write a short narrative.
    """
    totals = Counter((log.get("incident_type") or "Unknown").title() for log in logs)
    worker_stats: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "warning": 0, "danger": 0, "last_time": ""}
    )
    hourly = Counter()

    for log in logs:
        worker_id = int(log.get("worker_id") or log.get("maker_id") or 0)
        incident_type = (log.get("incident_type") or "Unknown").title()
        created_at = log.get("created_at_utc") or ""
        hour = created_at[11:13] if len(created_at) >= 13 else "unknown"
        hourly[hour] += 1

        stat = worker_stats[worker_id]
        stat["total"] += 1
        if incident_type == "Warning":
            stat["warning"] += 1
        elif incident_type == "Danger":
            stat["danger"] += 1
        if created_at and created_at > stat["last_time"]:
            stat["last_time"] = created_at

    return {
        "total": len(logs),
        "warning": int(totals.get("Warning", 0)),
        "danger": int(totals.get("Danger", 0)),
        "workers": [
            {"worker_id": worker_id, **stat}
            for worker_id, stat in sorted(worker_stats.items())
        ],
        "top_hours": [
            {"hour": hour, "count": count}
            for hour, count in hourly.most_common(5)
        ],
        "recent_danger_samples": [
            {
                "worker_id": log.get("worker_id") or log.get("maker_id"),
                "incident_type": log.get("incident_type"),
                "created_at": log.get("created_at_utc"),
            }
            for log in logs
            if (log.get("incident_type") or "").lower() == "danger"
        ][-8:],
    }


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
   <h3>작업자별 위험 상황 발생 현황</h3>
   <table>
     <thead><tr><th>Worker ID</th><th>총 건수</th><th>Warning</th><th>Danger</th><th>마지막 발생 시각</th></tr></thead>
     <tbody>
       <tr><td>{{worker_id}}</td><td>{{total}}</td><td><span class="badge warn">{{warn}}</span></td><td><span class="badge danger">{{danger}}</span></td><td>{{time}}</td></tr>
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
- 전날 대비, 전주 대비, 증가율, 감소율, 퍼센트 변화처럼 입력 데이터에 없는 비교 분석 금지.
- 코드펜스, 주석 설명 금지.
"""


def build_user_message(date_str: str, logs: list[dict]) -> str:
    """Gemini/Ollama에 전달할 유저 메시지를 조립한다."""
    facts = _build_report_facts(logs)
    return (
        f"[리포트 대상 날짜] {date_str} (Asia/Seoul)\n"
        "[집계 데이터(JSON) — 아래 숫자만 신뢰하고 그대로 사용]\n"
        f"{json.dumps(facts, ensure_ascii=False)}\n"
        "[작성 지시]\n"
        "- 위 집계 데이터만 사용해서 안전 리포트를 작성하세요.\n"
        "- 비교 기준 데이터가 없으므로 전날 대비/증가율/감소율 표현을 쓰지 마세요.\n"
        "- 임의의 수학 문제, 일반 설명, 마크다운 문장을 절대 출력하지 마세요.\n"
        "- 반드시 <h2>로 시작하는 HTML fragment만 출력하세요."
    )
