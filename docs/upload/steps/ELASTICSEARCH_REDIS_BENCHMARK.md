# Elasticsearch / Redis Background Job Benchmark

## 목적

대량 사고 로그가 쌓였을 때 현재 PostgreSQL 기반 조회가 어느 정도 느려지는지 확인하고, Elasticsearch read model을 붙였을 때 실제로 의미 있는 개선이 있는지 검증했다. 추가로 LLM 리포트 생성처럼 오래 걸리는 API 작업을 Redis background job으로 분리해, 프론트 요청이 긴 작업 완료를 기다리지 않도록 개선했다.

## 최신 재측정 데이터

- 대상 테이블: `incident_logs`
- 전체 행 수: `100,391`
- 원본 저장 형식 mock 행 수: `100,000`
- 기존 benchmark prefix 행 수: `0`
- mock snapshot path 형식: `/Users/haechan/Desktop/pobiga/ai/ai_project/snapshots/YYYY-MM-DD/realtime_forklift_YYYYMMDD_HHMMSS_micro.jpg`
- 기준 날짜: `2026-05-24`
- 검색 조건:
  - `date = 2026-05-24`
  - `incident_type = Danger`
  - `maker_id = 4`
  - `limit = 20`

## 선택도 개선 재측정: 1000개 날짜에 분산된 300,000건

한 날짜에 100,000건이 몰리면 `q=2026-05-24`가 너무 많은 문서에 매칭되어 검색 엔진의 장점을 보기 어렵다. 그래서 mock 데이터를 다시 구성했다.

- 기존 mock 삭제: `id > 391` 기준으로 이전 mock `100,000건` 제거
- 새 mock 생성: `1000개 날짜 x 날짜당 300건 = 300,000건`
- insert 방식: 날짜 순서가 아니라 전체 record를 shuffle한 뒤 bulk insert
- 전체 `incident_logs`: `300,391건`
- 날짜 종류: `1000개`
- `q=2026-05-24`가 snapshot path에 포함된 행: `301건`
- `Danger + maker_id=4 + q=2026-05-24` 매칭 행: `21건`
- Elasticsearch 재색인:
  - 문서 수: `300,391`
  - 소요 시간: `12.138s`
  - 처리량: `24,748.8 docs/sec`

이 테스트에서는 `target_date` 필터를 일부러 빼고, 전체 300,391건 중에서 `snapshot_path` 문자열에 `2026-05-24`가 들어간 로그를 찾았다.

파일:

- `metrics/incident_search_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json`
- `metrics/incident_search_api_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json`

### Direct Backend Search

| Backend | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| PostgreSQL `ILIKE` | 99.146ms | 97.764ms | 105.804ms | 132.647ms | 전체 문자열 scan 비용이 커짐 |
| Elasticsearch `wildcard` | 13.200ms | 8.577ms | 23.024ms | 268.776ms | p50 기준 약 11.4배 빠름 |

### FastAPI API Search

| API | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| `/incident-logs/search/postgres?q=2026-05-24` | 53.152ms | 50.789ms | 61.816ms | 126.458ms | API 레벨에서도 문자열 검색 비용이 보임 |
| `/incident-logs/search/elasticsearch?q=2026-05-24` | 19.547ms | 17.666ms | 23.137ms | 69.149ms | p50 기준 약 2.9배 빠름 |

결론: 날짜가 path에 문자열로 들어간 로그 검색처럼, 전체 로그 중 일부 날짜의 문자열을 검색창에서 찾는 상황에서는 Elasticsearch read model이 PostgreSQL `ILIKE`보다 명확히 유리하다.

## PostgreSQL 기준 측정

파일:

- `metrics/incident_search_pg_vs_es_original_mock_100k_filter_20260524.json`
- `metrics/incident_search_pg_vs_es_original_mock_100k_text_date_20260524.json`

| 항목 | 평균 | p50 | p95 | 최대 | 해석 |
|---|---:|---:|---:|---:|---|
| PostgreSQL 구조화 검색 | 8.779ms | 8.089ms | 10.755ms | 38.506ms | 100k 행에서도 단순 필터 검색은 충분히 빠름 |
| PostgreSQL 문자열 검색 `2026-05-24` | 36.863ms | 34.677ms | 49.323ms | 64.799ms | `ILIKE` 문자열 검색에서는 비용이 커짐 |

## Elasticsearch 색인

실행:

```bash
python -m server.scripts.index_incident_logs_to_elasticsearch --reset --batch-size 2000
```

결과:

- 색인 문서 수: `100,391`
- 소요 시간: `6.407s`
- 처리량: `15,669.1 docs/sec`
- 인덱스명: `incident_logs_v1`

## PostgreSQL vs Elasticsearch 검색 비교

### 1. 단순 구조화 검색

파일: `metrics/incident_search_pg_vs_es_original_mock_100k_filter_20260524.json`

| Backend | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| PostgreSQL | 8.779ms | 8.089ms | 10.755ms | 38.506ms | 충분히 빠름 |
| Elasticsearch | 7.181ms | 6.171ms | 10.259ms | 76.086ms | 비슷하거나 약간 빠르지만 차이는 작음 |

결론: `date`, `incident_type`, `maker_id` 같은 구조화 필터만 쓴다면 Elasticsearch를 붙이는 실익은 작다. 이 경우 PostgreSQL 인덱스와 페이지네이션이 우선이다.

### 2. 문자열 검색 `q=2026-05-24`

파일: `metrics/incident_search_pg_vs_es_original_mock_100k_text_date_20260524.json`

| Backend | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| PostgreSQL `ILIKE` | 36.863ms | 34.677ms | 49.323ms | 64.799ms | 문자열 조건에서 느려짐 |
| Elasticsearch | 6.770ms | 6.410ms | 9.745ms | 14.640ms | p50 기준 약 5.4배 빠름 |

결론: 사고 로그 검색 화면에서 사용자가 날짜 문자열을 검색창에 입력하고, 이 값이 `snapshot_path` 같은 문자열 필드 안에 들어 있는 데이터를 찾아야 한다면 Elasticsearch가 의미 있다.

### 3. 실제 FastAPI 검색 API 비교

파일: `metrics/incident_search_api_pg_vs_es_original_mock_100k_text_date_20260524.json`

| API | 평균 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|
| `/incident-logs/search/postgres` | 24.499ms | 22.609ms | 28.165ms | 72.228ms |
| `/incident-logs/search/elasticsearch` | 18.291ms | 16.702ms | 23.304ms | 48.833ms |

API 레벨에서는 직렬화, 네트워크, FastAPI middleware 비용이 섞여 격차가 줄어든다. 그래도 문자열 검색에서는 Elasticsearch endpoint가 평균과 p50에서 더 낮게 나왔다.

## 선택도 개선 재측정: 날짜 1000개에 분산된 300,000건

이전 벤치마크는 특정 날짜에 데이터가 몰려 있어 날짜 문자열 검색의 선택도가 낮았다. 실제 로그 검색에 더 가까운 조건을 만들기 위해 mock 데이터를 다음과 같이 다시 구성했다.

- 전체 mock 데이터: `300,000건`
- 날짜 종류: `1000개`
- 날짜별 데이터: `300건`
- 삽입 순서: 날짜별 정렬이 아니라 랜덤 셔플
- 전체 `incident_logs`: `300,391건`
- Elasticsearch 문서 수: `300,391건`

### 문자열 검색 `q=2026-05-24`, 날짜 필터 없음

파일: `metrics/incident_search_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json`

| Backend | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| PostgreSQL `ILIKE` | 99.146ms | 97.764ms | 105.804ms | 132.647ms | 전체 문자열 스캔 비용이 큼 |
| Elasticsearch | 13.200ms | 8.577ms | 23.024ms | 268.776ms | p50 기준 약 11.4배 빠름 |

API 레벨 측정 파일: `metrics/incident_search_api_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json`

| API | 평균 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|
| `/incident-logs/search/postgres` | 53.152ms | 50.789ms | 61.816ms | 126.458ms |
| `/incident-logs/search/elasticsearch` | 19.547ms | 17.666ms | 23.137ms | 69.149ms |

결론: 날짜가 문자열로 `snapshot_path` 안에 들어 있고, 사용자가 검색창에 `2026-05-24`처럼 입력하는 경우에는 Elasticsearch가 유리하다.

### 날짜 컬럼 인덱스 검색 `target_date=2026-05-24`

파일: `metrics/incident_search_pg_date_index_vs_es_date_filter_20260524.json`

추가한 PostgreSQL 인덱스:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_created_id
ON incident_logs (date, created_at DESC, id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_type_maker_created_id
ON incident_logs (date, incident_type, maker_id, created_at DESC, id DESC);
```

| Backend | 평균 | p50 | p95 | 최대 | 결론 |
|---|---:|---:|---:|---:|---|
| PostgreSQL 날짜 인덱스 | 0.480ms | 0.097ms | 0.660ms | 28.448ms | 날짜 필터는 PostgreSQL이 압도적으로 빠름 |
| Elasticsearch 날짜 필터 | 5.376ms | 4.528ms | 7.960ms | 70.134ms | 날짜만 찾기에는 오버헤드가 큼 |

실행 계획에서도 `idx_incident_logs_date_type_maker_created_id` 인덱스를 실제로 사용했다.

결론: 날짜 컬럼으로 정확히 필터링하는 기능은 Elasticsearch가 아니라 PostgreSQL B-tree 인덱스가 정답이다. Elasticsearch의 의미는 날짜 필터 자체가 아니라, 날짜 필터와 함께 자유 검색어/부분 문자열/키워드 검색이 섞일 때 커진다.

## 추가된 API

```http
GET /incident-logs/search/postgres
GET /incident-logs/search/elasticsearch
```

공통 query parameter:

- `target_date`: `YYYY-MM-DD`
- `incident_type`: `Warning` 또는 `Danger`
- `maker_id`: 정수
- `q`: 문자열 검색어
- `limit`: 기본 `20`, 최대 `500`
- `offset`: 기본 `0`

## Redis Background Job

기존 동기 리포트 생성:

```http
POST /reports/generate?target_date=2026-05-15
```

문제:

- API 요청이 LLM 생성 완료까지 계속 대기한다.
- Ollama/LLM 호출이 1분 이상 걸릴 수 있다.
- 프론트에서는 요청 timeout, 화면 멈춤, 중복 클릭 문제가 생기기 쉽다.

개선된 비동기 요청:

```http
POST /reports/generate-async?target_date=2026-05-15
GET /jobs/{job_id}
```

검증 결과:

- `/reports/generate-async` 응답 시간: `21.139ms`
- 반환 job id: `f2eed22d5e7a428999d6056b7b2e41cb`
- 최종 상태: `done`
- 생성 report id: `5`
- 실제 LLM 생성은 Redis worker가 background에서 처리

## 왜 Celery가 아니라 Redis Queue인가

Celery는 production-grade 분산 작업 큐로 좋지만, 현재 단계에서는 과하다.

Celery를 바로 붙이면 필요한 것:

- Celery worker 별도 프로세스
- broker 설정
- result backend 설정
- task discovery 구조
- retry, ack, serialization 정책
- 별도 worker process/service 분리
- 운영 로그/모니터링 추가

현재 프로젝트에서 당장 필요한 것:

- 오래 걸리는 리포트 생성을 API 응답 경로에서 떼어내기
- job 상태를 조회하기
- Redis를 이용해 background job 구조를 보여주기
- Mac Only 성능 비교와 포트폴리오 설명 가능성을 확보하기

따라서 이번 단계에서는 `Redis list + Redis hash + FastAPI worker loop`가 더 적절하다. 추후 작업량이 많아지고 worker를 여러 대로 늘려야 할 때 Celery/RQ/Arq 같은 전용 worker framework로 확장하는 편이 좋다.

## 결론

- 대량 데이터에서 `/incident-logs` 전체 반환은 반드시 개선해야 한다. 전체 반환 대신 검색 API + 페이지네이션이 필요하다.
- 단순 구조화 검색은 PostgreSQL이 충분히 빠르다. Elasticsearch가 항상 답은 아니다.
- 문자열/키워드 검색이 들어가는 로그 검색 화면에서는 Elasticsearch가 확실히 의미 있다.
- LLM 리포트 생성은 동기 API로 두면 안 된다. Redis background job으로 분리하는 방향이 맞다.
- 이 구조는 “AI 추론은 무거운 작업으로 분리하고, 백엔드는 검색/큐/상태조회로 안정성을 확보했다”는 백엔드 포트폴리오 포인트가 된다.
