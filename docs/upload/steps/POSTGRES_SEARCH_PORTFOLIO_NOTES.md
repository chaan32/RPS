# PostgreSQL Date Index Search Portfolio Notes

## 한 줄 요약

대량 사고 로그 검색에서 Elasticsearch와 PostgreSQL을 비교한 뒤, 날짜처럼 구조화된 검색은 PostgreSQL 복합 인덱스로 처리하도록 설계하여 날짜 필터 검색 p50 응답 시간을 `0.097ms` 수준까지 개선했다.

## 포트폴리오 짧은 설명

사고 로그 조회 API에서 날짜, 위험 유형, 장비 ID 기준 검색 성능을 개선하기 위해 PostgreSQL B-tree 복합 인덱스를 설계했다. Elasticsearch는 스냅샷 경로와 로그 메시지 같은 문자열 검색에 적합하다고 판단했고, 날짜 필터처럼 명확한 구조화 데이터는 PostgreSQL 인덱스를 사용하는 방식으로 검색 책임을 분리했다.

## 문제 상황

대량의 사고 로그가 쌓이면 관리자 페이지에서 특정 날짜의 위험 로그를 조회해야 한다. 처음에는 Elasticsearch를 붙이면 모든 검색이 빨라질 것처럼 보였지만, 실제로는 검색 대상이 무엇인지에 따라 적합한 저장소가 달랐다.

- `q=2026-05-24`: `snapshot_path` 문자열 안에 포함된 날짜 텍스트 검색
- `target_date=2026-05-24`: `incident_logs.date` 컬럼 자체를 기준으로 하는 날짜 필터 검색

두 검색은 겉으로는 같은 날짜를 찾는 것처럼 보이지만, 내부적으로는 완전히 다른 문제다.

## 해결 방향

날짜 검색은 PostgreSQL의 `date` 컬럼에 인덱스를 적용하고, 문자열 검색은 Elasticsearch를 사용하도록 분리했다.

적용한 PostgreSQL 인덱스:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_created_id
ON incident_logs (date, created_at DESC, id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incident_logs_date_type_maker_created_id
ON incident_logs (date, incident_type, maker_id, created_at DESC, id DESC);
```

첫 번째 인덱스는 날짜별 최신 로그 목록 조회에 사용하고, 두 번째 인덱스는 실제 운영 화면에서 자주 쓰는 `date + incident_type + maker_id + 최신순` 조건에 맞췄다.

## 검증 데이터

실제 검색 상황을 더 가깝게 만들기 위해 mock 로그를 한 날짜에 몰아넣지 않고 여러 날짜에 섞어 넣었다.

- 전체 mock 데이터: `300,000건`
- 날짜 종류: `1000개`
- 날짜별 데이터: 약 `300건`
- 삽입 순서: 날짜별 정렬이 아니라 랜덤 셔플
- 전체 `incident_logs`: `300,391건`
- Elasticsearch 인덱스 문서 수: `300,391건`

## 성능 결과

`target_date=2026-05-24`, `incident_type=Danger`, `maker_id=4`, `limit=20` 기준으로 측정했다.

| 검색 방식 | 평균 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|
| PostgreSQL 날짜 인덱스 | `0.480ms` | `0.097ms` | `0.660ms` | `28.448ms` |
| Elasticsearch 날짜 필터 | `5.376ms` | `4.528ms` | `7.960ms` | `70.134ms` |

PostgreSQL 실행 계획에서도 `idx_incident_logs_date_type_maker_created_id` 인덱스를 실제로 사용하는 것을 확인했다.

## 기술 선택 근거

Elasticsearch는 모든 검색을 대체하는 도구가 아니라, 검색어 기반의 텍스트 검색에 강한 별도 검색 엔진이다. 날짜, 상태, 장비 ID처럼 명확한 컬럼 조건은 PostgreSQL 인덱스가 더 단순하고 빠르며 운영 비용도 낮다.

따라서 이 프로젝트에서는 다음처럼 역할을 나눴다.

| 검색 유형 | 선택 기술 | 이유 |
|---|---|---|
| 날짜 필터 | PostgreSQL B-tree Index | 구조화된 컬럼 검색이므로 가장 빠르고 단순함 |
| 날짜 + 위험 유형 + 장비 ID | PostgreSQL Composite Index | 운영 화면의 주요 필터 조합에 적합함 |
| 스냅샷 경로 문자열 검색 | Elasticsearch | `snapshot_path` 안의 부분 문자열 검색에 적합함 |
| 로그 메시지/키워드 검색 | Elasticsearch | 텍스트 검색, 부분 검색, 랭킹 확장에 적합함 |

## 문제 해결 경험으로 쓰기 좋은 문장

처음에는 대량 로그 검색을 해결하기 위해 Elasticsearch 도입을 검토했지만, 벤치마크 결과 날짜 필터 검색은 PostgreSQL 복합 인덱스가 더 빠르다는 것을 확인했다. 이후 검색 조건을 구조화 검색과 문자열 검색으로 분리하고, 날짜 기반 조회는 PostgreSQL 인덱스로, 경로/키워드 검색은 Elasticsearch로 처리하도록 설계했다. 이를 통해 날짜 필터 검색 p50 응답 시간을 `0.097ms`까지 낮추고, 검색 기능의 성능과 운영 복잡도 사이의 균형을 맞췄다.

## 포트폴리오 담당 업무 항목 예시

- 사고 로그 조회 API의 검색 조건을 분석하고 PostgreSQL/Elasticsearch 역할을 분리
- `date`, `incident_type`, `maker_id`, `created_at` 기반 PostgreSQL 복합 인덱스 설계
- 30만건 규모 mock 데이터를 1000개 날짜에 랜덤 분산 삽입해 검색 성능 검증
- PostgreSQL `EXPLAIN ANALYZE`로 인덱스 사용 여부 확인
- 날짜 필터 검색 p50 `0.097ms`, p95 `0.660ms` 달성
- 문자열 검색은 Elasticsearch, 구조화 날짜 검색은 PostgreSQL로 분리하여 검색 아키텍처 정리

## 면접에서 설명할 때

Elasticsearch를 붙였다고 해서 모든 검색을 Elasticsearch로 넘기는 것이 아니라, 검색 조건의 성격을 먼저 나눴다고 설명하면 좋다. 날짜, 상태, 장비 ID처럼 컬럼이 명확한 조건은 PostgreSQL 인덱스가 더 빠르고 안정적이었다. 반대로 스냅샷 경로 안의 날짜 문자열이나 로그 본문 키워드처럼 부분 문자열 검색이 필요한 경우에는 Elasticsearch가 더 적합했다. 그래서 이 프로젝트에서는 두 저장소를 경쟁 관계가 아니라 역할 분담 관계로 설계했다.
