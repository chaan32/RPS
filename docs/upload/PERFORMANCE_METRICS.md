# RPS Performance Metrics

이 문서는 GitHub와 포트폴리오에 넣을 성능 지표만 모은 업로드용 문서입니다.

## 1. Model Metrics

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Custom YOLO | 97.721% | 96.449% | 97.123% | 96.785% |
| YOLO-Pose worker detection | 98.333% | 100.000% | 99.167% | 99.582% |
| Fusion V1 overall | 92.188% | 93.499% | 82.287% | 84.927% |
| Fusion V2 combined danger | 99.696% | 99.119% | 99.301% | 99.210% |
| Fusion V2 forklift danger | 99.683% | 98.126% | 98.919% | 98.521% |
| Fusion V2 dropzone danger | 99.708% | 99.504% | 99.448% | 99.476% |

## 2. Inference FPS

| 단계 | 내용 | FPS | 직전 대비 FPS 향상 | 1 frame 처리시간 | 직전 대비 처리시간 감소 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 초기 serial 처리 | 3.145 FPS | - | 0.320s | - |
| 1 | 카메라별 병렬 처리 | 5.488 FPS | +74.5% | 0.183s | 43.0% 감소 |
| 2 | 모델별 병렬 처리 | 6.335 FPS | +15.4% | 0.158s | 13.5% 감소 |
| 3 | Custom YOLO 이미지 크기 `640 -> 512` | 7.364 FPS | +16.2% | 0.136s | 14.2% 감소 |
| 4 | Pose 2프레임 1회 추론 + cache 재사용 | 10.148 FPS | +37.8% | 0.099s | 27.0% 감소 |

초기 대비 최종 성능:

- FPS: `3.145 -> 10.148`, 약 `222.7%` 향상
- Frame 처리시간: `0.320s -> 0.099s`, 약 `69.1%` 감소

## 3. 45-run Benchmark

| Mode | Runs | Avg FPS | FPS Range | Avg Loop | Worker Rate | Forklift Rate | Prediction Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 15 | 5.443 | 4.755-5.846 | 185.259ms | 1.000 | 0.980 | 0.978 |
| camera_parallel | 15 | 8.982 | 7.689-10.089 | 112.098ms | 1.000 | 0.985 | 0.987 |
| model_parallel | 15 | 10.148 | 9.035-11.065 | 98.966ms | 1.000 | 0.985 | 0.989 |

## 4. Search Benchmark

| 검색 유형 | 방식 | 평균 | p50 | p95 | 결론 |
| --- | --- | ---: | ---: | ---: | --- |
| 날짜 컬럼 필터 | PostgreSQL Index | 0.480ms | 0.097ms | 0.660ms | 구조화 날짜 검색에 가장 적합 |
| 날짜 컬럼 필터 | Elasticsearch Filter | 5.376ms | 4.528ms | 7.960ms | 날짜만 찾기에는 오버헤드가 큼 |
| snapshot_path 문자열 검색 | PostgreSQL ILIKE | 99.146ms | 97.764ms | 105.804ms | 전체 문자열 scan 비용 큼 |
| snapshot_path 문자열 검색 | Elasticsearch | 13.200ms | 8.577ms | 23.024ms | 부분 문자열 검색에 유리 |

## 5. Redis Background Job

| 항목 | 개선 전 | 개선 후 | 개선 효과 |
| --- | ---: | ---: | --- |
| 리포트 생성 API 응답 | Ollama 생성 완료까지 약 71초 대기 | Redis job 등록 후 즉시 `job_id` 반환 | 긴 작업을 API 응답 경로에서 분리 |
| 리포트 목록 응답 크기 | 97,761 bytes | 729 bytes | 약 99.25% 감소 |

## 6. Backend API Load Test

요약 API와 검색 API 개선 이후, 15초 동안 endpoint별 동시성 4로 부하를 걸어 측정했습니다.

| Endpoint | Requests | Mean | p50 | p95 | p99 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/workers` | 1,720 | 34.780ms | 24.848ms | 75.674ms | 154.314ms | 0 |
| `/incident-logs/search/postgres` | 1,481 | 40.381ms | 30.273ms | 79.652ms | 175.950ms | 0 |
| `/reports` | 1,780 | 33.597ms | 24.044ms | 72.936ms | 157.536ms | 0 |
| `/reports/summary` | 1,641 | 36.436ms | 25.889ms | 79.093ms | 157.010ms | 0 |

## 7. 근거 파일

원본 측정 파일은 `benchmark/file/metrics/`에 보관했습니다. 큰 학습 데이터셋은 `benchmark/file/fusion_v2/`에 별도 보관했습니다.

