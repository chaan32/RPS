# RPS Performance Benchmark Summary

이 문서는 포트폴리오와 GitHub README에 반영할 핵심 성능 지표를 모은 요약입니다. 원본 측정 파일은 `benchmark/file/metrics/`에 보관했습니다.

## 실시간 추론 성능

초기 구조는 cam1 pose, cam2 pose, custom YOLO, fusion 계산을 대부분 한 루프에서 순차 실행했습니다. 이후 카메라 단위 병렬 처리, 모델 단위 병렬 처리, custom YOLO 입력 크기 조정, pose skip/cache를 적용했습니다.

최종 설정:

```env
DETECTION_PARALLEL_MODE=model_parallel
POSE_IMGSZ=640
CUSTOM_IMGSZ=512
POSE_EVERY_N_FRAMES=2
```

| 단계 | 내용 | FPS | 직전 대비 FPS 향상 | 1 frame 처리시간 | 직전 대비 처리시간 감소 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 초기 serial 처리 | 3.145 FPS | - | 0.320s | - |
| 1 | 카메라별 병렬 처리 | 5.488 FPS | +74.5% | 0.183s | 43.0% 감소 |
| 2 | 모델별 병렬 처리 | 6.335 FPS | +15.4% | 0.158s | 13.5% 감소 |
| 3 | Custom YOLO 이미지 크기 `640 -> 512` | 7.364 FPS | +16.2% | 0.136s | 14.2% 감소 |
| 4 | Pose 2프레임 1회 추론 + cache 재사용 | 10.148 FPS | +37.8% | 0.099s | 27.0% 감소 |

최종적으로 초기 대비 FPS는 약 `222.7%` 향상되었고, frame 처리 시간은 약 `69.1%` 감소했습니다.

## 45회 반복 벤치마크

시나리오 1, 2, 3을 대상으로 serial / camera parallel / model parallel 방식을 각각 5회씩 실행했습니다.

| Mode | Runs | Avg FPS | FPS Range | Avg Loop | Worker Rate | Forklift Rate | Prediction Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 15 | 5.443 | 4.755-5.846 | 185.259ms | 1.000 | 0.980 | 0.978 |
| camera_parallel | 15 | 8.982 | 7.689-10.089 | 112.098ms | 1.000 | 0.985 | 0.987 |
| model_parallel | 15 | 10.148 | 9.035-11.065 | 98.966ms | 1.000 | 0.985 | 0.989 |

## 모델 성능

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Custom YOLO | 97.721% | 96.449% | 97.123% | 96.785% |
| YOLO-Pose worker detection | 98.333% | 100.000% | 99.167% | 99.582% |
| Fusion V1 overall | 92.188% | 93.499% | 82.287% | 84.927% |
| Fusion V2 combined danger | 99.696% | 99.119% | 99.301% | 99.210% |
| Fusion V2 forklift danger | 99.683% | 98.126% | 98.919% | 98.521% |
| Fusion V2 dropzone danger | 99.708% | 99.504% | 99.448% | 99.476% |

## PostgreSQL / Elasticsearch 검색 비교

대량 사고 로그 300,391건에서 날짜 기반 조회와 문자열 검색을 분리해 측정했습니다.

| 검색 유형 | 방식 | 평균 | p50 | p95 | 결론 |
| --- | --- | ---: | ---: | ---: | --- |
| 날짜 컬럼 필터 | PostgreSQL Index | 0.480ms | 0.097ms | 0.660ms | 구조화 날짜 검색에 가장 적합 |
| 날짜 컬럼 필터 | Elasticsearch Filter | 5.376ms | 4.528ms | 7.960ms | 날짜만 찾기에는 오버헤드가 큼 |
| snapshot_path 문자열 검색 | PostgreSQL ILIKE | 99.146ms | 97.764ms | 105.804ms | 전체 문자열 scan 비용 큼 |
| snapshot_path 문자열 검색 | Elasticsearch | 13.200ms | 8.577ms | 23.024ms | 부분 문자열 검색에 유리 |

정리하면 날짜/작업자/위험유형처럼 구조화된 조회는 PostgreSQL 복합 인덱스가 적합하고, snapshot path나 키워드 기반 자유 검색에는 Elasticsearch read model이 유리합니다.

## Redis Background Job

Ollama 기반 리포트 생성은 오래 걸리는 작업이므로 동기 API 응답 경로에서 분리했습니다.

| 항목 | 개선 전 | 개선 후 | 개선 효과 |
| --- | ---: | ---: | --- |
| 리포트 생성 API 응답 | Ollama 생성 완료까지 약 71초 대기 | Redis job 등록 후 즉시 `job_id` 반환 | 긴 작업을 API 응답 경로에서 분리 |
| 리포트 목록 응답 크기 | 97,761 bytes | 729 bytes | 약 99.25% 감소 |

## 원본 근거 파일

| 파일 | 설명 |
| --- | --- |
| `benchmark/file/metrics/REPORT.md` | pose skip/cache 최종 45회 벤치마크 리포트 |
| `benchmark/file/metrics/aggregate_by_mode.csv` | serial/camera_parallel/model_parallel 평균 |
| `benchmark/file/metrics/aggregate_by_mode_scenario.csv` | 시나리오별 평균 |
| `benchmark/file/metrics/incident_search_pg_date_index_vs_es_date_filter_20260524.json` | 날짜 컬럼 인덱스 검색 비교 |
| `benchmark/file/metrics/incident_search_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json` | 문자열 검색 direct 비교 |
| `benchmark/file/metrics/incident_search_api_pg_vs_es_shuffled_300k_text_date_no_date_filter_20260524.json` | 문자열 검색 API 비교 |
| `benchmark/file/metrics/backend_service_latency_after_summary_20260526.json` | 백엔드 API 부하 측정 |

