# Performance Timing Baseline - 2026-05-19

## 목적

성능 개선 전후를 같은 기준으로 비교하기 위해 프레임 단위 시간을 `perf.*` 표준 키로 기록한다.
기존 처리 로직에 `print`를 흩뿌리지 않고, `server/utils/perf.py`의 계측 helper를 호출해서 JSONL에 구조화된 값을 남긴다.

## 적용 방식

- `server/utils/perf.py`
  - `add_duration_ms(...)`: 시작/종료 시각으로 표준 ms 필드 추가
  - `add_camera_timings(...)`: `DetectionPipeline`의 cam별 timing을 표준 `perf.*` 키로 매핑
  - `StageTimer`: 이후 신규 코드에서 사용할 수 있는 context-manager 기반 timer
- `model/fusion/runtime/realtime_camera.py`
  - 기존 raw metric은 유지
  - 같은 row에 `perf.*` 표준 metric만 추가
- `input/media/tools/test/summarize_pipeline_metrics.py`
  - 기본 요약 대상을 `perf.*` 표준 필드로 변경
  - CSV에 metric 설명 컬럼 추가

## 표준 측정 파일

- Raw JSONL: `metrics/perf_baseline_s01_standard_20260519.jsonl`
- Summary CSV: `metrics/perf_baseline_s01_standard_20260519_summary.csv`

## 현재 기준선

조건:

- Scenario: `scenario_01_user_current`
- RTSP: local MediaMTX `rtsp://localhost:8555/cam1`, `rtsp://localhost:8555/cam2`
- Runtime: Mac Only, `HEADLESS=1`, `--duration 35`
- Rows: 124
- Duration: 34.92s
- Effective FPS: 3.55

| Metric | Mean ms | P50 ms | P95 ms | 해석 |
|---|---:|---:|---:|---|
| `perf.loop.total_ms` | 282.93 | 269.94 | 340.48 | 한 루프 전체 |
| `perf.pipeline.extract.cam1_ms` | 143.44 | 135.64 | 170.32 | cam1 전체 검출 |
| `perf.pipeline.extract.cam2_ms` | 137.97 | 132.93 | 172.57 | cam2 전체 검출 |
| `perf.model.pose.cam1_ms` | 79.94 | 74.48 | 92.99 | cam1 worker pose |
| `perf.model.pose.cam2_ms` | 76.53 | 73.38 | 96.58 | cam2 worker pose |
| `perf.model.custom_yolo.cam1_ms` | 44.39 | 42.60 | 60.03 | cam1 forklift/box 모델 |
| `perf.model.custom_yolo.cam2_ms` | 44.30 | 41.76 | 58.08 | cam2 forklift/box 모델 |
| `perf.vision.aruco.cam1_ms` | 18.84 | 18.37 | 22.04 | cam1 ArUco |
| `perf.vision.aruco.cam2_ms` | 16.91 | 16.64 | 19.18 | cam2 ArUco |
| `perf.output.publish_dispatch_ms` | 0.13 | 0.11 | 0.24 | alert dispatch 준비 |

## 해석

현재 병목은 명확히 모델 추론이다.

- Pose 합계 평균: 약 156.47ms
- Custom YOLO 합계 평균: 약 88.69ms
- ArUco 합계 평균: 약 35.75ms
- Fusion/후처리/dispatch는 현재 기준에서 거의 무시 가능한 수준

따라서 1차 성능 개선은 DB나 fusion 모델이 아니라 다음 순서가 맞다.

1. cam1/cam2 추론 병렬화
2. pose/custom YOLO 호출 횟수 줄이기
3. ArUco 검출 주기 조절 또는 캘리브레이션 후 캐싱 가능한 부분 분리
4. 필요 시 pose 모델 export/CoreML/MPS 최적화

## 주의

이번 `scenario_01_user_current` 런에서는 worker가 감지되지 않아 `n_workers=0`, `n_predictions=0`이었다.
따라서 이 기준선은 "full risk decision 비용"이 아니라 "현재 영상 입력에서 카메라 2개를 대상으로 매 프레임 pose/custom YOLO/ArUco를 돌리는 비용"으로 해석해야 한다.
