# E2E Scenario Baseline - 2026-05-19

검증일: 2026-05-19

목적: Unity 녹화 프레임을 RTSP로 흘려보낸 뒤, backend realtime pipeline이 worker/forklift 감지, BEV 좌표 변환, fusion 예측, 위험 알림 API, MQTT, DB 저장, snapshot 저장까지 정상 수행하는지 확인한다.

## 검증 흐름

```text
Unity scenario cam1/cam2 frames
-> FFmpeg RTSP publisher
-> MediaMTX rtsp://localhost:8554/cam1, /cam2
-> model.fusion.runtime.realtime_camera
-> YOLO pose worker detection
-> custom YOLO forklift/box detection
-> homography world coordinates
-> global tracking / fusion / early warning
-> FastAPI /send-alert
-> MQTT forklift/4/vibration
-> FastAPI /incident-logs/with-snapshot
-> PostgreSQL incident_logs
-> snapshots/YYYY-MM-DD/*.jpg
-> metrics/*.jsonl, metrics/*.csv
```

## 실행 구성

기존 1122 서버는 건드리지 않고, 검증용 FastAPI 서버를 1123에 별도로 실행했다.

```bash
DISABLE_FUSION_SUBPROCESS=1 \
LLM_BACKEND=local \
OLLAMA_HOST=http://127.0.0.1:11434 \
LOCAL_LLM_MODEL=qwen3:8b \
SERVER_METRICS_PATH=metrics/e2e_server_requests.jsonl \
SERVER_PORT=1123 \
conda run -n venv python -m uvicorn server.main:app --host 127.0.0.1 --port 1123
```

RTSP publisher:

```bash
conda run -n venv python input/media/tools/stream_collision_scenario_rtsp.py \
  --scenario scenario_01_user_current \
  --rtsp-base rtsp://localhost:8554
```

Realtime pipeline:

```bash
HEADLESS=1 \
WORKER_WORLD_BOUNDS=none \
POSE_MODEL_PATH=model/yolo/yolo11s-pose.pt \
BEST_MODEL_PATH=model/yolo/best_forklift_box_colab.pt \
CAMERA_RTSP_URL_1=rtsp://localhost:8554/cam1 \
CAMERA_RTSP_URL_2=rtsp://localhost:8554/cam2 \
FUSION_SERVER_URL=http://127.0.0.1:1123 \
MQTT_BROKER=127.0.0.1 \
LOCAL_SNAPSHOT_PATH=/Users/haechan/Desktop/pobiga/ai/ai_project/snapshots \
conda run -n venv python -m model.fusion.runtime.realtime_camera \
  --no-audio \
  --no-prompt \
  --duration 60 \
  --run-label e2e_scenario_01 \
  --metrics-path metrics/e2e_scenario_01.jsonl
```

## 산출물

- `metrics/e2e_scenario_01.jsonl`
- `metrics/e2e_scenario_01_summary.csv`
- `metrics/e2e_scenario_02.jsonl`
- `metrics/e2e_scenario_02_summary.csv`
- `metrics/e2e_scenario_03.jsonl`
- `metrics/e2e_scenario_03_summary.csv`
- `metrics/e2e_scenarios_combined_summary.csv`
- `metrics/e2e_server_requests.jsonl`
- `metrics/e2e_server_requests_summary.csv`

기존 2026-05-18 scenario 1 결과는 아래에 보관했다.

- `metrics/archive/e2e_scenario_01_20260518.jsonl`
- `metrics/archive/e2e_scenario_01_20260518_summary.csv`

## 통합 결과

| Scenario | Rows | Duration | Effective FPS | Cam OK | Worker frames | Forklift frames | Prediction frames | Loop mean | Loop p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e2e_scenario_01 | 192 | 59.843s | 3.208 | 100.00% | 100.00% | 100.00% | 97.92% | 312.478ms | 411.985ms |
| e2e_scenario_02 | 196 | 59.802s | 3.277 | 100.00% | 100.00% | 100.00% | 97.96% | 305.861ms | 316.737ms |
| e2e_scenario_03 | 188 | 59.703s | 3.149 | 100.00% | 99.47% | 100.00% | 97.87% | 318.619ms | 395.955ms |

주요 병목:

| Scenario | cam1 pose mean | cam2 pose mean | cam1 custom YOLO mean | cam2 custom YOLO mean | fusion forward mean |
|---|---:|---:|---:|---:|---:|
| e2e_scenario_01 | 99.876ms | 96.997ms | 37.879ms | 38.778ms | 2.121ms |
| e2e_scenario_02 | 96.870ms | 94.712ms | 38.799ms | 38.696ms | 1.862ms |
| e2e_scenario_03 | 99.213ms | 100.594ms | 40.250ms | 40.867ms | 2.074ms |

판단:

- 전체 FPS 병목은 fusion 모델이 아니라 카메라별 pose/custom YOLO inference에 있다.
- fusion forward는 평균 약 2ms 수준이라 현재 기준에서 주요 병목이 아니다.
- cam read, tracking, refinement, early warning, publish dispatch는 YOLO 대비 매우 작다.

## 서버/API/DB 확인

검증용 서버 요청 로그:

- `/send-alert`: 3회, HTTP 200
- `/incident-logs/with-snapshot`: 3회, HTTP 200
- API request duration mean: 13.55ms
- API request duration p95: 35.20ms
- API request duration max: 44.27ms

DB 저장 확인:

| id | maker_id | incident_type | status | created_at |
|---:|---:|---|---|---|
| 20 | 4 | Danger | success | 2026-05-19 16:05:17 |
| 21 | 4 | Danger | success | 2026-05-19 16:05:27 |
| 22 | 4 | Danger | success | 2026-05-19 16:05:37 |

Snapshot:

- `snapshots/2026-05-19/realtime_forklift_20260519_160517_279142.jpg`
- `snapshots/2026-05-19/realtime_forklift_20260519_160527_180455.jpg`
- `snapshots/2026-05-19/realtime_forklift_20260519_160537_362738.jpg`

## 확인된 정상 동작

- RTSP cam1/cam2 입력 수신 정상
- 두 카메라 모두 60초 동안 read 실패 없음
- worker와 forklift가 fusion 입력 직전까지 유지됨
- world coordinate 기반 prediction이 대부분의 프레임에서 생성됨
- danger 구간에서 `/send-alert` 호출됨
- 위험 로그와 snapshot이 DB에 저장됨

## 종료 안정성 후속 수정

Baseline 실행 당시에는 realtime process 종료 시 `Segmentation fault: 11`이 발생했다.
원인은 RTSP frame reader 백그라운드 스레드가 `cv2.VideoCapture.read()` 중일 수 있는데,
메인 스레드가 곧바로 `VideoCapture.release()`를 호출하는 구조였다. macOS + OpenCV/FFmpeg 조합에서는 이 종료 순서가 native crash로 이어질 수 있다.

수정:

- `input/media/camera.py`의 `VideoStream`이 reader thread 객체를 보관한다.
- `stop()`에서 `_stopped=True` 설정 후 reader thread를 `join()`으로 먼저 기다린다.
- thread가 멈춘 뒤 `VideoCapture.release()`를 호출한다.
- 최신 frame 참조를 비워 종료 시 native buffer 참조가 남지 않게 한다.

수정 후 검증:

| Run | Duration | Alert/DB 포함 | Exit code | 결과 |
|---|---:|---|---:|---|
| segfault_check | 5s | 아니오 | 0 | 정상 종료 |
| segfault_check_alert | 12s | 예, DB id 23/24 저장 | 0 | 정상 종료 |
| segfault_check_60 | 60s | 예, DB id 25~30 저장 | 0 | 정상 종료 |

## 남은 이슈

1. RTSP 종료 시 H.264 decoder warning
   - 60초 검증은 exit code 0으로 정상 종료했다.
   - 종료 직후 `Missing reference picture`, `decode_slice_header error` 경고가 출력될 수 있다.
   - 이는 RTSP publisher 종료/프레임 경계에서 발생하는 FFmpeg decoder warning이며, 현재 프로세스 crash는 아니다.

2. FPS는 아직 목표치보다 낮음
   - 현재 세 시나리오 기준 약 3.15~3.28 FPS.
   - 목표 6 FPS 이상을 위해 cam1/cam2 inference 병렬화, frame queue, pose/custom model 호출 분리 검토가 필요하다.

3. scenario 3 프레임 관리 확인 필요
   - `scenario_03_opposite_worker`는 현재 `customize_current_motion`과 frame hash가 동일하다.
   - 의도한 "아까처럼" 시나리오라면 문제는 아니지만, 완전히 독립된 세 번째 배치를 원한다면 별도 녹화/저장이 필요하다.

## 다음 작업

1. 성능 개선 1차 목표
   - 현재 3.2 FPS -> 6 FPS
   - 우선순위: cam1/cam2 inference 병렬화

2. RTSP 종료 경고 정리
   - 필요하면 publisher 종료 순서와 FFmpeg GOP/keyframe 설정을 조정한다.

3. 포트폴리오용 결과 정리
   - `metrics/e2e_scenarios_combined_summary.csv`를 기준선으로 사용
   - "fusion 병목이 아니라 detection 병목"이라는 결론을 수치로 설명
