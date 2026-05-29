# Mac Only Benchmark Runbook

목적: Mac 로컬 프로세스만으로 RTSP 입력부터 YOLO/Fusion/DB/MQTT/리포트까지 실행하고, 프레임별 모듈 소요시간을 `metrics/*.jsonl`에 남긴다.

## 0. 전제

- `conda activate venv`가 가능해야 한다.
- Mac에 PostgreSQL, Mosquitto, MediaMTX, FFmpeg, Ollama가 직접 설치되어 있어야 한다.
- `calibration/cam1_homography.json`, `calibration/cam2_homography.json`이 이미 있어야 자동 캡처 대기 없이 바로 돈다.
- `.env`의 `DATABASE_URL`은 Mac PostgreSQL 주소여야 한다. 보통 `127.0.0.1:5432`.
- `.env`의 `POSE_MODEL_PATH`는 `model/yolo/yolo11s-pose.pt`를 사용한다.

## 1. Mac 인프라 실행

터미널 1:

```bash
brew services start postgresql@16
brew services start mosquitto
mediamtx
```

이미 실행 중이면 `brew services start ...`는 다시 실행해도 된다. `mediamtx`는 foreground로 켜두는 편이 로그 확인에 좋다.

터미널 2:

```bash
ollama serve
```

Ollama 앱으로 이미 떠 있으면 생략 가능하다.

## 2. RTSP Publisher 실행

터미널 3:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project
conda activate venv

python input/media/tools/stream_collision_scenario_rtsp.py \
  --scenario scenario_01_center_crossing \
  --rtsp-base rtsp://localhost:8554
```

이 프로세스가 Unity 녹화 프레임을 `rtsp://localhost:8554/cam1`, `rtsp://localhost:8554/cam2`로 publish한다.

## 3. FastAPI 서버 실행

터미널 4:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project
conda activate venv

DISABLE_FUSION_SUBPROCESS=1 \
LLM_BACKEND=local \
OLLAMA_HOST=http://127.0.0.1:11434 \
LOCAL_LLM_MODEL=qwen3:8b \
SERVER_METRICS_PATH=metrics/mac_only_server_requests.jsonl \
SERVER_PORT=1122 \
python -m uvicorn server.main:app --host 127.0.0.1 --port 1122
```

`DISABLE_FUSION_SUBPROCESS=1`은 서버가 realtime 추론을 자동으로 중복 실행하지 않게 하는 옵션이다.

## 4. Realtime/Fusion Benchmark 실행

터미널 5:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project
conda activate venv

HEADLESS=1 \
CAMERA_RTSP_URL_1=rtsp://localhost:8554/cam1 \
CAMERA_RTSP_URL_2=rtsp://localhost:8554/cam2 \
FUSION_SERVER_URL=http://127.0.0.1:1122 \
MQTT_BROKER=127.0.0.1 \
LOCAL_SNAPSHOT_PATH=/Users/haechan/Desktop/pobiga/ai/ai_project/snapshots \
POSE_MODEL_PATH=model/yolo/yolo11s-pose.pt \
EXTRACT_MODE=model_parallel \
RISK_ENGINE=v2 \
python -m model.fusion.runtime.realtime_camera \
  --no-audio \
  --no-prompt \
  --duration 60 \
  --run-label mac_only_s01 \
  --metrics-path metrics/mac_only_s01.jsonl
```

저장되는 주요 지표:

- `cam_read_ms`: RTSP 최신 프레임 복사
- `cam1_pose_track_ms`, `cam2_pose_track_ms`: YOLO pose worker 검출
- `cam1_aruco_detect_ms`, `cam2_aruco_detect_ms`: ArUco 감지
- `cam1_custom_yolo_ms`, `cam2_custom_yolo_ms`: forklift/box custom YOLO
- `cross_camera_ms`: 카메라 간 worker id 전파
- `refine_ms`: detection refinement
- `global_track_ms`: 전역 좌표 tracker
- `fusion_forward_ms`: Fusion 모델 forward
- `early_warning_ms`: TTC/거리 기반 조기 경보
- `publish_dispatch_ms`: MQTT/DB 발행 스레드 dispatch 및 JPEG encode
- `loop_total_ms`: 한 프레임 전체 루프 시간

## 5. 결과 요약

```bash
python input/media/tools/test/summarize_pipeline_metrics.py \
  metrics/mac_only_s01.jsonl \
  --run-label mac_only_s01 \
  --csv-out metrics/mac_only_s01_summary.csv
```

서버 API 요청 시간 요약:

```bash
python input/media/tools/test/summarize_pipeline_metrics.py \
  metrics/mac_only_server_requests.jsonl \
  --field duration_ms \
  --csv-out metrics/mac_only_server_requests_summary.csv
```

이 결과를 같은 필드 기준으로 반복 측정하면 Mac 로컬 실행에서 어떤 모듈이 병목인지 판단할 수 있다.

## 6. 프론트에서 LLM 리포트 생성 시간 측정

프론트 개발 서버:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project/frontend
npm run dev
```

브라우저에서 `http://localhost:5173` 접속 후 기록/리포트 화면에서 `리포트 생성하기`를 누른다.

프론트 콘솔에는 다음 값이 출력된다.

- `requestId`: 서버 로그와 매칭하기 위한 요청 ID
- `clickToResponseMs`: 버튼 클릭부터 `/reports/generate` 응답 도착까지
- `responseToRenderMs`: 응답 도착 후 React 렌더 완료까지
- `clickToRenderMs`: 버튼 클릭부터 화면 표시 완료까지
- `htmlBytes`: 리포트 HTML 크기

브라우저 콘솔에서 다시 확인:

```javascript
window.__pobigaReportMetrics
JSON.parse(localStorage.getItem('pobigaReportMetrics') || '[]')
```

서버 로그에서 같은 요청 확인:

```bash
tail -n 20 metrics/mac_only_server_requests.jsonl
```

프론트의 `requestId`와 서버 로그의 `request_id`가 같은 행을 보면 된다.
