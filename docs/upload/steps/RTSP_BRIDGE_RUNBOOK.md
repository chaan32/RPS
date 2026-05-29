# Unity Collision Scenario RTSP Bridge

## 목적

Unity에서 녹화한 `cam1`, `cam2` 프레임을 실제 CCTV RTSP 입력처럼 백엔드에 넣기 위한 Mac 로컬 실행 절차다.

```text
Unity recorded frames
  -> FFmpeg publisher
  -> MediaMTX RTSP server
  -> backend realtime_camera
  -> YOLO + homography + fusion
```

이 방식을 쓰면 백엔드는 파일 경로가 아니라 `rtsp://localhost:8554/cam1`, `rtsp://localhost:8554/cam2`를 입력으로 받는다. 나중에 실제 CCTV로 바꿀 때도 입력 계층을 크게 바꾸지 않아도 된다.

## 1. MediaMTX 실행

터미널 1:

```bash
mediamtx
```

이미 다른 프로세스가 `8554` 포트를 쓰고 있으면 먼저 종료해야 한다.

```bash
lsof -nP -iTCP:8554 -sTCP:LISTEN
```

## 2. Unity scenario를 RTSP로 publish

터미널 2:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project
conda activate venv

python input/media/tools/stream_collision_scenario_rtsp.py \
  --scenario scenario_01_user_current \
  --rtsp-base rtsp://localhost:8554
```

브릿지가 publish하는 주소:

```text
rtsp://localhost:8554/cam1
rtsp://localhost:8554/cam2
```

다른 시나리오는 `--scenario` 값만 바꿔 실행한다.

```bash
python input/media/tools/stream_collision_scenario_rtsp.py --scenario scenario_02_swapped_positions --rtsp-base rtsp://localhost:8554
python input/media/tools/stream_collision_scenario_rtsp.py --scenario scenario_03_blind_corner_merge --rtsp-base rtsp://localhost:8554
python input/media/tools/stream_collision_scenario_rtsp.py --scenario scenario_04_dropzone_box --rtsp-base rtsp://localhost:8554
```

## 3. RTSP 수신 확인

```bash
ffprobe -rtsp_transport tcp -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  rtsp://localhost:8554/cam1
```

`cam2`도 같은 방식으로 확인한다.

```bash
ffprobe -rtsp_transport tcp -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  rtsp://localhost:8554/cam2
```

## 4. 백엔드 realtime 실행

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project
conda activate venv

HEADLESS=1 \
CAMERA_RTSP_URL_1=rtsp://localhost:8554/cam1 \
CAMERA_RTSP_URL_2=rtsp://localhost:8554/cam2 \
FUSION_SERVER_URL=http://127.0.0.1:1122 \
MQTT_BROKER=127.0.0.1 \
LOCAL_SNAPSHOT_PATH=/Users/haechan/Desktop/pobiga/ai/ai_project/snapshots \
python -m model.fusion.runtime.realtime_camera \
  --no-audio \
  --no-prompt \
  --duration 60 \
  --run-label rtsp_local_scenario \
  --metrics-path metrics/rtsp_local_scenario.jsonl
```

## 5. 종료

- 브릿지 터미널에서 `Ctrl-C`
- MediaMTX 터미널에서 `Ctrl-C`

## 주의할 점

- MediaMTX의 같은 path에는 publisher가 하나만 붙을 수 있다. `/cam1`, `/cam2`에 이미 다른 publisher가 붙어 있으면 새 브릿지가 끊긴다.
- 현재 브릿지는 Unity가 저장한 `frame_0000.jpg`부터 연속된 JPG 시퀀스를 사용한다. 프레임 번호가 끊기면 publish 전에 에러를 낸다.
- 백엔드는 항상 `rtsp://localhost:8554/cam1`, `rtsp://localhost:8554/cam2`를 읽는 Mac 로컬 실행 기준이다.
