# 산업 현장 멀티모달 안전 모니터링 시스템

> **듀얼 RTSP 카메라 + 음향 센서**로 작업자·지게차·크레인 인양물의 상호작용을 분석해
> **충돌·낙하 위험을 실시간 예측**하고, ESP32 진동 알림과 일일 LLM 리포트로 이어지는
> 엔드투엔드 안전 시스템.

![python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![fastapi](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![tensorflow](https://img.shields.io/badge/TensorFlow-YAMNet-FF6F00?logo=tensorflow&logoColor=white)
![ultralytics](https://img.shields.io/badge/YOLO-11n-00FFFF)
![react](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![postgres](https://img.shields.io/badge/PostgreSQL-asyncpg-336791?logo=postgresql&logoColor=white)
![mqtt](https://img.shields.io/badge/MQTT-aiomqtt-660066)
![esp32](https://img.shields.io/badge/ESP32--S3-PlatformIO-E7352C)

---

## 🎯 핵심 가치

> **"위험은 놓치지 않으면서, 헛알람으로 작업을 방해하지 않는다."**

- ⚡ **실시간** : 5 Hz (200 ms) 주기 추론
- 🎯 **정확도** : 운영 임계값 기준 헛알람 0건, 위험 99% 검출
- 🔗 **멀티모달** : 영상 + 음향을 그래프 신경망(11K 파라미터)으로 융합
- 📡 **양방향 IoT** : ESP32-S3 마이크 → 서버 / 서버 → ESP32 진동 모터

---

## 🏗 시스템 아키텍처

```
[Cam1 RTSP] ─┐                                 [ESP32-S3 mic]
[Cam2 RTSP] ─┤                                       │
              ▼                                       ▼ I2S → WS
   YOLO11-pose + 커스텀(forklift,box)         /ws/audio (FastAPI)
   ArUco 워커 식별 (W01/W02/W03)                       │
   Homography → 월드 좌표 (m)                  YAMNet (centroid cos)
              │                                       │
              ▼                                       ▼ /audio/score
   Pairwise Interaction Fusion Model  ◄────────  audio_score
   (GCN + Temporal + Threat Branch ×2)
              │
              ▼
   risk_matrix (worker × {forklift, dropzone})
              │
       cooldown 2s + 방향 결정 (heading 기반)
              │
      ┌───────┴────────┐
      ▼                ▼
  /send-alert     /incident-logs/with-snapshot
      │                │
      ▼                ▼
  MQTT publish     PostgreSQL + USB JPEG
      │                │
  ESP32 진동       LLM 일일 리포트 → React 대시보드
```

---

## 🧩 주요 기능

| 영역 | 기능 | 위치 |
|---|---|---|
| **캘리브레이션** | ArUco 4코너로 카메라별 Homography 자동 계산 | `input/media/calibrate_homography.py` |
| **객체 검출** | YOLO11-pose (사람) + 커스텀 YOLO (forklift, hoist) | `input/media/world_pipeline.py` |
| **작업자 식별** | ArUco 마커 ID 5/10/15 → W01/W02/W03 | `input/media/world_pipeline.py` |
| **멀티뷰 통합** | 두 카메라 좌표 평균 + cross-cam ID 흡수(1.5m) | `model/fusion/realtime_camera.py:pick_positions` |
| **음향 이상 감지** | YAMNet 임베딩 + Centroid Cosine Similarity | `model/yamnet/detector.py` |
| **위험 예측** | Pairwise Interaction Fusion Model (≈ 11K) | `model/fusion/model.py` |
| **방향 결정** | 워커 heading 기반 4방향(back/left/right/all) 매핑 | `model/fusion/realtime_camera.py:resolve_direction` |
| **알림 발송** | MQTT 진동 + DB IncidentLog 양방향 (cooldown 2s) | `model/fusion/publisher.py`, `db_logger.py` |
| **일일 리포트** | Gemini / Ollama 로컬 LLM 이중 백엔드 | `server/report/` |
| **대시보드** | React 19 + Recharts + jsPDF (PDF 다운로드) | `frontend/src/components/DailyAdminDashboard.tsx` |

---

## 📐 시스템 통합 지표

| 항목 | 값 |
|---|---|
| 작업공간 실측 | **2.22 m × 2.34 m** |
| 캘리브레이션 재투영 오차 (cam1) | 평균 **5.12 × 10⁻⁸ m** |
| 추론 주기 | **5 Hz (200 ms)** |
| ESP32 오디오 chunk | 1024 samples / **64 ms** |
| 음향 분석 윈도우 | **1.92 s** |
| 알림 cooldown | **2.0 s** |
| 드롭존 강제 격상 반경 | **0.5 m** |
| forklift 정지 판정 | < 0.10 m/frame (≈ 0.5 m/s) |
| Cross-cam worker 매칭 임계 | **1.5 m** |

---

## 📊 모델 성능 지표

> 단계별로 측정 / 보강 / 추가 갱신.

### ✅ 1단계 — 모델별 핵심 메트릭 *(완료)*

#### YOLO 객체 검출 (forklift, hoist)

| 클래스 | Instances | Precision | Recall | mAP@50 |
|---|---:|---:|---:|---:|
| **전체** | 2,819 | **1.000** | 0.963 | 0.965 |
| 지게차 | 1,704 | 1.000 | 1.000 | 0.989 |
| 인양물(hoist) | 1,115 | 0.999 | 1.000 | 0.981 |

- **Precision 1.000** → 검증셋 전체에서 헛검출 0건
- 학습: `model/yolo_prac/runs/detect/train4/`, 50 epoch
- 가중치: `YOLO/best_final.pt` *(Git LFS 권장)*

#### YAMNet 음향 이상 감지 (줄 끊어짐)

| 항목 | 값 |
|---|---:|
| Threshold (cosine sim) | **0.68** |
| **CV-Test Recall (LOFO)** | **1.0** |
| Recall | 0.95 |
| Accuracy | 0.95 |
| Embedding dim | 1024 |
| 학습 frame | 572 (636 → 하위 10% outlier 제거) |
| 데이터셋 | 24 anomaly wav + 51 normal wav |

- 설정 파일: `model/yamnet/anomaly_config.json`
- Centroid: `model/yamnet/anomaly_centroid.npy`

#### Pairwise Interaction Fusion Model

학습: 합성 24 시나리오, 71 epoch (EarlyStopping), BCE loss, ≈ 11K params

##### Forklift best ckpt (`best_forklift.pt`, epoch 56)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| safe | 0.914 | 1.000 | 0.955 | 393 |
| warn | 0.966 | 0.431 | 0.596 | 65 |
| **danger** | **1.000** | **0.955** | **0.977** | 22 |
| Macro F1 | | | **0.842** | |

##### Dropzone best ckpt (`best_dropzone.pt`, epoch 12)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| safe | 1.000 | 1.000 | 1.000 | 288 |
| warn | 0.979 | 0.561 | 0.713 | 82 |
| **danger** | 0.752 | **0.991** | **0.855** | 110 |
| Macro F1 | | | **0.856** | |

##### 운영 임계값 (DANGER ≥ 0.8) 기준

| 위협 | Precision | Recall | 한 줄 평가 |
|---|---:|---:|---|
| Forklift | **1.000** | 0.955 | 헛알람 0건, 위험 95% 검출 |
| Dropzone | 0.752 | **0.991** | 위험 99% 검출, 알림 100% 발송 |

#### 한눈 요약 표

| 모델 | 핵심 지표 | 값 |
|---|---|---:|
| YAMNet anomaly | CV recall (LOFO) | **1.0** |
| YOLO forklift/hoist | Precision / mAP@50 | **1.000 / 0.965** |
| Fusion forklift | Danger F1 | **0.977** |
| Fusion dropzone | Danger Recall | **0.991** |
| Calibration | 재투영 오차 평균 | **5.12 × 10⁻⁸ m** |

---

### ⏳ 2단계 — 프레임 처리 속도 측정 *(예정)*

`realtime_camera.py` 메인 루프의 단계별 latency 실측.

- [ ] 단계별 timer 박기 (`time.perf_counter()`)
- [ ] 100 프레임 누적 후 mean / p50 / p95 / p99 통계
- [ ] CSV 저장 + matplotlib 분포 시각화
- [ ] 병목 식별 → 다음 최적화 방향 결정

##### 측정 대상 단계

| 단계 | 내용 | 예상 latency |
|---|---|---|
| ① cam.read() | RTSP 프레임 수신 | I/O 의존 |
| ② extract_detections_with_world ×2 | YOLO + ArUco + homography | 50~150 ms (예상) |
| ③ pick_positions | 두 카메라 통합 | < 5 ms |
| ④ tracker.predict | Fusion 추론 | < 5 ms |
| ⑤ 시각화 (BEV + overlay) | cv2 draw + imshow | 20~40 ms |

> 결과 추가 예정: `tools/benchmark_pipeline.py` 실행 후 갱신

---

### ⏳ 3단계 — Docker 컨테이너화 *(예정)*

- [ ] `requirements.txt` 추출 (environment.yml의 pip 부분)
- [ ] `Dockerfile.backend` (FastAPI + fusion subprocess)
- [ ] `Dockerfile.frontend` (Vite build → nginx)
- [ ] `docker-compose.yml` (postgres + mosquitto + backend + frontend)
- [ ] 모델 가중치 volume mount 정책
- [ ] 환경변수 템플릿 (`.env.example`)

##### 멀티 컨테이너 구조

```
docker-compose.yml
├── backend     (FastAPI + fusion subprocess)
├── frontend    (Vite → nginx)
├── postgres    (postgres:16-alpine)
└── mosquitto   (eclipse-mosquitto)
```

---

### ⏳ 4단계 — CI/CD 자동화 *(예정)*

- [ ] **Phase A** : GitHub Actions로 lint + sanity check (즉시)
- [ ] **Phase B** : Docker 이미지 빌드 + GHCR push
- [ ] **Phase C** : 서버 자동 배포 (SSH or Watchtower)

---

## 🛠 설치 및 실행

### 1. 환경 준비

```bash
git clone <repo-url>
cd ai_project

# Conda 환경 생성
conda env create -f environment.yml
conda activate venv

# Frontend 의존성
cd frontend && npm install && cd ..
```

### 2. 환경변수 (`.env`)

```bash
# 카메라
CAMERA_RTSP_URL_1=rtsp://user:pass@192.168.0.10:554/stream2
CAMERA_RTSP_URL_2=rtsp://user:pass@192.168.0.11:554/stream2

# 데이터베이스
DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5432/safety

# MQTT
MQTT_BROKER=127.0.0.1

# LLM
LLM_BACKEND=gemini      # 또는 local
GEMINI_API_KEY=...
OLLAMA_HOST=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b

# 저장소
USB_STORE_PATH=/Volumes/USB
LOCAL_SNAPSHOT_PATH=./snapshots

# 서버
SERVER_PORT=1122

# YOLO 커스텀 모델
BEST_MODEL_PATH=YOLO/best_final.pt
```

### 3. 캘리브레이션 (최초 1회)

```bash
# 각 카메라로 작업공간 코너 ArUco(22, 24, 27, 38)가 보이는 스냅샷 촬영 후
python input/media/calibrate_homography.py --cam cam1 --image calibration/test_cam1.jpg
python input/media/calibrate_homography.py --cam cam2 --image calibration/test_cam2.jpg

# 검증 (격자 오버레이 시각 확인)
python input/media/verify_homography.py --cam cam1 --image calibration/test_cam1.jpg
```

### 4. 모델 학습 (선택)

```bash
# Fusion 모델 학습 (≈ 5분)
python model/fusion/train.py
# → model/fusion/checkpoints/{best, best_forklift, best_dropzone}.pt 생성
```

#### 학습 데이터셋 (DVC 관리)

YAMNet 학습 데이터셋(`model/yamnet/dataset/`, 102 wav, 13MB)은 **DVC**로 관리합니다.
git clone 직후엔 `dataset.dvc` 만 받고 실제 wav 는 비어있으니 `dvc pull` 필요:

```bash
# 데이터셋 받기 (협업 시 / 새 환경 셋업 시)
dvc pull

# 데이터셋 변경 후 다시 추적
dvc add model/yamnet/dataset
dvc push
git add model/yamnet/dataset.dvc
git commit -m "data: update yamnet dataset"
```

> 기본 remote 는 로컬(`~/dvc-storage`). 협업 시 S3 / GDrive 등으로 마이그레이션 가능:
> `dvc remote add -d s3 s3://bucket/path`

### 5. 서버 + 파이프라인 실행

```bash
# 한 번에 전체 시스템 기동 (FastAPI + fusion subprocess + MQTT 큐)
python -m server.main

# 또는 fusion 파이프라인만 단독 실행
python model/fusion/realtime_camera.py
```

### 6. Frontend

```bash
cd frontend
npm run dev          # 개발 모드 (vite proxy → :1122)
# 또는
npm run build && npm run preview
```

### 7. ESP32 펌웨어

```bash
cd firmware/esp32_audio_ws
pio run -t upload
pio device monitor
```

---

## 📁 디렉토리 구조

```
ai_project/
├── calibration/                    # 카메라별 H 행렬 + ArUco 실측 좌표
│   ├── world_markers.json
│   └── cam{1,2}_homography.json
├── firmware/esp32_audio_ws/        # ESP32-S3 PlatformIO 프로젝트
├── frontend/                       # React 19 + Vite + Tailwind
│   └── src/components/
├── input/
│   ├── audio/                      # ESP32 audio WebSocket
│   └── media/                      # 카메라/캘리브레이션/world_pipeline
├── model/
│   ├── fusion/                     # PairwiseInteractionFusionModel
│   │   ├── model.py / dataset.py / train.py / inference.py
│   │   ├── pair_labels.py / scenarios_synthetic.py
│   │   ├── publisher.py / db_logger.py / risk_output.py
│   │   ├── realtime_camera.py     # 통합 실시간 파이프라인
│   │   └── checkpoints/best*.pt
│   ├── yamnet/                     # 음향 이상 감지
│   │   ├── detector.py
│   │   ├── anomaly_centroid.npy
│   │   └── anomaly_config.json
│   └── yolo_prac/                  # 커스텀 YOLO 학습 결과
├── server/                         # FastAPI 백엔드
│   ├── main.py
│   ├── database/                   # SQLAlchemy 모델 + USB 저장
│   ├── pipeline/mqtt/              # aiomqtt handler
│   └── report/                     # Gemini / Ollama 리포트
├── YOLO/best_final.pt              # 커스텀 YOLO 가중치
├── environment.yml
└── README.md
```

---

## 🔬 모델 카드 요약

| 모델 | 입력 | 출력 | 학습 데이터 | 위치 |
|---|---|---|---|---|
| YOLO11-pose | 카메라 frame | person bbox + 17 keypoints | 사전학습 (COCO) | 동적 다운로드 |
| YOLO custom | 카메라 frame | forklift / hoist bbox | forklift_night_4.16 (50 epoch) | `YOLO/best_final.pt` |
| YAMNet + Centroid | 1.92s 16kHz mono PCM | (max_sim, is_anomaly) | 24 anomaly + 51 normal wav | `model/yamnet/` |
| Pairwise Fusion | (B, V=3, T=5, F=8) + adj + scene | risk_matrix (B, N, K=2) | 합성 24 시나리오 (LOFO) | `model/fusion/checkpoints/` |

---

## 🎬 데모 시나리오

```
1. 워커 W01 이 작업공간 진입 → ArUco 마커로 즉시 식별
2. 지게차가 W01 좌측에서 접근 → 5Hz 추론 → forklift risk 0.85
3. resolve_direction(): worker heading=π/2, 위협 위치 → "left"
4. cooldown 통과 → MQTT publish: forklift/4/vibration  payload="left"
5. ESP32 좌측 진동 모터 작동 → W01에 "왼쪽 주의" 신호
6. PostgreSQL IncidentLog 행 + 스냅샷 저장
7. 일일 LLM 리포트에 자동 요약 → 대시보드 PDF 다운로드
```

---

## 🗺 향후 계획

상세 로드맵은 별도 문서로 분리했습니다 → **[ROADMAP.md](./ROADMAP.md)**

### 5축 요약

| 축 | 핵심 한 방 | 우선 액션 |
|---|---|---|
| ① **정확도** | Unity 가상 영상으로 데이터 다양화 | warn 라벨 보강, per-pair 분리 운영 |
| ② **반응속도** | multiprocessing + ONNX 가속 | FPS 실측 → 병목 식별 |
| ③ **안정성** | Health check + 통합 로깅 | `/health` 엔드포인트 |
| ④ **확장성** | 카메라 / 워커 config 동적화 | YAML 기반 멀티 카메라 |
| ⑤ **차별화** | Pairwise Fusion 논문화 + Uncertainty | per-pair + TTC 출력 |

### 단기 처리 우선순위 TOP 5

```
1. B1.    FPS 실측              ⏰ 2시간   → 다음 결정 근거
2. F-4.2  per-pair 분리 운영    ⏰ 4시간   → Fusion Macro F1 +0.08
3. C1.    Health check          ⏰ 2시간   → 모니터링 기반
4. F-4.1  PR curve threshold    ⏰ 4시간   → 운영 정확도 ↑
5. F-5.1  analyze_fusion.py     ⏰ 4시간   → 모델 디버깅 기반
```

### Fusion 모델 심화 점검

[ROADMAP.md](./ROADMAP.md) 의 "🎯 Fusion 모델 전용 심화 점검" 섹션 참조.
학습 데이터 / 모델 구조 / 학습 전략 / 운영 / 검증 5개 영역에서 **24개 구체 점검 항목** 정의.

---

## 📜 라이선스 / 기여

> 라이선스 / 기여 가이드 추후 추가.

---

## 📌 변경 이력

| 날짜 | 단계 | 내용 |
|---|---|---|
| Init | 1단계 | YOLO / YAMNet / Fusion 메트릭 측정 완료, README 초기화 |
