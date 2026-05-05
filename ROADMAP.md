# 🗺 ROADMAP — 성능 개선 로드맵

> 본 시스템의 향후 발전 계획을 **5개 축** + **Fusion 모델 전용 심화 점검**으로 정리한 문서.
> 진행 시 체크박스(`[ ]` → `[x]`)로 상태를 갱신하고, 각 항목 하단에 결과 / 메트릭을 추가한다.

---

## 📐 5개 축 한눈에

```
                            ┌─────────────────┐
                            │   현재 시스템    │
                            └────────┬────────┘
                                     │
      ┌───────────┬───────────┬──────┴──────┬──────────────┐
      ▼           ▼           ▼             ▼              ▼
   ① 정확도    ② 반응속도   ③ 운영안정성  ④ 확장성     ⑤ 차별화
   (모델 품질)  (latency)   (장애 대응)   (대수/사이트)  (연구)
```

---

## ① 정확도 (Accuracy / F1 / Recall)

### 현재 진단

| 측면 | 상태 |
|---|---|
| YAMNet | CV-Test Recall 1.0 — ✅ 충분 |
| YOLO 객체 검출 | mAP@50 0.965, P 1.000 — ✅ 운영 수준 |
| Fusion warn 클래스 | F1 0.6대 — ⚠️ 약점 |
| 학습 데이터 | 합성 24 시나리오만 — ⚠️ 일반화 미검증 |

### 액션 (우선순위 ⭐ 많을수록 ROI 큼)

- [ ] **A1. warn 라벨 시나리오 5~10개 추가** ⭐⭐⭐ *(예상 1주)*
  - `pair_labels.py` 룰 보강 (조기 경고 거리/속도 임계 조정)
  - 합성 시나리오에 `s25_*` ~ `s30_*` 워닝 케이스 추가
  - 기대: warn F1 0.6 → 0.8
- [ ] **A2. per-pair 분리 운영** ⭐⭐⭐ *(예상 4시간)*
  - `realtime_camera.py`가 `best.pt` 1개만 로드 → `best_forklift.pt` + `best_dropzone.pt` 두 개 로드
  - forklift risk 는 forklift ckpt로, dropzone risk 는 dropzone ckpt로
  - 기대: Macro F1 0.77 → 0.85
- [ ] **A3. 멀티 워커(N=2,3) 학습** ⭐⭐ *(예상 1~2주)*
  - 현재 학습: N=1 worker만. 실시간은 N≤3 멀티
  - `scenario_generator.py` 확장 + `model.py` n_workers 인자 활용
- [ ] **A4. Unity 가상 영상으로 일반화 검증** ⭐⭐⭐ *(예상 2~3주)*
  - Unity URP로 작업공간 + 다양한 시나리오 영상 생성
  - 우리 파이프라인 INPUT으로 흘려서 OOD 성능 측정
- [ ] **A5. 라벨링 룰 정교화** ⭐⭐ *(예상 3~5일)*
  - TTC (Time-To-Collision) 추가
  - 가속도 / 사각지대 변수 도입
- [ ] **A6. GAT / Cross-Attention 도입** ⭐ *(예상 2~4주)*
  - GraphConv → Graph Attention Network
  - Threat Branch에 cross-attention
- [ ] **A7. SOTA 모델 비교** ⭐ *(예상 1주)*
  - YOLO11s/m, RT-DETR vs 현재 YOLO11n
  - mAP / 추론 속도 / 모델 크기 trade-off 표

---

## ② 반응 속도 (Latency / FPS)

### 현재 진단

| 측면 | 상태 |
|---|---|
| 메인 루프 | 5 Hz 고정 (200 ms sleep) |
| 추론 디바이스 | CPU 전용 (`device='cpu'`) |
| 카메라 처리 | 직렬 (cam1 → cam2) |
| 실측 latency | ❓ 미측정 |

### 액션

- [ ] **B1. FPS 실측** ⭐⭐⭐ *(예상 2시간)*
  - `time.perf_counter()` 단계별 timer
  - cam.read / extract_detections / pick_positions / tracker.predict / 시각화
  - 100 프레임 누적 mean / p50 / p95 / p99
- [ ] **B2. multiprocessing 분리** ⭐⭐⭐ *(예상 1주)*
  - 카메라 캡처 프로세스 × 2 + 추론 프로세스 + 시각화 프로세스
  - `multiprocessing.Queue`로 frame 스트림 전달
  - 기대: 5 Hz → 10 Hz
- [ ] **B3. ONNX / TensorRT 변환** ⭐⭐ *(예상 3~5일)*
  - YOLO 모델 → ONNX → 또는 TensorRT (NVIDIA GPU 시)
  - 기대: YOLO 추론 시간 50% 단축
- [ ] **B4. INT8 양자화** ⭐⭐ *(예상 1주)*
  - Post-Training Quantization
  - 정확도 1~2%p 손실 vs 속도 30%↑
- [ ] **B5. GPU 사용** ⭐⭐ *(환경 의존)*
  - CUDA / MPS / ROCm
  - `realtime_camera.py:device='cpu'` → `'cuda'`
  - 기대: 추론 5~10배 가속
- [ ] **B6. Frame Skip** ⭐ *(예상 2시간)*
  - 5Hz → 2.5Hz, 인접 프레임 보간
  - 부하 절반
- [ ] **B7. ArUco 캐싱** ⭐ *(예상 4시간)*
  - 코너 마커는 정적 → 매 프레임 재검출 불필요
  - 0.5초마다만 재검출

> ⚠️ **Celery / Kafka는 우리 규모에 과한 도구** (5Hz × 2cam = 10 msg/s).
> multiprocessing.Queue → 부족하면 redis streams → 그 이상에서 Kafka 검토.

---

## ③ 운영 안정성 (Robustness)

### 현재 진단

| 측면 | 상태 |
|---|---|
| RTSP 끊김 복구 | `VideoStream._update`에 부분 처리 |
| USB 미마운트 | 폴백 동작 ✅, 단 `serve_usb_image`는 폴백 미지원 |
| Health check | ❌ 없음 |
| 로깅 | print + loguru 혼재 |
| 모니터링 | ❌ 없음 |

### 액션

- [ ] **C1. Health check 엔드포인트** ⭐⭐⭐ *(예상 2시간)*
  - `/health` (DB 연결, MQTT 연결, fusion subprocess 살아 있나)
  - `/ready` (서비스 가능한 상태인가)
- [ ] **C2. RTSP 재연결 로직 강화** ⭐⭐ *(예상 1주)*
  - 한 카메라 죽어도 다른 카메라로 단독 운영
  - 지수 백오프 + 사고 알림 통합
- [ ] **C3. `serve_usb_image` 폴백 지원** ⭐⭐ *(예상 2시간)*
  - `LOCAL_SNAPSHOT_PATH` 까지 검사
  - USB 없는 환경에서도 프론트 이미지 표시
- [ ] **C4. 통합 로깅** ⭐⭐ *(예상 1일)*
  - loguru 일원화 + JSON 포맷 + level 표준화
  - print → log.info / log.warning 으로 마이그레이션
- [ ] **C5. Prometheus + Grafana** ⭐ *(예상 3일)*
  - p95 latency, error rate, alert count, fusion subprocess uptime
- [ ] **C6. 워치독** ⭐⭐ *(예상 4시간)*
  - fusion subprocess 죽으면 자동 재시작
  - server lifespan 에 watchdog 태스크 추가

---

## ④ 확장성 (Scalability)

### 현재 진단

| 측면 | 상태 |
|---|---|
| 카메라 | 2대 하드코딩 (`CAMERA_RTSP_URL_1/2`) |
| 작업자 | MAX_WORKERS = 3, Maker 1~5 시드 고정 |
| 머신 | 단일 머신 |

### 액션

- [ ] **D1. 카메라 config 동적화** ⭐⭐ *(예상 1일)*
  - YAML로 N대 정의 (`cameras.yaml`)
  - `realtime_camera.py` 에서 동적 로드
- [ ] **D2. Worker 동적 등록 API** ⭐⭐ *(예상 3일)*
  - 마커 ID → DB workers 테이블 mapping
  - `POST /workers` 로 신규 등록
- [ ] **D3. 멀티 사이트** ⭐ *(예상 1~2주)*
  - 현장별 calibration / DB 분리
- [ ] **D4. Redis Streams** ⭐ *(예상 1주)*
  - 카메라 N대 → 추론 워커 M개 분산
- [ ] **D5. Kubernetes** *(여건 시)*
  - 분산 추론 + 오토스케일

---

## ⑤ 차별화 (Research / 학술 가치)

### 현재 강점

- 11K 파라미터 경량 fusion 모델
- Pair-level 출력 (워커 × 위협)
- 합성 데이터로 LOFO 검증 가능
- 멀티모달 (영상 + 음향)

### 액션

- [ ] **E1. 워크샵 페이퍼 1편** *(1~2달)* — Pairwise Interaction Fusion for Industrial Safety
- [ ] **E2. Uncertainty Estimation** *(1달)* — Bayesian DL, 신뢰도 함께 출력
- [ ] **E3. TTC(Time-to-Collision) 예측** *(2주)*
- [ ] **E4. Self-Supervised Pretraining** *(1~2달)* — Unity 데이터로 마스킹된 위치 예측
- [ ] **E5. 오픈소스 공개** *(1주)* — GitHub + Docker Hub + 데모 영상

---

# 🎯 Fusion 모델 전용 심화 점검

> 학습 + 평가 + 운영 + 검증 4축에서 **현재 모델의 어떤 면을 더 봐야 하는지** 구체 체크리스트.

## 📊 현재 학습 결과 요약

| 시점 | F1 (Forklift) | F1 (Dropzone) | 비고 |
|---|---:|---:|---|
| Forklift best (epoch 56) | **0.842** | 0.595 | `best_forklift.pt` |
| Dropzone best (epoch 12) | 0.619 | **0.856** | `best_dropzone.pt` |
| Avg best (epoch 14) | ~0.74 | ~0.80 | `best.pt` (운영) |

> 💡 **관찰**: forklift 와 dropzone 의 학습 곡선이 **다른 epoch 에서 정점** → 한 모델로 양쪽 다 최고 성능 어려움.

---

## 🔍 F-1. 학습 데이터 / 라벨링 점검

### F-1.1 클래스 불균형 진단

```
forklift  validation support:  safe 393  /  warn  65  /  danger  22
dropzone  validation support:  safe 288  /  warn  82  /  danger 110
```

- [ ] **합성 24 시나리오 라벨 분포 정밀 분석**
  - 각 클래스 step 비율 정량화
  - safe 가 압도적으로 많음 → 모델이 "거의 safe" 로 편향될 위험
  - **점검 스크립트**: `python -m model.fusion.pair_labels` 출력의 `summarize_labels()` 결과 저장
- [ ] **클래스별 평가 metric 분리 출력**
  - 현재는 macro F1만 → confusion matrix 전체 출력 추가
  - warn → safe 오분류 vs warn → danger 오분류 비율 분리 (어느 방향이 위험한지)

### F-1.2 시나리오 다양성

```
SAFE         4개  ─┐
지게차 위험  10개  ├─ 총 24개
드롭존 위험  10개  ─┘
```

- [ ] **시나리오 30+ 으로 확장**
  - SAFE +6 (다양한 정상 상황: 작업자가 도구 사용, 멀리서 지나감 등)
  - 지게차 +5 (커브, 정지 후 출발, 후진 다양화)
  - 드롭존 +5 (서로 다른 인양물 위치, 작업자 우회 등)
- [ ] **각 시나리오의 jitter / noise 강도 다양화**
  - 현재 `jitter(std=0.02)` 고정 → 0.01, 0.05 변종 추가
  - 호모그래피 오차 시뮬레이션
- [ ] **Edge case 시나리오 추가**
  - 둘 다 위험 (forklift + dropzone 동시 trigger)
  - 위험 → 정상 → 위험 (간헐적 패턴)
  - 카메라 일시 끊김 (NaN 위치 구간)

### F-1.3 라벨링 룰 정교화

현재 룰 ([pair_labels.py](model/fusion/pair_labels.py)):
```python
FORK_DANGER_DIST  = 0.4 m
FORK_WARN_DIST    = 0.9 m
APPROACH_SPEED    = 0.05 m/s
DZ_WARN_BUFFER    = 0.2 m
AUDIO_DANGER_THR  = 0.65
```

- [ ] **TTC (Time-To-Collision) 라벨 도입**
  - `dist / closing_speed` 가 임계 이하면 danger
  - 거리 기반보다 위협 지속성 잘 잡음
- [ ] **forklift 정지 보정**
  - 현재: 정지 forklift 도 거리 < 0.4 면 danger 라벨
  - `realtime_camera.py` 에서는 정지면 무시하지만, **학습에서는 여전히 위험으로 학습 중**
  - **개선**: 라벨링 시 forklift 속도 < FORKLIFT_STATIC_SPEED 면 강제 safe
- [ ] **워커 facing 고려**
  - 현재: 워커가 위협을 보고 있어도 같은 라벨
  - 발표 정책: front 위협 = 알림 X (이미 시야)
  - **개선**: facing 일치도에 따라 라벨 약화

---

## 🧠 F-2. 모델 구조 점검

### F-2.1 GCN depth / hidden 차원

```python
# 현재
self.gconv1 = GraphConv(8, 24)
self.gconv2 = GraphConv(24, 24)
self.tconv  = Conv1d(24, 24, k=3)
self.gru    = GRU(24, 24)
hidden      = 24
```

- [ ] **Hidden 차원 sweep** (16 / 24 / 32 / 48)
  - 11K 파라미터 유지하면서 효과 큰 hidden 찾기
- [ ] **GCN layer depth** (1 / 2 / 3)
  - 노드가 3개라 2-layer로 충분할 수 있지만 검증 필요
- [ ] **Dropout 위치 / 비율 sweep** (0.1 / 0.3 / 0.5)
  - 현재 ThreatBranch head 에만 0.3
  - GRU 출력 / GCN 사이에도 추가 시도

### F-2.2 Threat Branch 분리 정책

현재: forklift / dropzone branch 가 **scene encoder + head 모두 분리**.

- [ ] **Branch 공유 변종 비교**
  - V1: scene encoder 공유 + head 만 분리
  - V2: scene encoder + head 모두 공유 (단일 출력 채널 2개)
  - V3: 현재 방식 (모두 분리)
  - 각 변종의 Macro F1 비교
- [ ] **Cross-Attention 도입**
  - worker_h 와 threat_h 사이 attention
  - 어떤 위협이 어떤 워커에 더 영향 주는지 학습
- [ ] **Threat 간 상호작용 모델링**
  - forklift 와 dropzone 이 동시에 가까울 때의 패턴
  - 현재는 두 분기가 완전 독립이라 상호작용 정보 잃음

### F-2.3 Temporal Window

- [ ] **T_WIN sweep** (3 / 5 / 10 / 20)
  - 현재 T_WIN=5 (1초)
  - dropzone 는 빠르게 학습 (epoch 12), forklift 는 늦음 (epoch 56) → 위협별로 적정 윈도우 다를 수 있음
- [ ] **Hop size 조정** (현재 stride=1)
  - stride=2 로 줄이면 학습 데이터 절반
  - 과적합 방지 효과 ?

---

## 🎲 F-3. 학습 전략 점검

### F-3.1 Loss / Optimizer

- [ ] **Class-Weighted BCE 시도**
  - support 비율 역수로 가중치 부여
  - warn 클래스 학습 강화
- [ ] **Focal Loss 시도**
  - `α(1-p)^γ * BCE`
  - 어려운 샘플 (경계 case) 집중 학습
- [ ] **AdamW vs Adam 비교**
  - 현재 Adam + weight_decay=1e-4
  - AdamW 가 가중치 감쇠를 더 깔끔하게 처리

### F-3.2 학습 안정화

- [ ] **Learning Rate Scheduler**
  - 현재 lr=1e-3 고정
  - CosineAnnealing 또는 ReduceLROnPlateau 시도
- [ ] **Gradient Clipping 모니터링**
  - 현재 `clip_grad_norm_(1.0)` → 실제 norm 분포 로깅
- [ ] **EarlyStopping patience 튜닝**
  - 현재 15 (per-pair 정체 시 중단)
  - 10 / 20 비교

### F-3.3 검증 전략

- [ ] **K-Fold CV 추가**
  - 24개 시나리오 → 5-fold (각 fold 5개 val)
  - 단일 split의 통계적 변동 줄이기
- [ ] **LOFO (Leave-One-File-Out)**
  - 24개라 LOFO 가능 (24 fold)
  - 각 fold 결과의 분산 확인 → robustness 정량
- [ ] **Hold-out test set 분리**
  - 현재 train / val 만. val 기준으로 best 선택 → val 과적합 위험
  - test 4~6개 시나리오 따로 떼서 final 평가만 사용

---

## 🚦 F-4. 운영 / 추론 점검

### F-4.1 Threshold 튜닝

현재: `WARN = 0.4`, `DANGER = 0.8` *(임의 설정)*

- [ ] **PR Curve 기반 임계값 산출**
  - validation 출력으로 PR curve 그리기
  - 운영 목표 (예: precision 1.0 유지하며 recall 최대)에 맞춰 threshold 결정
- [ ] **위협별 threshold 분리**
  - forklift Precision 1.0 (헛알람 0) → DANGER 0.8 OK
  - dropzone Precision 0.752 (warn↔danger 혼동) → DANGER 0.85~0.9 로 올려서 헛알람 줄일지 검토
- [ ] **F-Beta Score 기반 결정**
  - 안전 도메인은 recall 중요 → F2 (recall 가중) 최대화 threshold

### F-4.2 Per-pair 분리 운영 (A2 와 동일)

```python
# 현재 (best.pt 단일)
model = load_model("best.pt")

# 개선 후
model_f = load_model("best_forklift.pt")
model_d = load_model("best_dropzone.pt")
```

- [ ] **`realtime_camera.py` 두 모델 동시 로드**
  - 메모리 ×2 (10K params 라 미미)
  - forklift risk = `model_f(...)`[forklift_idx]
  - dropzone risk = `model_d(...)`[dropzone_idx]
- [ ] **A/B 비교**
  - 단일 모델 vs per-pair 분리 → 같은 검증셋에서 Macro F1 차이
  - 운영 latency 차이 (모델 2개 forward)

### F-4.3 후처리 룰 vs 학습 라벨

현재 [realtime_camera.py](model/fusion/realtime_camera.py) 가 모델 출력에 후처리 룰 적용:

| 룰 | 영향 | 학습 반영? |
|---|---|---|
| 정지 forklift trigger 무시 | forklift risk 무력화 | ❌ 학습은 정지도 위험으로 봄 |
| dropzone 0.5m 강제 1.0 | dropzone risk 덮어씀 | ❌ |
| front 알림 차단 | direction 단계에서 막음 | ❌ (학습엔 없음) |

- [ ] **후처리 룰을 학습 라벨에 반영**
  - 정지 forklift → safe 라벨
  - dropzone 0.5m → danger 라벨
  - 모델이 이 패턴을 직접 학습하면 후처리 룰 제거 가능 (코드 단순화 + 일관성)
- [ ] **하이브리드 평가**
  - 모델 단독 정확도 vs (모델 + 룰) 정확도
  - 룰의 기여도 정량화

---

## 🧪 F-5. 검증 / 분석 도구

### F-5.1 진단 스크립트 추가

- [ ] **`tools/analyze_fusion.py` 작성** *(예상 4시간)*
  - 검증셋 모든 윈도우에 대한 prediction 저장
  - confusion matrix (per-pair × 3-class) 시각화
  - PR curve / F1 curve
  - 잘못 분류된 케이스 리스트 + 시각화 (어떤 시나리오의 어떤 시점)
- [ ] **`tools/error_analysis.py` 작성**
  - "이 시점에 왜 모델이 틀렸나" 사례 분석
  - 입력 텐서 dump → 시나리오 재현
- [ ] **Risk timeline 시각화 도구**
  - 각 시나리오의 시간축에 따른 risk_f, risk_d, label 동시 plot
  - 발표 자료 + debugging 용도

### F-5.2 일반화 성능 검증

- [ ] **OOD (Out-Of-Distribution) 테스트셋 구성**
  - 합성 시나리오에 없는 패턴 (예: 동일 사람 이중 검출, 극단적 노이즈)
  - 실제 영상 (있다면) 또는 Unity 영상
- [ ] **Robustness 테스트**
  - 입력 좌표에 noise 추가 (homography 오차 시뮬레이션)
  - 일부 frame 누락 시뮬레이션
  - 모델 출력의 안정성 확인

### F-5.3 모델 인터프리터빌리티

- [ ] **Attention weight 시각화** (GAT 도입 후)
  - 어떤 노드가 어떤 노드에 주목하는지
- [ ] **GCN node embedding 시각화**
  - t-SNE / UMAP 으로 worker / forklift / dropzone 임베딩 분포 확인
- [ ] **Saliency Map**
  - 입력 feature 중 어떤 것이 risk 결정에 가장 영향 큰지
  - "audio 가 trigger 했나, 거리가 trigger 했나" 등

---

## 🎯 Fusion 모델 우선 처리 TOP 7

```
1. F-4.2 per-pair 분리 운영           ⭐⭐⭐  4시간    Macro F1 +0.08
2. F-4.1 PR Curve 기반 threshold      ⭐⭐⭐  4시간    운영 정확도 ↑
3. F-1.3 정지 forklift 라벨 반영      ⭐⭐⭐  1일     룰-학습 일관성
4. F-5.1 analyze_fusion.py 작성       ⭐⭐⭐  4시간    diagnosis 기반
5. F-1.2 시나리오 30+ 확장            ⭐⭐⭐  1주     warn F1 ↑
6. F-3.3 LOFO CV                      ⭐⭐    3일     robustness 정량
7. F-5.2 OOD 테스트                   ⭐⭐    1주     일반화 성능
```

---

## 🚀 통합 우선순위 (전체)

### TOP 5 — ROI 압도적 *(2주 내 완료 가능)*

```
1. B1.    FPS 실측              ⏰ 2시간    → 다음 결정 근거
2. F-4.2  per-pair 분리 운영    ⏰ 4시간    → Fusion Macro F1 +0.08
3. C1.    Health check          ⏰ 2시간    → 모니터링 기반
4. F-4.1  PR curve threshold    ⏰ 4시간    → 운영 정확도 ↑
5. F-5.1  analyze_fusion.py     ⏰ 4시간    → 모델 디버깅 기반
```

### 중기 — 본격 도약 *(1~2달)*

```
6. A1 / F-1.2  warn 라벨 + 시나리오 확장   ⏰ 1주    → warn F1 0.6 → 0.8
7. A4         Unity 가상 영상              ⏰ 2~3주  → 일반화 검증
8. B2         multiprocessing              ⏰ 1주    → 5Hz → 10Hz
9. B3+B4      ONNX + 양자화                ⏰ 1.5주  → 추론 70% 단축
```

### 장기 — 차별화 *(2~6달)*

```
10. F-2.2     GAT / Cross-Attention        ⏰ 1달    → 모델 차별화
11. E1        워크샵 페이퍼                ⏰ 1~2달  → 학술 가치
12. D1+D2     카메라/워커 동적             ⏰ 2주    → B2B 가능
```

---

## 📌 변경 이력

| 날짜 | 변경 |
|---|---|
| Init | ROADMAP.md 분리 + Fusion 모델 5축 심화 점검 추가 |
