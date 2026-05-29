# Fusion V2 Deep Learning Plan

## 목적

V1은 멀티뷰 좌표 변환과 규칙 기반 위험 판단을 결합한 안정적인 실시간 파이프라인이다.  
V2는 V1을 보존한 상태에서, 위험 판단 부분을 좌표 시계열 딥러닝 모델로 확장하는 별도 실험 버전이다.

```text
V1: YOLO/Pose -> BEV 좌표 -> 거리/TTC/반경 조건 -> Warning/Danger
V2: YOLO/Pose -> BEV 좌표 -> 최근 N프레임 좌표 시퀀스 -> 딥러닝 위험도 예측
```

## V1 보존 원칙

- `model/fusion/` 런타임과 기존 규칙 기반 알림 로직은 수정하지 않는다.
- V2 코드는 `model/fusion_v2/`에만 추가한다.
- V1이 생성한 `fusion_risk.csv`를 teacher label로 사용해 V2 학습 데이터를 만든다.
- V2 성능이 충분히 검증되기 전까지 운영 알림은 V1을 기준으로 유지한다.

## 현재 V2 범위

### 입력

V2는 원본 영상이 아니라 BEV/world 좌표 시계열을 입력으로 사용한다.

- worker 위치, 속도
- forklift 위치, 속도
- forklift front hazard 위치, 속도
- box/dropzone 위치, 속도
- worker-forklift 거리
- worker-forklift hazard 거리
- worker-dropzone 거리
- 객체 존재 여부
- tracker outlier flag

### 라벨

초기 V2 라벨은 V1의 최종 판단을 teacher label로 사용한다.

- forklift target = `max(forklift_risk, early_warning_score)`
- dropzone target = `max(dropzone_risk, dropzone_forced)`

이 방식은 V1을 대체하기 위한 첫 단계다. 이후 사람이 직접 검수한 라벨이나 Unity ground truth 기반 라벨로 교체할 수 있다.

### 모델

현재 V2 baseline은 GRU 기반 시계열 모델이다.

```text
(B, T, F) 좌표 시퀀스
-> Linear + LayerNorm
-> GRU
-> MLP Head
-> [forklift_risk, dropzone_risk]
```

출력은 작업자 1명 기준 pair risk이며, 다중 작업자는 worker별 window를 각각 모델에 넣는다.

## 이번 실행 결과

이번 단계에서는 영상 재녹화가 아니라, V1이 산출하는 `fusion_risk.csv`와 동일한 좌표 스키마로 V2 학습용 시나리오를 대량 생성했다.  
V2가 학습하는 대상은 원본 이미지가 아니라 BEV/world 좌표 시계열이기 때문이다.

- 기존 Unity 검증 시나리오: 7개
- V2 추가 좌표 시나리오: 210개
- 총 시나리오 소스: 217개
- 생성 window: 42,136개
- window size: 24 frame
- feature dimension: 23
- 학습 epoch: 15
- best epoch: 11

### 라벨 분포

| 대상 | SAFE | WARNING | DANGER |
| --- | ---: | ---: | ---: |
| 지게차 충돌 위험 | 32,336 (76.7%) | 5,858 (13.9%) | 3,942 (9.4%) |
| DropZone 위험 | 31,440 (74.6%) | 1,940 (4.6%) | 8,756 (20.8%) |

### V1 규칙 기반 결과와 V2 딥러닝 결과 비교

| 대상 | Accuracy | Danger Precision | Danger Recall | Danger F1 |
| --- | ---: | ---: | ---: | ---: |
| 전체 class 일치율 | 98.48% | - | - | - |
| 지게차 충돌 위험 | 99.22% | 95.98% | 95.64% | 95.81% |
| DropZone 위험 | 99.53% | 98.86% | 98.88% | 98.87% |

이 결과는 V2가 V1 teacher label을 잘 따라간다는 의미다. 아직 사람이 검수한 독립 정답이나 실제 충돌 ground truth로 검증한 것은 아니므로, V2가 V1보다 더 정확하다고 결론 내리면 안 된다.

## 생성/학습 명령어

```bash
conda activate venv

python -m model.fusion_v2.generate_scenarios \
  --output-root model/fusion_v2/generated_scenarios \
  --count-per-kind 35 \
  --seed 20260527 \
  --clean

python -m model.fusion_v2.dataset \
  --input-root simulation/Recordings/collision_scenarios model/fusion_v2/generated_scenarios \
  --output model/fusion_v2/data/fusion_v2_dataset.npz \
  --window-size 24 \
  --stride 2 \
  --augment 1 \
  --noise-std 0.02

python -m model.fusion_v2.train \
  --dataset model/fusion_v2/data/fusion_v2_dataset.npz \
  --output-dir model/fusion_v2/checkpoints \
  --epochs 15 \
  --batch-size 256 \
  --seed 42

python -m model.fusion_v2.evaluate \
  --dataset model/fusion_v2/data/fusion_v2_dataset.npz \
  --checkpoint model/fusion_v2/checkpoints/best.pt \
  --output-dir model/fusion_v2/reports
```

## 현재 한계

- 아직 V2 라벨은 사람이 직접 만든 정답이 아니라 V1 teacher label이다.
- 현재 데이터는 Unity 시나리오 수가 적기 때문에 과적합 가능성이 크다.
- 특히 DropZone danger 시나리오는 현재 1개뿐이라, 시나리오 단위 hold-out을 하면 train 또는 validation 한쪽에 DropZone danger가 비는 문제가 생긴다.
- V2가 V1보다 좋아졌다고 말하려면 시나리오별 hold-out 평가와 신규 시나리오 검증이 필요하다.
- 현 단계에서는 "V2 딥러닝 모델 실험 기반 구축"으로 보는 것이 맞다.

## 다음 단계

1. Unity에서 더 많은 시나리오를 생성한다.
2. 시나리오별로 `SAFE / WARNING / DANGER` 라벨 기준을 명확히 고정한다.
3. V2를 hold-out 시나리오에서 평가한다.
4. V1 규칙 기반 판단과 V2 딥러닝 판단의 first warning/danger frame을 비교한다.
5. 충분히 안정화되면 runtime에 `--risk-engine v1|v2` 옵션으로 선택 적용한다.
