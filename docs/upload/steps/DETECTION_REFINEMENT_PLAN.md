# Detection Refinement Layer Plan

## 현재 문제

Unity 녹화 영상에서 worker와 forklift가 가까워지는 구간에 YOLO pose 결과가 여러 개의 worker 후보를 만든다. 특히 forklift의 노란 프레임, 운전석, 수직 구조물이 사람 관절처럼 오인되면서 같은 프레임 안에 worker bbox가 2~3개씩 생긴다.

현재 흐름은 아래와 같다.

```text
cam1/cam2 frame
→ YOLO pose + custom YOLO raw detections
→ homography world 좌표 변환
→ pick_positions
→ BEV/fusion/early warning
```

`pick_positions()`는 fusion 입력용으로 최고 confidence worker 후보를 고르지만, 그 전 단계의 raw worker 후보가 overlay에 그대로 남아 있다. 그래서 화면에는 여러 개의 노란 worker bbox가 forklift 위에 겹쳐 보인다. 더 중요한 점은 raw 후보가 많아질수록 frame-by-frame 좌표가 흔들리고, fusion 입력도 불안정해질 가능성이 커진다.

## 근본 원인

1. `POSE_CONF_THRESHOLD = 0.01`로 아주 낮다.
   - 멀리 있거나 작게 보이는 worker를 살리기 위한 설정이다.
   - 대신 확신도 1~2%짜리 false positive까지 통과한다.

2. worker와 forklift가 충돌 직전까지 가까워진다.
   - forklift 구조물과 worker 신체가 이미지상 겹친다.
   - YOLO pose가 forklift 프레임을 사람 후보로 오인하기 쉽다.

3. raw detection과 fusion-ready detection이 분리되어 있지 않다.
   - 현업에서는 detector 결과를 바로 fusion에 넣지 않는다.
   - 후보 필터링, tracking, multi-view association을 거친 뒤 global track만 사용한다.

## 목표 구조

이번 단계에서 목표로 하는 구조는 아래와 같다.

```text
cam1/cam2 raw YOLO detections
→ DetectionRefiner
   - low-confidence false positive 제거
   - forklift와 과도하게 겹치는 worker 후보 제거
   - 카메라별 대표 worker 후보 1개 선택
   - 카메라별 대표 forklift/box 후보 선택
→ cross-camera/global position selection
→ BEV/fusion/early warning
```

장기적으로는 `DetectionRefiner` 뒤에 Kalman Filter 기반 global tracker를 붙이는 것이 맞다. 하지만 현재 Unity 벤치마크는 단일 worker와 단일 forklift 시나리오이므로, 1차 구현은 후보 정제 계층을 먼저 넣는다.

## 1차 구현 범위

1. `input/media/pipeline/refinement.py` 추가
   - `DetectionRefiner` 클래스 생성.
   - 입력: `{"cam1": detections, "cam2": detections}`.
   - 출력: 같은 구조의 refined detections.

2. worker 후보 정제
   - confidence가 너무 낮은 미식별 worker 후보 제거.
   - forklift bbox와 많이 겹치고 confidence가 낮은 worker 후보 제거.
   - ArUco/track/cross-camera ID가 있는 worker는 우선 보존.
   - 한 카메라에 worker 후보가 여러 개면 가장 신뢰도 높은 1개만 남김.

3. threat 후보 정제
   - forklift, box_1, box_2는 카메라별/클래스별 최고 confidence 후보만 남김.
   - 같은 물체가 중복 bbox로 그려지는 것을 줄임.

4. 적용 위치
   - offline diagnostic: `render_collision_fusion_diagnostics.py`
   - realtime runtime: `model/fusion/runtime/realtime_camera.py`
   - blindspot BEV/debug 영상: `render_blindspot_bev.py`
   - basic live runner: `input/media/pipeline/runner.py`

## 기대 효과

1. 카메라 overlay에서 forklift 주변에 worker bbox가 여러 개 겹치는 문제 감소.
2. BEV/fusion 입력으로 들어가는 worker 좌표의 프레임 단위 흔들림 감소.
3. early warning과 fusion risk가 raw false positive에 덜 흔들림.
4. 이후 Kalman/global tracker를 붙일 수 있는 구조적 기반 확보.

## 남는 한계

이 계층은 detector 자체를 재학습하는 것이 아니다. 따라서 worker가 완전히 가려지거나 YOLO pose가 실제 worker를 아예 검출하지 못하는 프레임은 완벽하게 복구할 수 없다. 그런 경우는 카메라 위치, worker 크기, pose 모델 fine-tuning, occlusion 시나리오 설계로 별도 개선해야 한다.

또한 현재 1차 구현은 단일 worker 벤치마크에 맞춘 보수적 정제다. 실제 다중 작업자 환경으로 확장하려면 worker ArUco ID 또는 appearance/trajectory 기반 multi-object association이 추가로 필요하다.

## 적용 결과

이번 1차 구현은 아래 파일에 반영했다.

- `input/media/pipeline/refinement.py`
- `input/media/pipeline/__init__.py`
- `input/media/pipeline/runner.py`
- `input/media/tools/test/check_blindspot_recording.py`
- `input/media/tools/test/render_blindspot_bev.py`
- `input/media/tools/test/render_collision_fusion_diagnostics.py`
- `model/fusion/runtime/realtime_camera.py`

검증은 `simulation/Recordings/collision_scenarios`의 3개 충돌 시나리오로 수행했다.

문제가 보였던 cam1 84프레임 기준 worker 후보 수는 다음처럼 줄었다.

```text
scenario_01_center_crossing: raw 3 → refined 1
scenario_02_dropzone_approach: raw 2 → refined 1
scenario_03_blind_corner_merge: raw 3 → refined 1
```

전체 fusion 진단 결과는 다음과 같다.

```text
scenario_01_center_crossing: both_frames 120/120, first_warning_frame 35, first_danger_frame 72
scenario_02_dropzone_approach: both_frames 120/120, first_warning_frame 12, first_danger_frame 60
scenario_03_blind_corner_merge: both_frames 120/120, first_warning_frame 40, first_danger_frame 85
```

생성된 확인 영상은 각 시나리오의 `diagnostics/fusion/fusion_cameras_bev.mp4`에 저장된다.

## 추가 확인: cam1/cam2 좌표 불일치

추가 검증 중 같은 worker에 대해 cam1과 cam2가 서로 다른 world 좌표를 내는 문제가 확인됐다. 캘리브레이션 reprojection error는 cam1 평균 약 0.45cm, cam2 평균 약 0.57cm 수준이라 homography 자체보다는, worker의 발/발목 기준점이 occlusion 때문에 잘못 잡히는 문제가 더 크다.

측정 결과는 다음과 같았다.

```text
cam1 worker error vs ground truth: mean 1.560m, median 1.387m, max 4.982m
cam2 worker error vs ground truth: mean 0.363m, median 0.180m, max 3.236m
cam1/cam2 worker disagreement: mean 1.370m, median 1.162m, max 4.823m
```

따라서 `DetectionRefiner`에 cross-camera disagreement 처리를 추가했다.

- 두 카메라 worker 좌표 차이가 `1.25m` 이하이면 둘 다 유지한다.
- `1.25m`를 넘으면 같은 worker로 보기 어렵거나 한쪽 기준점이 깨진 것으로 판단한다.
- 이때 `refine_score`가 높은 카메라의 worker만 남긴다.
- 점수 차이가 작으면 현재 Unity 벤치마크에서 worker를 더 안정적으로 보는 `cam2`를 우선한다.

적용 후 global fusion worker 좌표와 Unity ground truth의 오차는 다음과 같다.

```text
scenario_01_center_crossing: mean 0.483m, median 0.165m, p95 1.774m
scenario_02_dropzone_approach: mean 0.359m, median 0.203m, p95 1.242m
scenario_03_blind_corner_merge: mean 0.298m, median 0.178m, p95 1.003m
```

아직 충돌 직전 occlusion 구간에서는 1m 이상 튀는 프레임이 남아 있다. 다음 단계는 global tracker/Kalman smoothing을 붙여 한 프레임 좌표 튐을 이전 궤적으로 완화하는 것이다.

## Global Tracking / Smoothing 적용

좌표 튐 완화를 위해 `model/fusion/runtime/global_tracker.py`를 추가했다. 이 계층은 `DetectionRefiner`와 `pick_positions()` 뒤에서 동작하며, fusion/BEV/early warning에는 smoothed world 좌표를 전달한다.

```text
YOLO raw detections
→ DetectionRefiner
→ pick_positions
→ GlobalTrackManager
→ BEV / fusion / early warning
```

적용 범위는 다음과 같다.

- offline fusion diagnostic: `input/media/tools/test/render_collision_fusion_diagnostics.py`
- realtime fusion runtime: `model/fusion/runtime/realtime_camera.py`
- blindspot BEV diagnostic: `input/media/tools/test/render_blindspot_bev.py`

동작 방식은 다음과 같다.

- worker, forklift, dropzone을 각각 별도 world-coordinate track으로 유지한다.
- 새 측정값이 예측 위치에서 크게 벗어나면 outlier로 보고 약하게만 반영한다.
- 측정값이 정상 범위면 현재 측정값을 더 강하게 반영한다.
- 짧은 누락은 이전 속도 기반 예측으로 hold한다.

CSV에는 raw 좌표와 smoothed 좌표를 모두 저장한다.

- `raw_worker_x`, `raw_worker_y`
- `worker_x`, `worker_y`
- `raw_forklift_x`, `raw_forklift_y`
- `forklift_x`, `forklift_y`
- `worker_tracker_residual_m`, `worker_tracker_outlier`
- `forklift_tracker_residual_m`, `forklift_tracker_outlier`

또한 `coordinate_eval_summary.csv`와 각 시나리오별 `coordinate_eval.csv`를 생성해 ground truth 대비 좌표 성능을 자동 평가한다.

적용 후 worker 좌표 오차는 다음처럼 개선됐다.

```text
scenario_01_center_crossing: raw mean 0.483m → smoothed mean 0.435m, p95 1.774m → 1.345m
scenario_02_dropzone_approach: raw mean 0.359m → smoothed mean 0.325m, p95 1.242m → 1.127m
scenario_03_blind_corner_merge: raw mean 0.298m → smoothed mean 0.284m, p95 1.003m → 0.964m
```

알림 표기는 다음처럼 정리했다.

- 주황색: `WARNING`
- 빨간색: `DANGER`

`fusion_summary.csv`도 `caution_*` 대신 `warning_*` 필드를 사용한다.
