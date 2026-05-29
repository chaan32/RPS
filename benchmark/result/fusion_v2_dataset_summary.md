# Fusion V2 Dataset Archive Summary

이 문서는 GitHub에 직접 올리기 큰 Fusion V2 학습 데이터(`.npz`)의 의미와 보관 위치를 정리한 요약입니다.

## 보관 위치

| 파일 | 원본 위치 | 보관 위치 | 크기 | 용도 |
| --- | --- | --- | ---: | --- |
| `fusion_v2_dataset.npz` | `model/fusion_v2/data/fusion_v2_dataset.npz` | `benchmark/file/fusion_v2/fusion_v2_dataset.npz` | 38.47 MB | V1 teacher label 기반 초기 V2 학습 데이터 |
| `fusion_v2_geometry_future_dataset.npz` | `model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz` | `benchmark/file/fusion_v2/fusion_v2_geometry_future_dataset.npz` | 81.33 MB | 최종 V2 geometry future label 학습 데이터 |

## `.npz` 파일의 의미

`.npz`는 NumPy 배열을 여러 개 묶어 저장한 압축 파일입니다. Fusion V2에서는 프레임별 좌표 feature를 24프레임 sliding window로 묶고, 각 window의 위험 라벨을 함께 저장합니다.

| 배열 | 설명 |
| --- | --- |
| `x` | 모델 입력. `(window_count, 24, 23)` 형태의 좌표 시계열 feature |
| `y` | 모델 정답. `[forklift_target, dropzone_target]` 위험 라벨 |
| `scenario` | window가 나온 시나리오 이름 |
| `worker_id` | 해당 window의 작업자 id |
| `end_frame` | window 마지막 프레임 번호 |
| `augmented` | noise augmentation 여부 |
| `meta` | window size, stride, label mode, threshold 등 데이터셋 생성 설정 |

## 최종 학습 데이터

최종 포트폴리오 기준 데이터셋은 `fusion_v2_geometry_future_dataset.npz`입니다.

| 항목 | 값 |
| --- | ---: |
| 실제 Unity 녹화 시나리오 | 7개 |
| 절차적 synthetic 시나리오 | 450개 |
| 전체 시나리오 소스 | 457개 |
| 학습 window 수 | 89,176개 |
| window 크기 | 24 frames |
| stride | 2 |
| feature dimension | 23 |
| label mode | `geometry_future` |
| future horizon | 12 frames |
| augmentation | window당 noisy copy 1개 |
| noise std | 0.02 |

## Label 기준

`geometry_future` 데이터는 V1의 위험 점수를 정답으로 쓰지 않고, 절대좌표 기반 미래 위험 조건으로 label을 생성했습니다.

| Target | Warning | Danger | Future Horizon |
| --- | ---: | ---: | ---: |
| Forklift | 작업자와 forklift/FH 기준점 거리 <= 2.4m | <= 1.25m | 다음 12 frames |
| DropZone | 작업자와 dropzone 거리 <= 2.8m | <= 2.0m | 다음 12 frames |

## Git 관리 권장

`fusion_v2_geometry_future_dataset.npz`는 81MB라 GitHub 권장 파일 크기 50MB를 넘습니다. 따라서 일반 Git tracking 대상이 아니라 다음 방식으로 관리하는 것이 좋습니다.

- 로컬 보관: `benchmark/file/fusion_v2/`
- 원격 보관: Git LFS, DVC, 또는 외부 다운로드 링크
- GitHub README에는 데이터 생성 방식과 성능 지표만 문서화

