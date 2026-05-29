# Upload Asset Manifest

`docs/upload`는 GitHub 업로드용 문서와 문서에서 참조하는 대표 assets만 모은 폴더입니다.

## Documents

| 파일 | 설명 |
| --- | --- |
| `README.md` | GitHub 첫 화면용 프로젝트 소개 |
| `PERFORMANCE_METRICS.md` | 성능 지표 요약 |
| `FUSION_V2.md` | Fusion V2 딥러닝 모델 상세 |
| `ASSET_MANIFEST.md` | 업로드 asset 목록 |

## Architecture / Model Images

| 파일 | 설명 |
| --- | --- |
| `assets/system_architecture_main.png` | 전체 시스템 아키텍처 이미지 |
| `assets/system_architecture_sub.png` | 상세 처리 흐름 이미지 |
| `assets/fusion_v2_gru_window_flow.png` | Fusion V2 GRU sliding window 흐름 이미지 |
| `assets/fusion_v2_gru_window_flow.svg` | Fusion V2 GRU 흐름 원본 SVG |
| `assets/fusion_model_structure_no_audio.png` | 음성 입력 제외 Fusion 구조 이미지 |
| `assets/fusion_model_structure_no_audio.svg` | 음성 입력 제외 Fusion 구조 원본 SVG |
| `assets/model_metrics_and_fusion_structure.md` | 모델 성능 및 Fusion 구조 설명 원문 |

## Demo GIFs

| 파일 | 설명 |
| --- | --- |
| `assets/gifs/scenario_01_validation.gif` | Scenario 01 검증용 입력 |
| `assets/gifs/scenario_01_result.gif` | Scenario 01 모델 적용 결과 |
| `assets/gifs/scenario_02_validation.gif` | Scenario 02 검증용 입력 |
| `assets/gifs/scenario_02_result.gif` | Scenario 02 모델 적용 결과 |
| `assets/gifs/scenario_03_validation.gif` | Scenario 03 검증용 입력 |
| `assets/gifs/scenario_03_result.gif` | Scenario 03 모델 적용 결과 |
| `assets/gifs/fusion_v1_scenario_01.gif` | Fusion V1 결과 비교 |
| `assets/gifs/fusion_v2_scenario_01.gif` | Fusion V2 결과 비교 |

## Portfolio Images

| 파일 | 설명 |
| --- | --- |
| `assets/portfolio/rps_model_detection_overview.jpg` | cam1/cam2 검출 및 BEV 결과 예시 |
| `assets/portfolio/yolo_pose_worker_example.jpg` | YOLO-Pose 작업자 검출 예시 |
| `assets/portfolio/custom_yolo_validation_predictions_web.jpg` | Custom YOLO 검증 예시 |
| `assets/portfolio/rps_dropzone_danger_example.jpg` | DropZone 위험 판단 예시 |
| `assets/portfolio/fusion_training_summary.png` | Fusion 학습/평가 요약 이미지 |

## Benchmark Files

문서에 직접 넣기 큰 원본 측정 파일은 `benchmark/file/metrics/`에 보관했습니다.

| 경로 | 설명 |
| --- | --- |
| `benchmark/file/metrics/REPORT.md` | 최종 pose skip/cache 벤치마크 리포트 |
| `benchmark/file/metrics/aggregate_by_mode.csv` | 모드별 평균 성능 |
| `benchmark/file/metrics/aggregate_by_mode_scenario.csv` | 모드/시나리오별 평균 성능 |
| `benchmark/file/fusion_v2/*.npz` | Fusion V2 학습 데이터 archive |
