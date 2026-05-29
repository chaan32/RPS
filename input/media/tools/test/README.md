# Media Test And Diagnostic Tools

이 폴더는 운영 런타임 파일을 분류하려는 목적이 아니라, 검증/진단/평가용 스크립트만 한곳에 묶기 위한 공간이다.

상위 폴더 `input/media/tools/`에는 현재 실행 흐름에서 계속 쓰는 도구를 그대로 둔다.

## 파일 역할

- `audit_yolo_dataset_labels.py`: Unity synthetic YOLO label bbox가 비정상적으로 큰지 샘플링해서 확인한다.
- `check_blindspot_recording.py`: 녹화된 cam1/cam2 프레임 또는 mp4를 YOLO + Homography로 통과시켜 좌표 CSV와 진단 영상을 만든다.
- `evaluate_box_center_methods.py`: 공중/인양 박스의 중심 좌표 산출 방식을 비교한다.
- `render_blindspot_bev.py`: cam1/cam2 탐지 결과와 BEV 좌표를 한 화면 영상으로 렌더링한다.
- `render_collision_fusion_diagnostics.py`: collision scenario를 offline으로 돌려 cam1/cam2, BEV, fusion risk를 함께 렌더링한다.
- `show_dual_cam.py`: YOLO 없이 RTSP 두 채널이 열리는지만 화면으로 확인한다.
- `summarize_pipeline_metrics.py`: realtime benchmark JSONL을 읽어 모듈별 mean/p50/p95/max latency를 집계한다.
- `verify_homography.py`: calibration homography 결과를 격자/축 오버레이로 시각 검증한다.

## 남겨둔 상위 tools

- `stream_collision_scenario_rtsp.py`: Unity scenario를 MediaMTX RTSP로 publish하는 현재 E2E 검증/시연 입력 브릿지다.
- `install_custom_yolo_model.py`: Colab 학습 결과 모델을 프로젝트 모델 경로에 설치한다.
- `identify_markers.py`: ArUco marker 식별과 calibration 확인에 사용한다.
- `relabel_unity_dataset_by_color.py`: Unity dataset label 보정에 사용한다.
