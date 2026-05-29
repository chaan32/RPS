# Code Cleanup Audit

작성일: 2026-05-29

## 범위

실제 소스 코드 기준으로 확인했다. 생성물, 의존성, 런타임 산출물은 코드 감사 범위에서 제외하고 삭제/보관 후보로 분리했다.

- 확인한 소스 파일: 170개
- 주요 범위: `server/`, `model/`, `input/`, `frontend/src/`, `firmware/`, `portfolio_site/`, `simulation/Assets/Scripts/`
- 제외 범위: `.git/`, `.idea/`, `frontend/node_modules/`, `frontend/dist/`, `simulation/Library/`, `simulation/Recordings/`, `metrics/`, `snapshots/`, `runs/`, `__pycache__/`, `.pio/`

## 주석 정리 원칙

- 서비스/런타임 코드는 한국어 주석으로 통일한다.
- 주석은 "무엇을 하는지"보다 "왜 필요한지", "현재 정책이 무엇인지"를 설명한다.
- 오래된 TODO는 남기지 않고 현재 정책 또는 후속 작업 조건으로 바꾼다.
- CLI/검증 스크립트의 `print()`는 사용자가 터미널에서 직접 확인하는 목적이므로 유지한다.
- FastAPI route handler는 데코레이터로 참조되기 때문에 정적 검색에서 미사용처럼 보여도 삭제하지 않는다.

## 이번에 정리한 주석

- `server/main.py`: 앱 진입점, request metric middleware, router 등록 설명을 현재 구조에 맞게 수정
- `server/lifespan.py`: DB migration, worker seed, Fusion subprocess, MQTT/Redis task 역할을 명확히 정리
- `server/pipeline/mqtt/mqtt_handler.py`: MQTT producer 역할을 module/class docstring으로 명시하고 임시성 주석 제거
- `server/service/maker_service.py`: `maker` 명칭이 legacy alias임을 명확히 표시
- `server/api/__init__.py`: Spring 비교 설명 제거, FastAPI 라우터 패키지 설명으로 정리
- `model/fusion/runtime/realtime_camera.py`: YAMNet 중심의 오래된 흐름 설명을 RTSP multi-view, YOLO, Homography, Fusion V1/V2 흐름으로 교체
- `frontend/src/components/DailyAdminDashboard.tsx`: report timing console group 안의 중복 debug log 제거

## 바로 삭제하면 안 되는 항목

| 항목 | 이유 | 정리 방향 |
| --- | --- | --- |
| `server/api/makers.py` | 기존 `/makers` API 호환용. 현재 DB migration과 legacy alias 흐름과 연결되어 있음 | 프론트/외부 호출이 모두 `/workers`로 이동한 뒤 삭제 |
| `server/service/maker_service.py`의 `create_maker`, `list_makers` | `/makers` alias가 사용 중이면 필요 | `/makers` 제거 시 같이 삭제 |
| `server/database/models.py`의 `Maker = Worker` | legacy import 호환용 | legacy import 제거 후 삭제 |
| `input/audio/`, `model/yamnet/`, `server/api/audio.py` | 현재 V2에서는 필수는 아니지만 ESP32/YAMNet 오디오 확장 경로가 남아 있음 | 포트폴리오에서 오디오를 제외할지 확정 후 archive 또는 제거 |
| `server/lifespan.py`의 MQTT task, `server/pipeline/mqtt/` | 아두이노/MQTT 경고 장치 연동 경로 | 하드웨어 알림을 유지하면 필요, 완전히 REST 발행만 쓰면 제거 후보 |
| `server/api/aruco.py`, `server/service/aruco_service.py` | 이미지 업로드 기반 ArUco 디버그 API | 운영 API에서 쓰지 않으면 calibration CLI만 남기고 제거 가능 |

## 삭제 또는 Git 추적 제외 후보

아래 항목은 코드라기보다 생성물/실험 산출물이다. 로컬에서 보관할 수는 있지만 Git에는 올리지 않는 것이 좋다.

| 후보 | 성격 | 권장 처리 |
| --- | --- | --- |
| `frontend/dist/` | 프론트 빌드 산출물 | `.gitignore` 처리, 배포 산출물은 release/artifact로 관리 |
| `metrics/**/*.jsonl`, `metrics/**/*.csv` | 벤치마크 결과 | 대표 결과만 `docs/`에 요약, 원본은 필요 시 archive |
| `snapshots/` | 위험 상황 이미지 저장 결과 | Git 제외, README에는 대표 이미지만 압축해서 사용 |
| `simulation/Recordings/` | Unity 녹화 프레임/영상 | 최종 GIF/MP4만 별도 assets로 관리 |
| `runs/`, `model/yolo/runs/` | YOLO 학습 산출물 | Git 제외, best 모델과 성능표만 관리 |
| `model/fusion_v2/data/*.npz` | V2 학습 데이터. GitHub 50MB 경고 발생 | DVC/LFS 또는 외부 다운로드 링크로 이동 |
| `*.zip` 데이터셋 | 학습용 압축 파일 | Git 제외, 재생성 스크립트와 다운로드 경로만 문서화 |
| `__pycache__/`, `.pytest_cache/`, `.pio/` | 캐시/빌드 내부 파일 | 삭제 가능, Git 제외 |

## 함수/모듈 삭제 후보

삭제 전 확인이 필요한 후보만 정리한다. 이번 작업에서는 실제 삭제하지 않았다.

| 후보 | 현재 상태 | 삭제 조건 |
| --- | --- | --- |
| `frontend/src/api.ts`의 `fetchMakers` | 현재 컴포넌트는 `fetchWorkers`를 사용한다. exported legacy 함수로만 남아 있음 | 외부 import가 없고 `/makers` API를 제거하기로 하면 삭제 |
| `server/api/makers.py` 전체 | legacy `/makers` endpoint | `/workers` 전환 완료 후 삭제 |
| `server/service/maker_service.py`의 maker alias 2개 | legacy endpoint support | `server/api/makers.py` 삭제 시 함께 삭제 |
| `mqtt_consumer` | 현재는 MQTT 수신 확인 로그만 수행 | MQTT inbound 메시지를 더 이상 사용하지 않으면 삭제 또는 lifespan에서 비활성화 |
| `server/api/audio.py`의 `/audio/score` | V2 기본 경로에서는 `--no-audio` 사용 | 오디오 확장을 포트폴리오 범위에서 제외하면 제거 |
| `input/audio/run_yamnet_only.py` | 오디오 단독 실행 도구 | 오디오 실험을 보관하지 않기로 하면 archive |
| `server/api/aruco.py` | 디버그성 이미지 업로드 API | calibration을 CLI로만 운영하기로 하면 제거 |

## 남은 결정 사항

1. `/makers` 호환 API를 계속 유지할지, `/workers`로 완전히 정리할지 결정해야 한다.
2. YAMNet/ESP32 오디오 경로를 포트폴리오 Version 1/2 범위에 포함할지 결정해야 한다.
3. 대용량 학습 데이터와 벤치마크 산출물을 Git LFS/DVC/외부 스토리지 중 어디로 옮길지 결정해야 한다.
4. `simulation/Assets/Scripts/`는 Unity Editor menu와 직접 연결되므로 Unity에서 메뉴가 사라져도 되는지 확인한 뒤에만 정리해야 한다.
