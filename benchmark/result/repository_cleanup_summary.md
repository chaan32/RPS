# Repository Cleanup Summary

정리 기준은 "GitHub에 보여줄 문서와 대표 산출물은 남기고, 재생성 가능한 학습/실험 산출물은 제외"입니다.

## 정리 완료

| 영역 | 처리 |
| --- | --- |
| `docs/upload/` | GitHub 업로드용 README, Fusion V2 문서, 성능 지표 문서, asset manifest 생성 |
| `docs/upload/assets/` | README에 필요한 시스템 아키텍처, Fusion 구조 이미지, GIF, 대표 포트폴리오 이미지 복사 |
| `benchmark/result/` | Fusion V2 데이터셋 요약, 성능 벤치마크 요약, repository cleanup 요약 생성 |
| `benchmark/file/metrics/` | 성능 근거로 필요한 선별 metric 파일만 복사 |
| `benchmark/file/fusion_v2/` | 큰 `.npz` 데이터셋을 로컬 archive로 복사 |
| `simulation/Recordings/` | 최종 시나리오 1~4만 남기고 과거 진단/중간 녹화/이전 시나리오 제거 |
| `runs/`, `model/yolo/runs/` | YOLO 학습 중간 산출물 제거 |

## 최종 유지된 Unity 시나리오

| 경로 | 설명 |
| --- | --- |
| `simulation/Recordings/collision_scenarios/scenario_01_user_current` | 사용자 커스텀 배치 기반 충돌 위험 |
| `simulation/Recordings/collision_scenarios/scenario_02_swapped_positions` | 작업자/지게차 위치 반대 구도 |
| `simulation/Recordings/collision_scenarios/scenario_03_opposite_worker` | 반대 방향 접근 구도 |
| `simulation/Recordings/collision_scenarios/scenario_04_box_dropzone` | 인양물 DropZone 위험 구도 |

## Git 관리 정책

| 항목 | 정책 |
| --- | --- |
| `docs/upload/**` | GitHub 업로드 문서이므로 tracking 대상 |
| `benchmark/result/*.md` | 요약 문서이므로 tracking 대상 |
| `benchmark/file/metrics/**` | 선별된 작은 근거 파일이므로 tracking 대상 |
| `benchmark/file/fusion_v2/*.npz` | 50MB 이상 대형 학습 데이터라 Git tracking 제외 |
| `simulation/` | Unity 프로젝트/녹화 산출물이 커서 기본 Git tracking 제외 |
| `runs/`, `model/yolo/runs/` | 재생성 가능한 학습 산출물이므로 Git tracking 제외 |

## 주의

`portfolio_site/` 삭제 상태는 이번 정리 전부터 워킹트리에 존재하던 변경입니다. 이 정리 작업에서는 해당 디렉터리를 복구하거나 추가 수정하지 않았습니다.

