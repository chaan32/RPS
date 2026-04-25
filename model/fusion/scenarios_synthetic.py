"""
합성 시나리오 24개 정의 — 2m × 3m 작업공간 (ArUco 27/22/38/24 기준).

⚠️ 이 파일은 **실제 촬영 데이터가 들어오면 삭제 or 교체 대상**입니다.
실제 데이터 로더는 `scenarios_real.py` 같은 이름으로 새로 작성하고,
`scenario_generator.py::main()`의 import만 바꾸면 됩니다.

좌표계 (월드 좌표, 단위: m):
  ArUco 4개로 정의된 사각 작업공간:
    27번 (-2, 0)   38번 (0, 0)
    22번 (-2, 3)   24번 (0, 3)
  실측 가로 2m × 세로 3m

  레이아웃:
    - worker 라인  : 박스 안 수직, x ≈ -0.3
    - forklift 라인 : 박스 하단(y ≈ 0), x ∈ [-2, 0]
    - 드롭존       : center (-1.5, 2.0), radius 0.3

거리 임계 (작업공간 축소 반영):
    - forklift danger : dist < 0.4 m
    - forklift warn   : 0.4 ≤ dist < 0.9 m AND 접근 중
    - dropzone radius : 0.3 m
    - dz warn buffer  : 0.2 m

구성 (총 24개):
  - SAFE          : 4개  (안전 + false-alarm 필터)
  - 지게차 위험    : 10개 (작업자-지게차 충돌 관련)
  - 드롭존 위험    : 10개 (작업자-드롭존 진입 관련)
"""

from __future__ import annotations

import numpy as np

from scenario_generator import (
    Scenario,
    linear,
    piecewise,
    still,
    absent,
    jitter,
    audio_trace,
    crane_seq,
    labels_seq,
)


def build_synthetic_24(seed: int = 42) -> list[Scenario]:
    """24개 합성 시나리오 빌드. 랜덤 요소(jitter, audio noise)는 seed로 고정."""
    np.random.seed(seed)

    scenarios: list[Scenario] = []

    # ========================================================
    # SAFE (4) — 위험 없음 + false-alarm 필터
    # ========================================================
    scenarios.append(Scenario(
        name="s01_safe_fork_pass_worker_far",
        forklift=jitter(linear([-2, 0], [0, 0])),
        worker1=jitter(still([-0.3, 2.7])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        name="s02_safe_worker_walk_no_fork",
        forklift=absent(),
        worker1=jitter(linear([-0.3, 2.7], [-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        name="s03_safe_both_present_far",
        forklift=jitter(linear([-2, 0], [0, 0])),
        worker1=jitter(still([-0.3, 2.0])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 동측 ~0.7m 거리로 스쳐 지나감 → false-alarm 필터
        name="s04_safe_dropzone_brush_past",
        forklift=absent(),
        worker1=jitter(linear([0, 2.0], [-0.8, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 20, 0)]),
    ))

    # ========================================================
    # 지게차 위험 (10)
    # ========================================================
    scenarios.append(Scenario(
        # 서행 접근, 정지 작업자 — 끝 시점 dist ≈ 0.3m
        name="s05_fork_slow_approach_static_worker",
        forklift=jitter(linear([-2, 0], [-0.3, 0])),
        worker1=jitter(still([-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 10, 0), (10, 15, 1), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 일반 속도 접근, 작업자 동시 이동
        name="s06_fork_normal_approach_walking_worker",
        forklift=jitter(linear([-2, 0], [-0.3, 0])),
        worker1=jitter(linear([-0.3, 1.5], [-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 10, 0), (10, 15, 1), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 빠른 통과 (4초 안에 작업공간 횡단), 작업자 미인지
        name="s07_fork_fast_unaware",
        forklift=jitter(linear([-2, 0], [0, 0], t1=4.0)),
        worker1=jitter(still([-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(base=0.1),
        labels=labels_seq([(0, 2.5, 0), (2.5, 3.2, 1), (3.2, 4.0, 2), (4.0, 20, 1)]),
    ))
    scenarios.append(Scenario(
        # 작업자 +Y 응시, 지게차가 -X(서)에서 접근 → 우측 라인 통과
        name="s08_fork_collision_from_west",
        forklift=jitter(linear([-2, 0], [-0.8, 0])),
        worker1=jitter(still([-1.0, 0.2])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 11, 0), (11, 16, 1), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차가 +X(동)에서 접근
        name="s09_fork_collision_from_east",
        forklift=jitter(linear([0, 0], [-1.2, 0])),
        worker1=jitter(still([-1.0, 0.2])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 11, 0), (11, 16, 1), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 후진 (서쪽으로 reverse), 작업자가 후진 경로 상에 있음
        name="s10_fork_rear_backup",
        forklift=jitter(linear([-0.5, 0], [-1.5, 0])),
        worker1=jitter(still([-1.7, 0])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 8, 0), (8, 14, 1), (14, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 정면 충돌 — 지게차 동진 + 작업자 남행, 둘 다 (-0.3, ~0)에서 만남
        name="s11_fork_frontal_full",
        forklift=jitter(linear([-2, 0], [-0.3, 0])),
        worker1=jitter(linear([-0.3, 1.5], [-0.3, 0.15])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 12, 0), (12, 16, 1), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 정지, 작업자가 접근
        name="s12_fork_stop_worker_approach",
        forklift=jitter(still([-0.3, 0])),
        worker1=jitter(linear([-0.3, 1.8], [-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 11, 0), (11, 16, 1), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # T-junction에서 지게차가 가로길 → 세로길로 turn (worker 방향으로)
        name="s13_fork_curve_at_junction",
        forklift=jitter(piecewise([
            (0, [-2, 0]),
            (10, [-0.3, 0]),
            (20, [-0.3, 1.0]),
        ])),
        worker1=jitter(still([-0.3, 1.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 13, 0), (13, 17, 1), (17, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 접근 + 오디오 이상음 (멀티모달 결합)
        name="s14_fork_approach_with_audio_spike",
        forklift=jitter(linear([-2, 0], [-0.3, 0])),
        worker1=jitter(still([-0.3, 0.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(base=0.1, spike=(10, 20, 0.7)),
        labels=labels_seq([(0, 10, 0), (10, 15, 1), (15, 20, 2)]),
    ))

    # ========================================================
    # 드롭존 위험 (10)
    # 드롭존 center=(-1.5, 2.0), radius=0.3
    # ========================================================
    scenarios.append(Scenario(
        # 남쪽(T-junction)에서 드롭존 방향으로 이동
        name="s15_dz_enter_from_south_lifting",
        forklift=absent(),
        worker1=jitter(linear([-0.3, 1.5], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 15, 0), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 동쪽 측면 진입
        name="s16_dz_enter_from_east_lifting",
        forklift=absent(),
        worker1=jitter(linear([0, 2.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 16, 0), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 북쪽에서 내려오며 진입
        name="s17_dz_enter_from_north_lifting",
        forklift=absent(),
        worker1=jitter(linear([-1.5, 3.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 14, 0), (14, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 대각선 진입
        name="s18_dz_enter_diagonal_lifting",
        forklift=absent(),
        worker1=jitter(linear([-0.5, 3.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 15, 0), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 작업자가 드롭존 내부 정지, 중간에 인양 시작
        name="s19_dz_idle_to_lifting_worker_inside",
        forklift=absent(),
        worker1=jitter(still([-1.5, 2.0])),
        crane_state=crane_seq(0, changes=[(8, 1)]),
        audio=audio_trace(spike=(8, 20, 0.3)),
        labels=labels_seq([(0, 8, 1), (8, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 진입 + 오디오 이상음
        name="s20_dz_entry_with_audio_spike",
        forklift=absent(),
        worker1=jitter(linear([0, 2.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.05, spike=(8, 20, 0.9)),
        labels=labels_seq([(0, 16, 0), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 매우 느린 접근 (slow creep)
        name="s21_dz_slow_creep_entry",
        forklift=absent(),
        worker1=jitter(linear([-0.3, 2.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 15, 0), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 빠른 통과 (drop-zone 잠시 진입 후 탈출)
        name="s22_dz_quick_dash_through",
        forklift=absent(),
        worker1=jitter(piecewise([
            (0, [-0.3, 2.0]),
            (5, [-0.3, 2.0]),
            (10, [-2.0, 2.0]),
            (20, [-2.0, 2.0]),
        ])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 7.5, 0), (7.5, 9.5, 2), (9.5, 20, 0)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 중심 깊숙이 진입 (정중앙 도달)
        name="s23_dz_deep_inside_lifting",
        forklift=absent(),
        worker1=jitter(linear([-0.3, 2.0], [-1.5, 2.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 15, 0), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 경계 안쪽 정지 → 중간에 인양 시작 (idle warn → lift danger)
        name="s24_dz_static_edge_lift_starts",
        forklift=absent(),
        worker1=jitter(still([-1.25, 2.0])),
        crane_state=crane_seq(0, changes=[(10, 1)]),
        audio=audio_trace(spike=(10, 20, 0.4)),
        labels=labels_seq([(0, 10, 1), (10, 20, 2)]),
    ))

    return scenarios


if __name__ == "__main__":
    sc = build_synthetic_24()
    print(f"built {len(sc)} synthetic scenarios")
    cat_safe = [s for s in sc if "_safe_" in s.name]
    cat_fork = [s for s in sc if "_fork_" in s.name]
    cat_dz = [s for s in sc if "_dz_" in s.name]
    print(f"  SAFE       : {len(cat_safe)}")
    print(f"  지게차 위험 : {len(cat_fork)}")
    print(f"  드롭존 위험 : {len(cat_dz)}")
