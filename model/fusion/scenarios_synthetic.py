"""
합성 시나리오 24개 정의.

⚠️ 이 파일은 **실제 촬영 데이터가 들어오면 삭제 or 교체 대상**입니다.
실제 데이터 로더는 `scenarios_real.py` 같은 이름으로 새로 작성하고,
`scenario_generator.py::main()`의 import만 바꾸면 됩니다.

구성 (총 24개):
  - SAFE            : 4개  (안전 + false-alarm 필터)
  - 지게차 위험      : 10개 (작업자-지게차 충돌 관련)
  - 드롭존 위험      : 10개 (작업자-드롭존 진입 관련)

각 시나리오는 `scenario_generator.Scenario` 객체 반환.
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
        forklift=jitter(linear([-3, 1], [1, 1])),
        worker1=jitter(still([0, 5])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        name="s02_safe_worker_walk_no_fork",
        forklift=absent(),
        worker1=jitter(linear([0, 5], [0, 2])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        name="s03_safe_both_present_far",
        forklift=jitter(linear([-3, 1], [1, 1])),
        worker1=jitter(still([0, 4])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 20, 0)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 경계 밖 ~0.2m 스쳐 지나감 → false-alarm 필터용
        name="s04_safe_dropzone_brush_past",
        forklift=absent(),
        worker1=jitter(linear([-1.0, 3.5], [-1.0, 5.0])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 20, 0)]),
    ))

    # ========================================================
    # 지게차 위험 (10) — 작업자-지게차 충돌 관련
    # ========================================================
    scenarios.append(Scenario(
        name="s05_fork_slow_approach_static_worker",
        forklift=jitter(linear([-3, 1], [-0.5, 1])),
        worker1=jitter(still([0, 1.2])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 10, 0), (10, 16, 1), (16, 20, 2)]),
    ))
    scenarios.append(Scenario(
        name="s06_fork_normal_approach_walking_worker",
        forklift=jitter(linear([-3, 1], [0, 1])),
        worker1=jitter(linear([0, 3], [0, 1.1])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 8, 0), (8, 14, 1), (14, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 빠르게 4초 안에 접근 후 정지 (TTC 짧음)
        name="s07_fork_fast_unaware",
        forklift=jitter(linear([-3, 1], [0.5, 1], t1=4.0)),
        worker1=jitter(still([0, 1.2])),
        crane_state=crane_seq(0),
        audio=audio_trace(base=0.1),
        labels=labels_seq([(0, 2.5, 0), (2.5, 3.3, 1), (3.3, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 작업자(+Y 응시) 기준 좌측(-X)에서 접근
        name="s08_fork_left_collision",
        forklift=jitter(linear([-3, 1], [-0.3, 1])),
        worker1=jitter(still([0, 1])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 12, 0), (12, 17, 1), (17, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 작업자(+Y 응시) 기준 우측(+X)에서 접근
        name="s09_fork_right_collision",
        forklift=jitter(linear([1, 1], [0.3, 1])),
        worker1=jitter(still([0, 1])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 12, 0), (12, 17, 1), (17, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 후진, 작업자가 후방
        name="s10_fork_rear_backup",
        forklift=jitter(linear([-0.5, 1], [-1.5, 1])),
        worker1=jitter(still([-1, 1])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 8, 0), (8, 14, 1), (14, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 정면 충돌 — 지게차 전진 + 작업자 세로길 하강
        name="s11_fork_frontal_full",
        forklift=jitter(linear([-3, 1], [0, 1])),
        worker1=jitter(linear([0, 3], [0, 1.05])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 10, 0), (10, 15, 1), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 정지 상태, 작업자가 서서히 접근
        name="s12_fork_stop_worker_approach",
        forklift=jitter(still([-0.8, 1])),
        worker1=jitter(linear([0, 3], [0, 1.3])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 9, 0), (9, 15, 1), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차가 T-junction에서 방향 전환
        name="s13_fork_curve_at_junction",
        forklift=jitter(piecewise([
            (0, [-3, 1]),
            (8, [0, 1]),
            (14, [0.5, 1]),
            (20, [0.5, 1]),
        ])),
        worker1=jitter(still([0, 1.8])),
        crane_state=crane_seq(0),
        audio=audio_trace(),
        labels=labels_seq([(0, 6, 0), (6, 12, 1), (12, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 지게차 접근 + 오디오 이상음 복합
        name="s14_fork_approach_with_audio_spike",
        forklift=jitter(linear([-3, 1], [-0.3, 1])),
        worker1=jitter(still([0, 1])),
        crane_state=crane_seq(0),
        audio=audio_trace(base=0.1, spike=(10, 20, 0.7)),
        labels=labels_seq([(0, 10, 0), (10, 15, 1), (15, 20, 2)]),
    ))

    # ========================================================
    # 드롭존 위험 (10) — 작업자-드롭존 진입 관련
    # ========================================================
    scenarios.append(Scenario(
        # 남쪽(T-junction)에서 드롭존으로 이동
        name="s15_dz_enter_from_south_lifting",
        forklift=absent(),
        worker1=jitter(linear([0, 1], [-2, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 14, 0), (14, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 동쪽에서 측면 진입
        name="s16_dz_enter_from_east_lifting",
        forklift=absent(),
        worker1=jitter(linear([0, 3.5], [-2, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 12, 0), (12, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 북쪽에서 내려오며 진입
        name="s17_dz_enter_from_north_lifting",
        forklift=absent(),
        worker1=jitter(linear([-2, 5], [-2, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 10, 0), (10, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 대각선 진입
        name="s18_dz_enter_diagonal_lifting",
        forklift=absent(),
        worker1=jitter(linear([-0.5, 5], [-2, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 12, 0), (12, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 작업자가 드롭존 내부에 있고 중간에 인양 시작
        name="s19_dz_idle_to_lifting_worker_inside",
        forklift=absent(),
        worker1=jitter(still([-2, 3.5])),
        crane_state=crane_seq(0, changes=[(8, 1)]),
        audio=audio_trace(spike=(8, 20, 0.3)),
        labels=labels_seq([(0, 8, 1), (8, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 진입 + 오디오 이상음 멀티모달 결합
        name="s20_dz_entry_with_audio_spike",
        forklift=absent(),
        worker1=jitter(linear([0, 3.5], [-2, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.05, spike=(8, 20, 0.9)),
        labels=labels_seq([(0, 12, 0), (12, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 매우 느린 접근
        name="s21_dz_slow_creep_entry",
        forklift=absent(),
        worker1=jitter(linear([0, 3.5], [-2, 3.5], t0=0, t1=20)),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 15, 0), (15, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 빠르게 드롭존 통과 (짧지만 위험)
        name="s22_dz_quick_dash_through",
        forklift=absent(),
        worker1=jitter(piecewise([
            (0, [0, 3.5]),
            (6, [0, 3.5]),
            (10, [-3, 3.5]),
            (20, [-3, 3.5]),
        ])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 7, 0), (7, 9, 2), (9, 20, 0)]),
    ))
    scenarios.append(Scenario(
        # 드롭존 중심 깊숙이 진입
        name="s23_dz_deep_inside_lifting",
        forklift=absent(),
        worker1=jitter(linear([0, 3.5], [-2.3, 3.5])),
        crane_state=crane_seq(1),
        audio=audio_trace(base=0.3),
        labels=labels_seq([(0, 10, 0), (10, 20, 2)]),
    ))
    scenarios.append(Scenario(
        # 경계 근처 정지, 중간에 인양 시작
        name="s24_dz_static_edge_lift_starts",
        forklift=absent(),
        worker1=jitter(still([-1.3, 3.5])),
        crane_state=crane_seq(0, changes=[(10, 1)]),
        audio=audio_trace(spike=(10, 20, 0.4)),
        labels=labels_seq([(0, 10, 0), (10, 20, 1)]),
    ))

    return scenarios


if __name__ == "__main__":
    # 단독 실행 시 시나리오 목록 요약 출력
    sc = build_synthetic_24()
    print(f"built {len(sc)} synthetic scenarios")
    cat_safe = [s for s in sc if "_safe_" in s.name]
    cat_fork = [s for s in sc if "_fork_" in s.name]
    cat_dz = [s for s in sc if "_dz_" in s.name]
    print(f"  SAFE       : {len(cat_safe)}")
    print(f"  지게차 위험 : {len(cat_fork)}")
    print(f"  드롭존 위험 : {len(cat_dz)}")
