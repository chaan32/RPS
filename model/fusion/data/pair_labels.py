"""
Pair-level 위험 라벨 생성: Scenario → (T, N, K) risk matrix.

Pairwise Interaction Fusion Model의 학습 target.
N = worker 수 (현재 1), K = threat 수 (forklift, dropzone) = 2.

라벨 값 (soft, BCE 학습용):
  0.0  = safe
  0.5  = warning
  1.0  = danger

규칙 (간단한 룰 기반):

[worker × forklift]
  - 지게차 부재(NaN)               → 0.0
  - 거리 < 0.9 m                   → 1.0  (danger)
  - 0.9 ≤ 거리 < 2.0 AND 접근 중   → 0.5  (warning)
  - 그 외                          → 0.0

[worker × dropzone]
  - 내부 (signed_dist < 0) AND 인양 중       → 1.0  (danger)
  - 내부 AND idle                            → 0.5  (warning, 인양 시작 전 사전 경고)
  - 0 ≤ signed_dist < 0.5 AND 접근 AND 인양  → 0.5  (warning, 경계 접근)
  - 그 외                                    → 0.0

threat 인덱스 (graph_input.py와 동일):
  [0] forklift
  [1] dropzone
"""

from __future__ import annotations

import numpy as np

from .scenario_generator import (
    Scenario,
    RATE,
    DZ_CENTER,
    DZ_RADIUS,
)
from ..graph_input import THREAT_FORKLIFT, THREAT_DROPZONE, K_THREATS


# ── 임계값 (작업공간 2m × 3m 기준) ────────────────
FORK_DANGER_DIST = 0.4   # m, 이하면 danger
FORK_WARN_DIST = 0.9     # m, danger~warn 사이
APPROACH_SPEED = 0.05    # m/s, 접근 판정 임계 (작은 공간이라 낮춤)
DZ_WARN_BUFFER = 0.2     # m, 드롭존 경계 밖 warning 버퍼

# 오디오 격상 임계 (YAMnet max_sim 기준)
AUDIO_WARN_THRESHOLD = 0.4    # 이 이상이면 최소 warning
AUDIO_DANGER_THRESHOLD = 0.65 # 이 이상이면 최소 danger

DT = 1.0 / RATE          # time-step (s)


# ── helper ──────────────────────────────────────────
def _velocity(pos: np.ndarray) -> np.ndarray:
    """전진 차분 (T, 2) → (T, 2). NaN 위치 그대로 NaN 반환."""
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / DT
        vel[0] = vel[1]
    return vel


def _closing_speed(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """
    a 기준 b 방향으로의 닫히는 속도 (양수 = 접근 중).
    Returns: (T,) float
    """
    rel = pos_b - pos_a                           # b - a
    dist = np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-6
    unit = rel / dist                             # (T, 2) unit vector a→b
    vel_a = _velocity(pos_a)
    vel_b = _velocity(pos_b)
    rel_vel = vel_a - vel_b                       # 상대속도 (a 기준 b가 어떻게 보이나)
    # closing = 작업자가 위협 쪽으로 가는 속도 - 위협이 작업자에게서 멀어지는 속도
    # 즉, a가 b로 다가가는 속도
    closing = (vel_a * unit).sum(axis=-1) - (vel_b * unit).sum(axis=-1)
    return closing.astype(np.float32)


# ── 핵심: pair 라벨 계산 ───────────────────────────
def compute_pair_labels(scenario: Scenario) -> np.ndarray:
    """
    Scenario → (T, N=1, K=2) soft 라벨.

    값: {0.0, 0.5, 1.0}
    """
    T = scenario.worker1.shape[0]
    N = 1
    labels = np.zeros((T, N, K_THREATS), dtype=np.float32)

    w_pos = scenario.worker1.astype(np.float32)
    f_pos = scenario.forklift.astype(np.float32)
    crane = scenario.crane_state.astype(np.int32)

    # ── (worker × forklift) ──
    fork_present = ~np.isnan(f_pos).any(axis=1)
    if fork_present.any():
        # NaN 구간은 거리 0으로 두되, 라벨은 fork_present로 마스킹
        f_pos_safe = np.where(np.isnan(f_pos), 0.0, f_pos)
        d_wf = np.linalg.norm(w_pos - f_pos_safe, axis=-1)
        closing_wf = _closing_speed(w_pos, f_pos_safe)

        is_danger = fork_present & (d_wf < FORK_DANGER_DIST)
        is_warning = (
            fork_present
            & (d_wf >= FORK_DANGER_DIST)
            & (d_wf < FORK_WARN_DIST)
            & (closing_wf > APPROACH_SPEED)
        )
        labels[is_warning, 0, THREAT_FORKLIFT] = 0.5
        labels[is_danger, 0, THREAT_FORKLIFT] = 1.0
    # 부재 시 전부 0.0 (이미 zeros_like 초기화)

    # ── (worker × dropzone) ──
    dz_pos = np.tile(DZ_CENTER.astype(np.float32), (T, 1))
    d_wd = np.linalg.norm(w_pos - dz_pos, axis=-1)
    signed_dist_wd = d_wd - DZ_RADIUS              # 음수 = 내부
    closing_wd = _closing_speed(w_pos, dz_pos)
    crane_active = (crane != 0)

    inside = signed_dist_wd < 0.0
    near_outside_approaching = (
        (signed_dist_wd >= 0.0)
        & (signed_dist_wd < DZ_WARN_BUFFER)
        & (closing_wd > APPROACH_SPEED)
    )

    dz_danger = inside & crane_active                            # 인양 중 내부
    dz_warning_idle_inside = inside & ~crane_active              # idle 내부
    dz_warning_near = near_outside_approaching & crane_active    # 인양 중 경계 접근

    labels[dz_warning_idle_inside, 0, THREAT_DROPZONE] = 0.5
    labels[dz_warning_near, 0, THREAT_DROPZONE] = 0.5
    labels[dz_danger, 0, THREAT_DROPZONE] = 1.0

    # ── (오디오 격상 modifier) ──
    # 오디오 이상음은 위협원을 특정하지 못하므로 양 pair 모두에 동시 적용.
    # 거리 기반 라벨과 max로 결합 → 오디오만으로도 위험 격상 가능.
    audio = scenario.audio.astype(np.float32)
    audio_mod = np.zeros(T, dtype=np.float32)
    audio_mod[audio >= AUDIO_WARN_THRESHOLD] = 0.5
    audio_mod[audio >= AUDIO_DANGER_THRESHOLD] = 1.0

    labels[:, 0, THREAT_FORKLIFT] = np.maximum(labels[:, 0, THREAT_FORKLIFT], audio_mod)
    labels[:, 0, THREAT_DROPZONE] = np.maximum(labels[:, 0, THREAT_DROPZONE], audio_mod)

    return labels


def compute_pair_labels_batch(scenarios: list[Scenario]) -> np.ndarray:
    """
    여러 Scenario → (B, T, N, K) 라벨.
    """
    out = np.stack([compute_pair_labels(s) for s in scenarios], axis=0)
    return out


# ── 라벨 분포 요약 ──────────────────────────────────
def summarize_labels(labels: np.ndarray, threshold_warn: float = 0.4,
                     threshold_danger: float = 0.7) -> dict:
    """
    (T, N, K) 또는 (B, T, N, K) 라벨 → 클래스별 step 분포.
    """
    flat = labels.reshape(-1, labels.shape[-2], labels.shape[-1])
    total = flat.shape[0]
    out = {}
    threat_names = {THREAT_FORKLIFT: "forklift", THREAT_DROPZONE: "dropzone"}
    for t_idx, t_name in threat_names.items():
        col = flat[:, 0, t_idx]
        n_safe = int((col < threshold_warn).sum())
        n_warn = int(((col >= threshold_warn) & (col < threshold_danger)).sum())
        n_danger = int((col >= threshold_danger).sum())
        out[t_name] = {
            "safe": n_safe, "warn": n_warn, "danger": n_danger,
            "total": total,
            "ratio_safe": n_safe / total,
            "ratio_warn": n_warn / total,
            "ratio_danger": n_danger / total,
        }
    return out


# ── Sanity check ───────────────────────────────────
def _sanity_check():
    from scenarios_synthetic import build_synthetic_24

    scenarios = build_synthetic_24()
    print(f"loaded {len(scenarios)} scenarios\n")

    # 단일 시나리오 라벨 흐름 출력
    targets = [
        "s01_safe_fork_pass_worker_far",          # safe 전체
        "s05_fork_slow_approach_static_worker",   # 지게차 위험
        "s11_fork_frontal_full",                  # 정면 충돌
        "s15_dz_enter_from_south_lifting",        # 드롭존 진입
        "s19_dz_idle_to_lifting_worker_inside",   # idle→lift 전이
        "s22_dz_quick_dash_through",              # 짧은 진입/탈출
    ]
    for name in targets:
        s = next(s for s in scenarios if s.name == name)
        lab = compute_pair_labels(s)               # (T, 1, 2)
        fork_seq = lab[:, 0, THREAT_FORKLIFT]
        dz_seq = lab[:, 0, THREAT_DROPZONE]
        # 5초마다 간격으로 표본 출력
        print(f"[{s.name}]")
        print(f"  forklift label timeline (sampled @ t=0,5,10,15,20s):"
              f" {[f'{fork_seq[i]:.1f}' for i in [0, 25, 50, 75, 99]]}")
        print(f"  dropzone label timeline:                              "
              f" {[f'{dz_seq[i]:.1f}' for i in [0, 25, 50, 75, 99]]}")
        print()

    # 전체 분포
    all_labels = compute_pair_labels_batch(scenarios)   # (24, 100, 1, 2)
    print(f"[batch labels shape] {all_labels.shape}\n")
    summary = summarize_labels(all_labels)
    print("[overall distribution]")
    for k, v in summary.items():
        print(f"  {k:10s}: safe={v['ratio_safe']:.1%}  "
              f"warn={v['ratio_warn']:.1%}  danger={v['ratio_danger']:.1%}  "
              f"(total {v['total']} steps)")


if __name__ == "__main__":
    _sanity_check()
