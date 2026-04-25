"""
시나리오 데이터 스키마 + 유틸리티.

이 파일은 **실제 데이터로 교체되어도 그대로 유지**되는 공통 레이어.
시나리오 소스(합성 vs 실제 영상)는 별도 파일로 분리:
  - scenarios_synthetic.py  : 합성 24개 (교체 대상)
  - scenarios_real.py       : 실제 영상 데이터 로더 (생성 예정)

두 로더 모두 `list[Scenario]`를 반환하면 downstream (graph_input, pair_labels,
dataset, train)은 소스에 무관하게 동일하게 동작.

좌표계 (월드 좌표, 단위: m):
  ArUco 4개로 정의된 사각 작업공간 (실측 2m × 3m):
    27번: (-2, 0)    38번: (0, 0)      ← 하단 (forklift 라인)
    22번: (-2, 3)    24번: (0, 3)      ← 상단

  레이아웃:
    - 작업공간   : x ∈ [-2, 0], y ∈ [0, 3]
    - worker 라인 : 박스 안 수직, x ≈ -0.3, y가 3→0 방향 이동
    - forklift 라인 : 박스 하단 가장자리, y = 0, x가 -2→0 (우측) 이동
    - 드롭존     : center (-1.5, 2.0), radius 0.3

  카메라:
    - cam1 : worker 영역 (수직 라인)
    - cam2 : forklift 영역. 인양물에 의해 worker 시야 가려짐 (사각지대)
    - 두 카메라 모두 동일 절대 좌표계로 통일 (homography)

샘플링: 5 Hz × 20초 = 100 step (기본값).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── 전역 상수 ────────────────────────────────────────
RATE = 5
DURATION = 20.0
N_STEPS = int(RATE * DURATION)  # 100
TIME_AXIS = np.linspace(0, DURATION, N_STEPS)

DZ_CENTER = np.array([-1.5, 2.0])
DZ_RADIUS = 0.3

# 작업공간 경계 (sanity check / 시각화용)
WORKSPACE_X = (-2.0, 0.0)
WORKSPACE_Y = (0.0, 3.0)


# ── Scenario 데이터 클래스 ───────────────────────────
@dataclass
class Scenario:
    """
    시나리오 단위 데이터. 합성/실데이터 어느 쪽에서 생성되든 동일한 형식.

    Attributes:
        name:        고유 식별자 (파일명에 사용)
        forklift:    (N, 2) — 지게차 위치 (없으면 전체 NaN)
        worker1:     (N, 2) — 작업자 위치
        crane_state: (N,)  int — 0=idle, 1=lifting
        audio:       (N,)  float ∈ [0, 1]
        labels:      (N,)  int ∈ {0, 1, 2} — scene-level sanity label (선택)
    """
    name: str
    forklift: np.ndarray
    worker1: np.ndarray
    crane_state: np.ndarray
    audio: np.ndarray
    labels: Optional[np.ndarray] = None

    def __post_init__(self):
        self.forklift = np.asarray(self.forklift, dtype=np.float32)
        self.worker1 = np.asarray(self.worker1, dtype=np.float32)
        self.crane_state = np.asarray(self.crane_state, dtype=np.int32)
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int32)

        n = self.worker1.shape[0]
        assert self.forklift.shape == (n, 2), f"{self.name}: forklift shape"
        assert self.worker1.shape == (n, 2), f"{self.name}: worker1 shape"
        assert self.crane_state.shape == (n,), f"{self.name}: crane shape"
        assert self.audio.shape == (n,), f"{self.name}: audio shape"
        if self.labels is not None:
            assert self.labels.shape == (n,), f"{self.name}: labels shape"


# ── 궤적 primitive (합성용) ─────────────────────────
def linear(p0, p1, t0: float = 0.0, t1: float = DURATION) -> np.ndarray:
    """p0→p1 선형 이동. [t0, t1] 구간 밖은 끝값으로 고정."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    out = np.zeros((N_STEPS, 2))
    i0, i1 = int(t0 * RATE), int(t1 * RATE)
    out[:i0] = p0
    out[i1:] = p1
    if i1 > i0:
        a = np.linspace(0, 1, i1 - i0).reshape(-1, 1)
        out[i0:i1] = p0 + a * (p1 - p0)
    return out


def piecewise(waypoints) -> np.ndarray:
    """waypoints: [(t_sec, [x, y]), ...] — 구간별 선형 보간."""
    out = np.zeros((N_STEPS, 2))
    for k in range(len(waypoints) - 1):
        t0, p0 = waypoints[k]
        t1, p1 = waypoints[k + 1]
        i0, i1 = int(t0 * RATE), int(t1 * RATE)
        if i1 > i0:
            a = np.linspace(0, 1, i1 - i0).reshape(-1, 1)
            out[i0:i1] = np.array(p0) + a * (np.array(p1) - np.array(p0))
    out[int(waypoints[-1][0] * RATE):] = waypoints[-1][1]
    return out


def still(p) -> np.ndarray:
    return np.tile(np.array(p, dtype=float), (N_STEPS, 1))


def absent() -> np.ndarray:
    """엔티티가 씬에 없음을 NaN으로 표시."""
    return np.full((N_STEPS, 2), np.nan)


def jitter(path: np.ndarray, std: float = 0.02) -> np.ndarray:
    """센서 노이즈 시뮬레이션."""
    if np.isnan(path).all():
        return path
    return path + np.random.normal(0, std, path.shape)


def audio_trace(base: float = 0.05, noise: float = 0.02, spike=None) -> np.ndarray:
    """spike = (t0_sec, t1_sec, value) 또는 None."""
    a = np.clip(base + np.random.normal(0, noise, N_STEPS), 0, 1)
    if spike is not None:
        t0, t1, v = spike
        i0, i1 = int(t0 * RATE), int(t1 * RATE)
        a[i0:i1] = np.clip(v + np.random.normal(0, 0.03, i1 - i0), 0, 1)
    return a


def crane_seq(default: int = 0, changes=None) -> np.ndarray:
    """default로 시작, changes=[(t_sec, new_state), ...]로 스위칭."""
    c = np.full(N_STEPS, default, dtype=int)
    if changes:
        for t, state in changes:
            c[int(t * RATE):] = state
    return c


def labels_seq(segments) -> np.ndarray:
    """segments=[(t0_sec, t1_sec, class_id), ...] scene-level sanity 라벨."""
    y = np.zeros(N_STEPS, dtype=int)
    for t0, t1, c in segments:
        y[int(t0 * RATE):int(t1 * RATE)] = c
    return y


# ── Save / Load ─────────────────────────────────────
def save_scenarios_npz(scenarios: list[Scenario], path: str) -> None:
    """
    Scenario 리스트를 하나의 .npz로 저장.
    키 형식: {name}__forklift, {name}__worker1, {name}__crane_state,
             {name}__audio, {name}__labels
    """
    out = {}
    for s in scenarios:
        out[f"{s.name}__forklift"] = s.forklift
        out[f"{s.name}__worker1"] = s.worker1
        out[f"{s.name}__crane_state"] = s.crane_state
        out[f"{s.name}__audio"] = s.audio
        if s.labels is not None:
            out[f"{s.name}__labels"] = s.labels
    np.savez_compressed(path, **out)


def load_scenarios_npz(path: str) -> list[Scenario]:
    """저장된 .npz → Scenario 리스트 복원."""
    data = dict(np.load(path, allow_pickle=True))
    names = sorted({k.split("__")[0] for k in data if "__" in k})
    scenarios = []
    for name in names:
        scenarios.append(Scenario(
            name=name,
            forklift=data[f"{name}__forklift"],
            worker1=data[f"{name}__worker1"],
            crane_state=data[f"{name}__crane_state"],
            audio=data[f"{name}__audio"],
            labels=data.get(f"{name}__labels"),
        ))
    return scenarios


# ── Main entry ──────────────────────────────────────
def main():
    """
    현재는 합성 시나리오(24개)를 `fusion_train_24.npz`로 저장.
    실제 데이터 사용 시 아래 import만 교체하면 됨.
    """
    # === 시나리오 소스: 합성 (나중에 실제 데이터로 교체) ===
    from scenarios_synthetic import build_synthetic_24
    scenarios = build_synthetic_24(seed=42)
    # === 교체 예시: ===
    # from scenarios_real import build_from_video_dir
    # scenarios = build_from_video_dir("path/to/recordings")

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "fusion_train_24.npz")
    save_scenarios_npz(scenarios, out_path)

    print(f"saved {len(scenarios)} scenarios → {out_path}")
    print(f"shape per scenario: {N_STEPS} steps @ {RATE}Hz, {DURATION}s")
    for s in scenarios:
        label_dist = (
            np.bincount(s.labels, minlength=3).tolist() if s.labels is not None else "-"
        )
        print(f"  {s.name:50s}  labels={label_dist}")


if __name__ == "__main__":
    main()
