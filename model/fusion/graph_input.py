"""
Scenario (위치·오디오·크레인 상태) → Graph 모델 입력 텐서 변환.

출력 shape (단일 시나리오, T=N_STEPS 기준):
  nodes  : (V=3, T, F=8)  — 노드 feature
  adj    : (T, V, V)      — 거리 기반 adjacency
  scene  : (T, 2)         — [audio, crane_active] 전역 context

노드 순서 (고정):
  [0] worker1
  [1] forklift
  [2] dropzone

Feature 순서 (F=8):
  [0] x, [1] y                    — 위치 (m)
  [2] vx, [3] vy                  — 속도 (m/s)
  [4] is_worker                   — 타입 one-hot
  [5] is_forklift
  [6] is_dropzone
  [7] size                        — worker/forklift=0, dropzone=radius

부재(absent) 엔티티 처리:
  - NaN 위치는 sentinel (999, 999)로 치환
  - Gaussian adjacency kernel이 거리 ~∞ → adj ≈ 0으로 자동 무시
  - 타입 one-hot은 유지 (e.g., 지게차 부재 시에도 is_forklift=1)
"""

from __future__ import annotations

import numpy as np

from .data.scenario_generator import (
    Scenario,
    RATE,
    DURATION,
    N_STEPS,
    DZ_CENTER,
    DZ_RADIUS,
)


# ── 고정 인덱스 ─────────────────────────────────────
# 노드 순서
NODE_WORKER = 0
NODE_FORKLIFT = 1
NODE_DROPZONE = 2
V_NODES = 3

# 피처 차원
F_NODE = 8
# scene: [audio, crane_active, dist_w_to_f, dist_w_to_dz, fork_present, closing_w_to_f]
F_SCENE = 6

# 위협 타입별 scene feature 인덱스 (모델이 분리해서 사용)
SCENE_IDX_FORKLIFT = [0, 2, 4, 5]   # audio, dist_w_f, fork_present, closing_w_f
SCENE_IDX_DROPZONE = [0, 1, 3]       # audio, crane_active, dist_w_dz

# 위협 순서 (pair label / 모델 출력에서 사용)
THREAT_FORKLIFT = 0
THREAT_DROPZONE = 1
K_THREATS = 2

# sentinel 위치 (absent 엔티티용, adj에서 실질 무시됨)
_ABSENT_SENTINEL = np.array([999.0, 999.0], dtype=np.float32)


# ── 내부 helper ─────────────────────────────────────
def _sanitize_positions(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    NaN 위치를 sentinel로 치환.
    Returns:
      pos_clean: (T, 2) — NaN 없음
      present:   (T,)  bool — True면 해당 step에 엔티티 존재
    """
    pos = pos.astype(np.float32, copy=True)
    nan_mask = np.isnan(pos).any(axis=1)
    pos[nan_mask] = _ABSENT_SENTINEL
    return pos, ~nan_mask


def _velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    """단순 전진 차분. 첫 step은 두번째 값으로 pad."""
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / dt
        vel[0] = vel[1]
    return vel


# ── 핵심 변환 함수 ────────────────────────────────── 시작점
def to_graph_input(
    scenario: Scenario,
    dt: float = 1.0 / RATE, # 5프레임으로 지정했으니까 RATE=5
    dist_sigma: float = 1.0,
    dropzone_center: np.ndarray | None = None,
    dropzone_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 시나리오 → (nodes, adj, scene) 텐서.

    Args:
      scenario: Scenario 객체
      dt:       time-step 간격 (초). 기본 0.2s @ 5Hz.
      dist_sigma: Adjacency Gaussian kernel의 bandwidth (m). ; sigma 작을수록 근거리만 연결, 클수록 멀어도 연결.
      dropzone_center: (2,) 동적 dropzone 좌표. None이면 DZ_CENTER 상수 사용. (T, 2) 형태로 시간별 다른 위치도 허용 (인양물 이동)
      dropzone_radius: dropzone 반경. None이면 DZ_RADIUS 상수 사용.

    Returns:
      nodes: (V=3, T, F=8)  float32
      adj:   (T, V, V)      float32 — self-loop 0
      scene: (T, 2)         float32 — [audio, crane_active]
    """
    # 100 프레임 짜리인걸 확인, 시계열 길이 T 확정
    T = scenario.worker1.shape[0]

    # 동적 dropzone 처리
    if dropzone_center is None:
        dropzone_center = DZ_CENTER # 비어 있다면 디폴드 값 사용 
    dz_arr = np.asarray(dropzone_center, dtype=np.float32) #numpy 배열로 변환하기 


    # dropzone이 고정이라 디폴트 드롭존으로 설정하고 dz_arr를 만들었는데, 그렇게되면 [x,y] 형태가 되어 버려서 그걸 (T,2)행렬 형태로 만듦
    if dz_arr.ndim == 1:                      # (2,) → (T, 2)
        dz_arr = np.tile(dz_arr, (T, 1))
    dz_radius_val = float(DZ_RADIUS if dropzone_radius is None else dropzone_radius)

    # ── 각 엔티티 위치 정리 ──
    # 내부 헬퍼 함수를 통해서, NaN 값을 999.0, 999.0으로 치환하기 
    w_pos, _ = _sanitize_positions(scenario.worker1)
    f_pos, _ = _sanitize_positions(scenario.forklift)
    # 드롭존: 동적 좌표 적용
    dz_pos = dz_arr

    # ── 속도 계산 ──
    w_vel = _velocity(w_pos, dt)
    f_vel = _velocity(f_pos, dt)
    dz_vel = np.zeros_like(dz_pos)

    # ── 노드 feature 구성 ──
    nodes = np.zeros((V_NODES, T, F_NODE), dtype=np.float32)

    # worker1
    nodes[NODE_WORKER, :, 0:2] = w_pos
    nodes[NODE_WORKER, :, 2:4] = w_vel
    nodes[NODE_WORKER, :, 4] = 1.0   # is_worker
    nodes[NODE_WORKER, :, 7] = 0.0   # size

    # forklift
    nodes[NODE_FORKLIFT, :, 0:2] = f_pos
    nodes[NODE_FORKLIFT, :, 2:4] = f_vel
    nodes[NODE_FORKLIFT, :, 5] = 1.0  # is_forklift
    nodes[NODE_FORKLIFT, :, 7] = 0.0  # size

    # dropzone
    nodes[NODE_DROPZONE, :, 0:2] = dz_pos
    nodes[NODE_DROPZONE, :, 2:4] = dz_vel
    nodes[NODE_DROPZONE, :, 6] = 1.0  # is_dropzone
    nodes[NODE_DROPZONE, :, 7] = dz_radius_val

    # ── Adjacency (거리 기반 Gaussian kernel) ── 
    # 어떤 게 어떤 거랑 가까운지 계산 즉, 매 시점마다 워커-지게차-드롭존 사이 거리를 재서, 가까울수록 큰 값 멀수록 작은 값으로 변환

    # positions per time-step: (T, V, 2) - node에서 위치 정보만 빼기 
    pos_tv = nodes[:, :, 0:2].transpose(1, 0, 2)  # (T, V, 2)
    # pairwise diff: (T, V, V, 2) - 모든 노드 페어 사이의 차이 벡터 계산 함 
    diff = pos_tv[:, :, None, :] - pos_tv[:, None, :, :]
    # 차이를 거리로 
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)  # (T, V, V)
    # 거리를 adjacency로 변환 
    adj = np.exp(-dist / dist_sigma)
    # self-loop 제거 
    eye = np.eye(V_NODES, dtype=np.float32)
    adj = adj * (1.0 - eye[None, :, :])
    # (100, 3, 3) 형태로 나옴
    # 가까운 노드일수록 더 강하게 영향을 준다고 연산을 해야하기 때문에 필요 함 

    # ── Scene context ──
    # pair-wise 명시 feature (모델 학습 부담 줄이기 위해 directly 제공)
    
    # 지게차가 보였는지 체크하기. 보임 -> 1 안 보임 -> 0
    fork_present_per_step = (~np.isnan(scenario.forklift).any(axis=1)).astype(np.float32)
    # 작업자-지게차 거리
    dist_w_f = np.linalg.norm(w_pos - f_pos, axis=-1).astype(np.float32)
    # 작업자-드롭존 거리
    dist_w_dz = np.linalg.norm(w_pos - dz_pos, axis=-1).astype(np.float32)

    # 부재 시 거리는 큰 값(20.0)으로 cap (sentinel 1000 그대로면 normalize 안 됨)
    dist_w_f = np.where(fork_present_per_step > 0, dist_w_f, 20.0)

    # closing speed: worker→forklift 방향으로 접근하는 속도 (양수=접근)
    rel = f_pos - w_pos
    rel_norm = np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-6
    unit = rel / rel_norm
    rel_vel = w_vel - f_vel
    closing_w_f = (rel_vel * unit).sum(axis=-1).astype(np.float32)
    closing_w_f = np.where(fork_present_per_step > 0, closing_w_f, 0.0)

    scene = np.stack(
        [
            scenario.audio.astype(np.float32),          # 오디오 score 시계열
            scenario.crane_state.astype(np.float32),    # 크레인 활성 시계열
            dist_w_f,                                  # 작업자 - 지게차 거리 시계열
            dist_w_dz,                                 # 작업자 - 드롭존 거리 시계열
            fork_present_per_step,                     # 지계차 존재 시계열
            closing_w_f                                # 작업자 - 지게차 접근 속도 시계열
        ],
        axis=1,
      )  # (T, 6)

    return nodes, adj, scene


# ── 배치 변환 ───────────────────────────────────────
def to_graph_batch(
    scenarios: list[Scenario],
    dt: float = 1.0 / RATE,
    dist_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    여러 시나리오를 한 번에 변환 (전체 길이 유지, 윈도우 안 나눔).

    Returns:
      nodes: (B, V, T, F)
      adj:   (B, T, V, V)
      scene: (B, T, 2)
    """
    outs = [to_graph_input(s, dt=dt, dist_sigma=dist_sigma) for s in scenarios]
    nodes = np.stack([o[0] for o in outs], axis=0)
    adj = np.stack([o[1] for o in outs], axis=0)
    scene = np.stack([o[2] for o in outs], axis=0)
    return nodes, adj, scene


# ── Sanity check ───────────────────────────────────
def _sanity_check():
    from scenarios_synthetic import build_synthetic_24

    scenarios = build_synthetic_24()
    print(f"loaded {len(scenarios)} scenarios\n")

    # 단일 변환 검증
    s = scenarios[0]
    nodes, adj, scene = to_graph_input(s)
    print(f"[single scenario: {s.name}]")
    print(f"  nodes shape : {nodes.shape}  (expected ({V_NODES}, {N_STEPS}, {F_NODE}))")
    print(f"  adj shape   : {adj.shape}    (expected ({N_STEPS}, {V_NODES}, {V_NODES}))")
    print(f"  scene shape : {scene.shape}  (expected ({N_STEPS}, {F_SCENE}))")
    print(f"  nodes dtype : {nodes.dtype}")
    print()

    # 값 범위 점검
    print(f"  node positions (worker, t=0)   : {nodes[NODE_WORKER, 0, 0:2]}")
    print(f"  node positions (forklift, t=0) : {nodes[NODE_FORKLIFT, 0, 0:2]}")
    print(f"  node positions (dropzone, t=0) : {nodes[NODE_DROPZONE, 0, 0:2]}")
    print(f"  adj (t=0) matrix:\n{adj[0]}")
    print(f"  scene (t=0): audio={scene[0, 0]:.3f}, crane={scene[0, 1]:.0f}")
    print()

    # absent 엔티티(지게차 없는 시나리오) 점검
    absent_sc = next(s for s in scenarios if s.name == "s02_safe_worker_walk_no_fork")
    nodes_a, adj_a, _ = to_graph_input(absent_sc)
    w_pos = nodes_a[NODE_WORKER, 50, 0:2]
    f_pos = nodes_a[NODE_FORKLIFT, 50, 0:2]
    dist_wf = np.linalg.norm(w_pos - f_pos)
    adj_wf = adj_a[50, NODE_WORKER, NODE_FORKLIFT]
    print(f"[absent forklift: {absent_sc.name}]")
    print(f"  forklift position (sentinel): {f_pos}")
    print(f"  worker-forklift distance    : {dist_wf:.1f} m")
    print(f"  worker-forklift adjacency   : {adj_wf:.6f}  (should be ≈ 0)")
    print()

    # 배치 변환 검증
    nodes_b, adj_b, scene_b = to_graph_batch(scenarios)
    print(f"[batch]")
    print(f"  nodes : {nodes_b.shape}")
    print(f"  adj   : {adj_b.shape}")
    print(f"  scene : {scene_b.shape}")


if __name__ == "__main__":
    _sanity_check()
