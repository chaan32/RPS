"""
PyTorch Dataset + 슬라이딩 윈도우.

각 시나리오(T=100, 20초 @ 5Hz)를 T_WIN=20 (4초) 윈도우로 절단.
stride=5 (1초, 75% overlap) → 시나리오당 약 17 윈도우.

학습 라벨: **윈도우 마지막 시점**의 pair 라벨 사용
  → "최근 4초 context로 지금 위험도 판정"이라는 의미.

데이터 분할: 카테고리(SAFE / forklift / dropzone)별 stratified split.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.scenario_generator import Scenario, N_STEPS
from ..graph_input import to_graph_input
from ..data.pair_labels import compute_pair_labels


# ── 윈도우 설정 ─────────────────────────────────────
T_WIN = 5       # 1초 @ 5Hz (5프레임)
STRIDE = 1      # 0.2초 (80% overlap, 학습 샘플 풍부하게)


def _scenario_category(name: str) -> str:
    """시나리오 이름에서 카테고리 추출."""
    if "_safe_" in name:
        return "safe"
    if "_fork_" in name:
        return "forklift"
    if "_dz_" in name:
        return "dropzone"
    return "other"


# ── PyTorch Dataset ────────────────────────────────
class FusionDataset(Dataset):
    """
    Scenario 리스트 → 슬라이딩 윈도우 단위 샘플.

    각 샘플: (nodes_w, adj_w, scene_w, label)
      nodes_w : (V, T_WIN, F_node)
      adj_w   : (T_WIN, V, V)
      scene_w : (T_WIN, F_scene)
      label   : (N, K)   윈도우 끝 시점의 pair 라벨
    """
    def __init__(
        self,
        scenarios: list[Scenario],
        t_win: int = T_WIN,
        stride: int = STRIDE,
    ):
        self.t_win = t_win
        self.stride = stride
        self.windows: list[dict] = []

        for s in scenarios:
            nodes, adj, scene = to_graph_input(s)             # (V,T,F),(T,V,V),(T,F_s)
            labels = compute_pair_labels(s)                    # (T, N, K)
            T_total = nodes.shape[1]

            for start in range(0, T_total - t_win + 1, stride):
                end = start + t_win
                self.windows.append({
                    "scenario": s.name,
                    "nodes":  nodes[:, start:end, :].copy(),
                    "adj":    adj[start:end, :, :].copy(),
                    "scene":  scene[start:end, :].copy(),
                    "label":  labels[end - 1].copy(),          # (N, K) at end step
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return (
            torch.from_numpy(w["nodes"]).float(),
            torch.from_numpy(w["adj"]).float(),
            torch.from_numpy(w["scene"]).float(),
            torch.from_numpy(w["label"]).float(),
        )


# ── 시나리오 단위 stratified split ──────────────────
def split_scenarios(
    scenarios: list[Scenario],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Scenario], list[Scenario]]:
    """
    카테고리별로 균등하게 train/val 분할.

    24 시나리오 (SAFE 4 + 지게차 10 + 드롭존 10) 기준:
      - SAFE: 3 train + 1 val
      - 지게차: 8 train + 2 val
      - 드롭존: 8 train + 2 val
    """
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_cat.setdefault(_scenario_category(s.name), []).append(s)

    train, val = [], []
    for cat, lst in by_cat.items():
        idx = rng.permutation(len(lst))
        n_val = max(1, int(round(len(lst) * val_ratio)))
        val_idx = set(idx[:n_val].tolist())
        for i, s in enumerate(lst):
            (val if i in val_idx else train).append(s)
    return train, val


# ── Sanity check ───────────────────────────────────
def _sanity_check():
    from scenarios_synthetic import build_synthetic_24

    scenarios = build_synthetic_24()
    train_sc, val_sc = split_scenarios(scenarios)
    print(f"split: {len(train_sc)} train / {len(val_sc)} val\n")
    print(f"[train scenarios]")
    for s in train_sc:
        print(f"  {_scenario_category(s.name):8s}  {s.name}")
    print(f"\n[val scenarios]")
    for s in val_sc:
        print(f"  {_scenario_category(s.name):8s}  {s.name}")

    # Dataset
    train_ds = FusionDataset(train_sc)
    val_ds = FusionDataset(val_sc)
    print(f"\n[windows] train={len(train_ds)}  val={len(val_ds)}")
    print(f"  expected per scenario: (100 - 20)/5 + 1 = 17 windows\n")

    # 한 샘플 shape 점검
    nodes, adj, scene, label = train_ds[0]
    print(f"[sample 0]")
    print(f"  scenario : {train_ds.windows[0]['scenario']}")
    print(f"  nodes : {tuple(nodes.shape)}  (expect (3, 20, 8))")
    print(f"  adj   : {tuple(adj.shape)}    (expect (20, 3, 3))")
    print(f"  scene : {tuple(scene.shape)}  (expect (20, 2))")
    print(f"  label : {tuple(label.shape)}  (expect (1, 2))")
    print(f"  label values: {label.numpy()}")

    # 라벨 클래스 분포 (윈도우 끝 시점 기준)
    all_labels = np.stack([w["label"] for w in train_ds.windows])  # (W, N, K)
    print(f"\n[train window label distribution]")
    for t_idx, name in [(0, "forklift"), (1, "dropzone")]:
        col = all_labels[:, 0, t_idx]
        n_safe = (col < 0.4).sum()
        n_warn = ((col >= 0.4) & (col < 0.7)).sum()
        n_dang = (col >= 0.7).sum()
        total = len(col)
        print(f"  {name:10s}: safe={n_safe} ({n_safe/total:.1%})  "
              f"warn={n_warn} ({n_warn/total:.1%})  "
              f"danger={n_dang} ({n_dang/total:.1%})")


if __name__ == "__main__":
    _sanity_check()
