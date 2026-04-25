"""
Pairwise Interaction Fusion Model.

Lite Graph Model 스타일 (Social-STGCNN의 graph + temporal 아이디어 차용).
출력은 trajectory 예측이 아니라 pair-level 충돌/진입 확률.

입력:
  nodes : (B, V=3, T, F_node=8)   — V=[worker, forklift, dropzone]
  adj   : (B, T, V, V)             — 거리 기반 adjacency
  scene : (B, T, F_scene=6)        — 공유 scene context (모델 내부에서 분리)

출력:
  risk_matrix : (B, N, K=2)  ∈ [0, 1]
    [b, i, 0] = worker_i vs forklift collision probability
    [b, i, 1] = worker_i vs dropzone overlap probability

핵심 설계:
  - GCN + temporal encoder는 모든 노드에 SHARED (공통 표현 학습)
  - Scene encoder + pair head는 위협 타입별로 SEPARATED
    → forklift 출력과 dropzone 출력이 섞이지 않음

파라미터: 약 11K.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graph_input import (
    V_NODES,
    F_NODE,
    F_SCENE,
    K_THREATS,
    SCENE_IDX_FORKLIFT,
    SCENE_IDX_DROPZONE,
)


# ── Graph Conv 레이어 ──────────────────────────────
class GraphConv(nn.Module):
    """시간축마다 노드 간 메시지 전달. H' = ReLU(norm(A+I) · H · W)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x  : (B, V, T, in_dim)
        # adj: (B, T, V, V)
        B, V, T, _ = x.shape

        x_t = x.transpose(1, 2)                 # (B, T, V, in_dim)
        eye = torch.eye(V, device=x.device).view(1, 1, V, V)
        adj_self = adj + eye
        deg = adj_self.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj_self / deg               # (B, T, V, V)
        x_agg = torch.matmul(adj_norm, x_t)     # (B, T, V, in_dim)
        x_out = self.linear(x_agg)              # (B, T, V, out_dim)
        return x_out.transpose(1, 2)            # (B, V, T, out_dim)


# ── 위협 타입별 head 컴포넌트 ──────────────────────
class ThreatBranch(nn.Module):
    """
    한 위협 타입에 대한 scene encoder + pair head.

    입력:
      worker_h : (B, N, H)   공유 GCN+temporal로 추출된 worker 표현
      threat_h : (B, H)       해당 threat 노드 표현
      scene_t  : (B, T, F_scene_subset)  타입별 scene 부분집합

    출력:
      risk : (B, N) ∈ [0, 1]
    """
    def __init__(self, hidden: int, scene_in_dim: int, head_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.scene_proj = nn.Sequential(
            nn.Linear(scene_in_dim, hidden),
            nn.ReLU(),
        )
        self.scene_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.hidden = hidden

    def forward(
        self,
        worker_h: torch.Tensor,   # (B, N, H)
        threat_h: torch.Tensor,   # (B, H)
        scene_t: torch.Tensor,    # (B, T, F_subset)
    ) -> torch.Tensor:
        B, N, H = worker_h.shape
        # Scene 시간축 요약
        s = self.scene_proj(scene_t)               # (B, T, H)
        _, s_gru = self.scene_gru(s)               # (1, B, H)
        s_summary = s_gru.squeeze(0)               # (B, H)
        # Worker × threat × scene 결합
        t_exp = threat_h.unsqueeze(1).expand(B, N, H)
        s_exp = s_summary.unsqueeze(1).expand(B, N, H)
        feat = torch.cat([worker_h, t_exp, s_exp], dim=-1)   # (B, N, 3H)
        logits = self.head(feat).squeeze(-1)                  # (B, N)
        return torch.sigmoid(logits)


# ── Pairwise Interaction Fusion Model ─────────────
class PairwiseInteractionFusionModel(nn.Module):
    """
    상호작용 행렬 출력 모델.

    Args:
      n_node_feat:  노드 피처 차원 (default 8)
      hidden:       내부 hidden dim (default 24)
      n_workers:    forward 시 worker 노드 수 (default 1)
      n_threats:    forward 시 threat 노드 수 (default 2: forklift, dropzone)
    """
    def __init__(
        self,
        n_node_feat: int = F_NODE,
        hidden: int = 24,
        n_workers: int = 1,
        n_threats: int = K_THREATS,
    ):
        super().__init__()
        self.n_workers = n_workers
        self.n_threats = n_threats
        self.hidden = hidden

        # ── Spatial GCN (공유) ──
        self.gconv1 = GraphConv(n_node_feat, hidden)
        self.gconv2 = GraphConv(hidden, hidden)

        # ── Temporal encoder (공유, 모든 노드에 적용) ──
        self.tconv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

        # ── 위협 타입별 분기 ──
        self.branch_forklift = ThreatBranch(
            hidden=hidden, scene_in_dim=len(SCENE_IDX_FORKLIFT),
        )
        self.branch_dropzone = ThreatBranch(
            hidden=hidden, scene_in_dim=len(SCENE_IDX_DROPZONE),
        )

        # 학습 시 scene index를 GPU로 옮기기 위한 buffer
        self.register_buffer(
            "_idx_f", torch.tensor(SCENE_IDX_FORKLIFT, dtype=torch.long),
        )
        self.register_buffer(
            "_idx_d", torch.tensor(SCENE_IDX_DROPZONE, dtype=torch.long),
        )

    def forward(
        self,
        nodes: torch.Tensor,    # (B, V, T, F_node)
        adj: torch.Tensor,      # (B, T, V, V)
        scene: torch.Tensor,    # (B, T, F_scene)
    ) -> torch.Tensor:
        B, V, T, _ = nodes.shape

        # ── Spatial GCN (2-layer) ──
        h = torch.relu(self.gconv1(nodes, adj))
        h = torch.relu(self.gconv2(h, adj))      # (B, V, T, H)

        # ── Temporal encoder per node (shared) ──
        h = h.reshape(B * V, T, self.hidden)
        h_t = h.transpose(1, 2)
        h_t = self.tconv(h_t)
        h_t = h_t.transpose(1, 2)
        _, h_gru = self.gru(h_t)
        h_node = h_gru.squeeze(0).view(B, V, self.hidden)   # (B, V, H)

        # ── 노드 인덱싱 ──
        # [0..N-1] = workers, [N] = forklift, [N+1] = dropzone
        N = self.n_workers
        workers = h_node[:, :N]                 # (B, N, H)
        forklift_h = h_node[:, N]               # (B, H)
        dropzone_h = h_node[:, N + 1]           # (B, H)

        # ── Scene 분리 ──
        scene_f = scene.index_select(-1, self._idx_f)   # (B, T, len_f)
        scene_d = scene.index_select(-1, self._idx_d)   # (B, T, len_d)

        # ── Per-threat 분기 ──
        risk_f = self.branch_forklift(workers, forklift_h, scene_f)   # (B, N)
        risk_d = self.branch_dropzone(workers, dropzone_h, scene_d)   # (B, N)

        # ── 결합 → (B, N, K=2) ──
        return torch.stack([risk_f, risk_d], dim=-1)


# ── Sanity check ───────────────────────────────────
def _sanity_check():
    import numpy as np
    from scenarios_synthetic import build_synthetic_24
    from graph_input import to_graph_batch

    scenarios = build_synthetic_24()
    nodes_np, adj_np, scene_np = to_graph_batch(scenarios)
    print(f"input shapes:")
    print(f"  nodes: {nodes_np.shape}")
    print(f"  adj  : {adj_np.shape}")
    print(f"  scene: {scene_np.shape}\n")

    T_WIN = 20
    nodes_w = nodes_np[:, :, :T_WIN, :]
    adj_w = adj_np[:, :T_WIN, :, :]
    scene_w = scene_np[:, :T_WIN, :]

    model = PairwiseInteractionFusionModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    nodes_t = torch.from_numpy(nodes_w).float()
    adj_t = torch.from_numpy(adj_w).float()
    scene_t = torch.from_numpy(scene_w).float()

    model.eval()
    with torch.no_grad():
        risk = model(nodes_t, adj_t, scene_t)
    print(f"output shape: {risk.shape}")
    print(f"output range: [{risk.min().item():.4f}, {risk.max().item():.4f}]")


if __name__ == "__main__":
    _sanity_check()