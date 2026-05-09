"""
Pairwise Interaction Fusion Model 학습 스크립트.

Loss:    BCE (soft labels {0.0, 0.5, 1.0}에 직접 적용)
Optim:   Adam, lr=1e-3
Metric:  per-pair / per-class precision, recall, F1, accuracy
Output:  checkpoints/best.pt  (validation F1 macro 평균 기준 최고)

실행:
  python -m model.fusion.training.train                       # 합성 24개만
  python -m model.fusion.training.train --unity-dir <path>    # 합성 + Unity
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ..data.scenarios_synthetic import build_synthetic_24
from ..data.scenarios_unity import load_unity_scenarios
from .dataset import FusionDataset, split_scenarios
from ..model import PairwiseInteractionFusionModel
from ..graph_input import THREAT_FORKLIFT, THREAT_DROPZONE


# ── 하이퍼파라미터 ──────────────────────────────────
EPOCHS = 120
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# Early stopping: avg macro F1이 PATIENCE epoch 동안 개선 없으면 중단
EARLY_STOP_PATIENCE = 15

# 라벨/예측 임계값 (3-class 변환용)
THRESH_WARN = 0.4
THRESH_DANGER = 0.7

CLASS_NAMES = ["safe", "warn", "danger"]
THREAT_NAMES = {THREAT_FORKLIFT: "forklift", THREAT_DROPZONE: "dropzone"}


# ── helper ──────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_3way(probs: np.ndarray) -> np.ndarray:
    """확률 → 0=safe, 1=warn, 2=danger."""
    out = np.zeros_like(probs, dtype=np.int64)
    out[(probs >= THRESH_WARN) & (probs < THRESH_DANGER)] = 1
    out[probs >= THRESH_DANGER] = 2
    return out


# ── BCE on prob ─────────────────────────────────────
class BCEOnProb(nn.Module):
    """모델 출력이 이미 sigmoid 통과한 확률이므로 nn.BCELoss 사용."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
        return self.loss(pred, target)


# ── 학습/평가 루프 ──────────────────────────────────
def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    all_pred = []   # list of (W, K) np
    all_label = []  # list of (W, K) np

    for nodes, adj, scene, label in loader:
        nodes = nodes.to(device)
        adj = adj.to(device)
        scene = scene.to(device)
        label = label.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(nodes, adj, scene)        # (B, N=1, K=2)
            loss = criterion(pred, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        b = label.size(0)
        total_loss += loss.item() * b
        total_n += b

        # (B, N, K) → (B*N, K)
        all_pred.append(pred.detach().cpu().numpy().reshape(-1, pred.size(-1)))
        all_label.append(label.detach().cpu().numpy().reshape(-1, label.size(-1)))

    avg_loss = total_loss / total_n
    pred_arr = np.concatenate(all_pred, axis=0)     # (W, K)
    label_arr = np.concatenate(all_label, axis=0)   # (W, K)
    return avg_loss, pred_arr, label_arr


def compute_metrics_per_pair(pred_arr: np.ndarray, label_arr: np.ndarray) -> dict:
    """
    pred_arr, label_arr: (W, K)
    Returns: {threat_name: {accuracy, per_class_precision/recall/f1, macro_f1}}
    """
    out = {}
    for t_idx, t_name in THREAT_NAMES.items():
        p_3 = to_3way(pred_arr[:, t_idx])
        l_3 = to_3way(label_arr[:, t_idx])
        acc = accuracy_score(l_3, p_3)
        prec, rec, f1, support = precision_recall_fscore_support(
            l_3, p_3, labels=[0, 1, 2], zero_division=0,
        )
        out[t_name] = {
            "accuracy": acc,
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
            "macro_f1": float(np.mean(f1)),
        }
    return out


def format_metrics_compact(metrics: dict) -> str:
    """1줄 요약: f1_safe/warn/danger (macro)."""
    lines = []
    for name, m in metrics.items():
        f1 = m["f1"]
        lines.append(
            f"{name[0]}: f1={f1[0]:.2f}/{f1[1]:.2f}/{f1[2]:.2f} (M={m['macro_f1']:.2f})"
        )
    return "  ".join(lines)


def format_metrics_detail(metrics: dict, prefix: str = "") -> None:
    print(f"{prefix}{'pair':<10s} {'class':<8s} {'precision':>10s} {'recall':>8s} "
          f"{'f1':>8s} {'support':>9s} {'accuracy':>10s}")
    for name, m in metrics.items():
        for c in range(3):
            cls = CLASS_NAMES[c]
            acc_str = f"{m['accuracy']:.4f}" if c == 0 else ""
            print(f"{prefix}{name if c == 0 else '':<10s} {cls:<8s} "
                  f"{m['precision'][c]:>10.4f} {m['recall'][c]:>8.4f} "
                  f"{m['f1'][c]:>8.4f} {m['support'][c]:>9d} {acc_str:>10s}")
        print(f"{prefix}{'':<10s} {'macro_f1':<8s} {'':>10s} {'':>8s} "
              f"{m['macro_f1']:>8.4f}")


# ── Main ────────────────────────────────────────────
def main(unity_dir: str | None = None):
    set_seed(SEED)
    # checkpoints/ 는 fusion/ root 에 있음 (이 파일은 fusion/training/ 에 있으므로 부모로 한 번 올라감)
    fusion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(fusion_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 페어별 best 체크포인트 + 평균 best (backward compat)
    pair_names = list(THREAT_NAMES.values())  # ["forklift", "dropzone"]
    ckpt_paths = {
        name: os.path.join(ckpt_dir, f"best_{name}.pt") for name in pair_names
    }
    ckpt_paths["avg"] = os.path.join(ckpt_dir, "best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # 데이터 — 합성 24 + (옵션) Unity 시나리오
    scenarios = build_synthetic_24(seed=SEED)
    print(f"[data] synthetic: {len(scenarios)}")
    if unity_dir:
        unity_scenarios = load_unity_scenarios(unity_dir)
        scenarios = scenarios + unity_scenarios
        print(f"[data] +unity     : {len(unity_scenarios)}  →  total: {len(scenarios)}")
    train_sc, val_sc = split_scenarios(scenarios, val_ratio=0.2, seed=SEED)
    train_ds = FusionDataset(train_sc)
    val_ds = FusionDataset(val_sc)
    print(f"train scenarios: {len(train_sc)}  windows: {len(train_ds)}")
    print(f"val   scenarios: {len(val_sc)}   windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 모델
    model = PairwiseInteractionFusionModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    criterion = BCEOnProb()

    # 페어별 best/patience 추적
    best_f1   = {name: -1.0 for name in pair_names}
    best_epoch = {name: 0   for name in pair_names}
    patience  = {name: 0   for name in pair_names}
    best_avg_f1, best_avg_epoch = -1.0, 0

    print(f"{'epoch':>5} {'tr_loss':>8} {'val_loss':>9}  {'tr_metrics':<35s}  "
          f"{'val_metrics':<35s}  time  saved")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_pred, tr_label = run_epoch(
            model, train_loader, device, optimizer, criterion,
        )
        val_loss, val_pred, val_label = run_epoch(
            model, val_loader, device, None, criterion,
        )
        elapsed = time.time() - t0

        tr_m = compute_metrics_per_pair(tr_pred, tr_label)
        val_m = compute_metrics_per_pair(val_pred, val_label)

        # 페어별 best 갱신
        saved_tags = []
        for name in pair_names:
            f1 = val_m[name]["macro_f1"]
            if f1 > best_f1[name]:
                best_f1[name] = f1
                best_epoch[name] = epoch
                patience[name] = 0
                torch.save({
                    "epoch": epoch,
                    "pair": name,
                    "model_state": model.state_dict(),
                    "val_metrics": val_m,
                }, ckpt_paths[name])
                saved_tags.append(f"*{name[0]}")
            else:
                patience[name] += 1

        # 평균 best (backward compat: best.pt)
        avg_f1 = float(np.mean([val_m[n]["macro_f1"] for n in pair_names]))
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_avg_epoch = epoch
            torch.save({
                "epoch": epoch,
                "pair": "avg",
                "model_state": model.state_dict(),
                "val_metrics": val_m,
            }, ckpt_paths["avg"])
            saved_tags.append("*avg")

        marker = " " + "".join(saved_tags) if saved_tags else ""
        print(f"{epoch:5d} {tr_loss:8.4f} {val_loss:9.4f}  "
              f"{format_metrics_compact(tr_m):<35s}  "
              f"{format_metrics_compact(val_m):<35s}  "
              f"{elapsed:5.2f}s{marker}")

        # Early stopping: 모든 페어가 patience 초과로 정체
        if all(patience[name] >= EARLY_STOP_PATIENCE for name in pair_names):
            print(f"\n[EarlyStopping] 모든 페어가 {EARLY_STOP_PATIENCE} epoch 정체 "
                  f"→ epoch {epoch}에서 중단")
            break

    # ── 학습 종료 요약 ──
    print()
    print(f"best per-pair F1:")
    for name in pair_names:
        print(f"  {name:<10s} F1={best_f1[name]:.4f} @ ep{best_epoch[name]}  "
              f"→ {ckpt_paths[name]}")
    print(f"  {'avg':<10s} F1={best_avg_f1:.4f} @ ep{best_avg_epoch}  "
          f"→ {ckpt_paths['avg']}")
    print()

    # ── 최종 상세 평가: 페어별 best ckpt 각각 로드 ──
    print(f"=== Final detail metrics @ per-pair best ckpts ===\n")
    for name in pair_names:
        state = torch.load(ckpt_paths[name], map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        val_loss, val_pred, val_label = run_epoch(
            model, val_loader, device, None, criterion,
        )
        val_m = compute_metrics_per_pair(val_pred, val_label)
        print(f"[{name.upper()} best ckpt]  epoch={state['epoch']}  val_loss={val_loss:.4f}")
        # 자기 페어 metric만 강조 (다른 페어는 참고용)
        format_metrics_detail({name: val_m[name]}, prefix="  ")
        other = [n for n in pair_names if n != name][0]
        print(f"  (참고: {other} F1={val_m[other]['macro_f1']:.4f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unity-dir", default=None,
        help="Unity 가 export 한 JSON 디렉터리 (예: DangerSimulation/Assets/Output)",
    )
    args = parser.parse_args()
    main(unity_dir=args.unity_dir)
