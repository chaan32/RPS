"""
train.py 재현 + epoch별 history(loss, macro F1) 저장.

- best.pt는 건드리지 않음 (별도 best_history.pt에 저장).
- history는 JSON으로 저장 → plot_history.py 가 읽음.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import (
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, SEED,
    BCEOnProb, set_seed, run_epoch, compute_metrics_per_pair,
)
from ..data.scenarios_synthetic import build_synthetic_24
from .dataset import FusionDataset, split_scenarios
from ..model import PairwiseInteractionFusionModel
from ..graph_input import THREAT_FORKLIFT, THREAT_DROPZONE

THREAT_NAMES = {THREAT_FORKLIFT: "forklift", THREAT_DROPZONE: "dropzone"}


def main():
    set_seed(SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    fusion_root = os.path.dirname(here)   # training/ → fusion/
    history_path = os.path.join(here, "history.json")
    ckpt_path = os.path.join(fusion_root, "checkpoints", "best_history.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    scenarios = build_synthetic_24(seed=SEED)
    train_sc, val_sc = split_scenarios(scenarios, val_ratio=0.2, seed=SEED)
    train_ds = FusionDataset(train_sc)
    val_ds = FusionDataset(val_sc)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PairwiseInteractionFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = BCEOnProb()

    history = {
        "epoch": [],
        "tr_loss": [], "val_loss": [],
        "tr_f1_forklift": [], "val_f1_forklift": [],
        "tr_f1_dropzone": [], "val_f1_dropzone": [],
        "tr_acc_forklift": [], "val_acc_forklift": [],
        "tr_acc_dropzone": [], "val_acc_dropzone": [],
    }

    best_macro_f1 = -1.0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_pred, tr_label = run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_pred, val_label = run_epoch(model, val_loader, device, None, criterion)
        elapsed = time.time() - t0

        tr_m = compute_metrics_per_pair(tr_pred, tr_label)
        val_m = compute_metrics_per_pair(val_pred, val_label)

        history["epoch"].append(epoch)
        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["tr_f1_forklift"].append(tr_m["forklift"]["macro_f1"])
        history["val_f1_forklift"].append(val_m["forklift"]["macro_f1"])
        history["tr_f1_dropzone"].append(tr_m["dropzone"]["macro_f1"])
        history["val_f1_dropzone"].append(val_m["dropzone"]["macro_f1"])
        history["tr_acc_forklift"].append(tr_m["forklift"]["accuracy"])
        history["val_acc_forklift"].append(val_m["forklift"]["accuracy"])
        history["tr_acc_dropzone"].append(tr_m["dropzone"]["accuracy"])
        history["val_acc_dropzone"].append(val_m["dropzone"]["accuracy"])

        avg_macro = float(np.mean([val_m[n]["macro_f1"] for n in THREAT_NAMES.values()]))
        improved = avg_macro > best_macro_f1
        marker = " *" if improved else ""

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS or improved:
            print(f"ep{epoch:3d} tr={tr_loss:.3f} val={val_loss:.3f} "
                  f"trF1=[{tr_m['forklift']['macro_f1']:.3f},{tr_m['dropzone']['macro_f1']:.3f}] "
                  f"valF1=[{val_m['forklift']['macro_f1']:.3f},{val_m['dropzone']['macro_f1']:.3f}] "
                  f"{elapsed:.1f}s{marker}")

        if improved:
            best_macro_f1 = avg_macro
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
            }, ckpt_path)

    total = time.time() - t_start
    print(f"\nbest avg macro F1 = {best_macro_f1:.4f}  total={total:.1f}s")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"history → {history_path}")


if __name__ == "__main__":
    main()
