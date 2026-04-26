"""
페어별 best 도입 결과 종합 시각화.

4개 서브플롯:
  (1) Loss 곡선 (train vs val)
  (2) Val Macro F1 곡선 (forklift, dropzone) + 페어별 best epoch 표시
  (3) 페어별 Macro F1 비교: 원본 단일 best vs 페어별 best
  (4) 페어별 best ckpt의 클래스별 P/R/F1
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "history.json")
CKPT_DIR = os.path.join(HERE, "checkpoints")
OUT = os.path.join(HERE, "results_summary.png")

# 원본 단일 best.pt @ epoch 83 (덮어쓰기 전 값, 대화 기록에서 확보)
ORIG = {
    "epoch": 83,
    "forklift": {
        "macro_f1": 0.8388,
        "precision": [0.9097, 1.0000, 1.0000],
        "recall":    [1.0000, 0.4154, 0.9545],
        "f1":        [0.9527, 0.5870, 0.9767],
    },
    "dropzone": {
        "macro_f1": 0.7338,
        "precision": [0.9290, 1.0000, 0.7333],
        "recall":    [1.0000, 0.2439, 1.0000],
        "f1":        [0.9632, 0.3922, 0.8462],
    },
}


def load_ckpt(name):
    return torch.load(os.path.join(CKPT_DIR, name), map_location="cpu", weights_only=True)


def main():
    with open(HIST) as f:
        h = json.load(f)
    ep = np.array(h["epoch"])

    ckpt_f = load_ckpt("best_forklift.pt")
    ckpt_d = load_ckpt("best_dropzone.pt")
    new_f = ckpt_f["val_metrics"]["forklift"]
    new_d = ckpt_d["val_metrics"]["dropzone"]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # ── (1) Loss 곡선 ─────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, h["tr_loss"], label="train", color="#1f77b4")
    ax.plot(ep, h["val_loss"], label="val", color="#d62728")
    min_val_ep = int(ep[np.argmin(h["val_loss"])])
    ax.axvline(min_val_ep, ls="--", c="gray", alpha=0.5,
               label=f"min val_loss @ ep{min_val_ep}")
    ax.set_title("(1) Loss curve — overfitting visible after ep~60",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── (2) Val Macro F1 곡선 ────────────────────
    ax = fig.add_subplot(gs[0, 1])
    val_ff = np.array(h["val_f1_forklift"])
    val_fd = np.array(h["val_f1_dropzone"])
    ax.plot(ep, val_ff, label="val F1 forklift", color="#2ca02c")
    ax.plot(ep, val_fd, label="val F1 dropzone", color="#ff7f0e")
    best_f_ep = int(ep[np.argmax(val_ff)])
    best_d_ep = int(ep[np.argmax(val_fd)])
    ax.axvline(best_f_ep, ls="--", c="#2ca02c", alpha=0.6,
               label=f"forklift best @ ep{best_f_ep}")
    ax.axvline(best_d_ep, ls="--", c="#ff7f0e", alpha=0.6,
               label=f"dropzone best @ ep{best_d_ep}")
    ax.set_ylim(0, 1)
    ax.set_title("(2) Val Macro F1 — pairs peak at different epochs",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("macro F1")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    # ── (3) 페어별 Macro F1 비교 ─────────────────
    ax = fig.add_subplot(gs[1, 0])
    labels = ["Forklift", "Dropzone"]
    orig_vals = [ORIG["forklift"]["macro_f1"], ORIG["dropzone"]["macro_f1"]]
    new_vals = [new_f["macro_f1"], new_d["macro_f1"]]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, orig_vals, w, label="Original (single best.pt @ ep83)",
                color="#888888")
    b2 = ax.bar(x + w/2, new_vals, w, label="Per-pair best (dual ckpt)",
                color="#2ca02c")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Macro F1")
    delta_f = new_vals[0] - orig_vals[0]
    delta_d = new_vals[1] - orig_vals[1]
    ax.set_title(f"(3) Per-pair Macro F1: forklift {delta_f:+.3f}, dropzone {delta_d:+.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3, axis="y")

    # ── (4) 페어별 best ckpt 클래스별 P/R/F1 ─────
    ax = fig.add_subplot(gs[1, 1])
    classes = ["safe", "warn", "danger"]
    metrics = ["P", "R", "F1"]
    # 막대 그룹: forklift safe-P, safe-R, safe-F1, warn-P, ... 6 cluster x 3 bars
    n_groups = 6   # (forklift_safe, forklift_warn, forklift_danger, dz_safe, dz_warn, dz_danger)
    n_bars = 3     # P/R/F1
    group_labels = []
    group_data = {"P": [], "R": [], "F1": []}
    for pair_name, m in [("F", new_f), ("D", new_d)]:
        for ci, cls in enumerate(classes):
            group_labels.append(f"{pair_name}\n{cls}")
            group_data["P"].append(m["precision"][ci])
            group_data["R"].append(m["recall"][ci])
            group_data["F1"].append(m["f1"][ci])
    x = np.arange(n_groups)
    w = 0.27
    colors = {"P": "#1f77b4", "R": "#d62728", "F1": "#9467bd"}
    for i, met in enumerate(metrics):
        offset = (i - 1) * w
        ax.bar(x + offset, group_data[met], w, label=met, color=colors[met])
    # forklift / dropzone 구분선
    ax.axvline(2.5, c="black", lw=0.8, alpha=0.5)
    ax.text(1.0, 1.06, "Forklift (best_forklift.pt)", ha="center", fontsize=10,
            fontweight="bold", color="#2ca02c")
    ax.text(4.0, 1.06, "Dropzone (best_dropzone.pt)", ha="center", fontsize=10,
            fontweight="bold", color="#ff7f0e")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("score")
    ax.set_title("(4) Per-class Precision / Recall / F1 (per-pair best ckpts)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Fusion Model — 페어별 best 도입 결과  "
        f"(forklift best ep{ckpt_f['epoch']}, dropzone best ep{ckpt_d['epoch']})",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
