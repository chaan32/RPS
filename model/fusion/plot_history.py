"""history.json 읽어서 학습 곡선 + train/val gap 분석."""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "history.json")
OUT_PNG = os.path.join(HERE, "history_curves.png")


def main():
    with open(HIST) as f:
        h = json.load(f)

    ep = np.array(h["epoch"])
    tr_loss = np.array(h["tr_loss"])
    val_loss = np.array(h["val_loss"])
    tr_ff = np.array(h["tr_f1_forklift"]); val_ff = np.array(h["val_f1_forklift"])
    tr_fd = np.array(h["tr_f1_dropzone"]); val_fd = np.array(h["val_f1_dropzone"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) Loss
    ax = axes[0, 0]
    ax.plot(ep, tr_loss, label="train", color="#1f77b4")
    ax.plot(ep, val_loss, label="val",   color="#d62728")
    best_val_loss_ep = int(ep[np.argmin(val_loss)])
    ax.axvline(best_val_loss_ep, ls="--", c="gray", alpha=0.5,
               label=f"min val_loss @ ep{best_val_loss_ep}")
    ax.set_title("Loss (BCE)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) Forklift macro F1
    ax = axes[0, 1]
    ax.plot(ep, tr_ff, label="train", color="#1f77b4")
    ax.plot(ep, val_ff, label="val",   color="#d62728")
    best_ep_f = int(ep[np.argmax(val_ff)])
    ax.axvline(best_ep_f, ls="--", c="gray", alpha=0.5,
               label=f"best val F1 @ ep{best_ep_f}")
    ax.set_title("Forklift  macro F1"); ax.set_xlabel("epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # (3) Dropzone macro F1
    ax = axes[1, 0]
    ax.plot(ep, tr_fd, label="train", color="#1f77b4")
    ax.plot(ep, val_fd, label="val",   color="#d62728")
    best_ep_d = int(ep[np.argmax(val_fd)])
    ax.axvline(best_ep_d, ls="--", c="gray", alpha=0.5,
               label=f"best val F1 @ ep{best_ep_d}")
    ax.set_title("Dropzone  macro F1"); ax.set_xlabel("epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # (4) Train-Val gap
    ax = axes[1, 1]
    ax.plot(ep, tr_loss - val_loss, label="loss gap (tr - val)", color="#9467bd")
    ax.plot(ep, tr_ff - val_ff,    label="F1 gap (forklift)",   color="#2ca02c")
    ax.plot(ep, tr_fd - val_fd,    label="F1 gap (dropzone)",   color="#ff7f0e")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_title("Train - Val gap (positive = overfit signal)")
    ax.set_xlabel("epoch"); ax.set_ylabel("gap")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"saved → {OUT_PNG}")

    # ── 텍스트 요약 ────────────────────────────────
    avg_val_f1 = (val_ff + val_fd) / 2
    best_avg_ep = int(ep[np.argmax(avg_val_f1)])

    print()
    print("=" * 70)
    print("학습 진행 요약")
    print("=" * 70)
    print(f"총 epoch: {len(ep)}")
    print(f"min train loss: {tr_loss.min():.4f} (epoch {int(ep[tr_loss.argmin()])})")
    print(f"min  val  loss: {val_loss.min():.4f} (epoch {best_val_loss_ep})")
    print(f"max val F1 forklift : {val_ff.max():.4f} (epoch {best_ep_f})")
    print(f"max val F1 dropzone : {val_fd.max():.4f} (epoch {best_ep_d})")
    print(f"max val F1 평균     : {avg_val_f1.max():.4f} (epoch {best_avg_ep})")
    print()

    print("=" * 70)
    print("Train vs Val Gap (오버피팅 점검)")
    print("=" * 70)

    final = -1
    print(f"{'구간':<12}{'tr_loss':>9}{'val_loss':>10}{'gap':>9}"
          f"{'tr_F1_avg':>11}{'val_F1_avg':>12}{'F1_gap':>9}")
    for label, idx in [
        ("초반 (ep10)", 9), ("중반 (ep30)", 29),
        ("ep60",       59), ("best (ep83)", 82),
        ("최종 (last)", final),
    ]:
        tr_f_avg  = (tr_ff[idx]  + tr_fd[idx])  / 2
        val_f_avg = (val_ff[idx] + val_fd[idx]) / 2
        print(f"{label:<12}{tr_loss[idx]:>9.4f}{val_loss[idx]:>10.4f}"
              f"{tr_loss[idx]-val_loss[idx]:>+9.4f}"
              f"{tr_f_avg:>11.4f}{val_f_avg:>12.4f}"
              f"{tr_f_avg-val_f_avg:>+9.4f}")

    # 진단
    print()
    last_loss_gap = val_loss[-1] - tr_loss[-1]
    last_f1_gap = ((tr_ff[-1] + tr_fd[-1]) - (val_ff[-1] + val_fd[-1])) / 2
    print("=" * 70)
    print("진단")
    print("=" * 70)
    print(f"최종 val_loss - tr_loss = {last_loss_gap:+.4f}")
    print(f"최종 F1_gap (tr - val)   = {last_f1_gap:+.4f}")
    if val_loss[-1] > val_loss[best_val_loss_ep] * 1.5:
        print("  → val_loss가 최저점 대비 50%+ 상승. 명확한 오버피팅.")
    if last_f1_gap > 0.10:
        print(f"  → train-val F1 gap > 0.10. 일반화 부족.")
    print(f"  → best.pt(epoch 83)는 avg val F1 최대 시점 — 학습은 더 일찍 끊는 게 정답.")
    print(f"  → early stopping 권장 epoch ≈ {best_avg_ep}")


if __name__ == "__main__":
    main()
