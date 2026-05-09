"""학습된 dual fusion 모델 진단 도구.

실행:
    python -m model.fusion.training.analyze_fusion
    python -m model.fusion.training.analyze_fusion --split all
    python -m model.fusion.training.analyze_fusion --ckpt-dir <path> --out-dir <path>

생성물 (out-dir 안에):
    confusion_matrix.png   - per-pair × 3-class 혼동 행렬
    pr_curves.png          - per-pair Precision-Recall 곡선 + AP
    misclassified.txt      - 오분류 윈도우 리스트 (디버깅용)
    summary.json           - 정량 지표 (accuracy / macro F1 / per-class F1 / support)

기본 동작:
    val split (전체 24 시나리오 중 ~5개) 에 대해 dual model 추론 후 분석.
    --split all 로 전체 시나리오 분석 가능 (학습 데이터 포함이라 낙관적 수치).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

from ..data.scenarios_synthetic import build_synthetic_24
from ..graph_input import THREAT_DROPZONE, THREAT_FORKLIFT
from ..inference import load_dual_model
from .dataset import FusionDataset, split_scenarios


# train.py 와 일관된 임계값
THRESH_WARN = 0.4
THRESH_DANGER = 0.7
CLASS_NAMES = ["safe", "warn", "danger"]
THREAT_NAMES = {THREAT_FORKLIFT: "forklift", THREAT_DROPZONE: "dropzone"}


def to_3way(probs: np.ndarray) -> np.ndarray:
    """확률 → 0=safe, 1=warn, 2=danger (train.py 와 동일 규칙)."""
    out = np.zeros_like(probs, dtype=np.int64)
    out[(probs >= THRESH_WARN) & (probs < THRESH_DANGER)] = 1
    out[probs >= THRESH_DANGER] = 2
    return out


def collect_predictions(model, scenarios, device: str = "cpu"):
    """val/test 시나리오를 윈도우 단위로 추론.

    Returns:
        preds:  (W, K) — W 개 윈도우의 K=2 위협 확률
        labels: (W, K) — 같은 모양의 정답 확률
        names:  list[str] — 각 윈도우의 시나리오 이름
    """
    dataset = FusionDataset(scenarios)

    all_pred = []
    all_label = []
    all_name = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            nodes, adj, scene, label = dataset[i]
            n = nodes.unsqueeze(0).to(device)
            a = adj.unsqueeze(0).to(device)
            s = scene.unsqueeze(0).to(device)

            pred = model(n, a, s)              # (1, N=1, K=2)
            pred_np = pred[0, 0].cpu().numpy()  # (K,)
            label_np = label[0].cpu().numpy()    # (K,)

            all_pred.append(pred_np)
            all_label.append(label_np)
            all_name.append(dataset.windows[i]["scenario"])

    return np.array(all_pred), np.array(all_label), all_name


def plot_confusion_matrices(preds: np.ndarray, labels: np.ndarray, output_path: Path):
    """per-pair × 3-class 혼동 행렬 (forklift / dropzone 두 subplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (pair_idx, pair_name) in enumerate(THREAT_NAMES.items()):
        p = to_3way(preds[:, pair_idx])
        l = to_3way(labels[:, pair_idx])
        cm = confusion_matrix(l, p, labels=[0, 1, 2])

        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{pair_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)

        # 셀에 숫자 + 적절한 색
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for r in range(3):
            for c in range(3):
                ax.text(
                    c, r, cm[r, c],
                    ha="center", va="center",
                    color="black" if cm[r, c] < thresh else "white",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Confusion Matrix per Pair", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(preds: np.ndarray, labels: np.ndarray, output_path: Path):
    """per-pair PR 곡선. positive = label >= THRESH_DANGER (binary)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for pair_idx, pair_name in THREAT_NAMES.items():
        binary_label = (labels[:, pair_idx] >= THRESH_DANGER).astype(int)
        scores = preds[:, pair_idx]

        if binary_label.sum() == 0:
            print(f"[skip] {pair_name}: positive 샘플 0 → PR 곡선 스킵")
            continue

        prec, rec, _ = precision_recall_curve(binary_label, scores)
        ap = average_precision_score(binary_label, scores)
        ax.plot(rec, prec, label=f"{pair_name} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves (positive = label ≥ {THRESH_DANGER})")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def collect_misclassified(
    preds: np.ndarray, labels: np.ndarray, scenario_names: list[str],
) -> list[dict]:
    """3-class 기준 오분류 윈도우 모음."""
    miss = []
    for i, scenario_name in enumerate(scenario_names):
        for pair_idx, pair_name in THREAT_NAMES.items():
            p_3 = to_3way(np.array([preds[i, pair_idx]]))[0]
            l_3 = to_3way(np.array([labels[i, pair_idx]]))[0]
            if p_3 != l_3:
                miss.append({
                    "scenario": scenario_name,
                    "pair": pair_name,
                    "pred_prob": float(preds[i, pair_idx]),
                    "pred_class": CLASS_NAMES[p_3],
                    "label_prob": float(labels[i, pair_idx]),
                    "label_class": CLASS_NAMES[l_3],
                    "delta": int(p_3) - int(l_3),
                })
    return miss


def compute_summary(preds: np.ndarray, labels: np.ndarray) -> dict:
    """정량 지표 (per-pair)."""
    summary = {}
    for pair_idx, pair_name in THREAT_NAMES.items():
        p_3 = to_3way(preds[:, pair_idx])
        l_3 = to_3way(labels[:, pair_idx])
        summary[pair_name] = {
            "accuracy": float(accuracy_score(l_3, p_3)),
            "f1_macro": float(
                f1_score(l_3, p_3, labels=[0, 1, 2], average="macro", zero_division=0)
            ),
            "f1_per_class": f1_score(
                l_3, p_3, labels=[0, 1, 2], average=None, zero_division=0
            ).tolist(),
            "support_per_class": [int((l_3 == c).sum()) for c in range(3)],
        }
    return summary


def write_misclassified(miss: list[dict], total: int, output_path: Path):
    """오분류 리스트를 가독성 있는 텍스트로 저장."""
    by_pair: dict[str, list[dict]] = {"forklift": [], "dropzone": []}
    for m in miss:
        by_pair[m["pair"]].append(m)

    rate = 100 * len(miss) / max(1, total)
    lines = [
        f"# 오분류 윈도우 리포트",
        f"#   total mistakes: {len(miss)} / {total}  ({rate:.1f}%)",
        f"#   total windows : {total // 2}  (× 2 pair = {total})",
        "",
    ]

    for pair_name, items in by_pair.items():
        lines.append(f"## {pair_name}  ({len(items)} mistakes)")
        if not items:
            lines.append("  (없음)")
            lines.append("")
            continue

        # 시나리오별 묶기
        by_scenario: dict[str, list[dict]] = {}
        for it in items:
            by_scenario.setdefault(it["scenario"], []).append(it)

        for sc_name, sc_items in sorted(by_scenario.items()):
            lines.append(f"  [{sc_name}]  ({len(sc_items)} windows)")
            for m in sc_items[:5]:  # 샘플 5개만
                lines.append(
                    f"    pred={m['pred_prob']:.2f} ({m['pred_class']:6s}) ↔ "
                    f"label={m['label_prob']:.2f} ({m['label_class']:6s})  "
                    f"Δ={m['delta']:+d}"
                )
            if len(sc_items) > 5:
                lines.append(f"    ... ({len(sc_items) - 5} more)")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir", default="model/fusion/checkpoints",
        help="체크포인트 디렉터리 (best_forklift.pt + best_dropzone.pt)",
    )
    parser.add_argument(
        "--out-dir", default="model/fusion/training",
        help="결과 파일 저장 위치",
    )
    parser.add_argument(
        "--split", choices=["val", "all"], default="val",
        help="val (학습 안 본 시나리오만) / all (전체, 학습 데이터 포함)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 모델 로드
    print(f"[load] dual model from {args.ckpt_dir}")
    model = load_dual_model(args.ckpt_dir, device="cpu")

    # 2. 데이터셋 구축
    print(f"[data] building scenarios...")
    scenarios = build_synthetic_24()
    if args.split == "val":
        _, scenarios = split_scenarios(scenarios)
        print(f"[data] val split: {len(scenarios)} scenarios")
    else:
        print(f"[data] all scenarios: {len(scenarios)}")

    # 3. 추론
    print(f"[infer] running predictions...")
    preds, labels, names = collect_predictions(model, scenarios)
    print(f"[infer] total windows: {len(preds)}")

    # 4. 시각화 + 통계
    cm_path = out_dir / "confusion_matrix.png"
    pr_path = out_dir / "pr_curves.png"
    miss_path = out_dir / "misclassified.txt"
    summary_path = out_dir / "summary.json"

    plot_confusion_matrices(preds, labels, cm_path)
    print(f"[viz] confusion matrix → {cm_path}")

    plot_pr_curves(preds, labels, pr_path)
    print(f"[viz] PR curves → {pr_path}")

    miss = collect_misclassified(preds, labels, names)
    write_misclassified(miss, total=len(preds) * 2, output_path=miss_path)
    print(f"[viz] misclassified ({len(miss)}건) → {miss_path}")

    summary = compute_summary(preds, labels)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[viz] summary → {summary_path}")

    # 5. 콘솔 요약
    print(f"\n=== Summary ===")
    for pair_name, m in summary.items():
        f1s = m["f1_per_class"]
        sup = m["support_per_class"]
        print(
            f"  {pair_name:8s}  acc={m['accuracy']:.3f}  "
            f"macro_F1={m['f1_macro']:.3f}  "
            f"per_class_F1=[{f1s[0]:.2f}/{f1s[1]:.2f}/{f1s[2]:.2f}]  "
            f"support={sup}"
        )


if __name__ == "__main__":
    main()
