"""Compare Fusion V2 predictions against V1 teacher labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FusionV2Dataset
from .inference import load_checkpoint
from .schema import DANGER_THRESHOLD, SAFE_THRESHOLD, THREAT_NAMES


def _class(scores: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.int64)
    out[(scores >= SAFE_THRESHOLD) & (scores < DANGER_THRESHOLD)] = 1
    out[scores >= DANGER_THRESHOLD] = 2
    return out


def _metrics(prob: np.ndarray, target: np.ndarray) -> dict:
    pc = _class(prob)
    tc = _class(target)
    out: dict[str, object] = {"class_accuracy": float((pc == tc).mean())}
    for i, threat in enumerate(THREAT_NAMES):
        pred_d = prob[:, i] >= DANGER_THRESHOLD
        true_d = target[:, i] >= DANGER_THRESHOLD
        tp = int(np.logical_and(pred_d, true_d).sum())
        fp = int(np.logical_and(pred_d, ~true_d).sum())
        fn = int(np.logical_and(~pred_d, true_d).sum())
        tn = int(np.logical_and(~pred_d, ~true_d).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        out[threat] = {
            "danger_accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
            "danger_precision": float(precision),
            "danger_recall": float(recall),
            "danger_f1": float(f1),
            "support_danger": int(true_d.sum()),
            "predicted_danger": int(pred_d.sum()),
        }
    return out


def evaluate(dataset: Path, checkpoint: Path, output_dir: Path, batch_size: int = 256) -> dict:
    model, payload = load_checkpoint(checkpoint)
    mean = payload["mean"].astype(np.float32)
    std = payload["std"].astype(np.float32)
    ds = FusionV2Dataset(dataset, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    logit = np.concatenate(logits)
    target = np.concatenate(targets)
    prob = 1.0 / (1.0 + np.exp(-logit))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _metrics(prob, target)
    with (output_dir / "v1_v2_comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    data = np.load(dataset, allow_pickle=True)
    scenarios = data["scenario"].astype(str)
    workers = data["worker_id"].astype(str)
    frames = data["end_frame"]
    with (output_dir / "v1_v2_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "worker_id", "end_frame",
            "v1_forklift_target", "v2_forklift_prob",
            "v1_dropzone_target", "v2_dropzone_prob",
        ])
        for i in range(len(prob)):
            writer.writerow([
                scenarios[i], workers[i], int(frames[i]),
                round(float(target[i, 0]), 4), round(float(prob[i, 0]), 4),
                round(float(target[i, 1]), 4), round(float(prob[i, 1]), 4),
            ])
    print(f"[fusion-v2] summary: {output_dir / 'v1_v2_comparison_summary.json'}")
    print(f"[fusion-v2] predictions: {output_dir / 'v1_v2_predictions.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fusion V2 vs V1 teacher")
    parser.add_argument("--dataset", type=Path, default=Path("model/fusion_v2/data/fusion_v2_dataset.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/fusion_v2/checkpoints/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/fusion_v2/reports"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    summary = evaluate(args.dataset, args.checkpoint, args.output_dir, args.batch_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
