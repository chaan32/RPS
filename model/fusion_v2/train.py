"""Train Fusion V2 deep-learning risk predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FusionV2Dataset, build_dataset, load_metadata
from .model import TemporalRiskPredictor
from .schema import DANGER_THRESHOLD, FEATURE_COLUMNS, SAFE_THRESHOLD, THREAT_NAMES


def _split_by_scenario(npz_path: Path, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    scenarios = data["scenario"].astype(str)
    augmented = data["augmented"].astype(np.int8)

    unique = sorted(set(scenarios.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_scenarios = set(unique[:n_val])

    train_idx = np.where(~np.isin(scenarios, list(val_scenarios)))[0]
    val_idx = np.where((np.isin(scenarios, list(val_scenarios))) & (augmented == 0))[0]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _normalizer(npz_path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"][indices].astype(np.float32)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float32)
    std = x.reshape(-1, x.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _score_to_class(scores: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.int64)
    out[(scores >= SAFE_THRESHOLD) & (scores < DANGER_THRESHOLD)] = 1
    out[scores >= DANGER_THRESHOLD] = 2
    return out


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_prob = 1.0 / (1.0 + np.exp(-pred))
    pred_class = _score_to_class(pred_prob)
    true_class = _score_to_class(target)

    out: dict[str, object] = {
        "class_accuracy": float((pred_class == true_class).mean()),
    }
    for idx, threat in enumerate(THREAT_NAMES):
        pred_danger = pred_prob[:, idx] >= DANGER_THRESHOLD
        true_danger = target[:, idx] >= DANGER_THRESHOLD
        tp = int(np.logical_and(pred_danger, true_danger).sum())
        fp = int(np.logical_and(pred_danger, ~true_danger).sum())
        fn = int(np.logical_and(~pred_danger, true_danger).sum())
        tn = int(np.logical_and(~pred_danger, ~true_danger).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        out[threat] = {
            "danger_accuracy": float(accuracy),
            "danger_precision": float(precision),
            "danger_recall": float(recall),
            "danger_f1": float(f1),
            "support_danger": int(true_danger.sum()),
        }
    return out


def _run_epoch(
    model: TemporalRiskPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        # Danger/warning samples matter more than safe samples in this project.
        sample_weight = 1.0 + 2.0 * y.max(dim=1, keepdim=True).values
        loss_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss_raw * sample_weight).mean()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(float(loss.item()))
        preds.append(logits.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(preds) if preds else np.zeros((0, 2), dtype=np.float32),
        np.concatenate(targets) if targets else np.zeros((0, 2), dtype=np.float32),
    )


def train(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.build_dataset or not args.dataset.exists():
        build_dataset(
            input_root=args.input_root if isinstance(args.input_root, list) else [args.input_root],
            output_path=args.dataset,
            window_size=args.window_size,
            stride=args.stride,
            augment=args.augment,
            noise_std=args.noise_std,
            seed=args.seed,
            label_mode=args.label_mode,
            future_horizon_frames=args.future_horizon_frames,
            forklift_danger_m=args.forklift_danger_m,
            forklift_warning_m=args.forklift_warning_m,
            dropzone_danger_m=args.dropzone_danger_m,
            dropzone_warning_m=args.dropzone_warning_m,
        )

    train_idx, val_idx = _split_by_scenario(args.dataset, args.val_ratio, args.seed)
    mean, std = _normalizer(args.dataset, train_idx)
    train_ds = FusionV2Dataset(args.dataset, train_idx, mean, std)
    val_ds = FusionV2Dataset(args.dataset, val_idx, mean, std)

    if len(val_ds) == 0:
        raise RuntimeError("validation split is empty")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = TemporalRiskPredictor(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_payload = None
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_pred, val_target = _run_epoch(model, val_loader, device, None)
        val_metrics = _metrics(val_pred, val_target)
        mean_f1 = float(np.mean([
            val_metrics[t]["danger_f1"] for t in THREAT_NAMES
        ]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_danger_f1": mean_f1,
            "val_class_accuracy": val_metrics["class_accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={mean_f1:.3f} "
            f"val_acc={val_metrics['class_accuracy']:.3f}"
        )

        if mean_f1 >= best_f1:
            best_f1 = mean_f1
            best_payload = {
                "model_state": model.state_dict(),
                "mean": mean,
                "std": std,
                "feature_columns": list(FEATURE_COLUMNS),
                "model_config": {
                    "input_dim": len(FEATURE_COLUMNS),
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.layers,
                    "dropout": args.dropout,
                },
                "metadata": load_metadata(args.dataset),
                "epoch": epoch,
                "metrics": val_metrics,
            }
            torch.save(best_payload, args.output_dir / "best.pt")

    summary = {
        "dataset": str(args.dataset),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
        "feature_dim": len(FEATURE_COLUMNS),
        "best_epoch": int(best_payload["epoch"] if best_payload else -1),
        "best_metrics": best_payload["metrics"] if best_payload else {},
        "history": history,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[fusion-v2] saved checkpoint: {args.output_dir / 'best.pt'}")
    print(f"[fusion-v2] saved summary: {args.output_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fusion V2 risk predictor")
    parser.add_argument("--dataset", type=Path, default=Path("model/fusion_v2/data/fusion_v2_dataset.npz"))
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        default=[Path("simulation/Recordings/collision_scenarios")],
    )
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--augment", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--label-mode", choices=("teacher", "geometry_future"), default="teacher")
    parser.add_argument("--future-horizon-frames", type=int, default=12)
    parser.add_argument("--forklift-danger-m", type=float, default=1.25)
    parser.add_argument("--forklift-warning-m", type=float, default=2.4)
    parser.add_argument("--dropzone-danger-m", type=float, default=2.0)
    parser.add_argument("--dropzone-warning-m", type=float, default=2.8)
    parser.add_argument("--output-dir", type=Path, default=Path("model/fusion_v2/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
