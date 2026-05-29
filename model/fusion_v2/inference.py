"""Inference helper for Fusion V2 checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import TemporalRiskPredictor
from .schema import DANGER_THRESHOLD, SAFE_THRESHOLD


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[TemporalRiskPredictor, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    feature_columns = payload["feature_columns"]
    model_config = payload.get("model_config") or {
        "input_dim": len(feature_columns),
    }
    model = TemporalRiskPredictor(**model_config)
    model.load_state_dict(payload["model_state"])
    model.eval().to(device)
    return model, payload


def predict_window(
    model: TemporalRiskPredictor,
    payload: dict,
    window: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Return forklift/dropzone risk probabilities for one window."""
    mean = payload["mean"].astype(np.float32)
    std = payload["std"].astype(np.float32)
    x = ((window.astype(np.float32) - mean) / std)[None]
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().to(device))
        return torch.sigmoid(logits)[0].cpu().numpy()


def risk_level(score: float) -> str:
    if score >= DANGER_THRESHOLD:
        return "DANGER"
    if score >= SAFE_THRESHOLD:
        return "WARNING"
    return "SAFE"
