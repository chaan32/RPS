"""Fusion V2: deep-learning based risk prediction.

Version 1 remains in ``model.fusion`` and the runtime rule layer.  This package
is intentionally separate so V2 experiments cannot break the working pipeline.
"""

from .model import TemporalRiskPredictor

__all__ = ["TemporalRiskPredictor"]
