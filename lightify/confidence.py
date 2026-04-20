"""Confidence scoring.

- Validates success_count <= usage_count
- Normalizes to [0, 1] with documented formula
- Platt-style sigmoid shaping (NOTE: A/B params are NOT fit to data —
  they are hand-tuned defaults. Calibrate on held-out set for production.)
"""
from __future__ import annotations

import math
import time

from lightify.types import MemoryItem, Tier

# Tier weights — configurable, should be calibrated per deployment
TIER_WEIGHTS = {Tier.SMALL: 0.55, Tier.MID: 0.70, Tier.FRONTIER: 0.90}

# Formula weights
W_TIER = 0.35
W_SUCCESS = 0.40
W_RECENCY = 0.25

# Calibration parameters (Platt scaling: P(correct) = 1 / (1 + exp(A*s + B)))
# Default: identity mapping (A=-1, B=0 gives sigmoid that passes through 0.5 at s=0)
_PLATT_A = -5.0  # steepness
_PLATT_B = 2.5   # shift


def compute_raw_confidence(item: MemoryItem) -> float:
    """Compute uncalibrated confidence in [0, 1]."""
    item.validate()

    tier_w = TIER_WEIGHTS.get(item.source_tier, 0.70)
    usage = max(1, item.usage_count)
    success_rate = min(item.success_count, usage) / usage

    now = int(time.time())
    age_days = max(0, (now - item.created_ts)) / 86400.0
    recency = 1.0 / (1.0 + age_days)

    raw = W_TIER * tier_w + W_SUCCESS * success_rate + W_RECENCY * recency
    return max(0.0, min(1.0, raw))


def calibrate(raw_score: float) -> float:
    """Apply Platt scaling to convert raw score to calibrated probability."""
    return 1.0 / (1.0 + math.exp(_PLATT_A * raw_score + _PLATT_B))


def compute_confidence(item: MemoryItem, calibrated: bool = True) -> float:
    """Compute confidence, optionally calibrated."""
    raw = compute_raw_confidence(item)
    if calibrated:
        return calibrate(raw)
    return raw
