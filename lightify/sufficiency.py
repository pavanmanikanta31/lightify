"""CSE — Context Sufficiency Estimation.

Formal definition (from OpenAI reviewer):
  S(C, q) = 1 iff Coverage(C, q) >= sigma AND Phi(C) >= sigma_conf

If insufficient, triggers deeper retrieval or model tier escalation.
"""
from __future__ import annotations

from lightify.types import ContextCapsule


# Thresholds — should be calibrated per deployment
COVERAGE_THRESHOLD = 0.3     # sigma: fraction of query tokens in context
CONFIDENCE_THRESHOLD = 0.35  # sigma_conf: minimum context confidence
MIN_ITEMS = 1                # minimum number of context items


def estimate_sufficiency(capsule: ContextCapsule) -> bool:
    """Estimate whether context is sufficient for cheap inference.

    Returns True if context is likely sufficient, False if more
    retrieval or tier escalation is needed.
    """
    if capsule.num_items < MIN_ITEMS:
        return False
    if capsule.coverage < COVERAGE_THRESHOLD:
        return False
    if capsule.context_confidence < CONFIDENCE_THRESHOLD:
        return False
    return True


def sufficiency_score(capsule: ContextCapsule) -> float:
    """Continuous sufficiency score in [0, 1] for routing decisions."""
    if capsule.num_items == 0:
        return 0.0

    # Weighted combination of coverage and confidence
    cov_norm = min(1.0, capsule.coverage / COVERAGE_THRESHOLD) if COVERAGE_THRESHOLD > 0 else 1.0
    conf_norm = min(1.0, capsule.context_confidence / CONFIDENCE_THRESHOLD) if CONFIDENCE_THRESHOLD > 0 else 1.0
    item_norm = min(1.0, capsule.num_items / max(1, MIN_ITEMS))

    return 0.4 * cov_norm + 0.4 * conf_norm + 0.2 * item_norm
