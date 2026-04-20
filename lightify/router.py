"""CDDR — Confidence-Driven Dynamic Routing.

Formal definition (from OpenAI reviewer):
  R(q, C) = t_j where j = min{m : Phi(C) >= tau_m OR m = 3}
  with thresholds tau_1 > tau_2 > 0

Supports both serial cascade and parallel dispatch (Meta reviewer).
"""
from __future__ import annotations

from lightify.types import ContextCapsule, RouteDecision, Tier
from lightify.sufficiency import sufficiency_score

# Routing thresholds: confidence needed to use each tier
# tau_1 > tau_2 > 0 — higher confidence = cheaper tier acceptable
# Tier-1 is local (free), so be aggressive — try it whenever context is decent
TAU_TIER1 = 0.45   # local model is free, low risk to try
TAU_TIER2 = 0.30   # moderate confidence for mid model (Sonnet)
# Below tau_2 → frontier (tier3)


class Router:
    def __init__(
        self,
        tau_tier1: float = TAU_TIER1,
        tau_tier2: float = TAU_TIER2,
        parallel_dispatch: bool = False,
    ):
        self.tau_tier1 = tau_tier1
        self.tau_tier2 = tau_tier2
        self.parallel_dispatch = parallel_dispatch

    def route(self, capsule: ContextCapsule) -> RouteDecision:
        """Route query to appropriate model tier based on context confidence.

        Formal routing function:
            R(q, C) = Tier1  if  Phi(C) >= tau_1 and S(C,q) = 1
                    = Tier2  if  Phi(C) >= tau_2
                    = Tier3  otherwise
        """
        phi = capsule.context_confidence
        suff = sufficiency_score(capsule)
        has_conflicts = len(capsule.conflicts) > 0

        # Conflicts always escalate to at least Tier-2
        if has_conflicts:
            if phi >= self.tau_tier2:
                return RouteDecision(
                    tier=Tier.MID,
                    reason=f"conflicts detected ({len(capsule.conflicts)}), escalating to mid",
                    parallel=self.parallel_dispatch,
                )
            return RouteDecision(
                tier=Tier.FRONTIER,
                reason=f"conflicts + low confidence ({phi:.2f}), using frontier",
                parallel=self.parallel_dispatch,
            )

        if phi >= self.tau_tier1 and suff >= 0.7:
            return RouteDecision(
                tier=Tier.SMALL,
                reason=f"high confidence ({phi:.2f}), sufficient context ({suff:.2f})",
                parallel=self.parallel_dispatch,
            )

        if phi >= self.tau_tier2:
            return RouteDecision(
                tier=Tier.MID,
                reason=f"moderate confidence ({phi:.2f})",
                parallel=self.parallel_dispatch,
            )

        return RouteDecision(
            tier=Tier.FRONTIER,
            reason=f"low confidence ({phi:.2f}), using frontier",
            parallel=self.parallel_dispatch,
        )
