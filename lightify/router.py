"""CDDR — Confidence-Driven Dynamic Routing, with per-action overlay.

Formal definition:
  R(q, C) = t_j where j = min{m : Phi(C) >= tau_m OR m = 3}
  with thresholds tau_1 > tau_2 > 0

When enable_action_routing=True, a per-action classifier (action_router.py)
runs first. Its suggestion is combined with CDDR's suggestion via combine_tiers.
"""
from __future__ import annotations

from lightify.types import ContextCapsule, RouteDecision, Tier
from lightify.sufficiency import sufficiency_score
from lightify.action_router import classify_action, combine_tiers

TAU_TIER1 = 0.45
TAU_TIER2 = 0.30


class Router:
    def __init__(
        self,
        tau_tier1: float = TAU_TIER1,
        tau_tier2: float = TAU_TIER2,
        parallel_dispatch: bool = False,
        enable_action_routing: bool = False,
    ):
        self.tau_tier1 = tau_tier1
        self.tau_tier2 = tau_tier2
        self.parallel_dispatch = parallel_dispatch
        self.enable_action_routing = enable_action_routing

    def _cddr_tier(self, capsule: ContextCapsule) -> tuple[Tier, str]:
        phi = capsule.context_confidence
        suff = sufficiency_score(capsule)
        has_conflicts = len(capsule.conflicts) > 0

        if has_conflicts:
            if phi >= self.tau_tier2:
                return Tier.MID, f"conflicts detected ({len(capsule.conflicts)}), escalating to mid"
            return Tier.FRONTIER, f"conflicts + low confidence ({phi:.2f}), using frontier"

        if phi >= self.tau_tier1 and suff >= 0.7:
            return Tier.SMALL, f"high confidence ({phi:.2f}), sufficient context ({suff:.2f})"

        if phi >= self.tau_tier2:
            return Tier.MID, f"moderate confidence ({phi:.2f})"

        return Tier.FRONTIER, f"low confidence ({phi:.2f}), using frontier"

    def route(self, capsule: ContextCapsule, query: str | None = None) -> RouteDecision:
        """Route query to appropriate model tier.

        If enable_action_routing and query is provided, the per-action classifier
        runs first; its suggestion is combined with CDDR's suggestion.
        """
        cddr_tier, cddr_reason = self._cddr_tier(capsule)

        if not self.enable_action_routing or not query:
            return RouteDecision(
                tier=cddr_tier,
                reason=cddr_reason,
                parallel=self.parallel_dispatch,
            )

        action = classify_action(query)
        final_tier = combine_tiers(cddr_tier, action.suggested_tier, action.action_class)
        reason = (
            f"action={action.action_class} (->{action.suggested_tier.value}); "
            f"cddr={cddr_tier.value} ({cddr_reason}); final={final_tier.value}"
        )
        return RouteDecision(
            tier=final_tier,
            reason=reason,
            parallel=self.parallel_dispatch,
        )
