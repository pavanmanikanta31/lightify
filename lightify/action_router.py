"""Per-action tier routing (OpenHands gap).

Classifies a query by the *kind of action* the LLM needs to produce,
and suggests a minimum tier before CDDR confidence routing is applied.
The final tier is the max of action-suggested and CDDR-suggested tier.

Design goals:
- Zero dependencies (pure regex).
- Deterministic: the same query always maps to the same suggestion.
- Conservative: unknown action types fall through to CDDR unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lightify.types import Tier


@dataclass
class ActionDecision:
    suggested_tier: Tier
    action_class: str
    reason: str


# Patterns that consistently indicate cheap/mechanical operations.
# If any match, we suggest SMALL (local tier) even before CDDR runs.
_BASH_PATTERNS = [
    r"\b(ls|cat|grep|find|pwd|cd|head|tail|wc|df|du|ps)\b",
    r"\bkubectl (get|describe|logs)\b",
    r"\bgit (status|log|diff|show|blame|branch)\b",
    r"\bdocker (ps|images|logs|inspect)\b",
    r"what (does|do|is) .{0,40}\b(command|flag|option|arg|argument)\b",
    r"\bhow (do|can) i (list|show|check|see|print)\b",
]

# Patterns that indicate a short factual lookup.
_LOOKUP_PATTERNS = [
    r"^what is (the )?[a-z0-9 ]{3,40}\??$",
    r"^define\b",
    r"^(who|when|where) (is|was|were)\b",
    r"^how many\b",
    r"^convert \d",
]

# Patterns that indicate heavy reasoning, long-form writing,
# or multi-step planning. These force at least MID (Tier-2) as a floor.
_REASONING_PATTERNS = [
    r"\b(design|architect|plan|strategy|roadmap)\b.{0,40}\b(system|pipeline|service)",
    r"\bcompare (.+) (and|vs|versus) (.+)\b",
    r"\bwhy (does|do|did|is|are)\b.{20,}",
    r"\bhow would you (implement|design|build)\b",
    r"\bmulti[- ]step\b",
    r"\bexplain in detail\b",
    r"\bpros and cons\b",
    r"\btrade[- ]?offs?\b",
]

# Patterns that indicate code generation / refactor — non-trivial but often
# tractable at MID. Only promoted to FRONTIER when combined with "large",
# "entire", "full", "all".
_CODE_PATTERNS = [
    r"\b(write|generate|refactor|fix|implement)\b.{0,60}\b(function|method|class|test|module|file|service|component|pipeline)\b",
    r"\bunit tests? for\b",
    r"```",
]
_FRONTIER_CODE_HINTS = [r"\bentire\b", r"\bfull\b", r"\ball (of )?the\b", r"\bwhole\b"]


_compiled = {
    "bash": [re.compile(p, re.I) for p in _BASH_PATTERNS],
    "lookup": [re.compile(p, re.I) for p in _LOOKUP_PATTERNS],
    "reasoning": [re.compile(p, re.I) for p in _REASONING_PATTERNS],
    "code": [re.compile(p, re.I) for p in _CODE_PATTERNS],
    "frontier_code_hints": [re.compile(p, re.I) for p in _FRONTIER_CODE_HINTS],
}


def classify_action(query: str) -> ActionDecision:
    """Classify a query by action type and suggest a minimum tier."""
    q = query.strip()

    for pat in _compiled["bash"]:
        if pat.search(q):
            return ActionDecision(
                suggested_tier=Tier.SMALL,
                action_class="bash_or_shell_like",
                reason=f"matches mechanical shell pattern: {pat.pattern}",
            )

    for pat in _compiled["lookup"]:
        if pat.search(q):
            return ActionDecision(
                suggested_tier=Tier.SMALL,
                action_class="short_lookup",
                reason=f"matches short-lookup pattern: {pat.pattern}",
            )

    code_hit = any(p.search(q) for p in _compiled["code"])
    frontier_hint = any(p.search(q) for p in _compiled["frontier_code_hints"])
    if code_hit and frontier_hint:
        return ActionDecision(
            suggested_tier=Tier.FRONTIER,
            action_class="large_code",
            reason="code + scope hint (entire/full/all/whole)",
        )
    if code_hit:
        return ActionDecision(
            suggested_tier=Tier.MID,
            action_class="code",
            reason="code-generation query",
        )

    for pat in _compiled["reasoning"]:
        if pat.search(q):
            return ActionDecision(
                suggested_tier=Tier.MID,
                action_class="reasoning",
                reason=f"matches reasoning pattern: {pat.pattern}",
            )

    # Nothing matched — fall through to CDDR with no suggestion.
    return ActionDecision(
        suggested_tier=Tier.MID,
        action_class="unknown",
        reason="no action pattern matched; deferring to CDDR",
    )


_TIER_RANK = {Tier.SMALL: 0, Tier.MID: 1, Tier.FRONTIER: 2}


def combine_tiers(cddr_tier: Tier, action_tier: Tier, action_class: str) -> Tier:
    """Combine CDDR-selected tier with per-action suggestion.

    Policy:
      - For known-cheap actions (bash, short_lookup), trust the action router:
        prefer SMALL even when CDDR wanted MID. This is the novel savings path.
      - For all other action classes, take the MAX (more expensive) of the two:
        reasoning and large_code should not be downgraded by CDDR.
    """
    if action_class in ("bash_or_shell_like", "short_lookup"):
        return action_tier if _TIER_RANK[action_tier] <= _TIER_RANK[cddr_tier] else cddr_tier
    return cddr_tier if _TIER_RANK[cddr_tier] >= _TIER_RANK[action_tier] else action_tier
