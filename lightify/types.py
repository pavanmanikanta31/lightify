"""Shared types for Lightify."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(Enum):
    SMALL = "tier1"
    MID = "tier2"
    FRONTIER = "tier3"


@dataclass
class MemoryItem:
    id: int | None = None
    content: str = ""
    topic: str = ""
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    source_tier: Tier = Tier.MID
    created_ts: int = 0
    last_used_ts: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def validate(self) -> None:
        if self.success_count > self.usage_count:
            self.success_count = self.usage_count


@dataclass
class ContextCapsule:
    """Output of the context builder (Step 3)."""
    prompt: str = ""
    raw_items: list[MemoryItem] = field(default_factory=list)
    compressed_items: list[str] = field(default_factory=list)
    context_confidence: float = 0.0
    num_items: int = 0
    coverage: float = 0.0
    sufficient: bool = False
    conflicts: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class RouteDecision:
    tier: Tier = Tier.FRONTIER
    reason: str = ""
    parallel: bool = False


@dataclass
class ModelResponse:
    text: str = ""
    tier: Tier = Tier.FRONTIER
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    success: bool = False
    cost: float = 0.0


@dataclass
class PipelineResult:
    query: str = ""
    response: ModelResponse = field(default_factory=ModelResponse)
    capsule: ContextCapsule = field(default_factory=ContextCapsule)
    route: RouteDecision = field(default_factory=RouteDecision)
    tiers_attempted: list[Tier] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    cache_hit: bool = False
