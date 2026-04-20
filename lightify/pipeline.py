"""End-to-end Lightify inference pipeline.

Supports three variants (for ablation):
1. Caveman-only: single frontier model + compression
2. Hybrid Lightify: graph+memory retrieval + routing + CSE/MCD (no background)
3. Full Lightify: Hybrid + background refinement + SECR
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from lightify.compression import compress, SECREngine
from lightify.confidence import compute_confidence
from lightify.conflict import apply_conflict_penalties
from lightify.context_builder import build_context
from lightify.models.simulated import (
    QueryProfile,
    evaluate_response,
    simulate_inference,
)
from lightify.prompt_shaper import shape_prompt
from lightify.router import Router
from lightify.storage.sqlite_memory import MemoryStore
from lightify.sufficiency import estimate_sufficiency
from lightify.types import (
    ContextCapsule,
    ModelResponse,
    PipelineResult,
    Tier,
)


class Variant(Enum):
    CAVEMAN_ONLY = "caveman_only"
    HYBRID = "hybrid"
    FULL = "full"
    NAIVE_RAG = "naive_rag"


@dataclass
class PipelineConfig:
    variant: Variant = Variant.FULL
    top_k: int = 5
    max_cascade_steps: int = 3
    parallel_dispatch: bool = False


class LightifyPipeline:
    def __init__(self, store: MemoryStore, config: PipelineConfig | None = None):
        self.store = store
        self.config = config or PipelineConfig()
        self.router = Router(parallel_dispatch=self.config.parallel_dispatch)
        self.secr = SECREngine()
        self._response_cache: dict[str, PipelineResult] = {}

    def run(self, query: str, profile: QueryProfile | None = None) -> PipelineResult:
        """Execute the full Lightify pipeline for a query."""
        t_start = time.time()
        profile = profile or QueryProfile()

        # 1. Cache check
        cache_key = query.strip().lower()
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            cached.cache_hit = True
            return cached

        # 2. Context retrieval
        if self.config.variant == Variant.CAVEMAN_ONLY:
            return self._caveman_only(query, profile, t_start)
        elif self.config.variant == Variant.NAIVE_RAG:
            return self._naive_rag(query, profile, t_start)
        else:
            return self._lightify(query, profile, t_start)

    def _caveman_only(
        self, query: str, profile: QueryProfile, t_start: float
    ) -> PipelineResult:
        """Caveman-only: compress context + single frontier model."""
        candidates = self.store.get_all(limit=20)
        # Basic filtering + compression, no routing
        capsule = build_context(query, candidates, top_k=self.config.top_k)
        compressed_prompt = compress(capsule.prompt)

        response = simulate_inference(
            query, compressed_prompt, Tier.FRONTIER, profile, capsule.context_confidence
        )

        result = PipelineResult(
            query=query,
            response=response,
            capsule=capsule,
            tiers_attempted=[Tier.FRONTIER],
            total_latency_ms=response.latency_ms,
            total_tokens_in=response.tokens_in,
            total_tokens_out=response.tokens_out,
            total_cost=response.cost,
        )
        self._response_cache[query.strip().lower()] = result
        return result

    def _naive_rag(
        self, query: str, profile: QueryProfile, t_start: float
    ) -> PipelineResult:
        """Naive RAG: retrieve chunks + single frontier model, no compression."""
        candidates = self.store.search_fts(query, limit=self.config.top_k)
        # No compression, no routing — raw context + frontier
        ctx_parts = [item.content for item in candidates]
        raw_prompt = "Context:\n" + "\n".join(f"- {c}" for c in ctx_parts) + f"\n\nQuery: {query}"

        capsule = ContextCapsule(
            prompt=raw_prompt,
            raw_items=candidates,
            compressed_items=ctx_parts,
            context_confidence=0.5,
            num_items=len(candidates),
        )

        response = simulate_inference(
            query, raw_prompt, Tier.FRONTIER, profile, 0.5
        )

        return PipelineResult(
            query=query,
            response=response,
            capsule=capsule,
            tiers_attempted=[Tier.FRONTIER],
            total_latency_ms=response.latency_ms,
            total_tokens_in=response.tokens_in,
            total_tokens_out=response.tokens_out,
            total_cost=response.cost,
        )

    def _lightify(
        self, query: str, profile: QueryProfile, t_start: float
    ) -> PipelineResult:
        """Hybrid or Full Lightify pipeline."""
        # Step 2: Context selection (FTS + topic)
        fts_results = self.store.search_fts(query, limit=20)
        # Also try topic extraction (first word as heuristic)
        topic_guess = query.split()[0].lower() if query else ""
        topic_results = self.store.search_topic(topic_guess, limit=10)
        # Merge (dedup by id)
        seen_ids = set()
        candidates = []
        for item in fts_results + topic_results:
            if item.id not in seen_ids:
                candidates.append(item)
                seen_ids.add(item.id)

        # Step 3: Build context capsule
        capsule = build_context(query, candidates, top_k=self.config.top_k)

        # CSE: Sufficiency estimation
        capsule.sufficient = estimate_sufficiency(capsule)

        # MCD: Conflict detection
        capsule = apply_conflict_penalties(capsule)

        # CDPS: Shape prompt
        shaped_prompt = shape_prompt(capsule, query)

        # SECR (Full variant only): Apply learned compression rules
        if self.config.variant == Variant.FULL:
            shaped_prompt = self.secr.apply(shaped_prompt)
            self.secr.observe(shaped_prompt)

        # CDDR: Route to model tier
        route = self.router.route(capsule)

        # Execute with cascade
        tiers_attempted = []
        total_latency = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0
        final_response = None

        # Build cascade order starting from routed tier
        tier_order = {
            Tier.SMALL: [Tier.SMALL, Tier.MID, Tier.FRONTIER],
            Tier.MID: [Tier.MID, Tier.FRONTIER],
            Tier.FRONTIER: [Tier.FRONTIER],
        }
        cascade = tier_order[route.tier]

        for tier in cascade[:self.config.max_cascade_steps]:
            tiers_attempted.append(tier)
            response = simulate_inference(
                query, shaped_prompt, tier, profile, capsule.context_confidence
            )
            total_latency += response.latency_ms
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_cost += response.cost

            if evaluate_response(response, profile):
                final_response = response
                break
        else:
            # All tiers failed — use last response
            final_response = response

        # Background learning (Full variant only)
        if self.config.variant == Variant.FULL and final_response:
            self._background_update(capsule, final_response)

        result = PipelineResult(
            query=query,
            response=final_response,
            capsule=capsule,
            route=route,
            tiers_attempted=tiers_attempted,
            total_latency_ms=total_latency,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_cost=total_cost,
        )
        self._response_cache[query.strip().lower()] = result
        return result

    def _background_update(
        self, capsule: ContextCapsule, response: ModelResponse
    ) -> None:
        """Background learning: update memory usage stats and SECR rules."""
        for item in capsule.raw_items:
            if item.id is not None:
                self.store.update_usage(item.id, response.success)

        # SECR: evolve compression rules periodically
        self.secr.evolve()
