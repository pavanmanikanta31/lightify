"""Real Lightify pipeline — local models + Claude CLI for actual inference.

Three tiers:
  Tier-1: Local model (Ollama — Gemma/Phi/Llama) — $0, ~100-500ms
  Tier-2: Claude Sonnet (API) — $$
  Tier-3: Claude Opus (API) — $$$$$

Two modes:
1. WITHOUT Lightify: raw query → Claude Opus (baseline)
2. WITH Lightify: retrieve context → compress → shape → route to cheapest viable tier
"""
from __future__ import annotations

import time

from lightify.compression import compress, SECREngine
from lightify.conflict import apply_conflict_penalties
from lightify.context_builder import build_context
from lightify.models.claude_cli import invoke_claude
from lightify.models.ollama_local import invoke_ollama, _ollama_available
from lightify.prompt_shaper import shape_prompt
from lightify.router import Router
from lightify.storage.sqlite_memory import MemoryStore
from lightify.sufficiency import estimate_sufficiency
from lightify.types import ContextCapsule, ModelResponse, PipelineResult, Tier


class RealLightifyPipeline:
    """Full Lightify pipeline with real Claude CLI calls."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.router = Router()
        self.secr = SECREngine()

    def run_without_lightify(self, query: str) -> PipelineResult:
        """Baseline: raw query → Claude Opus. No context, no routing."""
        t_start = time.time()

        response = invoke_claude(
            prompt=query,
            tier=Tier.FRONTIER,
            max_turns=1,
            timeout_s=60,
        )

        return PipelineResult(
            query=query,
            response=response,
            tiers_attempted=[Tier.FRONTIER],
            total_latency_ms=response.latency_ms,
            total_tokens_in=response.tokens_in,
            total_tokens_out=response.tokens_out,
            total_cost=response.cost,
        )

    def run_with_lightify(self, query: str) -> PipelineResult:
        """Full Lightify: retrieve → compress → shape → route → Claude."""
        t_start = time.time()

        # Step 1: Retrieve context from memory
        fts_results = self.store.search_fts(query, limit=20)
        # Extract nouns for topic guess (skip common question words)
        _SKIP = {"what", "how", "why", "when", "where", "which", "who", "does",
                 "is", "are", "can", "do", "the", "a", "an", "of", "in", "for"}
        topic_words = [w.lower().rstrip("?.,!") for w in query.split()
                       if w.lower().rstrip("?.,!") not in _SKIP and len(w) > 2]
        topic_results = []
        for tw in topic_words[:3]:
            topic_results.extend(self.store.search_topic(tw, limit=5))

        seen_ids: set = set()
        candidates = []
        for item in fts_results + topic_results:
            if item.id not in seen_ids:
                candidates.append(item)
                seen_ids.add(item.id)

        # Step 2: Build context capsule (filter, rank, compress)
        capsule = build_context(query, candidates, top_k=5)

        # Step 3: CSE — sufficiency check
        capsule.sufficient = estimate_sufficiency(capsule)

        # Step 4: MCD — conflict detection
        capsule = apply_conflict_penalties(capsule)

        # Step 5: CDPS — shape prompt by confidence
        shaped_prompt = shape_prompt(capsule, query)

        # Step 6: SECR — apply learned compression rules
        shaped_prompt = self.secr.apply(shaped_prompt)
        self.secr.observe(shaped_prompt)

        # Step 7: CDDR — route to model tier
        route = self.router.route(capsule)

        # Step 8: Execute with cascade (local → Sonnet → Opus)
        tiers_attempted = []
        total_latency = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0
        final_response = None

        # If Ollama is not available, skip Tier-1 in the cascade
        has_local = _ollama_available()
        if has_local:
            tier_order = {
                Tier.SMALL: [Tier.SMALL, Tier.MID, Tier.FRONTIER],
                Tier.MID: [Tier.MID, Tier.FRONTIER],
                Tier.FRONTIER: [Tier.FRONTIER],
            }
        else:
            tier_order = {
                Tier.SMALL: [Tier.MID, Tier.FRONTIER],
                Tier.MID: [Tier.MID, Tier.FRONTIER],
                Tier.FRONTIER: [Tier.FRONTIER],
            }
        cascade = tier_order[route.tier]

        system_prompt = (
            "You are a helpful assistant. Answer based on the provided context. "
            "Be concise and accurate."
        )

        for tier in cascade:
            tiers_attempted.append(tier)

            if tier == Tier.SMALL:
                # Tier-1: local model (FREE)
                response = invoke_ollama(
                    prompt=shaped_prompt,
                    system_prompt=system_prompt,
                    timeout_s=30,
                )
            else:
                # Tier-2/3: Claude API
                response = invoke_claude(
                    prompt=shaped_prompt,
                    tier=tier,
                    system_prompt=system_prompt,
                    max_turns=1,
                    timeout_s=60,
                )

            total_latency += response.latency_ms
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_cost += response.cost

            if response.success:
                final_response = response
                break

        if final_response is None:
            final_response = response  # last attempt

        # Step 9: Background update
        for item in capsule.raw_items:
            if item.id is not None:
                self.store.update_usage(item.id, final_response.success)
        self.secr.evolve()

        # Auto-prune: keep memory store bounded
        self.store.prune(max_items=5000, min_confidence=0.05)

        return PipelineResult(
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
