"""Real Claude CLI model adapter — invokes `claude -p` for actual inference.

Tier mapping:
  Tier-1 (Small)    → claude --model haiku
  Tier-2 (Mid)      → claude --model sonnet
  Tier-3 (Frontier) → claude --model opus

Uses --bare to avoid CLAUDE.md/hooks/memory interference.
Uses --output-format json for token counts.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass

from lightify.types import ModelResponse, Tier

# Model aliases for Claude CLI
_TIER_TO_MODEL = {
    Tier.SMALL: "haiku",
    Tier.MID: "sonnet",
    Tier.FRONTIER: "opus",
}

def invoke_claude(
    prompt: str,
    tier: Tier,
    system_prompt: str | None = None,
    max_turns: int = 1,
    timeout_s: int = 120,
) -> ModelResponse:
    """Invoke Claude CLI and return structured response with metrics.

    Uses `claude -p --bare --output-format json` for clean, measurable calls.
    """
    model = _TIER_TO_MODEL[tier]
    t_start = time.time()

    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--output-format", "json",
        "--no-session-persistence",
        "--max-turns", str(max_turns),
    ]

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        latency_ms = (time.time() - t_start) * 1000

        # Parse JSON output
        stdout = result.stdout.strip()
        if not stdout:
            return ModelResponse(
                text=f"[{tier.value}] Empty response",
                tier=tier,
                latency_ms=latency_ms,
                success=False,
            )

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # If JSON parsing fails, treat raw text as response
            return ModelResponse(
                text=stdout[:2000],
                tier=tier,
                latency_ms=latency_ms,
                success=True,
            )

        # Extract from Claude CLI JSON output (real format)
        response_text = data.get("result", "")
        is_error = data.get("is_error", False)
        actual_cost = data.get("total_cost_usd", 0.0)
        actual_latency = data.get("duration_ms", latency_ms)

        # Token counts from usage block
        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # Use actual cost from Claude CLI (includes cache pricing)
        cost = actual_cost

        return ModelResponse(
            text=response_text,
            tier=tier,
            tokens_in=tokens_in + cache_creation + cache_read,
            tokens_out=tokens_out,
            latency_ms=actual_latency,
            success=not is_error and bool(response_text),
            cost=cost,
        )

    except subprocess.TimeoutExpired:
        latency_ms = (time.time() - t_start) * 1000
        return ModelResponse(
            text=f"[{tier.value}] Timeout after {timeout_s}s",
            tier=tier,
            latency_ms=latency_ms,
            success=False,
        )
    except Exception as e:
        latency_ms = (time.time() - t_start) * 1000
        return ModelResponse(
            text=f"[{tier.value}] Error: {e}",
            tier=tier,
            latency_ms=latency_ms,
            success=False,
        )


def evaluate_response_quality(response: ModelResponse, expected: str) -> float:
    """Simple quality score: fraction of expected keywords found in response."""
    if not response.success or not response.text:
        return 0.0
    resp_lower = response.text.lower()
    expected_words = set(expected.lower().split())
    if not expected_words:
        return 1.0 if response.success else 0.0
    found = sum(1 for w in expected_words if w in resp_lower)
    return found / len(expected_words)
