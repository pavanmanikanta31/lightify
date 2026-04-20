"""Ollama local model adapter — FREE Tier-1 inference.

Uses Ollama's REST API (http://localhost:11434) for local model inference.
Cost: $0. Latency: ~100-500ms depending on model and hardware.

Default model: gemma3:1b (815MB, fast on Apple Silicon)
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error

from lightify.types import ModelResponse, Tier

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"


def _ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _list_models() -> list[str]:
    """List available Ollama models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def invoke_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    timeout_s: int = 30,
) -> ModelResponse:
    """Invoke a local Ollama model. Cost: $0."""
    t_start = time.time()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())

        latency_ms = (time.time() - t_start) * 1000
        response_text = data.get("response", "")
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)

        return ModelResponse(
            text=response_text,
            tier=Tier.SMALL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            success=bool(response_text.strip()),
            cost=0.0,  # LOCAL = FREE
        )

    except urllib.error.URLError:
        latency_ms = (time.time() - t_start) * 1000
        return ModelResponse(
            text="[local] Ollama not available",
            tier=Tier.SMALL,
            latency_ms=latency_ms,
            success=False,
            cost=0.0,
        )
    except Exception as e:
        latency_ms = (time.time() - t_start) * 1000
        return ModelResponse(
            text=f"[local] Error: {e}",
            tier=Tier.SMALL,
            latency_ms=latency_ms,
            success=False,
            cost=0.0,
        )
