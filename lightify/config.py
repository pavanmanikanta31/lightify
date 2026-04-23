"""Lightify model configuration — manage which models serve each tier.

Config stored at ~/.lightify/config.json under the "models" key.
Users can swap models per tier without touching code.
"""
from __future__ import annotations

import json
import os

APP_DIR = os.environ.get("LIGHTIFY_HOME") or os.path.expanduser("~/.lightify")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODELS = {
    "tier1": {
        "provider": "ollama",
        "model": "gemma3:1b",
        "cost_per_1k": 0.0,
        "description": "Local (free, fast)",
    },
    "tier2": {
        "provider": "claude",
        "model": "sonnet",
        "cost_per_1k": 0.003,
        "description": "Claude Sonnet (API, mid-tier)",
    },
    "tier3": {
        "provider": "claude",
        "model": "opus",
        "cost_per_1k": 0.015,
        "description": "Claude Opus (API, frontier)",
    },
}

# ── Known models catalog ─────────────────────────────────────────────────

CATALOG = {
    # Local models (Ollama) — $0
    "gemma3:1b":     {"provider": "ollama", "size": "815MB",  "cost": "FREE", "speed": "fast",   "quality": "basic"},
    "gemma3:4b":     {"provider": "ollama", "size": "2.3GB",  "cost": "FREE", "speed": "medium", "quality": "good"},
    "phi3:mini":     {"provider": "ollama", "size": "2.2GB",  "cost": "FREE", "speed": "medium", "quality": "good"},
    "llama3.2:1b":   {"provider": "ollama", "size": "1.3GB",  "cost": "FREE", "speed": "fast",   "quality": "basic"},
    "llama3.2:3b":   {"provider": "ollama", "size": "2.0GB",  "cost": "FREE", "speed": "medium", "quality": "good"},
    "mistral:7b":    {"provider": "ollama", "size": "4.1GB",  "cost": "FREE", "speed": "slow",   "quality": "great"},
    "qwen2.5:1.5b":  {"provider": "ollama", "size": "986MB",  "cost": "FREE", "speed": "fast",   "quality": "good"},
    "deepseek-r1:1.5b": {"provider": "ollama", "size": "1.1GB", "cost": "FREE", "speed": "fast", "quality": "good"},

    # Claude API models — paid
    "haiku":  {"provider": "claude", "size": "API",  "cost": "$0.80/1M in",  "speed": "fast",   "quality": "good"},
    "sonnet": {"provider": "claude", "size": "API",  "cost": "$3/1M in",     "speed": "medium", "quality": "great"},
    "opus":   {"provider": "claude", "size": "API",  "cost": "$15/1M in",    "speed": "slow",   "quality": "best"},
}


def load_model_config() -> dict:
    """Load model config from ~/.lightify/config.json, falling back to defaults."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("models", DEFAULT_MODELS)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_MODELS


def save_model_config(models: dict) -> None:
    """Save model config to ~/.lightify/config.json."""
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}
    cfg["models"] = models
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_tier_model(tier: str) -> tuple[str, str]:
    """Return (provider, model_name) for a tier."""
    models = load_model_config()
    t = models.get(tier, DEFAULT_MODELS.get(tier, {}))
    return t.get("provider", "claude"), t.get("model", "sonnet")


# ── Budget ────────────────────────────────────────────────────────────────
# Spend is measured against the `trace` table in SQLite. A budget of 0
# disables the cap (default, preserves prior behavior).

DEFAULT_BUDGET = {
    "max_daily_usd": 0.0,   # 0 = unlimited
    "on_exceed": "block",   # "block" | "degrade"
}


def load_budget_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return {**DEFAULT_BUDGET, **cfg.get("budget", {})}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_BUDGET)


def save_budget_config(budget: dict) -> None:
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}
    cfg["budget"] = {**DEFAULT_BUDGET, **budget}
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
