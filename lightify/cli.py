#!/usr/bin/env python3
"""
Lightify CLI — latency-aware, memory-driven multi-model inference.

Usage:
    lightify init              First-time setup: verify Claude CLI, create memory DB
    lightify query "question"  Ask a question through Lightify (routed to cheapest viable model)
    lightify query --raw "q"   Ask without Lightify (raw Claude Opus baseline)
    lightify query --compare   Run both WITH and WITHOUT, show side-by-side comparison
    lightify bench             Run the full benchmark suite (8 queries, WITH vs WITHOUT)
    lightify status            Show memory store stats, SECR rules, config
    lightify memory list       List recent memory items
    lightify memory add "..."  Add a fact to memory
    lightify memory seed       Seed memory with default knowledge base
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

# ── Constants ─────────────────────────────────────────────────────────────

APP_DIR = os.environ.get("LIGHTIFY_HOME") or os.path.expanduser("~/.lightify")
DB_PATH = os.path.join(APP_DIR, "memory.db")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
VERSION = "1.0.0"

# ── Colors (ANSI) ────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[94m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[91m"
    RESET = "\033[0m"

    @staticmethod
    def ok(msg): return f"{C.GREEN}[ok]{C.RESET} {msg}"
    @staticmethod
    def warn(msg): return f"{C.YELLOW}[!!]{C.RESET} {msg}"
    @staticmethod
    def err(msg): return f"{C.RED}[err]{C.RESET} {msg}"
    @staticmethod
    def info(msg): return f"{C.BLUE}[..]{C.RESET} {msg}"
    @staticmethod
    def header(msg): return f"\n{C.BOLD}{C.CYAN}{msg}{C.RESET}"


def _ensure_app_dir():
    os.makedirs(APP_DIR, exist_ok=True)


def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict):
    _ensure_app_dir()
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _is_initialized() -> bool:
    return os.path.exists(DB_PATH) and os.path.exists(CONFIG_PATH)


# ── Commands ──────────────────────────────────────────────────────────────

def _load_banner() -> str:
    """Load banner from static file, render with Rich panel."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        banner_path = os.path.join(os.path.dirname(__file__), "banner.txt")
        with open(banner_path) as f:
            art = f.read().rstrip()

        inner = Text()
        inner.append(art + "\n\n", style="bold bright_blue")
        inner.append("Fast inference", style="bold bright_white")
        inner.append("  ·  ", style="dim")
        inner.append("Lower cost", style="bold bright_green")
        inner.append("  ·  ", style="dim")
        inner.append("Smarter routing", style="bold bright_cyan")
        inner.append("\n")

        console = Console()
        with console.capture() as capture:
            console.print(Panel(
                inner,
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(1, 2),
                subtitle=f"⚡ v{VERSION}",
                subtitle_align="right",
            ))
        return capture.get()
    except Exception:
        return f"lightify v{VERSION}"


def cmd_init(args):
    """First-time setup wizard."""
    print()
    print(_load_banner())

    _ensure_app_dir()

    # 1. Check Python version
    v = sys.version_info
    if v >= (3, 10):
        print(C.ok(f"Python {v.major}.{v.minor}.{v.micro}"))
    else:
        print(C.err(f"Python {v.major}.{v.minor} — need 3.10+"))
        sys.exit(1)

    # 2. Check Claude CLI
    claude_path = shutil.which("claude")
    if claude_path:
        try:
            ver = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
            print(C.ok(f"Claude CLI: {ver.stdout.strip()} ({claude_path})"))
        except Exception:
            print(C.ok(f"Claude CLI found at {claude_path}"))
    else:
        print(C.err("Claude CLI not found"))
        print(f"  Install: {C.DIM}https://claude.ai/code{C.RESET}")
        sys.exit(1)

    # 3. Test Claude CLI auth
    print(C.info("Testing Claude authentication..."))
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--output-format", "json",
             "--no-session-persistence", "--max-turns", "1",
             "Reply with exactly: OK"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout) if result.stdout else {}
        if data.get("is_error"):
            print(C.warn(f"Claude auth issue: {data.get('result', 'unknown')}"))
            print(f"  Run: {C.DIM}claude{C.RESET} and log in first")
            # Don't exit — let them continue, they can fix auth later
        else:
            print(C.ok("Claude authenticated"))
    except Exception as e:
        print(C.warn(f"Could not test auth: {e}"))

    # 4. Check Ollama (local models — Tier-1)
    ollama_path = shutil.which("ollama")
    if ollama_path:
        from lightify.models.ollama_local import _ollama_available, _list_models
        if _ollama_available():
            models = _list_models()
            print(C.ok(f"Ollama: {', '.join(models) if models else 'running (no models)'}"))
            if not models:
                print(f"  Pull a model: {C.DIM}ollama pull gemma3:1b{C.RESET}")
        else:
            print(C.warn("Ollama installed but not running"))
            print(f"  Start it: {C.DIM}ollama serve{C.RESET}")
    else:
        print(C.warn("Ollama not found (Tier-1 local inference disabled)"))
        print(f"  Install: {C.DIM}brew install ollama && ollama pull gemma3:1b{C.RESET}")
        print(f"  Without Ollama, queries route to Claude Sonnet (API, not free)")

    # 5. Create/verify memory DB
    from lightify.storage.sqlite_memory import MemoryStore
    print(C.info(f"Creating memory store at {DB_PATH}"))
    store = MemoryStore(DB_PATH)
    count = store.count()

    if count == 0:
        print(C.info("Seeding knowledge base..."))
        from benches.generate_data import seed_memory
        seeded = seed_memory(store)
        print(C.ok(f"Seeded {seeded} items"))
    else:
        print(C.ok(f"Memory store exists ({count} items)"))

    store.close()

    # 5. Save config
    _save_config({
        "version": VERSION,
        "db_path": DB_PATH,
        "initialized": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print()
    print(C.header("  Setup complete!"))
    print()
    print(f"  {C.BOLD}Try it:{C.RESET}")
    print(f"    lightify query \"What prevents true thread parallelism in Python?\"")
    print(f"    lightify query --compare \"How does Rust handle memory safety?\"")
    print(f"    lightify bench")
    print(f"    lightify status")
    print()


def _spinner(msg: str):
    """Context manager for a Rich spinner, falls back to plain print."""
    try:
        from rich.console import Console
        from rich.spinner import Spinner
        from rich.live import Live
        return Live(Spinner("dots", text=msg), refresh_per_second=10, transient=True)
    except ImportError:
        import contextlib
        @contextlib.contextmanager
        def _fallback():
            print(C.info(msg))
            yield
        return _fallback()


def cmd_query(args):
    """Run a query through Lightify (or baseline)."""
    if not _is_initialized():
        print(C.err("Not initialized. Run: lightify init"))
        sys.exit(1)

    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    if not query:
        print(C.err("No query provided"))
        sys.exit(1)

    from lightify.storage.sqlite_memory import MemoryStore
    from lightify.pipeline_real import RealLightifyPipeline
    from lightify.types import Tier

    store = MemoryStore(DB_PATH)
    pipeline = RealLightifyPipeline(
        store,
        action_routing=bool(getattr(args, "action_routing", False)),
    )

    # --model override: force a specific tier
    model_override = None
    if hasattr(args, "model") and args.model:
        model_map = {"haiku": Tier.SMALL, "sonnet": Tier.MID, "opus": Tier.FRONTIER}
        model_override = model_map.get(args.model)
        if not model_override:
            print(C.err(f"Unknown model: {args.model}. Use haiku, sonnet, or opus."))
            store.close()
            sys.exit(1)

    # Apply mode presets to router thresholds
    if hasattr(args, "fast") and args.fast:
        pipeline.router.tau_tier1 = 0.20  # almost always try local
        pipeline.router.tau_tier2 = 0.10
    elif hasattr(args, "cheap") and args.cheap:
        pipeline.router.tau_tier1 = 0.30  # prefer local, fall back to Sonnet
        pipeline.router.tau_tier2 = 0.15
    elif hasattr(args, "quality") and args.quality:
        pipeline.router.tau_tier1 = 0.90  # almost never use local
        pipeline.router.tau_tier2 = 0.70  # prefer Opus

    if args.compare:
        _run_compare(pipeline, query)
    elif args.baseline:
        _run_baseline(pipeline, query)
    elif model_override:
        _run_direct(pipeline, query, model_override)
    else:
        _run_lightify(pipeline, query)

    store.close()


def _run_lightify(pipeline, query: str):
    """Run query through Lightify pipeline."""
    with _spinner("Routing query through Lightify..."):
        result = pipeline.run_with_lightify(query)

    tiers = " → ".join(t.value for t in result.tiers_attempted)
    conflicts = len(result.capsule.conflicts) if result.capsule else 0
    conf = result.capsule.context_confidence if result.capsule else 0

    print(result.response.text or "(no response)")
    print()
    print(f"{C.DIM}─── lightify ───────────────────────────────────────────{C.RESET}")
    print(f"{C.DIM}  route: {tiers}  │  conf: {conf:.2f}  │  "
          f"conflicts: {conflicts}  │  "
          f"{result.total_tokens_in}+{result.total_tokens_out} tok  │  "
          f"{result.total_latency_ms:.0f}ms  │  "
          f"${result.total_cost:.4f}{C.RESET}")


def _run_baseline(pipeline, query: str):
    """Run query through raw Claude Opus (no Lightify)."""
    with _spinner("Sending to Claude Opus (baseline)..."):
        result = pipeline.run_without_lightify(query)

    print(result.response.text or "(no response)")
    print()
    print(f"{C.DIM}─── baseline (opus) ────────────────────────────────────{C.RESET}")
    print(f"{C.DIM}  {result.total_tokens_in}+{result.total_tokens_out} tok  │  "
          f"{result.total_latency_ms:.0f}ms  │  "
          f"${result.total_cost:.4f}{C.RESET}")


def _run_direct(pipeline, query: str, tier):
    """Run query through a specific model tier (--model override)."""
    from lightify.models.claude_cli import invoke_claude
    with _spinner(f"Sending to Claude {tier.value}..."):
        result = invoke_claude(prompt=query, tier=tier, max_turns=1, timeout_s=60)

    print(result.text or "(no response)")
    print()
    print(f"{C.DIM}─── direct ({tier.value}) ──────────────────────────────────────{C.RESET}")
    print(f"{C.DIM}  {result.tokens_in}+{result.tokens_out} tok  │  "
          f"{result.latency_ms:.0f}ms  │  "
          f"${result.cost:.4f}{C.RESET}")


def _run_compare(pipeline, query: str):
    """Run both WITH and WITHOUT, show side-by-side."""
    print(C.header("  Compare: Lightify vs Raw Claude"))
    print(f"  Query: {query}")
    print()

    # Without
    print(f"{C.YELLOW}━━━ WITHOUT Lightify (raw Opus) ━━━{C.RESET}")
    r_raw = pipeline.run_without_lightify(query)
    print(r_raw.response.text or "(no response)")
    print()

    # With
    print(f"{C.GREEN}━━━ WITH Lightify (routed) ━━━{C.RESET}")
    r_lit = pipeline.run_with_lightify(query)
    tiers = " → ".join(t.value for t in r_lit.tiers_attempted)
    conflicts = len(r_lit.capsule.conflicts) if r_lit.capsule else 0
    print(r_lit.response.text or "(no response)")
    print()

    # Comparison table
    print(C.header("  Results"))
    print(f"  {'':20} {'Without':>12}  {'With Lightify':>14}  {'Delta':>10}")
    print(f"  {'─'*60}")

    def row(label, v1, v2, fmt=".0f", unit="", lower_better=True):
        d = v2 - v1
        pct = (d / v1 * 100) if v1 else 0
        color = C.GREEN if (d < 0) == lower_better else C.RED
        print(f"  {label:20} {v1:>10{fmt}}{unit}  {v2:>12{fmt}}{unit}  "
              f"{color}{d:>+8{fmt}}{unit} ({pct:+.0f}%){C.RESET}")

    row("Tokens (in)", r_raw.total_tokens_in, r_lit.total_tokens_in, "d", "")
    row("Tokens (out)", r_raw.total_tokens_out, r_lit.total_tokens_out, "d", "")
    row("Latency (ms)", r_raw.total_latency_ms, r_lit.total_latency_ms, ".0f", "ms")
    row("Cost ($)", r_raw.total_cost, r_lit.total_cost, ".5f", "")

    savings = r_raw.total_cost - r_lit.total_cost
    print(f"\n  {C.BOLD}Route: {tiers}  │  Conflicts: {conflicts}  │  "
          f"Saved: ${savings:.4f}{C.RESET}")


def cmd_bench(args):
    """Run the full benchmark suite."""
    if not _is_initialized():
        print(C.err("Not initialized. Run: lightify init"))
        sys.exit(1)

    # Import and run the real benchmark
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benches.run_real_bench import run_benchmark
    run_benchmark()


def cmd_status(args):
    """Show system status."""
    if not _is_initialized():
        print(C.err("Not initialized. Run: lightify init"))
        sys.exit(1)

    from lightify.storage.sqlite_memory import MemoryStore

    cfg = _load_config()
    store = MemoryStore(DB_PATH)

    print(C.header("  Lightify Status"))
    print()
    print(f"  Version:      {VERSION}")
    print(f"  Config:       {CONFIG_PATH}")
    print(f"  Memory DB:    {DB_PATH}")
    print(f"  Items:        {store.count()}")
    print(f"  Initialized:  {cfg.get('created_at', 'unknown')}")
    print()

    # Show tier distribution
    items = store.get_all(limit=1000)
    tiers = {}
    for item in items:
        t = item.source_tier.value
        tiers[t] = tiers.get(t, 0) + 1
    print(f"  Memory by tier:")
    for t, c in sorted(tiers.items()):
        print(f"    {t}: {c} items")

    # Show top topics
    topics = {}
    for item in items:
        if item.topic:
            topics[item.topic] = topics.get(item.topic, 0) + 1
    if topics:
        print(f"\n  Top topics:")
        for topic, c in sorted(topics.items(), key=lambda x: -x[1])[:10]:
            print(f"    {topic}: {c}")

    store.close()


def cmd_memory(args):
    """Manage memory store."""
    if not _is_initialized():
        print(C.err("Not initialized. Run: lightify init"))
        sys.exit(1)

    from lightify.storage.sqlite_memory import MemoryStore
    from lightify.types import MemoryItem, Tier

    store = MemoryStore(DB_PATH)

    if args.memory_cmd == "list":
        items = store.get_all(limit=args.limit or 20)
        if not items:
            print(C.warn("Memory is empty. Run: lightify memory seed"))
        else:
            print(C.header(f"  Memory ({len(items)} items)"))
            print()
            for item in items:
                conf_bar = "█" * int(item.confidence * 10) + "░" * (10 - int(item.confidence * 10))
                print(f"  {C.DIM}[{item.id:3d}]{C.RESET} "
                      f"{C.CYAN}{item.topic:12}{C.RESET} "
                      f"{conf_bar} {item.confidence:.2f}  "
                      f"{item.content[:60]}{'...' if len(item.content) > 60 else ''}")

    elif args.memory_cmd == "add":
        text = " ".join(args.text) if args.text else ""
        if not text:
            print(C.err("Usage: lightify memory add \"your fact here\" --topic python"))
            store.close()
            sys.exit(1)
        topic = args.topic or ""
        item = MemoryItem(
            content=text,
            topic=topic,
            source_tier=Tier.FRONTIER,
            confidence=0.8,
        )
        item_id = store.insert(item)
        print(C.ok(f"Added to memory (id={item_id}, topic={topic or 'none'})"))

    elif args.memory_cmd == "seed":
        from benches.generate_data import seed_memory
        count = seed_memory(store)
        print(C.ok(f"Seeded {count} items (total: {store.count()})"))

    elif args.memory_cmd == "search":
        q = " ".join(args.text) if args.text else ""
        if not q:
            print(C.err("Usage: lightify memory search \"query\""))
            store.close()
            sys.exit(1)
        results = store.search_fts(q, limit=10)
        if not results:
            print(C.warn(f"No results for: {q}"))
        else:
            print(C.header(f"  Search: \"{q}\" ({len(results)} results)"))
            for item in results:
                print(f"  {C.DIM}[{item.id:3d}]{C.RESET} "
                      f"{C.CYAN}{item.topic:12}{C.RESET} "
                      f"{item.content[:70]}{'...' if len(item.content) > 70 else ''}")

    elif args.memory_cmd == "clear":
        confirm = input(f"{C.YELLOW}Clear all memory? [y/N]: {C.RESET}")
        if confirm.lower() == "y":
            store.close()
            os.remove(DB_PATH)
            MemoryStore(DB_PATH).close()
            print(C.ok("Memory cleared"))
            return  # store already closed
        else:
            print("Cancelled")

    else:
        print(C.err(f"Unknown memory command: {args.memory_cmd}"))
        print("  Commands: list, add, seed, search, clear")

    store.close()


def cmd_config(args):
    """View and manage model configuration per tier."""
    from lightify.config import (
        load_model_config, save_model_config, CATALOG, DEFAULT_MODELS,
    )

    if args.config_cmd == "show":
        models = load_model_config()
        print(C.header("  Model Configuration"))
        print()
        for tier in ["tier1", "tier2", "tier3"]:
            t = models.get(tier, {})
            provider = t.get("provider", "?")
            model = t.get("model", "?")
            desc = t.get("description", "")
            cost = CATALOG.get(model, {}).get("cost", "?")
            print(f"  {C.BOLD}{tier}{C.RESET}:  {C.CYAN}{model}{C.RESET} ({provider})  {C.DIM}{cost} — {desc}{C.RESET}")
        print()
        print(f"  Config: {CONFIG_PATH}")
        print(f"  Edit:   lightify config set tier1 <model>")

    elif args.config_cmd == "models":
        print(C.header("  Available Models"))
        print()
        print(f"  {'Model':<20} {'Provider':<10} {'Size':<10} {'Cost':<14} {'Speed':<8} {'Quality'}")
        print(f"  {'─'*75}")
        for name, info in CATALOG.items():
            print(f"  {name:<20} {info['provider']:<10} {info['size']:<10} "
                  f"{info['cost']:<14} {info['speed']:<8} {info['quality']}")
        print()
        print(f"  {C.BOLD}Local models{C.RESET} (Ollama, $0):")
        print(f"    Install: ollama pull <model>")
        print(f"    Use:     lightify config set tier1 <model>")
        print()
        print(f"  {C.BOLD}API models{C.RESET} (Claude, paid):")
        print(f"    Requires: claude CLI logged in")
        print(f"    Use:     lightify config set tier2 <model>")

    elif args.config_cmd == "set":
        if not args.tier or not args.model_name:
            print(C.err("Usage: lightify config set <tier1|tier2|tier3> <model>"))
            sys.exit(1)
        tier = args.tier
        model_name = args.model_name
        if tier not in ("tier1", "tier2", "tier3"):
            print(C.err(f"Invalid tier: {tier}. Use tier1, tier2, or tier3."))
            sys.exit(1)
        info = CATALOG.get(model_name)
        if not info:
            print(C.warn(f"Model '{model_name}' not in catalog. Setting anyway."))
            provider = "ollama" if ":" in model_name else "claude"
            info = {"provider": provider, "cost": "unknown"}
        else:
            provider = info["provider"]

        models = load_model_config()
        models[tier] = {
            "provider": provider,
            "model": model_name,
            "cost_per_1k": 0.0 if provider == "ollama" else 0.003,
            "description": f"{info.get('cost', '')} — {info.get('quality', '')}",
        }
        save_model_config(models)
        print(C.ok(f"{tier} → {model_name} ({provider})"))

    elif args.config_cmd == "reset":
        save_model_config(DEFAULT_MODELS)
        print(C.ok("Reset to defaults"))

    else:
        print(C.err(f"Unknown config command: {args.config_cmd}"))
        print("  Commands: show, models, set, reset")


# ── Argument parser ───────────────────────────────────────────────────────

def _print_help():
    """Print help in Docker/Claude/gh style."""
    print(f"Usage:  lightify <command> [options]")
    print()
    print(f"Route queries to the cheapest Claude model that can answer well,")
    print(f"using context from a persistent knowledge base.")
    print()
    print(f"Getting Started:")
    print(f"  init                           Set up Lightify (first-time)")
    print(f"  query <question>               Ask a question through Lightify")
    print()
    print(f"Commands:")
    print(f"  query <question>               Route query through Lightify pipeline")
    print(f"  query --baseline <question>    Send direct to Claude Opus (no Lightify)")
    print(f"  query --compare <question>     Side-by-side WITH vs WITHOUT comparison")
    print(f"  query --model sonnet <q>       Force a specific model (haiku/sonnet/opus)")
    print(f"  bench                          Run full benchmark suite (8 queries)")
    print(f"  status                         Show memory stats and configuration")
    print()
    print(f"Memory Commands:")
    print(f"  memory list                    Browse the knowledge base")
    print(f"  memory add <fact> --topic t    Add a fact to the knowledge base")
    print(f"  memory search <query>          Search the knowledge base")
    print(f"  memory seed                    Seed with default knowledge")
    print(f"  memory clear                   Clear all memory")
    print()
    print(f"Model Configuration:")
    print(f"  config show                    Show current model per tier")
    print(f"  config models                  List all available models")
    print(f"  config set tier1 gemma3:4b     Change which model serves a tier")
    print(f"  config reset                   Reset to defaults")
    print()
    print(f"Modes:")
    print(f"  --fast                         Optimize for speed (prefer local model)")
    print(f"  --cheap                        Optimize for cost (avoid API calls)")
    print(f"  --quality                      Optimize for quality (prefer Opus)")
    print()
    print(f"Options:")
    print(f"  -h, --help                     Show this help message")
    print(f"  -v, --version                  Print version ({VERSION})")
    print()
    print(f"Run 'lightify <command> --help' for more information on a command.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightify",
        description="Route queries to the cheapest Claude model that can answer well",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--version", action="store_true", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Set up Lightify (first-time)")

    # query
    p_query = sub.add_parser("query", help="Ask a question through Lightify")
    p_query.add_argument("query", nargs="+", help="Your question")
    p_query.add_argument("--baseline", action="store_true",
                         help="Send direct to Claude Opus (no Lightify)")
    p_query.add_argument("--compare", action="store_true",
                         help="Side-by-side WITH vs WITHOUT comparison")
    p_query.add_argument("--model", choices=["haiku", "sonnet", "opus"],
                         help="Force a specific model tier")
    mode = p_query.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true",
                      help="Optimize for speed (prefer local, minimal output)")
    mode.add_argument("--cheap", action="store_true",
                      help="Optimize for cost (local first, avoid Opus)")
    mode.add_argument("--quality", action="store_true",
                      help="Optimize for quality (prefer Opus, detailed output)")
    p_query.add_argument("--action-routing", action="store_true",
                         help="Enable per-action tier overlay (bash/lookup/code classifier)")

    # bench
    sub.add_parser("bench", help="Run full benchmark suite")

    # status
    sub.add_parser("status", help="Show memory stats and configuration")

    # memory
    p_mem = sub.add_parser("memory", help="Manage the knowledge base")
    p_mem.add_argument("memory_cmd", choices=["list", "add", "seed", "search", "clear"],
                       help="Memory subcommand")
    p_mem.add_argument("text", nargs="*", help="Text for add/search")
    p_mem.add_argument("--topic", default="", help="Topic tag (for add)")
    p_mem.add_argument("--limit", type=int, default=20, help="Max items to show")

    # config
    p_cfg = sub.add_parser("config", help="View and manage model tiers")
    p_cfg.add_argument("config_cmd", choices=["show", "models", "set", "reset"],
                       help="Config subcommand")
    p_cfg.add_argument("tier", nargs="?", help="Tier to set (tier1/tier2/tier3)")
    p_cfg.add_argument("model_name", nargs="?", help="Model name (e.g. gemma3:4b, sonnet)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"lightify {VERSION}")
        sys.exit(0)

    if args.help or not args.command:
        _print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "query": cmd_query,
        "bench": cmd_bench,
        "status": cmd_status,
        "memory": cmd_memory,
        "config": cmd_config,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        _print_help()


if __name__ == "__main__":
    main()
