#!/usr/bin/env bash
set -euo pipefail

# Lightify v2 — Self-installing setup
# Creates isolated venv, verifies Claude CLI, seeds memory store

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR"
PYTHON="$VENV_DIR/bin/python"
DB_PATH="$SCRIPT_DIR/lightify_memory.db"

echo "=== Lightify v2 Setup ==="

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi
echo "[ok] python3: $(python3 --version)"

# 2. Check/create venv
if [ ! -f "$PYTHON" ]; then
    echo "[..] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[ok] venv created at $VENV_DIR"
else
    echo "[ok] venv exists at $VENV_DIR"
fi

# 3. Install dependencies (rich for terminal UI)
echo "[..] Installing dependencies..."
"$PYTHON" -m pip install -q rich 2>/dev/null
echo "[ok] Dependencies installed"

# 4. Check Claude CLI
if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI not found. Install Claude Code: https://claude.com/claude-code"
    exit 1
fi
CLAUDE_VERSION=$(claude --version 2>&1 | head -1)
echo "[ok] claude CLI: $CLAUDE_VERSION"

# 5. Verify Claude CLI can run in print mode
echo "[..] Testing claude --print mode..."
TEST_OUT=$(claude -p --model haiku --output-format json --no-session-persistence --max-turns 1 "Reply with exactly: LIGHTIFY_OK" 2>/dev/null || true)
if echo "$TEST_OUT" | grep -q "LIGHTIFY_OK"; then
    echo "[ok] Claude CLI print mode works"
else
    echo "[warn] Claude CLI test returned unexpected output (may still work)"
    echo "       Output: $(echo "$TEST_OUT" | head -1)"
fi

# 6. Seed memory database
echo "[..] Seeding memory store at $DB_PATH..."
"$PYTHON" -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR')
from lightify.storage.sqlite_memory import MemoryStore
from benches.generate_data import seed_memory
store = MemoryStore('$DB_PATH')
count = seed_memory(store)
total = store.count()
store.close()
print(f'[ok] Seeded {count} items (total: {total})')
"

echo ""
echo "=== Setup complete ==="
echo "Run benchmarks: $PYTHON -m benches.run_real_bench"
echo "Memory DB: $DB_PATH"
