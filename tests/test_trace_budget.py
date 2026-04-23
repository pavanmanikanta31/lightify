"""Minimum-scope tests for the trace table and the remaining-budget math.

These tests do not make any API calls. They verify:
  - trace rows can be written and summed by timestamp
  - _remaining_budget_usd returns None when no cap, a positive number when
    below cap, and exactly 0 when at or over cap
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.pipeline_real import RealLightifyPipeline, _start_of_day_ts
from lightify.storage.sqlite_memory import MemoryStore


def _row(store, **kw):
    defaults = dict(
        query_hash="abc", tier_chosen="tier2", tier_reason="t",
        cascaded=False, cost_usd=0.01, tokens_in=100, tokens_out=50,
        latency_ms=250.0, success=True,
    )
    defaults.update(kw)
    store.insert_trace(**defaults)


def test_insert_and_sum_trace():
    db = tempfile.mktemp(suffix=".db")
    s = MemoryStore(db)
    _row(s, cost_usd=0.01)
    _row(s, cost_usd=0.02)
    _row(s, cost_usd=0.0)  # local tier, free
    total = s.spend_since(_start_of_day_ts())
    assert abs(total - 0.03) < 1e-9, f"expected 0.03, got {total}"
    s.close()


def test_spend_since_horizon_respected():
    db = tempfile.mktemp(suffix=".db")
    s = MemoryStore(db)
    _row(s, cost_usd=0.05)
    future = int(time.time()) + 10_000
    assert s.spend_since(future) == 0.0
    s.close()


def test_remaining_budget_none_when_unset():
    db = tempfile.mktemp(suffix=".db")
    s = MemoryStore(db)
    p = RealLightifyPipeline(s)
    p._budget = {"max_daily_usd": 0.0}
    assert p._remaining_budget_usd() is None
    s.close()


def test_remaining_budget_positive_when_under_cap():
    db = tempfile.mktemp(suffix=".db")
    s = MemoryStore(db)
    _row(s, cost_usd=0.30)
    p = RealLightifyPipeline(s)
    p._budget = {"max_daily_usd": 1.00}
    rem = p._remaining_budget_usd()
    assert rem is not None and abs(rem - 0.70) < 1e-9, f"expected 0.70, got {rem}"
    s.close()


def test_remaining_budget_zero_when_over_cap():
    db = tempfile.mktemp(suffix=".db")
    s = MemoryStore(db)
    _row(s, cost_usd=1.50)
    p = RealLightifyPipeline(s)
    p._budget = {"max_daily_usd": 1.00}
    assert p._remaining_budget_usd() == 0.0
    s.close()


if __name__ == "__main__":
    tests = [
        test_insert_and_sum_trace,
        test_spend_since_horizon_respected,
        test_remaining_budget_none_when_unset,
        test_remaining_budget_positive_when_under_cap,
        test_remaining_budget_zero_when_over_cap,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
