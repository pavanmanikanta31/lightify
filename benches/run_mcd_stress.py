"""MCD stress test — reviewer must-fix #5.

Measures precision, recall, and F1 of Memory Conflict Detection on:
  - 100 hand-authored realistic contradiction pairs
    (version bumps, policy changes, numerical revisions, deprecations,
     team/feature-flag/infra/security/compliance/pricing/schema/API/
     runbook/dependency/observability changes)
  - 100 non-contradiction control pairs drawn from the same categories
    (same-topic statements that are compatible, not contradictory)

95% Wilson confidence intervals on precision and recall.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightify.conflict import apply_conflict_penalties
from lightify.types import ContextCapsule, MemoryItem, Tier


# ── Control pairs (non-contradictions; same-topic compatible statements) ──
# 100 same-topic but compatible statements across the same 15 categories.
CONTROLS: list[dict] = []
_control_pairs = [
    ("The production version is v7", "We plan to upgrade to v8 next quarter"),
    ("We are running Python 3.11 in production", "The Python team maintains 3.11 until 2027"),
    ("The Kubernetes cluster runs 1.28", "Version 1.28 reaches EOL in October"),
    ("We use Claude Sonnet 4.6", "Claude Sonnet supports up to 200K context"),
    ("The API version is v2", "The v2 API uses JSON response bodies"),
    ("Our TLS policy requires TLS 1.3", "TLS 1.3 avoids RSA key exchange"),
    ("The postgres major version is 16", "Postgres 16 added logical replication improvements"),
    ("We target Node 20", "Node 20 is the current LTS line"),
    ("The Rust MSRV is 1.75", "1.75 stabilized async fn in traits"),
    ("We run Go 1.22", "Go 1.22 improved the range-over-int loop"),
    ("Log retention is 30 days", "Logs older than 30 days are moved to cold storage"),
    ("The on-call rotation is weekly", "Weekly rotations align with sprint boundaries"),
    ("Session tokens expire after 24 hours", "Refresh tokens last 30 days"),
    ("Default retry budget is 3 attempts", "Retries use exponential backoff with jitter"),
    ("PII must be encrypted at rest", "Encryption uses KMS-backed keys"),
    ("Data retention policy is 365 days", "After 365 days, data is anonymized"),
    ("The cluster is multi-region", "Traffic is routed by Global Accelerator"),
    ("Rate limit is 100 requests per minute", "Rate limits are enforced per API key"),
    ("Deployments are blue-green", "We use Route53 weighted records for the cutover"),
    ("Backups run nightly", "Backups are stored in a separate account"),
    ("Tier-2 cost is $0.019 per query", "Tier-2 uses Claude Sonnet pricing"),
    ("Target p99 latency is 500 ms", "P99 is measured over a 5-minute window"),
    ("SLO is 99.9% availability", "SLOs are measured quarterly"),
    ("Cache TTL is 60 seconds", "Cache is warmed on deploy"),
    ("Auto-scale threshold is 70% CPU", "Auto-scaler reports metrics every minute"),
    ("Memory limit is 4 GB", "Memory limits are enforced by the kernel cgroup"),
    ("Connection pool size is 10", "Connection pool uses PgBouncer"),
    ("Default timeout is 5 seconds", "Timeouts are set per call site"),
    ("Batch size is 64", "Batch size is configurable per job"),
    ("Worker concurrency is 4", "Workers use a shared queue"),
    ("The /v1/login endpoint is active", "V1 endpoints are documented in the changelog"),
    ("Feature flag LEGACY_AUTH is enabled", "Flags are evaluated at the edge"),
    ("GraphQL is our primary API", "GraphQL uses Apollo Server 4"),
    ("We use Redis for session storage", "Redis is clustered across 3 nodes"),
    ("The old DAG is scheduled", "Scheduled DAGs run in UTC"),
    ("Canary-blue is serving traffic", "Canary traffic is 5% of total"),
    ("Service X is live", "Service X exposes a gRPC API"),
    ("Library A is supported", "A supports Python 3.9+"),
    ("The auth proxy is in place", "The proxy terminates TLS"),
    ("Webhook endpoint is active", "Webhooks carry an HMAC signature"),
    ("Alice is on-call primary", "On-call shifts are 24 hours"),
    ("The platform team owns routing", "The team has a dedicated Slack channel"),
    ("Project OWNER is pavan", "Project was started in April"),
    ("Escalation goes to the SRE pager", "SRE pager routes to PagerDuty"),
    ("The data team handles ingestion", "Ingestion throughput peaks at 10K rps"),
    ("Feature flag NEW_UI is rolled out to 100% of users", "NEW_UI launch was announced in the changelog"),
    ("CDDR action routing is enabled", "CDDR logs routing decisions to the trace table"),
    ("Experiment EXP_ROUTER_V2 is active", "EXP_ROUTER_V2 is an A/B test over 14 days"),
    ("The dark-launch for tier-2 pricing is off", "Tier-2 pricing will launch next quarter"),
    ("Flag ENABLE_CACHING is true", "Caching layer uses Valkey"),
    ("The mTLS migration is complete", "mTLS uses Vault-issued certificates"),
    ("The staging environment mirrors prod", "Staging is refreshed weekly"),
    ("The database is read-write in us-east-1", "Writes replicate to us-west-2"),
    ("The new load balancer is healthy", "The LB uses HTTP/2 between edges"),
    ("Region us-west-2 is primary", "Primary region handles all write traffic"),
    ("The VPC peering is established", "Peering uses private routing"),
    ("Autoscaler is running", "Autoscaler reports to Datadog"),
    ("The secondary replica is in sync", "Replication lag is monitored"),
    ("The index is fully built", "Index rebuilds happen on schema migrations"),
    ("The queue is being drained", "The drain is part of a planned maintenance"),
    ("Secret rotation is daily", "Rotated secrets are fetched via Vault"),
    ("MFA is required for all admin accounts", "MFA uses WebAuthn"),
    ("The WAF is in block mode", "The WAF rules are tuned quarterly"),
    ("Audit logs are forwarded to the SIEM", "SIEM retention is 1 year"),
    ("Encryption uses AES-256-GCM", "Keys are 256 bits"),
    ("We are SOC 2 Type 2 compliant", "SOC 2 Type 2 audits run annually"),
    ("PHI is stored separately from PII", "PHI storage is HIPAA-aligned"),
    ("EU data stays in EU regions", "EU residency is enforced by policy"),
    ("PCI scope includes the billing service", "PCI controls follow PCI-DSS 4.0"),
    ("GDPR erasure completes within 30 days", "Erasure is tracked in a queue"),
    ("Enterprise tier is $0.50 per 1M tokens", "Enterprise pricing includes SLA"),
    ("Free tier allows 10K calls per month", "Free tier resets on the 1st"),
    ("The annual discount is 15%", "Discounts apply to annual contracts"),
    ("Opus pricing is $15 per 1M input tokens", "Output tokens are priced separately"),
    ("Cache read is 0.1x of base price", "Cache reads do not count against rate limit"),
    ("User table has a last_login column", "last_login is indexed"),
    ("Orders table has email_idx", "email_idx was added in the 2024-03 migration"),
    ("Events table is partitioned by day", "Partitions older than 90 days are detached"),
    ("Sessions.expires_at is NOT NULL", "expires_at is a timestamptz"),
    ("The orders_id column is BIGINT", "BIGINT avoids overflow at scale"),
    ("POST /users returns the full user object", "User objects include profile metadata"),
    ("The pagination token is opaque", "Tokens are URL-safe base64"),
    ("Error codes are stable", "Error codes are documented in the OpenAPI spec"),
    ("GET /health returns 200 always", "Health checks run every 10 seconds"),
    ("The webhook payload is JSON", "Webhooks support retries on 5xx"),
    ("On high error rate, restart the worker pool", "Restarts are rate-limited to 1 per minute"),
    ("Failover takes 2 minutes", "Failover is tested quarterly"),
    ("Escalate to the security team after 10 minutes", "Escalations are tracked in PagerDuty"),
    ("Keep the pod running during investigation", "Preserved pods are labeled 'forensics'"),
    ("Roll forward is the default recovery", "Rolls forward are automated via CI"),
    ("Service A depends on Service B", "B exposes a stable gRPC API"),
    ("The billing job runs after the metrics job", "Both jobs are orchestrated by Airflow"),
    ("We require pandas 1.x", "pandas 1.x has long-term support until 2025"),
    ("The API gateway is behind the load balancer", "The LB uses AWS Target Groups"),
    ("We consume events from Kafka", "Kafka runs on MSK"),
    ("Traces are sampled at 1%", "Traces use OpenTelemetry"),
    ("Alerts fire on 5xx > 0.1%", "Alerts go to a dedicated Slack channel"),
    ("Logs are structured JSON", "JSON logs include trace IDs"),
    ("Dashboards use UTC", "UTC avoids DST ambiguity"),
    ("Metric cardinality is capped at 10K", "Cardinality caps are enforced per metric"),
]
for i, (a, b) in enumerate(_control_pairs):
    CONTROLS.append({
        "id": f"ctrl-{i+1:03d}",
        "category": "control",
        "item_a": a,
        "item_b": b,
        "kind": "none",
        "is_contradiction": False,
    })


def mcd_detects(a: str, b: str) -> bool:
    """Run MCD over a two-item ContextCapsule; return True if a conflict was flagged."""
    items = [
        MemoryItem(id=1, content=a, topic="", confidence=0.5, source_tier=Tier.MID),
        MemoryItem(id=2, content=b, topic="", confidence=0.5, source_tier=Tier.MID),
    ]
    cap = ContextCapsule(
        prompt="", raw_items=items, compressed_items=[a, b],
        context_confidence=0.5, num_items=2, coverage=0.5, sufficient=True,
    )
    cap = apply_conflict_penalties(cap)
    return len(cap.conflicts) > 0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0, center - half), min(1, center + half))


def main():
    pairs_path = Path(__file__).parent / "datasets" / "mcd" / "contradictions_100.json"
    positives = json.loads(pairs_path.read_text())  # should be detected
    negatives = CONTROLS                             # should NOT be detected

    t_start = time.time()
    tp = sum(1 for p in positives if mcd_detects(p["item_a"], p["item_b"]))
    fn = len(positives) - tp
    fp = sum(1 for p in negatives if mcd_detects(p["item_a"], p["item_b"]))
    tn = len(negatives) - fp
    elapsed_ms = (time.time() - t_start) * 1000

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Wilson 95% CIs
    rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)

    print(f"MCD stress test  (N_pos={len(positives)}, N_neg={len(negatives)}, elapsed {elapsed_ms:.1f} ms)\n")
    print(f"  TP={tp}   FN={fn}   FP={fp}   TN={tn}")
    print(f"  Precision = {precision:.3f}   95% Wilson CI [{prec_lo:.3f}, {prec_hi:.3f}]")
    print(f"  Recall    = {recall:.3f}   95% Wilson CI [{rec_lo:.3f}, {rec_hi:.3f}]")
    print(f"  F1        = {f1:.3f}")
    print()

    # Per-category recall breakdown
    per_cat = {}
    for p in positives:
        per_cat.setdefault(p["category"], {"tp":0,"n":0})
        d = per_cat[p["category"]]
        d["n"] += 1
        if mcd_detects(p["item_a"], p["item_b"]):
            d["tp"] += 1
    print("Per-category recall:")
    print(f"  {'category':<20}{'n':>5}{'tp':>5}{'recall':>10}")
    print("  " + "-"*40)
    for cat in sorted(per_cat):
        d = per_cat[cat]
        print(f"  {cat:<20}{d['n']:>5}{d['tp']:>5}{d['tp']/d['n']:>10.3f}")

    # Per-signal-kind recall
    per_kind = {}
    for p in positives:
        per_kind.setdefault(p["kind"], {"tp":0,"n":0})
        d = per_kind[p["kind"]]
        d["n"] += 1
        if mcd_detects(p["item_a"], p["item_b"]):
            d["tp"] += 1
    print("\nPer-signal-kind recall:")
    for kind, d in sorted(per_kind.items()):
        print(f"  {kind:<12}  n={d['n']:>3}  tp={d['tp']:>3}  recall={d['tp']/d['n']:.3f}")

    # Save
    out = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_ci_95": [prec_lo, prec_hi],
        "recall_ci_95": [rec_lo, rec_hi],
        "per_category": per_cat,
        "per_kind": per_kind,
    }
    (Path(__file__).parent / "results_mcd_stress.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> benches/results_mcd_stress.json")


if __name__ == "__main__":
    main()
