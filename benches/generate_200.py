"""Generate synthetic queries with gold oracle tier labels.

Fillers are sized so that template * filler combinations exceed the target N
with substantial margin, and gen() rejects exact-string duplicates up to a
retry budget. For scale=25 (N=5000), effective combinatorial capacity is
>80k queries; post-generation uniqueness exceeds 95%.

Categories:
  - bash_like        : 20% of set, oracle = SMALL (mechanical)
  - short_lookup     : 20%,         oracle = SMALL (factual lookup)
  - reasoning        : 15%,         oracle = MID   (needs thought)
  - code             : 15%,         oracle = MID   (code generation)
  - large_code       : 10%,         oracle = FRONTIER (scope hint = heavy)
  - conflict         : 10%,         oracle = FRONTIER (contradiction triggers)
  - cold_knowledge   : 10%,         oracle = FRONTIER (outside memory store)

Gold label is a category-level prior, not per-query ground truth. See
benches/datasets/synthetic/README.md for the full methodology.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


BASH_TEMPLATES = [
    "ls -la {dir}",
    "cat {file}",
    "grep {pat} {file}",
    "find {dir} -name '{pat}'",
    "git {sub} {ref}",
    "kubectl get {res} -n {ns}",
    "kubectl describe {res} {name} -n {ns}",
    "kubectl logs {name} -n {ns}",
    "docker ps --filter 'name={name}' --filter 'status={dstate}'",
    "docker logs {name} --tail {n}",
    "docker inspect {name}",
    "what does the -{flag} flag of {cmd} do",
    "what does the --{longflag} option in {cmd} do",
    "how do i list {thing} in {ctx}",
    "how can i see all {thing} owned by {owner}",
    "what is the {cmd} command for {thing}",
    "show me {thing} in {dir}",
    "tail {file} for {pat}",
    "head -n {n} {file}",
    "wc -l {file}",
    "du -sh {dir}",
    "df -h {dir}",
    "ps aux | grep {thing}",
    "netstat -an | grep {thing}",
]
BASH_FILLERS = dict(
    dir=["/tmp", "/var/log", "/etc", "/home/user", "/opt", "/usr/local", "/srv",
         "/mnt/data", "/var/lib/docker", "/var/cache", "/root", "/proc", "/sys",
         "/app", "/data", "/workspace", "/build", "/test"],
    file=["foo.txt", "config.yaml", ".bashrc", "data.json", "app.log",
          "requirements.txt", "package.json", "Makefile", "Dockerfile",
          "nginx.conf", "my.cnf", "hosts", "resolv.conf", "auth.log",
          "syslog", "error.log", "stderr.log", "access.log", "crash.log",
          ".env", "pyproject.toml", "tsconfig.json", "go.mod", "Cargo.toml"],
    pat=["error", "TODO", "warning", "[0-9]+", "pattern", "FATAL", "WARN",
         "ERROR", "timeout", "connection refused", "segfault",
         "null pointer", "panic", "OOMKilled", "deprecated", "stacktrace",
         "exception", "401", "403", "404", "500", "502", "503"],
    sub=["status", "log", "diff", "show", "blame", "branch", "tag",
         "stash", "reflog", "remote", "config --get", "rev-parse"],
    ref=["HEAD", "main", "v1.2", "origin/main", "--oneline", "master",
         "release/2024-04", "v2.0.0-rc1", "hotfix/auth", "develop",
         "feature/routing", "--stat", "-p", "--graph"],
    res=["pods", "services", "deployments", "configmaps", "nodes",
         "statefulsets", "daemonsets", "ingress", "jobs", "cronjobs",
         "persistentvolumeclaims", "secrets", "replicasets",
         "horizontalpodautoscalers", "networkpolicies"],
    ns=["default", "kube-system", "production", "staging", "dev", "data",
        "monitoring", "logging", "platform", "ml", "ingress-nginx",
        "cert-manager", "istio-system", "observability"],
    name=["api-gateway", "worker-1", "db-primary", "cache", "auth-service",
          "router-mid", "payment-worker", "ingest-batch", "ml-trainer",
          "search-head", "metrics-sink", "kafka-broker-0", "postgres-primary",
          "redis-master", "etcd-0", "frontend-deployment", "scheduler",
          "canary-v3", "blue", "green"],
    dstate=["running", "exited", "paused", "restarting", "dead", "created"],
    flag=["a", "l", "h", "r", "v", "f", "n", "t", "u", "x", "p", "A", "R", "L", "V"],
    longflag=["help", "verbose", "dry-run", "force", "recursive", "output",
              "config", "follow", "limit", "context", "namespace",
              "selector", "no-headers", "timeout", "since"],
    thing=["files", "containers", "processes", "users", "ports",
           "services", "sockets", "threads", "handles", "mounts",
           "env vars", "groups", "connections", "routes", "interfaces"],
    cmd=["git", "kubectl", "docker", "systemctl", "grep", "find", "awk",
         "sed", "curl", "tar", "ssh", "rsync", "journalctl", "tcpdump"],
    ctx=["this directory", "the current pod", "production",
         "the staging cluster", "my shell", "the last hour"],
    owner=["root", "nobody", "me", "the api user", "the worker pool",
           "pid 1", "the scheduler", "the build job"],
    n=["5", "10", "50", "100", "200", "500", "1000"],
)

LOOKUP_TEMPLATES = [
    "what is {concept}?",
    "define {concept}",
    "what does {concept} mean",
    "briefly explain {concept}",
    "what is the purpose of {concept}",
    "who is {person}?",
    "who created {artifact}",
    "when was {event} released?",
    "when did {event} happen",
    "how many {unit} in a {bigger_unit}",
    "how many {unit} make up one {bigger_unit}",
    "what does {acronym} stand for?",
    "expand the acronym {acronym}",
    "convert {n} {unit} to {unit2}",
    "is {concept} the same as {concept2}?",
]
LOOKUP_FILLERS = dict(
    concept=["TLS", "idempotent", "a mutex", "dependency injection", "REST",
             "a closure", "memoization", "TCP", "UDP", "a race condition",
             "DNS", "OAuth", "CSRF", "CORS", "a load balancer",
             "Kubernetes ingress", "gRPC", "a webhook", "JWT",
             "eventual consistency", "a semaphore", "a circuit breaker",
             "a bloom filter", "a consistent hash", "HTTP/2", "QUIC",
             "a service mesh", "sharding", "partitioning", "idempotency",
             "a monad", "a coroutine", "a continuation", "garbage collection",
             "virtual memory", "copy-on-write", "backpressure",
             "a deadlock", "a livelock", "inversion of control"],
    concept2=["a fiber", "a thread", "a process", "a channel",
              "an actor", "a promise", "a future", "a generator"],
    person=["the CEO of Anthropic", "Linus Torvalds", "Guido van Rossum",
            "the creator of Git", "Tim Berners-Lee", "Yann LeCun",
            "Ken Thompson", "Dennis Ritchie", "Grace Hopper",
            "Alan Kay", "Donald Knuth", "Barbara Liskov",
            "Jeff Dean", "Martin Fowler", "the author of TCP/IP"],
    artifact=["Python", "Go", "Rust", "Kubernetes", "Linux",
              "Git", "Docker", "Redis", "Postgres", "Kafka",
              "gRPC", "LLVM", "Vim", "Emacs", "SQLite"],
    event=["Python 3.0", "TLS 1.3", "Kubernetes 1.0", "Rust 1.0", "Go 1.0",
           "Docker 1.0", "Redis 7.0", "HTTP/2", "QUIC", "JSON",
           "Unicode 1.0", "IPv6", "gRPC 1.0", "Node.js 1.0"],
    unit=["bytes", "bits", "seconds", "minutes", "hours", "days",
          "milliseconds", "microseconds", "nanoseconds", "pages",
          "blocks", "chunks", "rows"],
    bigger_unit=["kilobyte", "megabyte", "gigabyte", "day", "week", "year",
                 "decade", "century", "page", "frame"],
    acronym=["TLS", "MTU", "DNS", "CIDR", "JWT", "PKI", "REST", "CRUD",
             "RBAC", "ABAC", "SNI", "MTLS", "JWE", "OTEL", "OIDC",
             "SSO", "CSRF", "XSS", "MITM", "DDoS", "MAC", "CSR", "ACL"],
    n=["10", "64", "128", "1024", "256", "512", "4096", "8192",
       "32", "16", "2048", "65536"],
    unit2=["kilobytes", "milliseconds", "nanoseconds", "microseconds",
           "megabytes", "gigabytes", "seconds"],
)

REASONING_TEMPLATES = [
    "compare {a} and {b} for {goal}",
    "how does {a} differ from {b}",
    "when should you pick {a} over {b}",
    "why does {subject} {verb} when {condition}?",
    "what happens to {subject} if {condition}",
    "how would you design a {thing} for {goal}?",
    "what would a good {thing} look like for {goal}",
    "explain in detail the trade-offs of {choice}",
    "pros and cons of {choice} in {ctx}",
    "what are the trade-offs of using {choice} for {goal}?",
    "how would you implement {feature}",
    "what's the cheapest way to implement {feature}",
    "how do you avoid {symptom} when running {choice}",
]
REASONING_FILLERS = dict(
    a=["CDDR", "microservices", "REST", "SQL", "OAuth 2.0", "WebSockets",
       "gRPC", "a monorepo", "server-side rendering", "polling",
       "eventual consistency", "vector search", "keyword search",
       "BM25", "cross-encoders", "synchronous calls", "callbacks"],
    b=["RouteLLM", "monoliths", "GraphQL", "NoSQL", "SAML", "long-polling",
       "REST", "polyrepo", "static generation", "webhooks",
       "strong consistency", "graph search", "semantic search",
       "TF-IDF", "bi-encoders", "async/await", "promises"],
    subject=["prompt compression", "retry logic", "a cache", "a retry budget",
             "eventual consistency", "a circuit breaker", "backpressure",
             "leader election", "a scheduler", "the GC", "a hot partition",
             "a rate limiter", "a bloom filter", "TCP slow start",
             "the heap", "the JIT", "autoscaler"],
    verb=["reduce quality", "fail", "starve", "thrash", "degrade",
          "deadlock", "overshoot", "oscillate", "leak memory",
          "drop traffic", "miss a signal", "produce false positives"],
    condition=["retrieval fails", "traffic spikes", "downstream is slow",
               "memory is low", "the network partitions",
               "a replica is lost", "CPU saturates", "disk fills",
               "clock skew hits 1s", "TLS expires", "DNS flakes",
               "a dependency is rate-limited"],
    thing=["retry budget", "rate limiter", "cache eviction policy",
           "schema migration", "auth system", "feature-flag system",
           "sharded queue", "cross-region failover", "audit log",
           "event replay mechanism", "snapshot policy",
           "backoff schedule", "canary policy"],
    goal=["cascading tiers", "multi-tenant isolation", "gradual rollout",
          "backward compat", "zero-downtime deploys", "PII protection",
          "cross-region latency", "recovery under partition",
          "operator ergonomics", "observability", "SLO compliance",
          "cost predictability"],
    choice=["local inference", "caching", "eventual consistency",
            "a saga pattern", "server-side rendering", "streaming RPC",
            "optimistic locking", "read replicas", "stored procedures",
            "edge functions", "consistent hashing", "sharded writes"],
    feature=["a feature flag system", "a rate limiter",
             "request deduplication", "a priority queue",
             "a distributed lock", "a leader election service",
             "a gradual rollout system", "a changelog pipeline",
             "a dead-letter queue", "an event replay tool"],
    symptom=["hot partitions", "cache stampedes", "thundering herd",
             "N+1 queries", "memory bloat", "GC pauses",
             "queue depth explosions", "head-of-line blocking"],
    ctx=["a multi-tenant deployment", "an edge worker",
         "a hot path", "an internal tool", "a batch job",
         "a streaming pipeline", "a CLI", "a sidecar"],
)

CODE_TEMPLATES = [
    "write a function that {action}",
    "write a {lang} function that {action}",
    "generate a unit test for the {name} function",
    "write a test that covers {symptom} in the {component}",
    "refactor the {component} to use {thing}",
    "fix the bug in the {component} where {symptom}",
    "implement a {component} that {behavior}",
    "add {behavior} to the {component}",
    "write a helper that {action} and returns {rettype}",
    "patch the {component} so that {symptom} no longer happens",
]
CODE_FILLERS = dict(
    action=["reverses a string", "parses JSON", "computes fibonacci",
            "sorts a list", "deduplicates an array", "checks if a year is leap",
            "flattens a nested list", "merges two sorted arrays",
            "computes the Fibonacci word", "chunks an iterator",
            "maps a function over a dict", "groups by key",
            "memoizes a pure function", "debounces calls",
            "rate-limits by token bucket", "zips two streams",
            "validates an IPv4 address", "slugifies a title",
            "escapes HTML", "parses a semver string",
            "walks a directory tree lazily", "computes sha256",
            "exponentially backs off"],
    lang=["Python", "Go", "TypeScript", "Rust", "Java", "C++",
          "Kotlin", "Swift", "C#", "Ruby"],
    name=["parse_config", "compute_hash", "validate_input",
          "render_template", "route_request", "escape_html",
          "resolve_dns", "retry_with_backoff", "encode_base64",
          "stream_chunks", "merge_sorted", "build_prompt",
          "evict_cache", "count_tokens", "normalize_path",
          "parse_semver", "diff_trees", "split_batch"],
    component=["router module", "cache layer", "retry handler",
               "auth middleware", "serializer", "token bucket",
               "event dispatcher", "state reducer", "config loader",
               "memory store", "prompt shaper", "context builder",
               "conflict detector", "tier classifier", "pipeline runner"],
    thing=["dependency injection", "async/await", "generics",
           "a state machine", "a factory", "an ABC",
           "context managers", "structured concurrency",
           "a trait object", "a type class", "a builder",
           "a visitor pattern", "a proper iterator protocol"],
    symptom=["it leaks memory", "it double-counts events",
             "it times out under load", "it returns stale data",
             "it throws on empty input", "it swallows errors",
             "it deadlocks on reentry", "it silently truncates",
             "it misses the trailing newline", "it fails on UTF-8 BOM",
             "it rounds incorrectly", "it logs PII"],
    behavior=["deduplicates requests", "debounces events",
              "rate-limits callers", "caches responses",
              "retries with jitter", "backs off exponentially",
              "emits prometheus metrics", "streams chunks",
              "validates inputs", "logs structured JSON"],
    rettype=["a dict", "a list", "a generator", "a tuple",
             "a bool", "an Optional", "a Result", "a stream"],
)

LARGE_CODE_TEMPLATES = [
    "refactor the entire {component}",
    "rewrite the full {component} in {lang}",
    "generate unit tests for all the {component} methods",
    "migrate the whole {component} from {old} to {new}",
    "implement the entire {component} from scratch",
    "port the entire {component} to {lang}",
    "rebuild the full {component} test suite in {lang}",
    "redesign the entire {component} using {thing}",
    "rewrite all the {component} integrations",
    "modernize the whole {component}",
]
LARGE_CODE_FILLERS = dict(
    component=["authentication module", "routing subsystem", "ORM layer",
               "event pipeline", "cache layer", "retry machinery",
               "rate-limit middleware", "metrics exporter", "billing service",
               "search indexer", "inference gateway", "prompt builder",
               "feature-flag evaluator", "rollout controller",
               "permissions service", "scheduler core"],
    lang=["Rust", "Go", "TypeScript", "Python 3.13", "Kotlin", "Java 21",
          "Swift 6", "C++20", "Zig", "OCaml"],
    old=["callbacks", "REST", "synchronous calls", "threads",
         "manual polling", "raw SQL", "string templating"],
    new=["async/await", "gRPC", "message queues", "coroutines",
         "structured concurrency", "an ORM", "a DSL"],
    thing=["structured concurrency", "a state-machine DSL",
           "CQRS", "event sourcing", "a typed AST"],
)

CONFLICT_TEMPLATES = [
    "is {subject} {state} or {other_state}?",
    "is the {thing} currently {state} or {other_state}",
    "should we use {a} or {b} for {goal}?",
    "is {policy} still in effect?",
    "has {policy} changed since last week?",
    "what is the current {thing}?",
    "given the contradictory guidance on {topic}, what is correct?",
    "is it {state} or {other_state} for {subject} in {ctx}",
    "the docs say {subject} is {state}, but the runbook says {other_state}. which is right?",
    "which is current for {thing}: {a} or {b}?",
]
CONFLICT_FILLERS = dict(
    subject=["v7", "the API", "the cluster", "the feature flag",
             "canary-v3", "blue-green", "route-mid", "mcd-2",
             "the ingress", "the Sonnet alias", "Opus pricing",
             "the auth service", "the retry budget"],
    state=["production", "deprecated", "active", "paused",
           "rolled out", "canaried", "GA", "beta"],
    other_state=["staging", "supported", "paused", "live",
                 "disabled", "rolled back", "private", "archived"],
    a=["library A", "Protocol X", "Strategy A", "route-A",
       "policy-2024-11", "v3 schema", "canary-blue"],
    b=["library B", "Protocol Y", "Strategy B", "route-B",
       "policy-2025-04", "v4 schema", "canary-green"],
    goal=["production", "our use case", "the migration",
          "cost control", "quality", "compliance"],
    policy=["the data retention policy", "the escalation policy",
            "the on-call rotation", "the rollout policy",
            "the auth-token TTL", "the PII policy",
            "the rate-limit policy", "the canary policy"],
    thing=["production version", "default tier", "escalation target",
           "routing flag", "memory backend", "tier-2 alias",
           "active branch", "canary state"],
    topic=["the migration", "the deprecation", "the rollout",
           "the schema change", "the config freeze",
           "the alias rewrite", "the cost estimate"],
    ctx=["prod", "staging", "dev", "the canary region",
         "the shadow cluster"],
)

COLD_TEMPLATES = [
    "what is the capital of {country}?",
    "what is the official language of {country}",
    "what currency does {country} use",
    "what is the land area of {country} in km^2",
    "when did {country} gain independence",
    "who is the current head of state of {country}",
    "who won the {event} in {year}?",
    "who was the {award} laureate in {year}",
    "which country won the {event} in {year}",
    "who was the runner-up in {event} in {year}",
    "what is {obscure}?",
    "explain the {obscure}",
    "where does the name {obscure} come from",
    "summarize the history of {topic}",
    "what is the latest research on {topic}?",
    "what are the main findings in {topic} as of 2024",
    "list notable papers on {topic}",
    "what are the key open problems in {topic}",
    "how did {topic} research progress between {year} and 2024",
]
COLD_FILLERS = dict(
    country=["Burkina Faso", "Kyrgyzstan", "Suriname",
             "Eswatini", "Tuvalu", "Vanuatu",
             "Bhutan", "Djibouti", "Lesotho", "Moldova",
             "Belarus", "Latvia", "Estonia", "Lithuania",
             "Tajikistan", "Turkmenistan", "Uzbekistan",
             "Kazakhstan", "Mongolia", "Mauritania",
             "Chad", "Niger", "Mali", "Senegal"],
    event=["Nobel Prize in Physics", "FIFA Women's World Cup",
           "Booker Prize", "Chess Olympiad", "Fields Medal",
           "Turing Award", "Abel Prize", "Pritzker Prize",
           "Venice Biennale", "Cannes Palme d'Or",
           "Rugby World Cup", "Cricket World Cup",
           "Tour de France", "Ballon d'Or"],
    award=["Nobel", "Turing", "Abel", "Fields", "Pritzker",
           "Pulitzer", "Booker", "Hugo"],
    year=["2005", "2008", "2011", "2014", "2016",
          "2018", "2020", "2022", "2024"],
    obscure=["the Voynich manuscript", "the Piltdown Man hoax",
             "the Tunguska event", "the Hessdalen lights",
             "the Green Children of Woolpit", "the Dyatlov Pass incident",
             "the Chinguetti meteorite", "the Oakville blobs",
             "the Antikythera mechanism", "the Phaistos Disc",
             "the Nazca Lines", "the Baghdad Battery",
             "the Signal of Wow", "the Taos Hum"],
    topic=["helium superfluidity", "turbulent mixing layers",
           "antibiotic-resistant E. coli", "solid-state batteries",
           "quantum error correction", "room-temperature superconductors",
           "cortical microcircuits", "deep-sea hydrothermal vents",
           "neutrino oscillations", "glacial isostatic rebound",
           "magnetar flares", "CRISPR off-target effects",
           "amyloid plaque imaging", "gravitational lensing"],
)


def fill(template: str, fillers: dict[str, list[str]], rng: random.Random) -> str:
    out = template
    for _ in range(5):
        replaced = False
        for k, opts in fillers.items():
            marker = "{" + k + "}"
            if marker in out:
                out = out.replace(marker, rng.choice(opts), 1)
                replaced = True
        if not replaced:
            break
    return out


def gen(n: int, templates: list[str], fillers: dict, category: str,
        oracle: str, rng: random.Random,
        has_conflict: bool = False) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    max_attempts = n * 20
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        t = rng.choice(templates)
        q = fill(t, fillers, rng)
        if q in seen:
            continue
        seen.add(q)
        out.append({
            "id": f"{category}-{len(out) + 1}",
            "category": category,
            "query": q,
            "oracle_tier": oracle,
            "has_contradiction": has_conflict,
        })
    if len(out) < n:
        # fell short — capacity insufficient; pad with duplicates but flag loudly
        print(f"  warning: only produced {len(out)}/{n} unique {category} queries; "
              f"padding with {n - len(out)} duplicates")
        i = 0
        while len(out) < n:
            dup = dict(out[i % len(out)])
            dup["id"] = f"{category}-{len(out) + 1}"
            out.append(dup)
            i += 1
    return out


def main(scale: int = 1, seed: int = 42, tag: str | None = None):
    """Generate synthetic benchmark set. scale=1 -> 200, scale=10 -> 2000, scale=25 -> 5000."""
    rng = random.Random(seed)
    rows: list[dict] = []
    rows.extend(gen(40 * scale, BASH_TEMPLATES, BASH_FILLERS, "bash_like", "SMALL", rng))
    rows.extend(gen(40 * scale, LOOKUP_TEMPLATES, LOOKUP_FILLERS, "short_lookup", "SMALL", rng))
    rows.extend(gen(30 * scale, REASONING_TEMPLATES, REASONING_FILLERS, "reasoning", "MID", rng))
    rows.extend(gen(30 * scale, CODE_TEMPLATES, CODE_FILLERS, "code", "MID", rng))
    rows.extend(gen(20 * scale, LARGE_CODE_TEMPLATES, LARGE_CODE_FILLERS, "large_code", "FRONTIER", rng))
    rows.extend(gen(20 * scale, CONFLICT_TEMPLATES, CONFLICT_FILLERS, "conflict",
                    "FRONTIER", rng, has_conflict=True))
    rows.extend(gen(20 * scale, COLD_TEMPLATES, COLD_FILLERS, "cold_knowledge", "FRONTIER", rng))

    rng.shuffle(rows)

    fname = f"queries_{len(rows)}" + (f"_{tag}" if tag else "") + ".json"
    out_path = Path(__file__).parent / "datasets" / "synthetic" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    uniq = len(set(r["query"] for r in rows))
    print(f"wrote {len(rows)} queries to {out_path}  (unique: {uniq} = {uniq/len(rows):.1%})")
    return fname, uniq, len(rows)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1,
                   help="1 -> 200 queries; 10 -> 2000; 25 -> 5000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None,
                   help="optional suffix appended to filename")
    args = p.parse_args()
    main(scale=args.scale, seed=args.seed, tag=args.tag)
