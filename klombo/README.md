# Klombo

`Klombo` is a standalone memory and learning core for autonomous coding agents.

It is intentionally **not attached to CodeBot** in this build.
The goal of this package is to harden the learning layer first, prove that it
changes behavior over time, and only then expose integration surfaces.

## Purpose

Klombo should remember:

- what worked in a repo
- what failed repeatedly
- which commands are reliable
- which paths and file types matter
- which user preferences keep showing up
- how to resume interrupted work

Klombo should then use that memory to improve future planning context.

## Current Scope

This standalone v0.8 includes:

- append-only episode recording
- richer repo profile learning
- successful procedure extraction
- anti-pattern extraction from repeated failures
- user preference reinforcement
- per-repo preference scoping with global rollups
- semantic fact extraction from actual repo scans
- repo-family clustering for cautious transfer across similar repos
- mission resume state storage
- mission resume guidance with blocked-step recovery hints
- conflict-aware recovery planning for resume guidance
- operator review surfaces for recovery conflicts
- persisted operator review decisions for resumed missions
- mission-context invalidation for stale operator approvals
- planning-context retrieval with explainability
- confidence-weighted transfer controls across repo families
- decision-aware transfer learning from accepted and rejected reviews
- confidence decay and stale-memory pruning
- atomic writes and corruption quarantine
- scan-time architecture summaries with entrypoints, test dirs, and service boundaries
- ownership-zone extraction and lightweight dependency edges from repo scans
- realistic repo-shaped benchmark fixtures
- benchmark scaffolding for memory-on vs memory-off measurement
- benchmark history and regression tracking
- tamper-evident benchmark signing and verification
- external signing key sourcing via explicit key, provider, env var, or persisted fallback
- automated tests

## Not Included Yet

These are intentionally out of scope until the core is hardened:

- direct CodeBot integration
- live tool execution
- self-modifying code
- networked sync
- automatic write-back into another agent runtime
- privileged shell or filesystem operations outside Klombo's root

## Storage Layout

Klombo stores all state under a dedicated root:

```text
<root>/
  logs/
    episodes.jsonl
  quarantine/
    *.corrupt
  state/
    anti_patterns.json
    benchmark_runs.json
    benchmark_signing_key.txt
    missions.json
    operator_reviews.json
    preferences.json
    procedures.json
    repo_profiles.json
    transfer_reviews.json
```

## Core API

```python
from klombo import BenchmarkHarness, KlomboEngine
from klombo.fixtures import default_repo_scenarios

engine = KlomboEngine("./memory")

engine.record_episode(...)
engine.record_mission_state(...)
engine.record_operator_review(...)
engine.record_transfer_review(...)
engine.scan_repo("my-repo", "/path/to/repo")

context = engine.get_planning_context(
    repo_id="my-repo",
    request="Fix auth bug in src/auth",
    task_type="bugfix",
)

resume = engine.resume_context("mission_123")
engine.maintain_memories()
scenarios = default_repo_scenarios()

harness = BenchmarkHarness(
    engine,
    signing_key_env="KLOMBO_BENCHMARK_SIGNING_KEY",
    persist_generated_key=False,
)

# context now includes:
# - transfer_candidates
# - transfer_controls
# - explanations for transfer candidates
#
# resume now includes:
# - recovery_plan
# - conflicts
# - chosen_strategy
# - operator_review
```

If `KLOMBO_BENCHMARK_SIGNING_KEY` is set, benchmark history signing uses that
external key and does not need to persist a generated local key.

## Hardening Rules

Before Klombo attaches to any agent runtime, it should meet these rules:

1. No silent learning-path failures.
2. No mutation outside its configured storage root.
3. Benchmarks must show memory-on improvement over memory-off baselines.
4. Resume context must survive restarts.
5. Anti-patterns must reduce repeated failures on benchmark tasks.
6. Procedure scoring must be traceable and explainable.
7. Integration must be opt-in and disabled by default.
8. Corrupt state files must be quarantined instead of crashing the runtime.
9. Storage writes must be atomic inside Klombo's root.
10. Benchmark regressions must be visible before any integration is enabled.
11. Benchmark history must fail verification if tampered with.
12. Transfer across repos must be cautious and explainable, never silent.
13. Recovery conflicts should surface an explicit operator review path.
14. Operator-approved conflict resolutions should persist across restarts.
15. Transfer review outcomes should influence future cross-repo guidance.
16. Persisted approvals must be invalidated when mission context materially changes.

## Running Tests

From the `klombo/` directory:

```bash
python3 -m unittest discover -s tests -v
```

## Next Hardening Targets

- add deeper dependency extraction beyond lightweight import edges
- add integration adapters only after benchmark gains are stable
