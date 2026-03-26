# brain_core

`brain_core` is the Rust side of the future KlomboAGI cognition kernel.

Python should keep owning orchestration, tool adapters, storage, and the fast-iterating research loop.
Rust should own the deterministic, high-frequency pieces that benefit from stronger typing, lower latency,
and long-running state safety.

## Module Layout

```text
brain_core/
  Cargo.toml
  README.md
  src/
    lib.rs          # Python bindings and public kernel entrypoints
    types.rs        # Shared structs for retrieval, planning, transfer, learning
    retrieval.rs    # Memory ranking and nearest-neighbor style lookup
    plan_search.rs  # Candidate scoring and cheap search over next-step plans
    scoring.rs      # Reliability, causal benefit, and regression scoring
    transfer.rs     # Structural similarity and transfer mapping
    learning.rs     # Procedure extraction and anti-pattern synthesis
```

## Python Boundary

The intended ownership split is:

- Python:
  - mission intake
  - world-state persistence
  - CLI and daemon lifecycle
  - tool execution
  - experiment control
- Rust:
  - `retrieve_memory(query, memories, limit)`
  - `score_plan(goal, candidates, anti_patterns)`
  - `transfer_score(source_tags, target_tags)`
  - `reliability_score(successes, failures)`
  - `extract_learning(task_id, domain, actions, success, failure_reason)`

That gives KlomboAGI a fast kernel without forcing the whole repo to stop being a Python research codebase.

## First Integration Targets

1. Replace heuristic memory ranking in the planner and working-memory path.
2. Move causal-memory usefulness scoring into a deterministic kernel.
3. Score plan candidates against learned anti-patterns before execution.
4. Add transfer matching between past successful trajectories and new tasks.

## Why Rust Here

- safer long-running state than ad hoc Python objects
- deterministic kernels for eval reproducibility
- easy Python interop with `pyo3`
- good future path for parallel search, indexing, and retrieval

This crate is intentionally small right now: it is a scaffold for the boundary, not a claim that the "brain"
is solved.
