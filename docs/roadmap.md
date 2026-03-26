# KlomboAGI — Roadmap

## Current State (Stage 0)

- 100 ARC-AGI puzzles solved via strategy search (no LLM)
- Baby conversation interface with NARS truth values
- CognitionLoop orchestrator (10 phases)
- Knowledge graph on external storage (137KB brain)
- 299 tests passing
- No real learning. No transfer. No autonomy beyond strategy matching.

## Stage 1: General Task Executor

**Goal**: Complete unseen coding and research tasks with tools.

### Workstreams

1. **Hidden eval harness** (FIRST)
   - 50 tasks across 5 domains, never seen during development
   - Automated scoring: success, interventions, autonomy horizon
   - Baseline: run each task manually and record human performance

2. **Generalize planner**
   - Remove repo-specific assumptions from planner.py
   - General action schema: read, write, search, execute, ask
   - Subgoal decomposition without domain-specific heuristics

3. **Broaden tool layer**
   - Browser/web actions
   - API clients
   - Test runners
   - Git-aware workspace ops
   - Database queries

4. **Structured logging**
   - Every step emits: action, observation, decision, outcome
   - Stored as trajectory in datasets/trajectories/

### Gate
- >50% success on hidden eval suite
- <1 intervention per task average
- Works on 3+ domains

## Stage 2: Memory That Matters

**Goal**: Retrieved memory measurably improves future task success.

1. **Causal memory scoring**
   - Log every memory retrieval
   - Track whether it changed the decision
   - Track whether the changed decision improved the outcome

2. **Skill extraction from trajectories**
   - After successful task completion, extract reusable procedure
   - Store as named skill with preconditions and steps

3. **Failure pattern mining**
   - After failed task, extract what went wrong
   - Store as anti-pattern to avoid in future

### Gate
- Memory usefulness >30%
- Demonstrable improvement over 10+ episodes

## Stage 3: Transfer

**Goal**: Success on new repos and new task families without hand-tuned heuristics.

1. **Cross-domain eval suite**
   - Tasks in domains the system was never trained on
   - Measure transfer from known domains

2. **Abstract skill representation**
   - Skills stored as structural patterns, not domain-specific procedures
   - "Debug a failing test" applies to Python, JavaScript, Rust

### Gate
- >0.3 transfer ratio
- No domain-specific heuristics in the runtime

## Stage 4: Long-Horizon Autonomy

**Goal**: Run for hours, recover from failure, resume after interruption.

1. **Subgoal trees with uncertainty tracking**
2. **Checkpointed recovery** — save state, resume after crash
3. **Replanning under tool failure**
4. **Contradiction detection and handling**

### Gate
- 50+ step autonomy horizon
- >50% failure recovery rate
- Resume after simulated crash

## Stage 5: External Proof

**Goal**: Blind evals, outside replication, strong baselines.

1. **Publish eval suite** (without answers)
2. **Invite external testers**
3. **Compare against GPT-4, Claude, SWE-agent on same tasks**
4. **Publish results with full methodology**

### Gate
- Outside person reproduces results
- Outperforms baseline on at least one metric
- No gaming detected
