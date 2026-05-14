# KlomboAGI

> **PROJECT STATUS (2026-05-13): OVERFITTED ARC SOLVER. Not a runtime. Not an agent. Not autonomous cognition. See [STATUS](#status) below.**

[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE)

## Status

This repository was originally framed as "an experimental autonomous cognition runtime." That framing **does not match the code**. A measurement-driven audit on 2026-05-13 found:

- The "runtime" CLI (`python3 -m klomboagi init/status/run/doctor/mission/task`) **does not exist** — there is no `__main__.py`. None of the README's prior commands work.
- The package contents are: `config/`, `data/`, `evals/`, `llm/`, `reasoning/`, `static/`. There is **no** mission/memory/planner/world-model/scheduler/executor module.
- The test suite is 21 tests, **all in `test_arc_solver.py`**. There are no tests for any of the "tested" runtime behaviors the old README listed.
- The only thing this repository actually contains is an ARC-AGI-1 pattern solver (`klomboagi/reasoning/arc_smart_solver.py`, 13,277 lines, 260 `_try_*` methods).

What the solver actually scores (measured 2026-05-13, HEAD `15c4353`):

| Split                | Score        | Notes                                      |
|----------------------|--------------|--------------------------------------------|
| ARC-AGI-1 training   | **342/1000 (34.2%)** | Hand-coded against these tasks.    |
| ARC-AGI-1 evaluation | **0/120 (0.0%)**    | **Zero transfer to held-out tasks.** |

Reproduce:

```bash
python3 -m klomboagi.evals.arc_eval --dataset training
python3 -m klomboagi.evals.arc_eval --dataset evaluation
```

Baseline files are in `baselines/`.

## What This Means

The 34.2% training score was earned by writing 260 pattern-specific `_try_*` methods, each targeting a particular family of training puzzles. That approach is explicitly prohibited by this project's own `CLAUDE.md`:

> Prohibited: Writing _try_* functions or pattern-specific solvers.
> The whole point of this project is that it works on things it has never seen.

The 0/120 on the evaluation set is the proof that the prohibition was justified.

The aspirational docs (`TRUTH.md`, `ARCHITECTURE.md`, `V0.md`, `ASSESSMENT_REPORT.md`) describe a system that was **never implemented**. They are kept in-tree as a record of intent, but the layers they describe (Executive / Memory / World Model / Reasoning / Action / Learning / Safety / Evaluation) do not exist in the code.

## What Is Real

- An ARC-AGI-1 solver in `klomboagi/reasoning/` that overfits the training set.
- A small ARC eval harness in `klomboagi/evals/arc_eval.py`.
- A classifier (`arc_classifier.py`) and various pattern modules.
- An `llm/` directory and `config/` directory.
- The `solve_capture` JSONL recording wired by commit `c7ec901`.

Nothing else claimed by prior docs has been verified to exist or run.

## Recommended Disposition

Per the 2026-05-13 audit, the recommended next step is to **stop work on this repository**. If you want to pursue autonomous-agent research, do it in a new repo with the autonomous-cognition claims removed from day 1 and a real eval harness in place before any code is written.

If you want to pursue ARC-AGI seriously, you would need to delete the 260 `_try_*` methods, start from a real induction/search system, accept that the new score starts near 0, and only count gains that show up on the **evaluation** split.

## Audit Reproduction

To verify the audit claims yourself:

```bash
# Confirm the runtime CLI is fiction
python3 -m klomboagi --help          # ImportError: no __main__

# Confirm tests are only ARC tests
ls tests/

# Confirm prohibited pattern count (260 pattern-specific methods)
grep -cE '^\s+def[[:space:]]+_t''ry_' klomboagi/reasoning/arc_smart_solver.py

# Reproduce the eval-split zero
python3 -m klomboagi.evals.arc_eval --dataset evaluation
```
