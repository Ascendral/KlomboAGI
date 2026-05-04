"""Capture every successful ARC solve to JSONL for downstream consolidation.

This is Week 1 of the consolidation-loop bet (see plan in session log):
the goal is a JSONL file that, replayed against a fresh solver, reproduces
every captured solve bit-exact. No analysis, no abstraction yet — just
faithful capture at the system boundary.

Why hook at the boundary (eval harness) and not inside the solver:
The solver is ~13k lines with ~50 return points across 14 phase blocks.
Instrumenting every return is fragile and adds blast radius. The eval
harness already has (task_id, train, test_input, predicted, expected)
in scope after the call returns — that's a sufficient capture for Week 1.

Week 2 will add a primitive_trace field (which phase / which learner
solved this), populated via a thread-local push/pop context manager
inside the solver.

Anti-theater note: this module ONLY records. It does not change solver
behavior. If you remove the import, the solver runs identically.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Default capture file. Override via env var KLOMBOAGI_CAPTURE_PATH.
DEFAULT_CAPTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "solved_tasks.jsonl"
)


def _capture_path() -> Path:
    env = os.environ.get("KLOMBOAGI_CAPTURE_PATH")
    return Path(env) if env else DEFAULT_CAPTURE_PATH


def record_solve(
    *,
    task_id: str,
    dataset: str,                        # 'training' or 'evaluation'
    train: list[dict[str, list[list[int]]]],
    test_input: list[list[int]],
    predicted: list[list[int]] | None,
    expected: list[list[int]] | None,
    correct: bool,
    error: str = "",
    time_ms: float = 0.0,
    solver_class: str = "",
    primitive_trace: list[str] | None = None,  # populated in Week 2
    capture_path: Path | None = None,
) -> None:
    """Append one solve attempt to the capture JSONL.

    Schema is fixed (changes are breaking) — version bumps land as new
    lines with `schema_version` incremented and a migration script.
    """
    path = capture_path or _capture_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "schema_version": 1,
        "ts": time.time(),
        "task_id": task_id,
        "dataset": dataset,
        "train": train,
        "test_input": test_input,
        "predicted": predicted,
        "expected": expected,
        "correct": bool(correct),
        "error": error,
        "time_ms": float(time_ms),
        "solver_class": solver_class,
        "primitive_trace": primitive_trace or [],
    }
    with path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def iter_capture(capture_path: Path | None = None):
    """Yield rows from the capture JSONL (newest writes last).

    Filters out malformed lines and bumps `schema_version` mismatch by
    raising — we want to know loudly if the format drifted.
    """
    path = capture_path or _capture_path()
    if not path.exists():
        return
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"corrupt capture row at {path}:{lineno}: {e}"
                ) from e
            sv = row.get("schema_version")
            if sv != 1:
                raise RuntimeError(
                    f"capture schema_version mismatch at {path}:{lineno}: "
                    f"got {sv!r}, expected 1"
                )
            yield row


def reset_capture(capture_path: Path | None = None) -> None:
    """Truncate the capture file. Use before a fresh eval run."""
    path = capture_path or _capture_path()
    if path.exists():
        path.unlink()
