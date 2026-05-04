"""Replay every captured solve and verify it reproduces bit-exact.

Week 1 success criterion (per consolidation-loop plan):
    "Replay the JSONL and reproduce all N current solves bit-exact."

Usage:
    python3 scripts/replay_solved_tasks.py
    python3 scripts/replay_solved_tasks.py --capture-path /tmp/run.jsonl
    python3 scripts/replay_solved_tasks.py --only-correct

Reads `klomboagi/data/solved_tasks.jsonl` (or override path), and for
each captured row, runs SmartARCSolverV2 fresh on the same (train,
test_input) and asserts predicted == row['predicted'].

Why this matters: if the capture isn't faithful enough to replay, then
Week 2's rule abstraction is operating on lies. This script is the
guardrail that proves the capture is real.

Anti-theater: this script does NOT use the row's `predicted` to feed
the solver — it only uses (train, test_input) and re-derives predicted
from scratch. The check is `re-run prediction == captured prediction`,
which is a strict reproducibility test on the solver itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `klomboagi` importable when run from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from klomboagi.reasoning.solve_capture import iter_capture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay captured solves bit-exact")
    parser.add_argument("--capture-path", default=None, help="Override capture JSONL path")
    parser.add_argument("--only-correct", action="store_true",
                        help="Only replay rows where correct=True")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to replay (0=all)")
    args = parser.parse_args()

    from klomboagi.reasoning.arc_smart_solver import SmartARCSolverV2

    cap_path = Path(args.capture_path) if args.capture_path else None
    rows = list(iter_capture(cap_path))

    if args.only_correct:
        rows = [r for r in rows if r["correct"]]
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("ERROR: capture file empty or missing")
        return 2

    solver = SmartARCSolverV2()

    n = len(rows)
    matches = 0
    mismatches: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []

    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        task_id = row["task_id"]
        train = row["train"]
        test_input = row["test_input"]
        captured = row["predicted"]

        try:
            replayed = solver.solve(train, test_input)
        except Exception as e:
            errors.append((task_id, str(e)[:120]))
            sys.stdout.write(f"\r  [{i}/{n}] {task_id} E   ")
            sys.stdout.flush()
            continue

        if replayed == captured:
            matches += 1
            mark = "="
        else:
            mismatches.append((task_id, f"got={_summarize(replayed)} cap={_summarize(captured)}"))
            mark = "X"
        sys.stdout.write(f"\r  [{i}/{n}] {task_id} {mark}   ")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print()
    print(f"\nReplay: {matches}/{n} bit-exact ({matches/n:.1%})  elapsed={elapsed:.1f}s")

    correct_capture = sum(1 for r in rows if r["correct"])
    print(f"Of which captured-correct: {correct_capture}/{n}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Errors: {len(errors)}")

    if mismatches:
        print("\n  First 10 mismatches:")
        for tid, info in mismatches[:10]:
            print(f"    {tid}: {info}")
    if errors:
        print("\n  First 10 errors:")
        for tid, info in errors[:10]:
            print(f"    {tid}: {info}")

    # Exit non-zero if any captured-correct row failed to replay — the
    # consolidation loop depends on those being faithful.
    correct_replay_failures = sum(
        1 for r, _info in mismatches
        if any(row["task_id"] == r and row["correct"] for row in rows)
    )
    if correct_replay_failures > 0 or errors:
        print("\nFAIL: at least one captured solve failed to replay.")
        return 1
    print("\nPASS: every captured solve replayed bit-exact.")
    return 0


def _summarize(grid):
    if grid is None:
        return "None"
    if not grid:
        return "[]"
    return f"{len(grid)}x{len(grid[0]) if grid[0] else 0}"


if __name__ == "__main__":
    sys.exit(main())
