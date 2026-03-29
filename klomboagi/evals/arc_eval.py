"""
ARC-AGI-1 Evaluation — get a real score.

Runs the SmartARCSolver (V18 + learned strategy ordering) against
the full ARC-AGI-1 dataset via arckit.

Reports:
  - Overall accuracy (correct / total)
  - Per-task results (pass/fail + which strategy worked)
  - Failure analysis (what types of tasks fail and why)
  - Timing

This is the single most important benchmark for KlomboAGI.
A real number tells us exactly where we are.
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field


@dataclass
class ArcEvalResult:
    """Result of evaluating one ARC task."""
    task_id: str = ""
    correct: bool = False
    predicted: list[list[int]] | None = None
    expected: list[list[int]] | None = None
    strategy_used: str = ""
    time_ms: float = 0
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    error: str = ""


@dataclass
class ArcEvalReport:
    """Full evaluation report."""
    results: list[ArcEvalResult] = field(default_factory=list)
    total: int = 0
    correct: int = 0
    failed: int = 0
    errors: int = 0
    total_time_s: float = 0

    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    def summary(self) -> str:
        lines = [
            f"ARC-AGI-1 Evaluation: {self.correct}/{self.total} "
            f"({self.accuracy():.1%})",
            f"  Time: {self.total_time_s:.1f}s "
            f"({self.total_time_s / self.total * 1000:.0f}ms/task)" if self.total > 0 else "",
            f"  Correct: {self.correct}",
            f"  Failed: {self.failed}",
            f"  Errors: {self.errors}",
        ]

        # Size analysis
        correct_sizes = {}
        failed_sizes = {}
        for r in self.results:
            key = f"{r.input_shape}→{r.output_shape}"
            if r.correct:
                correct_sizes[key] = correct_sizes.get(key, 0) + 1
            else:
                failed_sizes[key] = failed_sizes.get(key, 0) + 1

        # Growth/shrink/same analysis
        same_size_correct = sum(1 for r in self.results if r.correct and r.input_shape == r.output_shape)
        same_size_total = sum(1 for r in self.results if r.input_shape == r.output_shape)
        diff_size_correct = sum(1 for r in self.results if r.correct and r.input_shape != r.output_shape)
        diff_size_total = sum(1 for r in self.results if r.input_shape != r.output_shape)

        lines.append(f"\n  Same-size tasks: {same_size_correct}/{same_size_total} "
                     f"({same_size_correct/same_size_total:.1%})" if same_size_total > 0 else "")
        lines.append(f"  Diff-size tasks: {diff_size_correct}/{diff_size_total} "
                     f"({diff_size_correct/diff_size_total:.1%})" if diff_size_total > 0 else "")

        # Show some failures
        failures = [r for r in self.results if not r.correct and not r.error]
        if failures:
            lines.append(f"\n  Sample failures:")
            for r in failures[:10]:
                lines.append(f"    {r.task_id}: {r.input_shape}→{r.output_shape} ({r.time_ms:.0f}ms)")

        return "\n".join(lines)


def run_arc_eval(max_tasks: int = 0, dataset: str = "training",
                 on_progress=None) -> ArcEvalReport:
    """
    Run ARC-AGI-1 evaluation.

    Args:
        max_tasks: 0 = all tasks, >0 = limit
        dataset: "training" (400 tasks) or "evaluation" (400 tasks)
        on_progress: callback(task_idx, total, task_id, correct)
    """
    import arckit
    import numpy as np
    from klomboagi.reasoning.arc_smart_solver import SmartARCSolver

    train_set, eval_set = arckit.load_data()
    tasks = train_set if dataset == "training" else eval_set

    if max_tasks > 0:
        tasks = list(tasks)[:max_tasks]

    solver = SmartARCSolver()
    report = ArcEvalReport()
    start = time.time()

    for idx, task in enumerate(tasks):
        task_start = time.time()
        result = ArcEvalResult(task_id=task.id)

        try:
            # Convert arckit format to solver format
            train_examples = []
            for ex in task.train:
                inp = np.array(ex[0]).tolist()
                out = np.array(ex[1]).tolist()
                train_examples.append({"input": inp, "output": out})

            # Get test input and expected output
            test_ex = task.test[0]
            test_input = np.array(test_ex[0]).tolist()
            test_expected = np.array(test_ex[1]).tolist()

            result.input_shape = (len(test_input), len(test_input[0]) if test_input else 0)
            result.output_shape = (len(test_expected), len(test_expected[0]) if test_expected else 0)
            result.expected = test_expected

            # Solve
            predicted = solver.solve(train_examples, test_input)
            result.predicted = predicted
            result.correct = (predicted == test_expected)

        except Exception as e:
            result.error = str(e)[:200]
            result.correct = False

        result.time_ms = (time.time() - task_start) * 1000
        report.results.append(result)
        report.total += 1
        if result.correct:
            report.correct += 1
        elif result.error:
            report.errors += 1
        else:
            report.failed += 1

        if on_progress:
            on_progress(idx + 1, len(tasks), task.id, result.correct)

    report.total_time_s = time.time() - start
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ARC-AGI-1 evaluation")
    parser.add_argument("--max", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--dataset", default="training", choices=["training", "evaluation"])
    args = parser.parse_args()

    def progress(idx, total, task_id, correct):
        mark = "✓" if correct else "✗"
        sys.stdout.write(f"\r  [{idx}/{total}] {task_id} {mark}   ")
        sys.stdout.flush()

    print(f"\n  ARC-AGI-1 Evaluation ({args.dataset} set)")
    print(f"  ═══════════════════════════════════════")

    report = run_arc_eval(max_tasks=args.max, dataset=args.dataset, on_progress=progress)
    print(f"\n\n{report.summary()}")
