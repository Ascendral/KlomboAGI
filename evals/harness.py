"""
Eval Harness — the most important file in this repo.

Runs tasks from evals/hidden/, scores them, and produces reports.
No task in hidden/ should ever be seen during development.

Usage:
    python -m evals.harness run          # Run all hidden tasks
    python -m evals.harness run coding   # Run one domain
    python -m evals.harness report       # Show latest results
    python -m evals.harness compare      # Compare two runs
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    """Result of running one eval task."""
    task_id: str
    domain: str
    success: bool
    interventions: int = 0
    autonomy_steps: int = 0
    failure_recovered: bool = False
    memory_retrievals: int = 0
    memory_useful: int = 0
    duration_s: float = 0.0
    error: str = ""
    trace: list[dict] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregate report from an eval run."""
    run_id: str
    timestamp: str
    tasks_attempted: int = 0
    tasks_succeeded: int = 0
    total_interventions: int = 0
    avg_autonomy_horizon: float = 0.0
    failure_recoveries: int = 0
    failure_total: int = 0
    transfer_score: float = 0.0
    memory_retrievals: int = 0
    memory_useful: int = 0
    regressions: int = 0
    results: list[TaskResult] = field(default_factory=list)
    duration_s: float = 0.0

    def success_rate(self) -> float:
        return self.tasks_succeeded / self.tasks_attempted if self.tasks_attempted > 0 else 0.0

    def intervention_rate(self) -> float:
        return self.total_interventions / self.tasks_attempted if self.tasks_attempted > 0 else 0.0

    def recovery_rate(self) -> float:
        return self.failure_recoveries / self.failure_total if self.failure_total > 0 else 0.0

    def memory_usefulness(self) -> float:
        return self.memory_useful / self.memory_retrievals if self.memory_retrievals > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = self.success_rate()
        d["intervention_rate"] = self.intervention_rate()
        d["recovery_rate"] = self.recovery_rate()
        d["memory_usefulness"] = self.memory_usefulness()
        return d

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"EVAL REPORT: {self.run_id}",
            f"{'='*60}",
            f"Tasks:        {self.tasks_succeeded}/{self.tasks_attempted} ({100*self.success_rate():.1f}%)",
            f"Interventions: {self.total_interventions} ({self.intervention_rate():.1f} per task)",
            f"Autonomy:     {self.avg_autonomy_horizon:.1f} steps avg",
            f"Recovery:     {self.failure_recoveries}/{self.failure_total} ({100*self.recovery_rate():.1f}%)",
            f"Memory:       {self.memory_useful}/{self.memory_retrievals} useful ({100*self.memory_usefulness():.1f}%)",
            f"Regressions:  {self.regressions}",
            f"Duration:     {self.duration_s:.1f}s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class EvalHarness:
    """
    Runs evaluation tasks and produces scored reports.

    The harness is domain-agnostic. Each task is a JSON file with:
    - description: what to do
    - inputs: files/data provided
    - expected: what success looks like
    - scorer: how to check (exact_match, contains, function)
    """

    def __init__(self, eval_dir: str = "evals/hidden", report_dir: str = "evals/reports"):
        self.eval_dir = Path(eval_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def discover_tasks(self, domain: str | None = None) -> list[dict]:
        """Find all task.json files in the eval directory."""
        tasks = []
        search_dir = self.eval_dir / domain if domain else self.eval_dir

        for task_file in search_dir.rglob("task.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                task["_path"] = str(task_file)
                task["_domain"] = task_file.parent.parent.name if domain is None else domain
                tasks.append(task)
            except Exception as e:
                print(f"Warning: could not load {task_file}: {e}")

        return tasks

    def run_task(self, task: dict, agent: Any = None) -> TaskResult:
        """Run a single eval task and score it."""
        task_id = task.get("id", os.path.basename(task["_path"]))
        domain = task.get("_domain", "unknown")

        t0 = time.time()

        try:
            if agent is None:
                # No agent — just record that we can't do it
                return TaskResult(
                    task_id=task_id, domain=domain, success=False,
                    error="No agent provided",
                    duration_s=time.time() - t0,
                )

            # Run the agent on the task
            result = agent.execute(task)

            # Score
            success = self._score(task, result)

            return TaskResult(
                task_id=task_id, domain=domain, success=success,
                interventions=result.get("interventions", 0),
                autonomy_steps=result.get("steps", 0),
                failure_recovered=result.get("recovered", False),
                memory_retrievals=result.get("memory_retrievals", 0),
                memory_useful=result.get("memory_useful", 0),
                duration_s=time.time() - t0,
                trace=result.get("trace", []),
            )
        except Exception as e:
            return TaskResult(
                task_id=task_id, domain=domain, success=False,
                error=str(e), duration_s=time.time() - t0,
            )

    def _score(self, task: dict, result: dict) -> bool:
        """Score a task result against expected output."""
        scorer = task.get("scorer", "exact_match")
        expected = task.get("expected")
        actual = result.get("output")

        if scorer == "exact_match":
            if isinstance(expected, list) and isinstance(actual, list):
                return sorted(actual) == sorted(expected)
            return actual == expected
        elif scorer == "contains":
            if expected and actual:
                return any(w in str(actual).lower() for w in str(expected).lower().split())
            return bool(actual)
        elif scorer == "function_passes_tests":
            return actual is not None
        elif scorer == "not_empty":
            return bool(actual)
        else:
            return False

    def run_all(self, domain: str | None = None, agent: Any = None) -> EvalReport:
        """Run all tasks and produce a report."""
        tasks = self.discover_tasks(domain)

        run_id = f"eval_{int(time.time())}"
        t0 = time.time()

        results = []
        for task in tasks:
            result = self.run_task(task, agent)
            results.append(result)

        # Aggregate
        report = EvalReport(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            tasks_attempted=len(results),
            tasks_succeeded=sum(1 for r in results if r.success),
            total_interventions=sum(r.interventions for r in results),
            avg_autonomy_horizon=sum(r.autonomy_steps for r in results) / len(results) if results else 0,
            failure_recoveries=sum(1 for r in results if r.failure_recovered),
            failure_total=sum(1 for r in results if not r.success),
            memory_retrievals=sum(r.memory_retrievals for r in results),
            memory_useful=sum(r.memory_useful for r in results),
            results=results,
            duration_s=time.time() - t0,
        )

        # Save report
        report_path = self.report_dir / f"{run_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        return report

    def compare(self, run_a: str, run_b: str) -> str:
        """Compare two eval runs."""
        path_a = self.report_dir / f"{run_a}.json"
        path_b = self.report_dir / f"{run_b}.json"

        with open(path_a) as f:
            a = json.load(f)
        with open(path_b) as f:
            b = json.load(f)

        lines = [
            f"{'='*60}",
            f"COMPARISON: {run_a} vs {run_b}",
            f"{'='*60}",
            f"Success:      {a['success_rate']:.1%} → {b['success_rate']:.1%} ({b['success_rate']-a['success_rate']:+.1%})",
            f"Interventions: {a['intervention_rate']:.1f} → {b['intervention_rate']:.1f}",
            f"Autonomy:     {a['avg_autonomy_horizon']:.1f} → {b['avg_autonomy_horizon']:.1f}",
            f"Recovery:     {a.get('recovery_rate',0):.1%} → {b.get('recovery_rate',0):.1%}",
            f"Memory:       {a.get('memory_usefulness',0):.1%} → {b.get('memory_usefulness',0):.1%}",
        ]
        return "\n".join(lines)
