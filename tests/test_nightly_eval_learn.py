from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.nightly_eval_learn import run_nightly


class _ExactMatchAgent:
    def execute(self, task: dict[str, object]) -> dict[str, object]:
        return {
            "output": task.get("expected"),
            "interventions": 0,
            "steps": 2,
            "recovered": False,
            "memory_retrievals": 1,
            "memory_useful": 1,
            "trace": [{"action": "answer"}],
        }


class NightlyEvalLearnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_processes_new_trajectories_once(self) -> None:
        trajectory_dir = self.root / "datasets" / "trajectories"
        trajectory_dir.mkdir(parents=True)
        (trajectory_dir / "success.json").write_text(
            json.dumps(
                {
                    "task_id": "task_success",
                    "domain": "coding",
                    "description": "Fix parser bug",
                    "success": True,
                    "steps": [
                        {"action": "search_files", "phase": "plan", "outcome": "success"},
                        {"action": "apply_patch", "phase": "act", "outcome": "success"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (trajectory_dir / "failure.json").write_text(
            json.dumps(
                {
                    "task_id": "task_failure",
                    "domain": "ops",
                    "description": "Deploy service",
                    "success": False,
                    "failure_point": 0,
                    "steps": [
                        {
                            "action": "run_command",
                            "phase": "act",
                            "decision_reason": "restart deployment",
                            "error": "permission denied",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        first = run_nightly(
            trajectory_dir=trajectory_dir,
            skill_dir=self.root / "datasets" / "skills",
            failure_dir=self.root / "datasets" / "failure_cases",
            memory_dir=self.root / "datasets" / "memory_events",
            nightly_dir=self.root / "datasets" / "nightly",
            skip_evals=True,
        )
        second = run_nightly(
            trajectory_dir=trajectory_dir,
            skill_dir=self.root / "datasets" / "skills",
            failure_dir=self.root / "datasets" / "failure_cases",
            memory_dir=self.root / "datasets" / "memory_events",
            nightly_dir=self.root / "datasets" / "nightly",
            skip_evals=True,
        )

        self.assertEqual(first.processed_trajectories, 2)
        self.assertEqual(first.extracted_skills, 1)
        self.assertEqual(first.extracted_anti_patterns, 1)
        self.assertEqual(second.processed_trajectories, 0)
        self.assertEqual(second.skipped_trajectories, 2)

    def test_runs_hidden_eval_suite_with_pluggable_agent(self) -> None:
        eval_dir = self.root / "evals" / "hidden" / "research" / "tiny_task"
        eval_dir.mkdir(parents=True)
        (eval_dir / "task.json").write_text(
            json.dumps(
                {
                    "id": "tiny_task",
                    "domain": "research",
                    "description": "Answer with the provided payload",
                    "expected": {"answer": 7},
                    "scorer": "exact_match",
                }
            ),
            encoding="utf-8",
        )

        report = run_nightly(
            trajectory_dir=self.root / "datasets" / "trajectories",
            skill_dir=self.root / "datasets" / "skills",
            failure_dir=self.root / "datasets" / "failure_cases",
            memory_dir=self.root / "datasets" / "memory_events",
            eval_dir=self.root / "evals" / "hidden",
            eval_report_dir=self.root / "evals" / "reports",
            nightly_dir=self.root / "datasets" / "nightly",
            agent=_ExactMatchAgent(),
        )

        self.assertIsNotNone(report.eval_summary)
        self.assertEqual(report.eval_summary["tasks_attempted"], 1)
        self.assertEqual(report.eval_summary["tasks_succeeded"], 1)

