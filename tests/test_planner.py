from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codeagi.reasoning.planner import Planner
from codeagi.storage.manager import StorageManager


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.planner = Planner(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def test_build_plan_with_tasks_creates_steps_for_each_task(self) -> None:
        mission = {"id": "m1", "description": "Build a parser"}
        tasks = [
            {"id": "t1", "description": "Write tokenizer", "status": "queued", "dependencies": []},
            {"id": "t2", "description": "Write AST builder", "status": "queued", "dependencies": ["t1"]},
        ]
        plan = self.planner.build_plan(mission, tasks)
        self.assertEqual(plan["mission_id"], "m1")
        self.assertEqual(len(plan["steps"]), 2)
        self.assertIn("Advance mission", plan["summary"])

    def test_build_plan_without_tasks_creates_bootstrap_step(self) -> None:
        mission = {"id": "m2", "description": "Explore codebase"}
        plan = self.planner.build_plan(mission, [])
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["status"], "ready")
        self.assertIn("Bootstrap", plan["summary"])

    def test_next_action_returns_ready_step(self) -> None:
        plan = {
            "steps": [
                {"status": "completed", "description": "Done", "task_id": "t1"},
                {"status": "ready", "description": "Next task", "task_id": "t2"},
            ]
        }
        action = self.planner.next_action(plan)
        self.assertEqual(action["type"], "execute_task")
        self.assertEqual(action["task_id"], "t2")

    def test_next_action_returns_mission_complete_when_all_done(self) -> None:
        plan = {
            "steps": [
                {"status": "completed", "description": "Done1", "task_id": "t1"},
                {"status": "completed", "description": "Done2", "task_id": "t2"},
            ]
        }
        action = self.planner.next_action(plan)
        self.assertEqual(action["type"], "mission_complete")

    def test_next_action_returns_resolve_blocker_when_blocked(self) -> None:
        plan = {
            "steps": [
                {"status": "completed", "description": "Done", "task_id": "t1"},
                {"status": "blocked", "description": "Stuck", "task_id": "t2", "blocked_reason": "Missing dep"},
            ]
        }
        action = self.planner.next_action(plan)
        self.assertEqual(action["type"], "resolve_blocker")

    def test_draft_task_search_heuristic(self) -> None:
        mission = {"id": "m3", "description": "Search the repo for TODO markers"}
        with patch("codeagi.reasoning.planner.llm_complete", return_value=None):
            result = self.planner.draft_task(mission)
        self.assertEqual(result["action_kind"], "search_files")
        self.assertIn("TODO", result["action_payload"]["pattern"])

    def test_draft_task_write_file_heuristic(self) -> None:
        mission = {"id": "m4", "description": "Write file output.txt with results"}
        with patch("codeagi.reasoning.planner.llm_complete", return_value=None):
            result = self.planner.draft_task(mission)
        self.assertEqual(result["action_kind"], "write_file")
        self.assertEqual(result["action_payload"]["path"], "output.txt")
