from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klomboagi.reasoning.critic import Critic
from klomboagi.storage.manager import StorageManager


class CriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["KLOMBOAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.critic = Critic(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("KLOMBOAGI_RUNTIME_ROOT", None)
        os.environ.pop("KLOMBOAGI_LONG_TERM_ROOT", None)
        os.environ.pop("KLOMBOAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def _make_args(self, *, valid: bool = True, action_type: str = "execute_task", tasks: list | None = None):
        mission = {"id": "m1", "description": "Build a system"}
        if tasks is None:
            tasks = [{"id": "t1", "description": "Step one", "status": "queued"}]
        plan = {"id": "plan1"}
        proposed_action = {"type": action_type, "description": "Do something", "task_id": "t1"}
        verification = {"valid": valid, "issues": [] if valid else ["Problem found"], "warnings": []}
        return dict(mission=mission, tasks=tasks, plan=plan, proposed_action=proposed_action, verification=verification)

    @patch("klomboagi.reasoning.critic.llm_complete", return_value=None)
    def test_critique_approves_valid_execute_task(self, _mock_llm) -> None:
        result = self.critic.critique(**self._make_args(valid=True, action_type="execute_task"))
        self.assertTrue(result["approved"])
        self.assertIn("grounded in the current world model", result["notes"][0])

    @patch("klomboagi.reasoning.critic.llm_complete", return_value=None)
    def test_critique_rejects_invalid_verification(self, _mock_llm) -> None:
        result = self.critic.critique(**self._make_args(valid=False))
        self.assertFalse(result["approved"])
        self.assertEqual(result["final_action"]["type"], "replan")

    @patch("klomboagi.reasoning.critic.llm_complete", return_value=None)
    def test_critique_rejects_premature_mission_complete(self, _mock_llm) -> None:
        tasks = [
            {"id": "t1", "description": "Done", "status": "completed"},
            {"id": "t2", "description": "Not done", "status": "queued"},
        ]
        result = self.critic.critique(**self._make_args(valid=True, action_type="mission_complete", tasks=tasks))
        self.assertFalse(result["approved"])
        self.assertEqual(result["final_action"]["type"], "replan")
        self.assertTrue(any("unfinished" in note.lower() for note in result["notes"]))

    @patch("klomboagi.reasoning.critic.llm_complete", return_value=None)
    def test_critique_approves_decompose_mission(self, _mock_llm) -> None:
        result = self.critic.critique(**self._make_args(valid=True, action_type="decompose_mission"))
        self.assertTrue(result["approved"])
        self.assertTrue(any("decomposition" in note.lower() for note in result["notes"]))

    @patch("klomboagi.reasoning.critic.llm_complete", return_value='{"approved": false, "reason": "dangerous action"}')
    def test_critique_llm_override_blocks_action(self, _mock_llm) -> None:
        result = self.critic.critique(**self._make_args(valid=True, action_type="execute_task"))
        self.assertFalse(result["approved"])
        self.assertEqual(result["final_action"]["type"], "replan")
        self.assertTrue(any("LLM critic blocked" in note for note in result["notes"]))
