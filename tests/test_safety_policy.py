from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from codeagi.safety.policy import PolicyEngine
from codeagi.reasoning.verifier import Verifier
from codeagi.storage.manager import StorageManager


class ExtendedPolicyTests(unittest.TestCase):
    """Additional safety policy tests beyond the basic ones in test_policy.py."""

    def setUp(self) -> None:
        self.policy = PolicyEngine()

    def test_blocks_sudo(self) -> None:
        result = self.policy.check_command("sudo apt install foo")
        self.assertFalse(result["allowed"])
        self.assertIn("not permitted", result["reason"])

    def test_blocks_pipe_metacharacter(self) -> None:
        result = self.policy.check_command("cat file.txt | grep secret")
        self.assertFalse(result["allowed"])
        self.assertIn("metacharacter", result["reason"].lower())

    def test_blocks_redirect_metacharacter(self) -> None:
        result = self.policy.check_command("echo hello > /etc/passwd")
        self.assertFalse(result["allowed"])

    def test_blocks_unknown_command(self) -> None:
        result = self.policy.check_command("my_custom_binary --flag")
        self.assertFalse(result["allowed"])
        self.assertIn("outside the allowed safe set", result["reason"])

    def test_allows_grep_command(self) -> None:
        result = self.policy.check_command("grep -r TODO .")
        self.assertTrue(result["allowed"])

    def test_blocks_python3_with_flags(self) -> None:
        result = self.policy.check_command("python3 -c 'import os; os.system(\"rm -rf /\")'")
        self.assertFalse(result["allowed"])

    def test_allows_python3_with_script(self) -> None:
        result = self.policy.check_command("python3 script.py")
        self.assertTrue(result["allowed"])

    def test_blocks_empty_command(self) -> None:
        result = self.policy.check_command([])
        self.assertFalse(result["allowed"])

    def test_allows_list_command(self) -> None:
        result = self.policy.check_command(["ls", "-la"])
        self.assertTrue(result["allowed"])

    def test_blocks_git_command(self) -> None:
        result = self.policy.check_command("git push origin main")
        self.assertFalse(result["allowed"])


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.verifier = Verifier(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def test_verify_valid_plan(self) -> None:
        result = self.verifier.verify(
            mission={"id": "m1", "description": "Test mission"},
            tasks=[{"id": "t1", "description": "Do task", "status": "queued"}],
            plan={"id": "plan1", "steps": [{"id": "s1", "status": "ready", "description": "Go", "task_id": "t1"}]},
            next_action={"type": "execute_task", "description": "Execute", "task_id": "t1"},
            world_entities={"m1": {"type": "mission", "attributes": {}}},
            world_relations=[],
        )
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)

    def test_verify_flags_missing_mission_entity(self) -> None:
        result = self.verifier.verify(
            mission={"id": "m1", "description": "Test mission"},
            tasks=[{"id": "t1", "description": "Do task", "status": "queued"}],
            plan={"id": "plan1", "steps": [{"id": "s1", "status": "ready", "description": "Go", "task_id": "t1"}]},
            next_action={"type": "execute_task", "description": "Execute", "task_id": "t1"},
            world_entities={},
            world_relations=[],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("missing from world entities" in issue for issue in result["issues"]))

    def test_verify_flags_missing_dependency_target(self) -> None:
        result = self.verifier.verify(
            mission={"id": "m1", "description": "Test"},
            tasks=[{"id": "t1", "description": "Do task", "status": "queued"}],
            plan={"id": "plan1", "steps": []},
            next_action={"type": "decompose_mission", "description": "Decompose", "task_id": None},
            world_entities={"m1": {"type": "mission", "attributes": {}}},
            world_relations=[{"type": "depends_on", "from": "t1", "to": "t_missing"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("missing task" in issue.lower() for issue in result["issues"]))

    def test_verify_flags_mission_complete_with_outstanding_tasks(self) -> None:
        result = self.verifier.verify(
            mission={"id": "m1", "description": "Test"},
            tasks=[
                {"id": "t1", "description": "Done", "status": "completed"},
                {"id": "t2", "description": "Not done", "status": "queued"},
            ],
            plan={"id": "plan1", "steps": []},
            next_action={"type": "mission_complete", "description": "All done", "task_id": None},
            world_entities={"m1": {"type": "mission", "attributes": {}}},
            world_relations=[],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("incomplete tasks" in issue.lower() for issue in result["issues"]))

    def test_verify_warns_on_blocked_step_without_reason(self) -> None:
        result = self.verifier.verify(
            mission={"id": "m1", "description": "Test"},
            tasks=[{"id": "t1", "description": "Do task", "status": "queued"}],
            plan={
                "id": "plan1",
                "steps": [{"id": "s1", "status": "blocked", "description": "Stuck", "task_id": "t1"}],
            },
            next_action={"type": "resolve_blocker", "description": "Fix it", "task_id": "t1"},
            world_entities={"m1": {"type": "mission", "attributes": {}}},
            world_relations=[],
        )
        self.assertTrue(len(result["warnings"]) > 0)
        self.assertTrue(any("blocked_reason" in w for w in result["warnings"]))
