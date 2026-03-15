from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from codeagi.reasoning.goal_refiner import GoalRefiner, SubtaskNode
from codeagi.storage.manager import StorageManager


class GoalRefinerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.refiner = GoalRefiner(self.storage, max_depth=2)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def _mission(self, desc: str = "Fix the login error", mid: str = "m1") -> dict[str, object]:
        return {"id": mid, "description": desc, "status": "active"}

    def test_decompose_bug_fix(self) -> None:
        nodes = self.refiner.decompose(self._mission("Fix the login error in auth module"))
        self.assertGreater(len(nodes), 1, "Bug fix should decompose into subtasks")

        root = next(n for n in nodes if n.depth == 0)
        self.assertEqual(root.status, "in_progress")

    def test_decompose_feature(self) -> None:
        nodes = self.refiner.decompose(self._mission("Add dark mode to the settings page"))
        self.assertGreater(len(nodes), 1)

    def test_decompose_refactor(self) -> None:
        nodes = self.refiner.decompose(self._mission("Refactor the database module"))
        self.assertGreater(len(nodes), 1)

    def test_decompose_research(self) -> None:
        nodes = self.refiner.decompose(self._mission("Investigate memory leak in worker"))
        self.assertGreater(len(nodes), 1)

    def test_decompose_test(self) -> None:
        nodes = self.refiner.decompose(self._mission("Write tests for the payment module"))
        self.assertGreater(len(nodes), 1)

    def test_generic_mission_stays_single(self) -> None:
        nodes = self.refiner.decompose(self._mission("Do something"))
        # No heuristic matches and LLM not available → single root node
        self.assertEqual(len(nodes), 1)
        root = nodes[0]
        self.assertEqual(root.status, "ready")

    def test_ready_nodes_are_leaves(self) -> None:
        nodes = self.refiner.decompose(self._mission("Fix the crash in parser"))
        ready = self.refiner.get_ready_nodes(nodes)
        self.assertGreater(len(ready), 0)
        for n in ready:
            self.assertEqual(n.subtasks, [])
            self.assertEqual(n.status, "ready")

    def test_complete_node_propagates(self) -> None:
        nodes = self.refiner.decompose(self._mission("Do something"))
        self.assertEqual(len(nodes), 1)

        self.refiner.complete_node(nodes, nodes[0].id, "Done")
        self.assertEqual(nodes[0].status, "completed")
        self.assertTrue(self.refiner.is_finished(nodes))

    def test_fail_node_skips_dependents(self) -> None:
        nodes = self.refiner.decompose(self._mission("Fix the login bug"))
        ready = self.refiner.get_ready_nodes(nodes)
        self.assertGreater(len(ready), 0)

        self.refiner.fail_node(nodes, ready[0].id, "Not found")
        skipped = [n for n in nodes if n.status == "skipped"]
        self.assertGreater(len(skipped), 0, "Dependents should be skipped")

    def test_fail_all_children_fails_root(self) -> None:
        nodes = self.refiner.decompose(self._mission("Fix the crash bug"))
        root = next(n for n in nodes if n.depth == 0)

        # Fail first ready, then complete/skip remaining
        node_map = {n.id: n for n in nodes}
        for child_id in root.subtasks:
            child = node_map[child_id]
            if child.status in ("ready", "pending"):
                self.refiner.fail_node(nodes, child_id, "fail")

        # All children should be terminal now, root should be failed
        self.assertTrue(self.refiner.is_finished(nodes))
        self.assertEqual(root.status, "failed")

    def test_summarize_output(self) -> None:
        nodes = self.refiner.decompose(self._mission("Fix the login error"))
        summary = self.refiner.summarize(nodes)
        self.assertIn("Fix the login error", summary)
        self.assertIn("in_progress", summary)

    def test_decomposition_strategy_logged(self) -> None:
        self.refiner.decompose(self._mission("Fix the crash", mid="m_log"))
        strategies = self.storage.decomposition_strategies.load(default=[])
        self.assertGreater(len(strategies), 0)
        self.assertEqual(strategies[0]["mission_id"], "m_log")

    def test_event_log_records_decomposition(self) -> None:
        self.refiner.decompose(self._mission("Fix the crash"))
        events = self.storage.paths.event_log_file.read_text().strip().split("\n")
        import json
        found = any(
            json.loads(line).get("event_type") == "goal.decomposed"
            for line in events
        )
        self.assertTrue(found, "Expected goal.decomposed event in log")

    def test_max_depth_respected(self) -> None:
        refiner = GoalRefiner(self.storage, max_depth=1)
        nodes = refiner.decompose(self._mission("Fix the crash in parser"))
        for node in nodes:
            self.assertLessEqual(node.depth, 1)

    def test_subtask_node_to_dict(self) -> None:
        node = SubtaskNode(
            id="test_1",
            description="Test node",
            parent_id=None,
            depth=0,
            created_at="2026-03-15T10:00:00Z",
        )
        d = node.to_dict()
        self.assertEqual(d["id"], "test_1")
        self.assertEqual(d["status"], "pending")
        self.assertIsInstance(d["dependencies"], list)

    def test_action_kinds_assigned(self) -> None:
        nodes = self.refiner.decompose(self._mission("Debug the authentication failure"))
        action_kinds = [n.action_kind for n in nodes if n.action_kind]
        self.assertGreater(len(action_kinds), 0, "Some nodes should have action_kind")
        valid_kinds = {"search_files", "read_file", "write_file", "apply_patch", "run_command"}
        for kind in action_kinds:
            self.assertIn(kind, valid_kinds)


if __name__ == "__main__":
    unittest.main()
