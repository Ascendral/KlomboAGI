from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from codeagi.core.loop import RuntimeLoop
from codeagi.core.mission import MissionManager
from codeagi.storage.manager import StorageManager


class CrossSessionResumptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.runtime_root = str(base / "runtime")
        self.long_term_root = str(base / "long_term")
        os.environ["CODEAGI_RUNTIME_ROOT"] = self.runtime_root
        os.environ["CODEAGI_LONG_TERM_ROOT"] = self.long_term_root
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def test_new_runtime_resumes_existing_mission(self) -> None:
        # --- Session 1: create mission and tasks, run one cycle ---
        storage1 = StorageManager.bootstrap()
        runtime1 = RuntimeLoop(storage1)
        missions1 = MissionManager(storage1)

        mission = missions1.create_mission("Resumable mission")
        first_task = missions1.create_task(mission.id, "Step one")
        second_task = missions1.create_task(mission.id, "Step two")

        payload1 = runtime1.run_cycle()
        self.assertEqual(payload1["status"], "active")
        self.assertEqual(payload1["mission"]["id"], mission.id)

        # Verify at least the first task completed
        tasks_after_s1 = missions1.list_tasks(mission.id)
        completed_ids_s1 = {t["id"] for t in tasks_after_s1 if t["status"] == "completed"}
        self.assertIn(first_task.id, completed_ids_s1)

        # Record how many tasks were completed in session 1
        completed_count_s1 = len(completed_ids_s1)

        # Tear down the first RuntimeLoop (simulate process exit)
        del runtime1

        # --- Session 2: new RuntimeLoop pointing to same storage ---
        storage2 = StorageManager.bootstrap()
        runtime2 = RuntimeLoop(storage2)
        missions2 = MissionManager(storage2)

        # Verify the new session can see the existing mission
        resumed_missions = missions2.list_missions()
        self.assertEqual(len(resumed_missions), 1)
        self.assertEqual(resumed_missions[0]["id"], mission.id)

        # Verify tasks persisted across sessions
        resumed_tasks = missions2.list_tasks(mission.id)
        self.assertEqual(len(resumed_tasks), 2)
        resumed_completed = {t["id"] for t in resumed_tasks if t["status"] == "completed"}
        self.assertEqual(resumed_completed, completed_ids_s1)

        # Run a second cycle — it should advance the mission
        payload2 = runtime2.run_cycle()

        # The second session should work with the same mission
        resumed_mission = missions2.get_mission(mission.id)
        tasks_after_s2 = missions2.list_tasks(mission.id)
        completed_count_s2 = sum(1 for t in tasks_after_s2 if t["status"] == "completed")

        # Session 2 should have advanced: either completed more tasks or finished the mission
        if completed_count_s1 < 2:
            self.assertGreater(completed_count_s2, completed_count_s1)
        else:
            # All tasks were already completed in session 1
            self.assertEqual(resumed_mission["status"], "completed")

    def test_resumed_session_preserves_working_memory(self) -> None:
        # Session 1
        storage1 = StorageManager.bootstrap()
        runtime1 = RuntimeLoop(storage1)
        missions1 = MissionManager(storage1)

        mission = missions1.create_mission("Memory persistence check")
        missions1.create_task(mission.id, "Record something")
        runtime1.run_cycle()

        wm_after_s1 = storage1.working_memory.load(default={})
        self.assertIn(mission.id, wm_after_s1)

        del runtime1

        # Session 2
        storage2 = StorageManager.bootstrap()
        wm_after_s2 = storage2.working_memory.load(default={})
        self.assertIn(mission.id, wm_after_s2)
        self.assertEqual(wm_after_s2[mission.id]["mission_id"], mission.id)
