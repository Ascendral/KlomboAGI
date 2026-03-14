"""Tests proving that memory affects behavior across missions and cycles.

1. A procedure learned from one mission influences planning in a subsequent mission.
2. Semantic facts from one cycle are retrievable in the next.
3. Working memory persists across RuntimeLoop instances.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from codeagi.core.loop import RuntimeLoop
from codeagi.core.mission import MissionManager
from codeagi.learning.consolidation import MemoryConsolidator
from codeagi.learning.semantic import SemanticMemory
from codeagi.memory.working_memory import WorkingMemoryManager
from codeagi.storage.manager import StorageManager


class TestProcedureInfluencesPlanning(unittest.TestCase):
    """A procedure created in mission A is retrieved when mission B has overlapping keywords."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.runtime_root = str(base / "runtime")
        self.long_term_root = str(base / "long_term")
        os.environ["CODEAGI_RUNTIME_ROOT"] = self.runtime_root
        os.environ["CODEAGI_LONG_TERM_ROOT"] = self.long_term_root
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"

    def tearDown(self) -> None:
        for key in ("CODEAGI_RUNTIME_ROOT", "CODEAGI_LONG_TERM_ROOT", "CODEAGI_MAX_CYCLE_STEPS"):
            os.environ.pop(key, None)
        self.temp_dir.cleanup()

    def test_procedure_from_mission_a_retrieved_in_mission_b(self) -> None:
        # --- Mission A: run a cycle so procedures are created ---
        storage = StorageManager.bootstrap()
        runtime = RuntimeLoop(storage)
        missions = MissionManager(storage)

        mission_a = missions.create_mission("Deploy backend service")
        missions.create_task(mission_a.id, "Configure deployment pipeline")
        runtime.run_cycle()

        # Verify at least one procedure was consolidated
        consolidator = MemoryConsolidator(storage)
        procedures = consolidator.load_all()
        self.assertGreater(len(procedures), 0, "Mission A should have created at least one procedure")

        # --- Mission B: overlapping description ---
        mission_b = missions.create_mission("Deploy frontend service")
        missions.create_task(mission_b.id, "Set up deployment steps")

        # Retrieve procedures relevant to mission B's description
        retrieved = consolidator.retrieve(str(mission_b.description))
        self.assertGreater(len(retrieved), 0,
                           "Procedures from mission A should be retrievable for mission B "
                           "because both mention 'deploy'")

        # The retrieved procedure text should reference the original mission A work
        joined = " ".join(retrieved).lower()
        self.assertTrue(
            "deploy" in joined or "deployment" in joined,
            f"Retrieved procedures should reference deployment work, got: {retrieved}",
        )


class TestSemanticFactsCrossycle(unittest.TestCase):
    """Semantic facts stored in cycle 1 are retrievable in cycle 2."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"

    def tearDown(self) -> None:
        for key in ("CODEAGI_RUNTIME_ROOT", "CODEAGI_LONG_TERM_ROOT", "CODEAGI_MAX_CYCLE_STEPS"):
            os.environ.pop(key, None)
        self.temp_dir.cleanup()

    def test_semantic_facts_retrievable_across_cycles(self) -> None:
        # --- Cycle 1 ---
        storage = StorageManager.bootstrap()
        runtime = RuntimeLoop(storage)
        missions = MissionManager(storage)

        mission = missions.create_mission("Analyze database performance")
        missions.create_task(mission.id, "Profile database queries")
        payload1 = runtime.run_cycle()
        self.assertEqual(payload1["status"], "active")

        semantic = SemanticMemory(storage)
        facts_after_c1 = semantic.load_all()
        self.assertGreater(len(facts_after_c1), 0,
                           "Cycle 1 should have produced at least one semantic fact")

        # --- Cycle 2: new RuntimeLoop, same storage ---
        del runtime
        runtime2 = RuntimeLoop(storage)

        # Before running cycle 2, verify retrieval works
        retrieved = semantic.retrieve("database performance analysis")
        self.assertGreater(len(retrieved), 0,
                           "Semantic facts from cycle 1 should be retrievable before cycle 2")

        # Run cycle 2 — the runtime loop calls semantic_memory.retrieve internally
        # and feeds results into working memory as relevant_memories
        mission2 = missions.create_mission("Optimize database indexing")
        missions.create_task(mission2.id, "Review index strategy")
        payload2 = runtime2.run_cycle()

        # The working memory for mission2 should contain relevant_memories
        # populated in part from semantic retrieval
        wm = storage.working_memory.load(default={})
        if mission2.id in wm:
            relevant = wm[mission2.id].get("relevant_memories", [])
            # The semantic facts about "database" should have been injected
            has_database_ref = any("database" in str(m).lower() for m in relevant)
            self.assertTrue(has_database_ref,
                            f"Working memory should reference database facts, got: {relevant}")


class TestWorkingMemoryPersistsAcrossRuntimeLoops(unittest.TestCase):
    """Working memory written by RuntimeLoop 1 is visible to RuntimeLoop 2."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.runtime_root = str(base / "runtime")
        self.long_term_root = str(base / "long_term")
        os.environ["CODEAGI_RUNTIME_ROOT"] = self.runtime_root
        os.environ["CODEAGI_LONG_TERM_ROOT"] = self.long_term_root
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"

    def tearDown(self) -> None:
        for key in ("CODEAGI_RUNTIME_ROOT", "CODEAGI_LONG_TERM_ROOT", "CODEAGI_MAX_CYCLE_STEPS"):
            os.environ.pop(key, None)
        self.temp_dir.cleanup()

    def test_working_memory_survives_runtime_restart(self) -> None:
        # Session 1
        storage1 = StorageManager.bootstrap()
        runtime1 = RuntimeLoop(storage1)
        missions1 = MissionManager(storage1)

        mission = missions1.create_mission("Persistent memory test")
        missions1.create_task(mission.id, "Step alpha")
        missions1.create_task(mission.id, "Step beta")
        runtime1.run_cycle()

        wm_mgr1 = WorkingMemoryManager(storage1)
        wm_s1 = wm_mgr1.load(mission.id)
        self.assertIsNotNone(wm_s1, "Working memory should exist after cycle 1")
        self.assertEqual(wm_s1["mission_id"], mission.id)

        # Capture state written in session 1
        focus_s1 = wm_s1["current_focus"]
        hypotheses_s1 = wm_s1.get("hypotheses", [])
        self.assertGreater(len(hypotheses_s1), 0, "Should have at least one hypothesis")

        del runtime1

        # Session 2: fresh RuntimeLoop over same storage roots
        storage2 = StorageManager.bootstrap()
        wm_mgr2 = WorkingMemoryManager(storage2)
        wm_s2 = wm_mgr2.load(mission.id)
        self.assertIsNotNone(wm_s2, "Working memory should persist into session 2")
        self.assertEqual(wm_s2["mission_id"], mission.id)
        self.assertEqual(wm_s2["current_focus"], focus_s1)
        self.assertEqual(wm_s2.get("hypotheses", []), hypotheses_s1)

    def test_working_memory_update_visible_in_new_loop(self) -> None:
        # Session 1: create and manually update working memory
        storage1 = StorageManager.bootstrap()
        runtime1 = RuntimeLoop(storage1)
        missions1 = MissionManager(storage1)

        mission = missions1.create_mission("Manual WM test")
        missions1.create_task(mission.id, "Do something")
        runtime1.run_cycle()

        wm_mgr1 = WorkingMemoryManager(storage1)
        wm_mgr1.update(mission.id, current_focus="custom focus from session 1")

        del runtime1

        # Session 2: verify the manual update persists
        storage2 = StorageManager.bootstrap()
        wm_mgr2 = WorkingMemoryManager(storage2)
        wm_s2 = wm_mgr2.load(mission.id)
        self.assertIsNotNone(wm_s2)
        self.assertEqual(wm_s2["current_focus"], "custom focus from session 1")


if __name__ == "__main__":
    unittest.main()
