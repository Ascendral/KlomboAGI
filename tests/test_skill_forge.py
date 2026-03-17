from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from klomboagi.learning.skill_forge import SkillForge
from klomboagi.storage.manager import StorageManager


class SkillForgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["KLOMBOAGI_MAX_CYCLE_STEPS"] = "3"
        # Point CODEBOT_HOME to temp dir so skills go to a test location
        self.codebot_home = str(base / "codebot_home")
        os.environ["CODEBOT_HOME"] = self.codebot_home
        self.storage = StorageManager.bootstrap()
        self.forge = SkillForge(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("KLOMBOAGI_RUNTIME_ROOT", None)
        os.environ.pop("KLOMBOAGI_LONG_TERM_ROOT", None)
        os.environ.pop("KLOMBOAGI_MAX_CYCLE_STEPS", None)
        os.environ.pop("CODEBOT_HOME", None)
        self.temp_dir.cleanup()

    def _seed_procedure(
        self,
        *,
        proc_id: str = "procedure_001",
        title: str = "Search files procedure for debugging",
        trigger: str = "Debug the login issue",
        confidence: float = 0.9,
        use_count: int = 5,
    ) -> dict[str, object]:
        procedure = {
            "id": proc_id,
            "mission_id": "mission_001",
            "title": title,
            "trigger": trigger,
            "steps": [
                "Review mission intent: Debug the login issue",
                "Execute action: search_files for error patterns",
                "Record outcome: completed search",
            ],
            "confidence": confidence,
            "use_count": use_count,
            "created_at": "2026-03-15T10:00:00Z",
            "updated_at": "2026-03-15T10:00:00Z",
        }
        procedures = self.storage.procedures.load(default=[])
        procedures.append(procedure)
        self.storage.procedures.save(procedures)
        return procedure

    def test_scan_promotes_nothing_when_no_procedures(self) -> None:
        promoted = self.forge.scan_and_promote()
        self.assertEqual(promoted, [])

    def test_scan_skips_low_confidence_procedures(self) -> None:
        self._seed_procedure(confidence=0.3, use_count=10)
        promoted = self.forge.scan_and_promote()
        self.assertEqual(promoted, [])

    def test_scan_skips_low_use_count_procedures(self) -> None:
        self._seed_procedure(confidence=0.95, use_count=1)
        promoted = self.forge.scan_and_promote()
        self.assertEqual(promoted, [])

    def test_scan_promotes_eligible_procedure(self) -> None:
        self._seed_procedure(confidence=0.9, use_count=5)
        promoted = self.forge.scan_and_promote()
        self.assertEqual(len(promoted), 1)

        skill = promoted[0]
        self.assertEqual(skill["author"], "klomboagi")
        self.assertEqual(skill["origin"], "promoted")
        self.assertEqual(skill["confidence"], 0.9)
        self.assertEqual(skill["use_count"], 5)
        self.assertIn("klomboagi_", skill["name"])

    def test_promoted_skill_written_to_shared_store(self) -> None:
        self._seed_procedure()
        promoted = self.forge.scan_and_promote()
        self.assertEqual(len(promoted), 1)

        skill_name = promoted[0]["name"]
        skill_path = Path(self.codebot_home) / "skills" / f"{skill_name}.json"
        self.assertTrue(skill_path.exists())

        data = json.loads(skill_path.read_text())
        self.assertEqual(data["author"], "klomboagi")
        self.assertIsInstance(data["steps"], list)
        self.assertGreater(len(data["steps"]), 0)

    def test_promoted_skill_has_valid_steps(self) -> None:
        self._seed_procedure()
        promoted = self.forge.scan_and_promote()
        skill = promoted[0]

        for step in skill["steps"]:
            self.assertIn("tool", step)
            self.assertIn("args", step)
            self.assertIsInstance(step["tool"], str)
            self.assertIsInstance(step["args"], dict)

    def test_duplicate_promotion_reinforces_instead(self) -> None:
        self._seed_procedure()
        # First promotion
        promoted1 = self.forge.scan_and_promote()
        self.assertEqual(len(promoted1), 1)

        skill_name = promoted1[0]["name"]
        skill_path = Path(self.codebot_home) / "skills" / f"{skill_name}.json"
        data_before = json.loads(skill_path.read_text())

        # Second scan should reinforce, not create duplicate
        promoted2 = self.forge.scan_and_promote()
        self.assertEqual(len(promoted2), 0)  # No new promotions

        data_after = json.loads(skill_path.read_text())
        self.assertGreater(data_after["use_count"], data_before["use_count"])
        self.assertGreater(data_after["confidence"], data_before["confidence"])

    def test_promote_one_force_promotes(self) -> None:
        # Low confidence, low use_count — would not auto-promote
        proc = self._seed_procedure(confidence=0.3, use_count=1)
        promoted = self.forge.promote_one(str(proc["id"]))
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted["author"], "klomboagi")

    def test_promote_one_nonexistent_returns_none(self) -> None:
        result = self.forge.promote_one("nonexistent_id")
        self.assertIsNone(result)

    def test_list_shared_skills(self) -> None:
        self._seed_procedure()
        self.forge.scan_and_promote()
        skills = self.forge.list_shared_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["author"], "klomboagi")

    def test_skill_name_generation(self) -> None:
        proc = self._seed_procedure(title="Search Files Procedure for Debugging")
        name = self.forge._procedure_to_skill_name(proc)
        self.assertTrue(name.startswith("klomboagi_"))
        self.assertRegex(name, r"^[a-z0-9_]+$")
        self.assertLessEqual(len(name), 64)

    def test_skill_name_sanitization(self) -> None:
        proc = self._seed_procedure(title="Bad!@#$ Name---With  Spaces")
        name = self.forge._procedure_to_skill_name(proc)
        self.assertRegex(name, r"^[a-z0-9_]+$")

    def test_action_to_tool_mapping(self) -> None:
        proc = self._seed_procedure()
        proc["steps"] = [
            "Review mission intent: Fix bug",
            "Execute action: search_files for patterns",
            "Record outcome: completed",
        ]
        self.storage.procedures.save([proc])

        promoted = self.forge.scan_and_promote()
        self.assertEqual(len(promoted), 1)

        steps = promoted[0]["steps"]
        tool_names = [s["tool"] for s in steps]
        # "search_files" maps to "grep"
        self.assertIn("grep", tool_names)

    def test_event_log_records_promotion(self) -> None:
        self._seed_procedure()
        self.forge.scan_and_promote()

        # Read event log
        events_file = self.storage.paths.event_log_file
        events = events_file.read_text().strip().split("\n")
        found = False
        for line in events:
            event = json.loads(line)
            if event.get("event_type") == "skillforge.promoted":
                found = True
                break
        self.assertTrue(found, "Expected skillforge.promoted event in log")

    def test_multiple_procedures_some_eligible(self) -> None:
        self._seed_procedure(proc_id="p1", confidence=0.9, use_count=5, title="Good procedure")
        self._seed_procedure(proc_id="p2", confidence=0.3, use_count=1, title="Weak procedure")
        self._seed_procedure(proc_id="p3", confidence=0.85, use_count=4, title="Another good one")

        promoted = self.forge.scan_and_promote()
        self.assertEqual(len(promoted), 2)  # p1 and p3 promoted, p2 skipped


class SkillForgeInteropTests(unittest.TestCase):
    """Test that KlomboAGI-written skills are compatible with CodeBot's format."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["KLOMBOAGI_MAX_CYCLE_STEPS"] = "3"
        self.codebot_home = str(base / "codebot_home")
        os.environ["CODEBOT_HOME"] = self.codebot_home
        self.storage = StorageManager.bootstrap()
        self.forge = SkillForge(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("KLOMBOAGI_RUNTIME_ROOT", None)
        os.environ.pop("KLOMBOAGI_LONG_TERM_ROOT", None)
        os.environ.pop("KLOMBOAGI_MAX_CYCLE_STEPS", None)
        os.environ.pop("CODEBOT_HOME", None)
        self.temp_dir.cleanup()

    def test_promoted_skill_matches_codebot_schema(self) -> None:
        """Promoted skills must have all fields CodeBot's loadSkills() expects."""
        proc = {
            "id": "proc_interop",
            "mission_id": "mission_interop",
            "title": "Interop test procedure",
            "trigger": "test interop",
            "steps": [
                "Review mission intent: test interop",
                "Execute action: read_file config.json",
            ],
            "confidence": 0.95,
            "use_count": 10,
            "created_at": "2026-03-15T10:00:00Z",
            "updated_at": "2026-03-15T10:00:00Z",
        }
        self.storage.procedures.save([proc])

        promoted = self.forge.scan_and_promote()
        self.assertEqual(len(promoted), 1)
        skill = promoted[0]

        # Required by CodeBot's loadSkills()
        self.assertIn("name", skill)
        self.assertIn("steps", skill)
        self.assertIsInstance(skill["name"], str)
        self.assertIsInstance(skill["steps"], list)
        self.assertGreater(len(skill["steps"]), 0)

        # Each step must have tool and args
        for step in skill["steps"]:
            self.assertIn("tool", step)
            self.assertIn("args", step)

        # Shared metadata
        self.assertEqual(skill["author"], "klomboagi")
        self.assertIn("confidence", skill)
        self.assertIn("use_count", skill)
        self.assertIn("origin", skill)
        self.assertIn("created_at", skill)
        self.assertIn("updated_at", skill)

    def test_codebot_authored_skill_readable_by_klomboagi(self) -> None:
        """Skills written by CodeBot should be readable by KlomboAGI."""
        skills_dir = Path(self.codebot_home) / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        codebot_skill = {
            "name": "lint-and-fix",
            "description": "Run linter and auto-fix issues",
            "steps": [
                {"tool": "execute", "args": {"command": "npm run lint -- --fix"}},
                {"tool": "git", "args": {"subcommand": "diff"}},
            ],
            "author": "codebot",
            "confidence": 0.7,
            "use_count": 3,
            "origin": "forged",
            "created_at": "2026-03-15T10:00:00Z",
            "updated_at": "2026-03-15T10:00:00Z",
        }
        (skills_dir / "lint-and-fix.json").write_text(json.dumps(codebot_skill, indent=2))

        skills = self.forge.list_shared_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["name"], "lint-and-fix")
        self.assertEqual(skills[0]["author"], "codebot")


if __name__ == "__main__":
    unittest.main()
