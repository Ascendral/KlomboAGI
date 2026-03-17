from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from klomboagi.evals.repo_runner import RepoEvalRunner
from klomboagi.storage.manager import StorageManager


class RepoEvalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["KLOMBOAGI_WORKSPACE_ROOT"] = str(base / "workspace")
        os.environ["KLOMBOAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.runner = RepoEvalRunner(self.storage)
        self.workspace_root = base / "workspace"

    def tearDown(self) -> None:
        for key in [
            "KLOMBOAGI_RUNTIME_ROOT",
            "KLOMBOAGI_LONG_TERM_ROOT",
            "KLOMBOAGI_WORKSPACE_ROOT",
            "KLOMBOAGI_MAX_CYCLE_STEPS",
        ]:
            os.environ.pop(key, None)
        self.temp_dir.cleanup()

    def test_repo_search_fixture_executes(self) -> None:
        result = self.runner.run_fixture("repo_search")
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["action_outcome"]["status"], "completed")
        self.assertEqual(result["action_outcome"]["details"]["matches"], ["repo/app.py"])

    def test_repo_patch_fixture_updates_workspace_file(self) -> None:
        result = self.runner.run_fixture("repo_patch")
        target = self.workspace_root / "repo" / "main.txt"
        self.assertEqual(result["action_outcome"]["status"], "completed")
        self.assertEqual(target.read_text(), "hello new world\n")
