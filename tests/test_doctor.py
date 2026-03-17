from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from klomboagi.interfaces.cli import main
from klomboagi.interfaces.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["KLOMBOAGI_WORKSPACE_ROOT"] = str(base / "workspace")

    def tearDown(self) -> None:
        for key in ["KLOMBOAGI_RUNTIME_ROOT", "KLOMBOAGI_LONG_TERM_ROOT", "KLOMBOAGI_WORKSPACE_ROOT"]:
            os.environ.pop(key, None)
        self.temp_dir.cleanup()

    def test_run_doctor_reports_writable_paths(self) -> None:
        report = run_doctor()
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["checks"]), 3)

    def test_doctor_cli_command(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["doctor"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
