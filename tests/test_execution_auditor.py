from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from codeagi.evals.execution_auditor import ActionTelemetry, ExecutionAuditor
from codeagi.storage.manager import StorageManager


class ExecutionAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"
        self.storage = StorageManager.bootstrap()
        self.auditor = ExecutionAuditor(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_MAX_CYCLE_STEPS", None)
        self.temp_dir.cleanup()

    def _telemetry(self, **kwargs: object) -> ActionTelemetry:
        defaults = {
            "action_type": "execute_task",
            "tool_name": "search_files",
            "success": True,
            "duration_ms": 100.0,
        }
        defaults.update(kwargs)
        return ActionTelemetry(**defaults)

    def test_record_stores_telemetry(self) -> None:
        self.auditor.record(self._telemetry())
        self.assertEqual(len(self.auditor.get_history()), 1)

    def test_record_persists_to_store(self) -> None:
        self.auditor.record(self._telemetry())
        data = self.storage.execution_telemetry.load(default=[])
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["tool_name"], "search_files")

    def test_detect_repeated_failure(self) -> None:
        for _ in range(3):
            self.auditor.record(self._telemetry(success=False, error="not found"))
        anomalies = self.auditor.detect()
        types = [a.anomaly_type for a in anomalies]
        self.assertIn("repeated_failure", types)

    def test_no_false_repeated_failure(self) -> None:
        self.auditor.record(self._telemetry(success=False))
        self.auditor.record(self._telemetry(success=True))
        self.auditor.record(self._telemetry(success=False))
        anomalies = self.auditor.detect()
        types = [a.anomaly_type for a in anomalies]
        self.assertNotIn("repeated_failure", types)

    def test_detect_loop(self) -> None:
        for _ in range(5):
            self.auditor.record(self._telemetry(tool_name="grep", success=True))
        anomalies = self.auditor.detect()
        types = [a.anomaly_type for a in anomalies]
        self.assertIn("loop_detected", types)

    def test_detect_error_cascade(self) -> None:
        # 6 successes then 4 failures across 2 tools
        for i in range(6):
            self.auditor.record(self._telemetry(success=True))
        for i in range(4):
            tool = "grep" if i % 2 == 0 else "write_file"
            self.auditor.record(self._telemetry(tool_name=tool, success=False, error="fail"))
        anomalies = self.auditor.detect()
        types = [a.anomaly_type for a in anomalies]
        self.assertIn("error_cascade", types)

    def test_get_tool_stats(self) -> None:
        self.auditor.record(self._telemetry(tool_name="grep", duration_ms=100))
        self.auditor.record(self._telemetry(tool_name="grep", duration_ms=200, success=False))
        stats = self.auditor.get_tool_stats("grep")
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["failures"], 1)
        self.assertEqual(stats["avg_duration_ms"], 150)

    def test_summarize(self) -> None:
        self.auditor.record(self._telemetry(tool_name="grep"))
        self.auditor.record(self._telemetry(tool_name="read_file"))
        summary = self.auditor.summarize()
        self.assertIn("grep", summary)
        self.assertIn("read_file", summary)

    def test_summarize_empty(self) -> None:
        self.assertIn("No action telemetry", self.auditor.summarize())

    def test_reset(self) -> None:
        self.auditor.record(self._telemetry())
        self.auditor.reset()
        self.assertEqual(len(self.auditor.get_history()), 0)

    def test_anomaly_logged_to_event_log(self) -> None:
        for _ in range(3):
            self.auditor.record(self._telemetry(success=False, error="fail"))

        import json
        events = self.storage.paths.event_log_file.read_text().strip().split("\n")
        found = any(
            json.loads(line).get("event_type") == "auditor.anomaly"
            for line in events
        )
        self.assertTrue(found, "Expected auditor.anomaly event in log")

    def test_telemetry_to_dict(self) -> None:
        t = self._telemetry(mission_id="m1")
        d = t.to_dict()
        self.assertEqual(d["action_type"], "execute_task")
        self.assertEqual(d["mission_id"], "m1")


if __name__ == "__main__":
    unittest.main()
