from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from klomboagi.brain_core import load_failure_patterns, retrieve_memory, score_plan_candidates


class BrainCoreBridgeTests(unittest.TestCase):
    def test_retrieve_memory_ranks_best_overlap_first(self) -> None:
        hits = retrieve_memory(
            "database indexing strategy",
            [
                "frontend typography polish",
                "database indexing strategy for write-heavy workloads",
                "deployment rollback checklist",
            ],
            limit=2,
        )
        self.assertEqual(hits[0][0], "database indexing strategy for write-heavy workloads")

    def test_score_plan_candidates_penalizes_known_failure_patterns(self) -> None:
        candidates = [
            {
                "id": "write",
                "description": "Write file release.md",
                "action_kind": "write_file",
                "action_payload": {},
                "estimated_cost": 0.1,
            },
            {
                "id": "read",
                "description": "Read file release.md",
                "action_kind": "read_file",
                "action_payload": {},
                "estimated_cost": 0.2,
            },
        ]
        scored = score_plan_candidates(
            "Write file release.md",
            candidates,
            anti_patterns=["Write file release.md permission denied"],
        )
        self.assertEqual(scored[0]["candidate_id"], "read")

    def test_load_failure_patterns_reads_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_dir = Path(temp_dir) / "failure_cases"
            failure_dir.mkdir(parents=True)
            os.environ["KLOMBOAGI_FAILURE_CASE_DIR"] = str(failure_dir)
            (failure_dir / "pattern.json").write_text(
                json.dumps(
                    {
                        "description": "Failed during write_file",
                        "trigger": "write release notes",
                        "failed_action": "write_file",
                        "error_type": "permission denied",
                        "avoidance": "read before writing",
                    }
                ),
                encoding="utf-8",
            )
            patterns = load_failure_patterns()
            self.assertEqual(len(patterns), 1)
            self.assertIn("permission denied", patterns[0])
            os.environ.pop("KLOMBOAGI_FAILURE_CASE_DIR", None)


if __name__ == "__main__":
    unittest.main()
