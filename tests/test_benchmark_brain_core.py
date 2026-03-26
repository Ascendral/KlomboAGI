from __future__ import annotations

import unittest

from scripts.benchmark_brain_core import benchmark


class BenchmarkBrainCoreTests(unittest.TestCase):
    def test_benchmark_returns_named_results(self) -> None:
        result = benchmark(iterations=5)
        self.assertIn("native_available", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["name"], "retrieve_memory")
        self.assertEqual(result["results"][1]["name"], "score_plan_candidates")


if __name__ == "__main__":
    unittest.main()
