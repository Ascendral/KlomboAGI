"""
ARC Solver Unit Tests — validates the 106-strategy solver pipeline.

Tests against real ARC-AGI-1 tasks that the solver can solve.
Also tests edge cases and ensures no regressions.
"""

import pytest
import arckit
import numpy as np
from klomboagi.reasoning.arc_solver import ARCSolver, ARCSolverV18
from klomboagi.reasoning.arc_smart_solver import SmartARCSolver
from klomboagi.reasoning.arc_objects import ObjectDetector


# Load dataset once for all tests
_train_set, _eval_set = arckit.load_data()
_TASK_MAP = {task.id: task for task in _train_set}


def _get_task(task_id: str):
    """Get a task by ID, return (train_examples, test_input, test_expected)."""
    task = _TASK_MAP[task_id]
    train_ex = [{"input": np.array(ex[0]).tolist(), "output": np.array(ex[1]).tolist()}
                for ex in task.train]
    test_ex = task.test[0]
    test_input = np.array(test_ex[0]).tolist()
    test_expected = np.array(test_ex[1]).tolist()
    return train_ex, test_input, test_expected


# ── Known-Passing Tasks ──
# These are tasks the solver MUST continue to solve correctly.

KNOWN_PASSING = [
    "00d62c1b", "0b148d64", "1cf80156", "1e0a9b12", "1f85a75f",
    "22eb0ac0", "23b5c85d", "25ff71a9", "2dee498d", "332efdb3",
]


class TestSmartSolver:
    """Tests for the full SmartARCSolver (V18 + learned strategy ordering)."""

    def setup_method(self):
        self.solver = SmartARCSolver()

    @pytest.mark.parametrize("task_id", KNOWN_PASSING)
    def test_known_passing(self, task_id):
        """Each known-passing task must continue to pass (no regressions)."""
        train_ex, test_input, test_expected = _get_task(task_id)
        predicted = self.solver.solve(train_ex, test_input)
        assert predicted == test_expected, f"Regression: task {task_id} no longer solved"

    def test_returns_none_for_unsolvable(self):
        """Solver should return None for tasks it can't solve, not crash."""
        # Create an impossible task — random grids
        train = [
            {"input": [[1, 2], [3, 4]], "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]},
            {"input": [[5, 6], [7, 8]], "output": [[1, 1, 1], [2, 2, 2], [3, 3, 3]]},
        ]
        test_input = [[0, 0], [0, 0]]
        result = self.solver.solve(train, test_input)
        assert result is None or isinstance(result, list)

    def test_identity_transform(self):
        """If all training outputs equal inputs, predict identity."""
        train = [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
        ]
        test_input = [[9, 0], [1, 2]]
        result = self.solver.solve(train, test_input)
        assert result == [[9, 0], [1, 2]]

    def test_color_replacement(self):
        """Simple color swap: 1→2, 2→1."""
        train = [
            {"input": [[1, 1], [2, 2]], "output": [[2, 2], [1, 1]]},
            {"input": [[1, 2], [1, 2]], "output": [[2, 1], [2, 1]]},
        ]
        test_input = [[1, 1, 1], [2, 2, 2]]
        result = self.solver.solve(train, test_input)
        # Should produce some result (may find different valid transform)
        assert result is not None, "Should find some transformation"

    def test_empty_input_handling(self):
        """Solver should handle empty or minimal inputs gracefully."""
        train = [{"input": [[0]], "output": [[0]]}]
        test_input = [[0]]
        result = self.solver.solve(train, test_input)
        assert result is not None  # Should at least return identity


class TestBaseSolver:
    """Tests for the base ARCSolver (V1 strategies only)."""

    def setup_method(self):
        self.solver = ARCSolver()

    def test_identity(self):
        train = [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        ]
        result = self.solver.solve(train, [[5, 6], [7, 8]])
        assert result == [[5, 6], [7, 8]]


class TestV18Solver:
    """Tests for ARCSolverV18 (all 106 strategies)."""

    def setup_method(self):
        self.solver = ARCSolverV18()

    def test_has_all_strategies(self):
        """V18 should have 90+ strategy methods."""
        strategies = [attr for attr in dir(self.solver) if attr.startswith("_try_")]
        assert len(strategies) >= 90, f"Expected 90+ strategies, got {len(strategies)}"

    def test_no_crashes_on_varied_sizes(self):
        """Run against tasks of different sizes without crashing."""
        sizes = [(3, 3), (5, 5), (10, 10), (15, 15), (3, 7), (7, 3)]
        for rows, cols in sizes:
            train = [
                {"input": [[0] * cols for _ in range(rows)],
                 "output": [[0] * cols for _ in range(rows)]},
            ]
            test_input = [[0] * cols for _ in range(rows)]
            # Should not crash
            result = self.solver.solve(train, test_input)
            assert result is not None


class TestObjectDetection:
    """Tests for ARC object detection/segmentation."""

    def setup_method(self):
        self.detector = ObjectDetector()

    def test_detect_single_object(self):
        """Should detect a single non-zero region."""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        objects = self.detector.detect(grid)
        assert len(objects) >= 1

    def test_detect_multiple_objects(self):
        """Should detect separate colored regions."""
        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
        ]
        objects = self.detector.detect(grid)
        assert len(objects) >= 2  # At least the distinct color groups

    def test_empty_grid(self):
        """Empty grid should return no objects."""
        grid = [[0, 0], [0, 0]]
        objects = self.detector.detect(grid)
        assert isinstance(objects, list)


class TestArcEvalIntegration:
    """Integration test: run the eval harness on a small sample."""

    def test_eval_runs_without_errors(self):
        """The eval harness should complete without errors."""
        from klomboagi.evals.arc_eval import run_arc_eval
        report = run_arc_eval(max_tasks=20)
        assert report.total == 20
        assert report.errors == 0
        assert report.correct + report.failed == report.total
        assert report.correct >= 1  # Should solve at least 1 out of 20
