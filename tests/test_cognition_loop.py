"""Tests for the integrated cognition loop."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from klomboagi.reasoning.cognition_loop import CognitionLoop, CognitionPhase


@pytest.fixture
def storage():
    s = MagicMock()
    def _load(key, default=None):
        if key == 'causal_graph':
            return {'nodes': [], 'edges': {}}
        return default if default is not None else []
    s.load_json = MagicMock(side_effect=_load)
    s.save_json = MagicMock()
    s.event_log = MagicMock()
    return s


@pytest.fixture
def loop(storage):
    return CognitionLoop(storage)


class TestBasicLoop:

    def test_runs_to_completion(self, loop):
        """The loop should complete without crashing."""
        result = loop.think({"description": "Fix the login bug"})
        assert result.phase == CognitionPhase.COMPLETE
        assert len(result.trace) > 0

    def test_trace_records_all_phases(self, loop):
        result = loop.think({"description": "Add a new feature"})
        phases = [entry["phase"] for entry in result.trace]
        assert "perceive" in phases
        assert "remember" in phases

    def test_novel_problem_goes_through_inquiry(self, loop):
        """With no past experiences, should hit inquiry phase."""
        result = loop.think({
            "description": "Something completely new",
            "referenced_entities": ["quantum_flux"],
            "known_entities": [],
        })
        phases = [entry["phase"] for entry in result.trace]
        assert "inquire" in phases

    def test_completes_with_hypothesis(self, loop):
        result = loop.think({"description": "Simple task"})
        assert result.hypothesis is not None


class TestWithPastExperience:

    def test_finds_similar_and_transfers(self, storage):
        """If past episodes exist, should find and transfer."""
        past_episodes = [
            {"id": "ep1", "description": "Fix auth bug",
             "actions": [{"type": "read"}, {"type": "edit"}, {"type": "test"}],
             "outcome": "completed", "success": True},
        ]
        storage.load_json = MagicMock(side_effect=lambda key, default=None:
            past_episodes if key == "episodes" else
            [] if key == "abstractions" else
            {"nodes": [], "edges": {}} if key == "causal_graph" else
            default or [])

        loop = CognitionLoop(storage)
        result = loop.think({
            "description": "Fix payment bug",
            "actions": [{"type": "read"}],
        })

        phases = [entry["phase"] for entry in result.trace]
        assert "remember" in phases


class TestNudge:

    def test_nudge_sends_back_to_revise(self, loop):
        """Human nudge should redirect to revision."""
        state = loop.think({"description": "Test problem"})

        # Nudge after completion
        loop.nudge(state, "dimensional")
        assert state.phase == CognitionPhase.REVISE
        assert "dimensional" in state.nudges

    def test_nudge_recorded_in_trace(self, loop):
        state = loop.think({"description": "Test"})
        loop.nudge(state, "reverse")
        assert any("nudge" in entry.get("phase", "") for entry in state.trace)


class TestSelfEvaluation:

    def test_evaluation_runs(self, loop):
        result = loop.think({"description": "Solve X"})
        assert result.evaluation is not None

    def test_known_facts_checked(self, loop):
        result = loop.think({
            "description": "What color is the sky?",
            "known_facts": ["the sky is blue"],
        })
        phases = [entry["phase"] for entry in result.trace]
        assert "evaluate" in phases


class TestLearning:

    def test_episode_saved_after_completion(self, storage):
        storage.load_json = MagicMock(side_effect=lambda key, default=None:
            [] if key in ("episodes", "abstractions", "knowledge_gaps") else
            {"nodes": [], "edges": {}} if key == "causal_graph" else
            default or [])

        loop = CognitionLoop(storage)
        loop.think({"description": "Learn from this"})

        # Should have saved episodes
        save_calls = [call for call in storage.save_json.call_args_list
                      if call[0][0] == "episodes"]
        assert len(save_calls) > 0

    def test_causal_model_updated(self, storage):
        storage.load_json = MagicMock(side_effect=lambda key, default=None:
            [] if key in ("episodes", "abstractions", "knowledge_gaps") else
            {"nodes": [], "edges": {}} if key == "causal_graph" else
            default or [])

        loop = CognitionLoop(storage)
        loop.think({"description": "Causal test"})

        # Should have saved causal graph
        save_calls = [call for call in storage.save_json.call_args_list
                      if call[0][0] == "causal_graph"]
        assert len(save_calls) > 0


class TestExplain:

    def test_explain_produces_readable_output(self, loop):
        result = loop.think({"description": "Explain this"})
        explanation = loop.explain(result)
        assert len(explanation) > 0
        assert "[perceive]" in explanation


class TestAlligatorIntegration:
    """Full integration test using the alligator problem."""

    def test_alligator_with_nudge(self, loop):
        # First attempt — system has no experience
        state = loop.think({
            "description": "Is an alligator greener or longer?",
            "known_facts": [
                "alligators are green",
                "alligators are about 12 feet long",
            ],
        })

        # The system completed but probably with low confidence
        assert state.phase == CognitionPhase.COMPLETE
        assert state.hypothesis is not None

        # Human nudges: "think dimensionally"
        loop.nudge(state, "dimensional")
        assert state.phase == CognitionPhase.REVISE

        # Run one more step to process the nudge
        loop._step(state)  # REVISE
        loop._step(state)  # HYPOTHESIZE
        loop._step(state)  # EVALUATE

        # Should have more attempts now
        assert state.attempts >= 1
        # Trace should show the nudge
        nudge_entries = [e for e in state.trace if "nudge" in e.get("phase", "").lower() or "nudge" in e.get("message", "").lower()]
        assert len(nudge_entries) > 0
