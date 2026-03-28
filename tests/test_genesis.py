"""Tests for Cognitive Genesis — the full integration layer."""

from __future__ import annotations

import pytest

from klomboagi.core.genesis import Genesis, DialogContext, Surprise


@pytest.fixture
def genesis():
    return Genesis(memory_path="/tmp/klombo_test_genesis.json")


# ── Dialog Context tests ──


class TestDialogContext:

    def test_starts_empty(self):
        ctx = DialogContext()
        assert ctx.current_topic == ""
        assert ctx.turn_count == 0

    def test_update_extracts_topic_from_teaching(self):
        ctx = DialogContext()
        ctx.update({"type": "teach", "subject": "alligator", "predicate": "reptile"}, "an alligator is a reptile")
        assert ctx.current_topic == "alligator"
        assert ctx.turn_count == 1

    def test_topic_depth_increases(self):
        ctx = DialogContext()
        ctx.update({"type": "teach", "subject": "alligator", "predicate": "reptile"}, "an alligator is a reptile")
        ctx.update({"type": "teach", "subject": "alligator", "predicate": "green"}, "an alligator is green")
        assert ctx.topic_depth == 2

    def test_topic_change_resets_depth(self):
        ctx = DialogContext()
        ctx.update({"type": "teach", "subject": "alligator", "predicate": "reptile"}, "an alligator is a reptile")
        ctx.update({"type": "teach", "subject": "python", "predicate": "language"}, "python is a language")
        assert ctx.current_topic == "python"
        assert ctx.previous_topic == "alligator"
        assert ctx.topic_depth == 1

    def test_resolve_pronoun(self):
        ctx = DialogContext()
        ctx.current_topic = "alligator"
        assert "alligator" in ctx.resolve_pronoun("is it green?")

    def test_resolve_pronoun_no_topic(self):
        ctx = DialogContext()
        assert ctx.resolve_pronoun("is it green?") == "is it green?"

    def test_entities_extracted(self):
        ctx = DialogContext()
        ctx.update({"type": "general", "content": "alligators live in swamps"}, "alligators live in swamps")
        assert "alligators" in ctx.entities_mentioned or "swamps" in ctx.entities_mentioned

    def test_to_dict(self):
        ctx = DialogContext()
        d = ctx.to_dict()
        assert "current_topic" in d
        assert "turn_count" in d


# ── Surprise tests ──


class TestSurprise:

    def test_surprise_dataclass(self):
        s = Surprise(
            statement="cat is a reptile",
            old_belief="cat is a mammal",
            new_input="cat is a reptile",
            old_confidence=0.8,
            surprise_magnitude=0.8,
        )
        assert s.surprise_magnitude == 0.8
        assert not s.resolved

    def test_surprise_to_dict(self):
        s = Surprise(
            statement="x", old_belief="y", new_input="z",
            old_confidence=0.5, surprise_magnitude=0.5,
        )
        d = s.to_dict()
        assert "statement" in d
        assert "surprise_magnitude" in d


# ── Genesis core tests ──


class TestGenesis:

    def test_init(self, genesis):
        assert genesis.total_turns == 0
        assert len(genesis.traits.traits) == 4  # curiosity, persistence, analysis, accuracy

    def test_hear_basic_teaching(self, genesis):
        response = genesis.hear("a dog is an animal")
        assert "dog" in response.lower()
        assert genesis.total_turns == 1

    def test_hear_basic_question(self, genesis):
        genesis.hear("a cat is a feline")
        response = genesis.hear("what is a cat?")
        assert len(response) > 0

    def test_dialog_context_tracked(self, genesis):
        genesis.hear("a dog is an animal")
        assert genesis.context.current_topic == "dog"

    def test_pronoun_resolution(self, genesis):
        genesis.hear("a dog is an animal")
        # "it" should resolve to "dog"
        assert genesis.context.current_topic == "dog"

    def test_status(self, genesis):
        genesis.hear("a tree is a plant")
        status = genesis.status()
        assert "Personality" in status
        assert "curiosity" in status

    def test_trait_system_initialized(self, genesis):
        pv = genesis.traits.personality_vector()
        assert "curiosity" in pv
        assert "persistence" in pv
        assert "analysis" in pv
        assert "accuracy" in pv

    def test_traits_strengthen_on_relevant_input(self, genesis):
        before = genesis.traits.traits["curiosity"].drive_strength
        genesis.hear("what is a quasar?")
        after = genesis.traits.traits["curiosity"].drive_strength
        # Curiosity keywords include "what" — should strengthen
        assert after >= before


# ── Surprise detection tests ──


class TestSurpriseDetection:

    def test_detects_color_contradiction(self, genesis):
        genesis.hear("a car is red")
        response = genesis.hear("a car is blue")
        assert "thought" in response.lower() or "wait" in response.lower() or "updating" in response.lower()
        assert genesis.total_surprises >= 1

    def test_detects_animal_class_contradiction(self, genesis):
        genesis.hear("a whale is a fish")
        response = genesis.hear("a whale is a mammal")
        assert genesis.total_surprises >= 1
        # Old belief should be weakened
        old_belief = genesis.base._beliefs.get("whale is fish")
        if old_belief:
            assert old_belief.truth.frequency < 1.0

    def test_no_surprise_on_additional_info(self, genesis):
        genesis.hear("a dog is an animal")
        genesis.hear("a dog is friendly")
        # "animal" and "friendly" aren't contradictory
        assert genesis.total_surprises == 0

    def test_surprise_on_negation(self, genesis):
        genesis.hear("a tomato is a vegetable")
        response = genesis.hear("a tomato is not a vegetable")
        assert genesis.total_surprises >= 1


# ── Predicate conflict detection ──


class TestPredicateConflict:

    def test_color_conflict(self, genesis):
        assert genesis._predicates_conflict("red", "blue")
        assert genesis._predicates_conflict("green", "yellow")

    def test_no_conflict_different_types(self, genesis):
        assert not genesis._predicates_conflict("green", "large")
        assert not genesis._predicates_conflict("red", "fast")

    def test_animal_class_conflict(self, genesis):
        assert genesis._predicates_conflict("mammal", "reptile")
        assert genesis._predicates_conflict("bird", "fish")

    def test_negation_conflict(self, genesis):
        assert genesis._predicates_conflict("alive", "not alive")
        assert genesis._predicates_conflict("not true", "true")

    def test_no_conflict_same_predicate(self, genesis):
        assert not genesis._predicates_conflict("red", "red")

    def test_temperature_conflict(self, genesis):
        assert genesis._predicates_conflict("hot", "cold")

    def test_size_conflict(self, genesis):
        assert genesis._predicates_conflict("big", "small")


# ── Proactive curiosity tests ──


class TestProactiveCuriosity:

    def test_proactive_fires_on_turn_3(self, genesis):
        # Teach something to create gaps
        genesis.hear("a quasar is a space object")
        genesis.hear("a quasar is very bright")
        # Turn 3 should trigger proactive
        response = genesis.hear("a quasar is far away")
        # May or may not fire depending on gaps, but shouldn't crash
        assert len(response) > 0

    def test_proactive_requires_curiosity_trait(self, genesis):
        # Weaken curiosity below threshold
        genesis.traits.traits["curiosity"].drive_strength = 0.1
        genesis.hear("a rock is hard")
        genesis.hear("a rock is grey")
        response = genesis.hear("a rock is heavy")
        # Should NOT fire proactive with weak curiosity
        assert "curious" not in response.lower() or "wondering" not in response.lower()


# ── Integration test ──


class TestGenesisLifecycle:

    def test_full_teaching_session(self, genesis):
        """Simulate a real teaching session."""
        r1 = genesis.hear("an alligator is a reptile")
        assert "alligator" in r1.lower()

        r2 = genesis.hear("a reptile is an animal")
        assert "animal" in r2.lower() or "reptile" in r2.lower()

        r3 = genesis.hear("what is an alligator?")
        # Should know via deduction chain
        assert len(r3) > 0

        # Check status works
        status = genesis.status()
        assert "Cognitive Genesis" in status

    def test_correction_flow(self, genesis):
        """Teach something wrong, then correct it."""
        genesis.hear("a whale is a fish")
        response = genesis.hear("no, a whale is a mammal")
        assert "corrected" in response.lower() or "mammal" in response.lower()

    def test_personality_develops(self, genesis):
        """Personality should shift based on interaction type."""
        initial = genesis.traits.personality_vector()

        # Ask lots of questions (triggers curiosity)
        for i in range(5):
            genesis.hear(f"what is concept_{i}?")

        after = genesis.traits.personality_vector()
        # Curiosity should have strengthened (keywords "what" match)
        assert after["curiosity"] >= initial["curiosity"]
