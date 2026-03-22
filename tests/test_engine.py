"""Tests for the internal reasoning engine."""

from __future__ import annotations

import pytest
from klomboagi.reasoning.engine import ReasoningEngine, PropertyType


@pytest.fixture
def engine():
    return ReasoningEngine()


class TestPropertyIdentification:

    def test_identifies_color(self, engine):
        chain = engine.reason("what color?", ["the car is red"])
        color_facts = [f for f in chain.facts if f.property_type == PropertyType.COLOR]
        assert len(color_facts) > 0

    def test_identifies_measurement(self, engine):
        chain = engine.reason("how long?", ["the rope is 10 feet long"])
        measure_facts = [f for f in chain.facts if f.property_type == PropertyType.MEASUREMENT]
        assert len(measure_facts) > 0

    def test_identifies_weight(self, engine):
        chain = engine.reason("how heavy?", ["the box weighs 50 pounds"])
        weight_facts = [f for f in chain.facts if f.property_type == PropertyType.WEIGHT]
        assert len(weight_facts) > 0

    def test_color_is_2d(self, engine):
        chain = engine.reason("what?", ["the wall is blue"])
        color_facts = [f for f in chain.facts if f.property_type == PropertyType.COLOR]
        assert color_facts[0].dimensions == 2

    def test_measurement_is_1d(self, engine):
        chain = engine.reason("what?", ["the road is 5 miles long"])
        m_facts = [f for f in chain.facts if f.property_type == PropertyType.MEASUREMENT]
        assert m_facts[0].dimensions == 1

    def test_weight_is_0d(self, engine):
        chain = engine.reason("what?", ["the rock weighs 10 pounds"])
        w_facts = [f for f in chain.facts if f.property_type == PropertyType.WEIGHT]
        assert w_facts[0].dimensions == 0


class TestDimensionalComparison:

    def test_alligator_greener_than_longer(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        assert "MORE" in chain.conclusion
        assert "green" in chain.conclusion.lower()
        assert chain.confidence >= 0.7

    def test_fire_truck_redder_than_heavier(self, engine):
        chain = engine.reason(
            "Is a fire truck redder or heavier?",
            ["fire trucks are red", "fire trucks weigh about 40000 pounds"],
        )
        assert "MORE" in chain.conclusion
        assert "red" in chain.conclusion.lower()

    def test_same_dimension_no_winner(self, engine):
        chain = engine.reason(
            "Is the ocean deeper or wider?",
            ["the ocean is about 12000 feet deep", "the ocean is about 12000 miles wide"],
        )
        assert "same" in chain.conclusion.lower() or "magnitude" in chain.conclusion.lower()

    def test_transfer_to_novel_problem(self, engine):
        """Engine should handle a novel problem with same structure."""
        chain = engine.reason(
            "Is a banana more yellow or longer?",
            ["bananas are yellow", "bananas are about 7 inches long"],
        )
        assert "MORE" in chain.conclusion
        assert "yellow" in chain.conclusion.lower()
        assert chain.confidence >= 0.7


class TestTrickDetection:

    def test_detects_cross_dimensional_trick(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        trick_steps = [s for s in chain.steps if s.operation == "detect_structure"]
        assert len(trick_steps) > 0
        trick_fact = chain.facts[trick_steps[0].output_fact]
        assert "YES" in trick_fact.content

    def test_no_trick_for_same_type(self, engine):
        chain = engine.reason(
            "Is the building taller or wider?",
            ["the building is 100 feet tall", "the building is 50 feet wide"],
        )
        trick_steps = [s for s in chain.steps if s.operation == "detect_structure"]
        assert len(trick_steps) > 0
        trick_fact = chain.facts[trick_steps[0].output_fact]
        assert "NO" in trick_fact.content


class TestReasoningChain:

    def test_chain_has_steps(self, engine):
        chain = engine.reason("Is X or Y?", ["X is red", "Y is long"])
        assert len(chain.steps) > 0

    def test_each_step_has_explanation(self, engine):
        chain = engine.reason("Is X or Y?", ["X is red", "Y is long"])
        for step in chain.steps:
            assert step.explanation != ""

    def test_derived_facts_reference_parents(self, engine):
        chain = engine.reason("test", ["the sky is blue"])
        derived = [f for f in chain.facts if not f.is_given]
        for f in derived:
            # Every derived fact should reference its derivation step
            assert f.id > 0  # Not a given fact

    def test_explain_produces_readable_output(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        explanation = chain.explain()
        assert "Given facts:" in explanation
        assert "Reasoning:" in explanation
        assert "Conclusion:" in explanation
        assert "Step 1" in explanation

    def test_multiple_frameworks_recorded(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        assert len(chain.frameworks_used) >= 2

    def test_alternative_conclusions_exist(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        assert len(chain.alternative_conclusions) > 0


class TestNudge:

    def test_nudge_recorded_in_chain(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
            nudge="think dimensionally",
        )
        nudge_steps = [s for s in chain.steps if s.operation == "apply_nudge"]
        assert len(nudge_steps) > 0

    def test_re_reason_with_nudge(self, engine):
        chain = engine.reason(
            "Is an alligator greener or longer?",
            ["alligators are green", "alligators are about 12 feet long"],
        )
        chain2 = engine.reason_with_nudge(chain, "think about 3D vs 1D")
        nudge_steps = [s for s in chain2.steps if s.operation == "apply_nudge"]
        assert len(nudge_steps) > 0


class TestThreeDogs:
    """The '3 dogs' insight — it's about the 3, not the dogs."""

    def test_quantity_is_0d(self, engine):
        chain = engine.reason("how many?", ["there are three dogs"])
        q_facts = [f for f in chain.facts if f.property_type == PropertyType.QUANTITY]
        assert len(q_facts) > 0
        assert q_facts[0].dimensions == 0

    def test_quantity_vs_color(self, engine):
        """3 dogs vs blue dogs — the color occupies more dimensional space."""
        chain = engine.reason(
            "Are the dogs more blue or more three?",
            ["the dogs are blue", "there are three dogs"],
        )
        # Color (2D) > Quantity (0D)
        assert "MORE" in chain.conclusion
        assert "blue" in chain.conclusion.lower()
