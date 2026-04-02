"""Rigorous tests for the CoreReasoner -- the actual AGI engine."""

import pytest
from klomboagi.reasoning.core_reasoner import CoreReasoner, Rel, Fact


@pytest.fixture
def reasoner():
    r = CoreReasoner()
    r.tell_many([
        ("dog", Rel.IS_A, "mammal"),
        ("cat", Rel.IS_A, "mammal"),
        ("whale", Rel.IS_A, "mammal"),
        ("snake", Rel.IS_A, "reptile"),
        ("eagle", Rel.IS_A, "bird"),
        ("penguin", Rel.IS_A, "bird"),
        ("mammal", Rel.IS_A, "animal"),
        ("reptile", Rel.IS_A, "animal"),
        ("bird", Rel.IS_A, "animal"),
        ("animal", Rel.IS_A, "living thing"),
        ("mammal", Rel.HAS_PROP, "warm-blooded"),
        ("reptile", Rel.HAS_PROP, "cold-blooded"),
        ("bird", Rel.HAS_PROP, "has feathers"),
        ("bird", Rel.CAN, "fly"),
        ("dog", Rel.HAS_PROP, "loyal"),
        ("whale", Rel.LOCATED, "ocean"),
        ("gravity", Rel.CAUSES, "falling"),
        ("falling", Rel.CAUSES, "impact"),
        ("impact", Rel.CAUSES, "damage"),
        ("fire", Rel.REQUIRES, "oxygen"),
        ("heat", Rel.CAUSES, "expansion"),
    ], confidence=0.9, source="test")
    r.tell_numeric("whale", "weight", 140000, "kg", 0.9, "test")
    r.tell_numeric("car", "weight", 1500, "kg", 0.9, "test")
    r.tell_numeric("human", "height", 1.7, "meters", 0.9, "test")
    r.tell_numeric("human", "weight", 70, "kg", 0.9, "test")
    r.forward_chain()
    return r


class TestForwardChaining:
    """Test that forward chaining derives correct conclusions."""

    def test_transitive_is_a(self, reasoner):
        """dog -> mammal -> animal should derive dog -> animal."""
        result = reasoner.ask("dog", Rel.IS_A, "animal")
        assert result.known
        assert result.confidence > 0.5

    def test_deep_transitivity(self, reasoner):
        """dog -> mammal -> animal -> living thing (3 hops)."""
        result = reasoner.ask("dog", Rel.IS_A, "living thing")
        assert result.known
        assert result.confidence > 0.3

    def test_property_inheritance(self, reasoner):
        """dog -> mammal + mammal has warm-blooded -> dog has warm-blooded."""
        result = reasoner.ask("dog", Rel.HAS_PROP, "warm-blooded")
        assert result.known

    def test_capability_inheritance(self, reasoner):
        """eagle -> bird + bird can fly -> eagle can fly."""
        result = reasoner.ask("eagle", Rel.CAN, "fly")
        assert result.known

    def test_confidence_degrades(self, reasoner):
        """Derived facts should have lower confidence than direct facts."""
        direct = None
        for f in reasoner.facts:
            if f.subject == "dog" and f.relation == Rel.IS_A and f.obj == "mammal":
                direct = f
                break
        derived = None
        for f in reasoner.derived:
            if f.subject == "dog" and f.relation == Rel.IS_A and f.obj == "animal":
                derived = f
                break
        assert direct is not None
        assert derived is not None
        assert derived.confidence < direct.confidence

    def test_no_false_derivations(self, reasoner):
        """Should NOT derive things that don't follow logically."""
        result = reasoner.ask("dog", Rel.IS_A, "reptile")
        assert not result.known

    def test_causal_chain(self, reasoner):
        """gravity -> falling -> impact -> damage (causal transitivity)."""
        result = reasoner.ask("gravity", Rel.CAUSES, "damage")
        assert result.known


class TestExceptions:
    """Test that deny() properly blocks inherited facts."""

    def test_deny_blocks_inheritance(self, reasoner):
        """Penguin can't fly even though bird can fly."""
        # Before deny
        result = reasoner.ask("penguin", Rel.CAN, "fly")
        assert result.known  # inherited from bird

        # Deny
        reasoner.deny("penguin", Rel.CAN, "fly")
        reasoner.forward_chain()

        # After deny
        result = reasoner.ask("penguin", Rel.CAN, "fly")
        assert not result.known

    def test_deny_doesnt_affect_others(self, reasoner):
        """Denying penguin fly shouldn't affect eagle fly."""
        reasoner.deny("penguin", Rel.CAN, "fly")
        reasoner.forward_chain()

        result = reasoner.ask("eagle", Rel.CAN, "fly")
        assert result.known

    def test_deny_survives_rechaining(self, reasoner):
        """Blocked facts shouldn't reappear after forward_chain."""
        reasoner.deny("penguin", Rel.CAN, "fly")
        reasoner.forward_chain()
        reasoner.forward_chain()  # run again

        result = reasoner.ask("penguin", Rel.CAN, "fly")
        assert not result.known


class TestQueries:
    """Test backward chaining / query answering."""

    def test_open_query(self, reasoner):
        """What is a dog? Should return mammal, animal, living thing."""
        result = reasoner.ask("dog", Rel.IS_A, "?")
        assert result.known
        assert "mammal" in result.answer

    def test_specific_query_true(self, reasoner):
        """Is a dog an animal? Yes."""
        result = reasoner.ask("dog", Rel.IS_A, "animal")
        assert result.known
        assert "Yes" in result.answer

    def test_specific_query_false(self, reasoner):
        """Is a dog a bird? Don't know (not false, just unknown)."""
        result = reasoner.ask("dog", Rel.IS_A, "bird")
        assert not result.known

    def test_honest_ignorance(self, reasoner):
        """Asking about something completely unknown."""
        result = reasoner.ask("quantum", Rel.IS_A, "?")
        assert not result.known
        assert "don't know" in result.answer.lower()


class TestNumericComparison:
    """Test magnitude reasoning."""

    def test_heavier(self, reasoner):
        """Whale is heavier than car."""
        result = reasoner.ask_compare("whale", "car", "weight")
        assert result.known
        assert "whale" in result.answer.lower()
        assert "greater" in result.answer.lower()

    def test_lighter(self, reasoner):
        """Car is lighter than whale."""
        result = reasoner.ask_compare("car", "whale", "weight")
        assert result.known
        assert "whale" in result.answer.lower()

    def test_unknown_comparison(self, reasoner):
        """Can't compare things we don't have data for."""
        result = reasoner.ask_compare("dog", "cat", "weight")
        assert not result.known
        assert "don't know" in result.answer.lower()

    def test_unit_conversion(self, reasoner):
        """Should handle unit conversion."""
        reasoner.tell_numeric("building", "height", 50, "meters", 0.9, "test")
        reasoner.tell_numeric("tree", "height", 100, "feet", 0.9, "test")
        result = reasoner.ask_compare("building", "tree", "height")
        assert result.known  # Should be able to compare meters vs feet


class TestTextLearning:
    """Test learning from natural language text."""

    def test_learns_is_a(self):
        r = CoreReasoner()
        facts = r.learn_from_text("A tiger is a big cat. Tigers are predators.")
        assert any(f.relation == Rel.IS_A and "tiger" in f.subject for f in facts)

    def test_learns_causes(self):
        r = CoreReasoner()
        facts = r.learn_from_text("Smoking causes cancer. Pollution causes asthma.")
        assert any(f.relation == Rel.CAUSES for f in facts)

    def test_learns_can(self):
        r = CoreReasoner()
        facts = r.learn_from_text("Dolphins can echolocate.")
        assert any(f.relation == Rel.CAN for f in facts)

    def test_learns_and_chains(self):
        """Learning should trigger forward chaining."""
        r = CoreReasoner()
        r.tell("mammal", Rel.IS_A, "animal", 0.9, "boot")
        r.tell("mammal", Rel.HAS_PROP, "warm-blooded", 0.9, "boot")
        r.learn_from_text("A dolphin is a mammal.")
        # Should now know dolphin is an animal
        result = r.ask("dolphin", Rel.IS_A, "animal")
        assert result.known

    def test_normalizes_plurals(self):
        r = CoreReasoner()
        facts = r.learn_from_text("Elephants are mammals.")
        assert any(f.subject == "elephant" for f in facts)


class TestGaps:
    """Test knowledge gap identification."""

    def test_finds_undefined_concepts(self, reasoner):
        gaps = reasoner.find_gaps()
        questions = [g for g in gaps]
        # "living thing" is referenced but has no further definition
        assert len(questions) > 0

    def test_finds_property_gaps(self):
        r = CoreReasoner()
        r.tell("quokka", Rel.IS_A, "mammal")
        r.forward_chain()
        gaps = r.find_gaps()
        # quokka has a category but we should want to know properties
        property_gaps = [g for g in gaps if "quokka" in g]
        assert len(property_gaps) > 0
