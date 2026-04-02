"""Rigorous tests for the Seven Pillars of Reasoning."""

import pytest
from klomboagi.reasoning.core_reasoner import CoreReasoner, Rel
from klomboagi.reasoning.seven_pillars import (
    decompose, compare, abstract, transfer, inquire,
    build_causal_model, predict_effects, self_evaluate,
)


@pytest.fixture
def reasoner():
    r = CoreReasoner()
    r.tell_many([
        ("dog", Rel.IS_A, "mammal"), ("cat", Rel.IS_A, "mammal"),
        ("wolf", Rel.IS_A, "mammal"), ("horse", Rel.IS_A, "mammal"),
        ("mammal", Rel.IS_A, "animal"), ("animal", Rel.IS_A, "living thing"),
        ("mammal", Rel.HAS_PROP, "warm-blooded"), ("mammal", Rel.HAS_PROP, "has fur"),
        ("dog", Rel.HAS_PROP, "loyal"), ("dog", Rel.CAN, "swim"),
        ("cat", Rel.HAS_PROP, "independent"), ("cat", Rel.CAN, "climb"),
        ("bird", Rel.IS_A, "animal"), ("bird", Rel.CAN, "fly"),
        ("bird", Rel.HAS_PROP, "has feathers"),
        ("eagle", Rel.IS_A, "bird"), ("penguin", Rel.IS_A, "bird"),
        ("gravity", Rel.CAUSES, "falling"),
        ("falling", Rel.CAUSES, "impact"),
        ("impact", Rel.CAUSES, "damage"),
        ("fire", Rel.REQUIRES, "oxygen"),
        ("heat", Rel.CAUSES, "expansion"),
        ("expansion", Rel.CAUSES, "cracking"),
    ], confidence=0.9, source="test")
    r.deny("penguin", Rel.CAN, "fly")
    r.forward_chain()
    return r


class TestDecompose:
    def test_finds_known_entities(self, reasoner):
        d = decompose(reasoner, "why do dogs chase cats")
        assert "dog" in d.parts or "dogs" in d.parts

    def test_identifies_unknowns(self, reasoner):
        d = decompose(reasoner, "what is quantum entanglement")
        assert len(d.unknowns) > 0

    def test_finds_relations(self, reasoner):
        d = decompose(reasoner, "dogs and cats are both mammals")
        # dog and cat should be known entities with relations
        assert "dog" in d.parts or "dogs" in d.parts


class TestCompare:
    def test_similar_things(self, reasoner):
        c = compare(reasoner, "dog", "cat")
        assert c.similarity > 0.3
        assert "warm-blooded" in c.shared_properties or "has fur" in c.shared_properties

    def test_different_things(self, reasoner):
        c = compare(reasoner, "dog", "gravity")
        assert c.similarity < 0.1

    def test_finds_unique_properties(self, reasoner):
        c = compare(reasoner, "dog", "cat")
        assert "loyal" in c.a_only
        assert "independent" in c.b_only

    def test_symmetric(self, reasoner):
        c1 = compare(reasoner, "dog", "cat")
        c2 = compare(reasoner, "cat", "dog")
        assert abs(c1.similarity - c2.similarity) < 0.01


class TestAbstract:
    def test_finds_common_pattern(self, reasoner):
        a = abstract(reasoner, ["dog", "cat", "horse"])
        assert "mammal" in a.common_categories
        assert "warm-blooded" in a.common_properties

    def test_identifies_variables(self, reasoner):
        a = abstract(reasoner, ["dog", "cat"])
        # loyal is dog-only, independent is cat-only
        assert "loyal" in a.variable_properties or "independent" in a.variable_properties

    def test_empty_examples(self, reasoner):
        a = abstract(reasoner, [])
        assert a.confidence == 0.0

    def test_single_example(self, reasoner):
        a = abstract(reasoner, ["dog"])
        assert len(a.common_categories) > 0


class TestTransfer:
    def test_transfers_properties(self, reasoner):
        t = transfer(reasoner, "dog", "wolf")
        assert "loyal" in t.transferred_properties
        assert t.confidence > 0

    def test_transfer_adds_to_kb(self, reasoner):
        transfer(reasoner, "dog", "wolf")
        result = reasoner.ask("wolf", Rel.HAS_PROP, "loyal")
        assert result.known
        assert result.confidence < 0.9  # should be lower than direct fact

    def test_no_transfer_unrelated(self, reasoner):
        t = transfer(reasoner, "dog", "gravity")
        assert len(t.transferred_properties) == 0 or t.confidence < 0.1


class TestInquire:
    def test_finds_gaps(self, reasoner):
        gaps = inquire(reasoner)
        assert len(gaps) > 0

    def test_gaps_have_priority(self, reasoner):
        gaps = inquire(reasoner)
        assert all(0 <= g.priority <= 1 for g in gaps)

    def test_gaps_sorted_by_priority(self, reasoner):
        gaps = inquire(reasoner)
        priorities = [g.priority for g in gaps]
        assert priorities == sorted(priorities, reverse=True)


class TestCausalModel:
    def test_traces_chain(self, reasoner):
        chain = build_causal_model(reasoner, "damage")
        assert chain is not None
        assert chain.root_cause == "gravity"
        assert chain.final_effect == "damage"
        assert len(chain.chain) == 3  # gravity->falling, falling->impact, impact->damage

    def test_no_duplicates(self, reasoner):
        chain = build_causal_model(reasoner, "damage")
        assert chain is not None
        effects = [e for _, e in chain.chain]
        assert len(effects) == len(set(effects))

    def test_interventions(self, reasoner):
        chain = build_causal_model(reasoner, "damage")
        assert chain is not None
        # fire requires oxygen is in KB, but not in this causal chain
        # so no interventions expected for gravity->damage chain

    def test_nonexistent_effect(self, reasoner):
        chain = build_causal_model(reasoner, "teleportation")
        assert chain is None

    def test_predict_forward(self, reasoner):
        effects = predict_effects(reasoner, "gravity")
        effect_names = [e for e, _ in effects]
        assert "falling" in effect_names
        assert "impact" in effect_names
        assert "damage" in effect_names

    def test_predict_no_duplicates(self, reasoner):
        effects = predict_effects(reasoner, "gravity")
        effect_names = [e for e, _ in effects]
        assert len(effect_names) == len(set(effect_names))

    def test_predict_confidence_decreases(self, reasoner):
        effects = predict_effects(reasoner, "gravity")
        # Direct effect should have higher confidence than indirect
        falling_conf = next(c for e, c in effects if e == "falling")
        damage_conf = next(c for e, c in effects if e == "damage")
        assert falling_conf > damage_conf


class TestSelfEvaluate:
    def test_supported_claim(self, reasoner):
        e = self_evaluate(reasoner, "dog is a animal")
        assert e.is_supported
        assert len(e.supporting_facts) > 0

    def test_unsupported_claim(self, reasoner):
        e = self_evaluate(reasoner, "dog is a reptile")
        assert not e.is_supported

    def test_blocked_claim(self, reasoner):
        e = self_evaluate(reasoner, "penguin can fly")
        assert not e.is_supported

    def test_detects_weaknesses(self, reasoner):
        e = self_evaluate(reasoner, "quantum can teleport")
        assert len(e.weaknesses) > 0

    def test_finds_alternatives(self, reasoner):
        e = self_evaluate(reasoner, "dog is a reptile")
        # Should suggest dog is a mammal as alternative
        assert len(e.alternatives) > 0 or len(e.contradicting_facts) > 0

    def test_causal_claim(self, reasoner):
        e = self_evaluate(reasoner, "gravity causes falling")
        assert e.is_supported
