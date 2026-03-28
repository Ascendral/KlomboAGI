"""Tests for the relation system — multi-directional reasoning."""

from __future__ import annotations

import pytest

from klomboagi.core.relations import (
    RelationStore, RelationType, Relation, InferenceResult,
    TRANSITIVE_RELATIONS, INVERSE_RELATIONS,
)
from klomboagi.core.genesis import Genesis


# ── RelationStore tests ──


class TestRelationStore:

    @pytest.fixture
    def store(self):
        return RelationStore()

    def test_add_relation(self, store):
        r = store.add("gravity", RelationType.CAUSES, "acceleration")
        assert r.source == "gravity"
        assert r.relation == RelationType.CAUSES
        assert r.target == "acceleration"

    def test_deduplicate(self, store):
        store.add("gravity", RelationType.CAUSES, "acceleration")
        store.add("gravity", RelationType.CAUSES, "acceleration")
        assert store.stats()["total_relations"] == 1

    def test_forward_lookup(self, store):
        store.add("gravity", RelationType.CAUSES, "acceleration")
        store.add("gravity", RelationType.CAUSES, "force")
        results = store.get_forward("gravity")
        assert len(results) == 2

    def test_forward_filtered(self, store):
        store.add("gravity", RelationType.CAUSES, "acceleration")
        store.add("gravity", RelationType.USES, "geometry")
        results = store.get_forward("gravity", RelationType.CAUSES)
        assert len(results) == 1
        assert results[0].target == "acceleration"

    def test_backward_lookup(self, store):
        store.add("gravity", RelationType.CAUSES, "acceleration")
        store.add("force", RelationType.CAUSES, "acceleration")
        results = store.get_backward("acceleration")
        assert len(results) == 2

    def test_get_all_about(self, store):
        store.add("energy", RelationType.ENABLES, "work")
        store.add("heat", RelationType.PART_OF, "energy")
        results = store.get_all_about("energy")
        assert len(results) == 2

    def test_query_flexible(self, store):
        store.add("a", RelationType.CAUSES, "b")
        store.add("c", RelationType.CAUSES, "d")
        store.add("a", RelationType.USES, "e")
        results = store.query(relation=RelationType.CAUSES)
        assert len(results) == 2

    def test_stats(self, store):
        store.add("a", RelationType.CAUSES, "b")
        store.add("a", RelationType.USES, "c")
        s = store.stats()
        assert s["total_relations"] == 2
        assert s["unique_concepts"] == 3


# ── Transitive Inference tests ──


class TestTransitiveInference:

    def test_simple_chain(self):
        store = RelationStore()
        store.add("a", RelationType.CAUSES, "b")
        store.add("b", RelationType.CAUSES, "c")
        inferred = store.infer_transitive()
        # Should derive: a causes c
        derived = [i for i in inferred if i.relation.source == "a" and i.relation.target == "c"]
        assert len(derived) == 1
        assert derived[0].relation.relation == RelationType.CAUSES

    def test_three_step_chain(self):
        store = RelationStore()
        store.add("a", RelationType.CAUSES, "b")
        store.add("b", RelationType.CAUSES, "c")
        store.add("c", RelationType.CAUSES, "d")
        inferred = store.infer_transitive(max_depth=3)
        targets = {i.relation.target for i in inferred if i.relation.source == "a"}
        assert "c" in targets
        assert "d" in targets

    def test_confidence_decreases_with_depth(self):
        store = RelationStore()
        store.add("a", RelationType.CAUSES, "b", confidence=0.8)
        store.add("b", RelationType.CAUSES, "c", confidence=0.8)
        inferred = store.infer_transitive()
        derived = [i for i in inferred if i.relation.source == "a" and i.relation.target == "c"]
        assert len(derived) == 1
        assert derived[0].relation.confidence < 0.8  # Should be lower

    def test_is_a_transitive(self):
        store = RelationStore()
        store.add("dog", RelationType.IS_A, "mammal")
        store.add("mammal", RelationType.IS_A, "animal")
        inferred = store.infer_transitive()
        derived = [i for i in inferred if i.relation.source == "dog" and i.relation.target == "animal"]
        assert len(derived) == 1

    def test_part_of_transitive(self):
        store = RelationStore()
        store.add("bit", RelationType.PART_OF, "byte")
        store.add("byte", RelationType.PART_OF, "memory")
        inferred = store.infer_transitive()
        derived = [i for i in inferred if i.relation.source == "bit" and i.relation.target == "memory"]
        assert len(derived) == 1

    def test_no_duplicate_derivation(self):
        store = RelationStore()
        store.add("a", RelationType.CAUSES, "b")
        store.add("b", RelationType.CAUSES, "c")
        store.add("a", RelationType.CAUSES, "c")  # Already known
        inferred = store.infer_transitive()
        # Should NOT re-derive a→c since it's already known
        derived = [i for i in inferred if i.relation.source == "a" and i.relation.target == "c"]
        assert len(derived) == 0

    def test_non_transitive_not_inferred(self):
        store = RelationStore()
        store.add("hot", RelationType.OPPOSITE_OF, "cold")
        store.add("cold", RelationType.OPPOSITE_OF, "warm")
        inferred = store.infer_transitive()
        # OPPOSITE_OF is NOT transitive — should NOT derive hot opposite_of warm
        assert len(inferred) == 0


# ── Cross-Relation Inference tests ──


class TestCrossRelationInference:

    def test_inheritance(self):
        store = RelationStore()
        store.add("dog", RelationType.IS_A, "mammal")
        store.add("mammal", RelationType.REQUIRES, "oxygen")
        inferred = store.infer_cross_relation()
        # dog should inherit "requires oxygen" from mammal
        derived = [i for i in inferred if i.relation.source == "dog" and i.relation.target == "oxygen"]
        assert len(derived) == 1
        assert derived[0].relation.relation == RelationType.REQUIRES

    def test_inherits_causes(self):
        store = RelationStore()
        store.add("gravity", RelationType.IS_A, "fundamental force")
        store.add("fundamental force", RelationType.CAUSES, "acceleration")
        inferred = store.infer_cross_relation()
        derived = [i for i in inferred if i.relation.source == "gravity"]
        assert any(d.relation.target == "acceleration" for d in derived)


# ── Integration with Genesis ──


class TestGenesisRelations:

    def test_teach_relations(self):
        g = Genesis(memory_path="/tmp/klombo_test_rel.json")
        result = g.teach_relations("cross-domain")
        assert "relations" in result.lower()
        assert g.relations.stats()["total_relations"] > 0

    def test_teach_everything(self):
        g = Genesis(memory_path="/tmp/klombo_test_everything.json")
        result = g.teach_everything()
        assert "beliefs" in result.lower()
        assert "relations" in result.lower()
        assert len(g.base._beliefs) > 200

    def test_what_connects(self):
        g = Genesis(memory_path="/tmp/klombo_test_connect.json")
        g.relations.add("gravity", RelationType.CAUSES, "acceleration")
        g.relations.add("mass", RelationType.CAUSES, "gravity")
        result = g.what_connects("gravity")
        assert "acceleration" in result
        assert "mass" in result

    def test_inference_runs(self):
        g = Genesis(memory_path="/tmp/klombo_test_infer.json")
        g.relations.add("a", RelationType.CAUSES, "b")
        g.relations.add("b", RelationType.CAUSES, "c")
        result = g.teach_relations("causal chains")
        assert "Inferred" in result


# ── Relation dataclass tests ──


class TestRelation:

    def test_to_dict(self):
        r = Relation(source="a", relation=RelationType.CAUSES, target="b")
        d = r.to_dict()
        assert d["source"] == "a"
        assert d["relation"] == "causes"

    def test_inverse(self):
        r = Relation(source="a", relation=RelationType.CAUSES, target="b")
        assert r.inverse() == "caused_by"

    def test_repr(self):
        r = Relation(source="gravity", relation=RelationType.CAUSES, target="acceleration")
        assert "causes" in repr(r)
