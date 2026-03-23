"""Tests for NARS truth value system."""

from __future__ import annotations

import pytest
from klomboagi.reasoning.truth import (
    TruthValue, EvidenceStamp, Belief,
    revision, deduction, induction, abduction, analogy,
    comparison, negation, temporal_projection, w2c, c2w,
)


class TestTruthValue:

    def test_single_observation(self):
        tv = TruthValue.from_single_observation(True)
        assert tv.frequency == 1.0
        assert 0.4 < tv.confidence < 0.6  # ~0.5 for k=1

    def test_evidence_counts(self):
        tv = TruthValue.from_evidence(positive=8, total=10)
        assert abs(tv.frequency - 0.8) < 0.01
        assert tv.confidence > 0.8

    def test_confidence_never_reaches_one(self):
        tv = TruthValue.from_evidence(positive=1000, total=1000)
        assert tv.confidence < 1.0

    def test_expectation(self):
        tv = TruthValue(0.8, 0.9)
        assert tv.expectation > 0.5  # Positive belief

    def test_zero_evidence(self):
        tv = TruthValue.from_evidence(0, 0)
        assert tv.frequency == 0.5  # Unknown
        assert tv.confidence == 0.0  # No evidence


class TestRevision:

    def test_accumulates_evidence(self):
        tv1 = TruthValue.from_single_observation(True)
        tv2 = TruthValue.from_single_observation(True)
        combined = revision(tv1, tv2)
        assert combined.confidence > tv1.confidence
        assert combined.frequency == 1.0

    def test_contradicting_evidence_lowers_frequency(self):
        tv1 = TruthValue(1.0, 0.9)  # Strong positive
        tv2 = TruthValue.from_single_observation(False)  # One negative
        combined = revision(tv1, tv2)
        assert combined.frequency < tv1.frequency

    def test_contradicting_evidence_raises_confidence(self):
        tv1 = TruthValue(1.0, 0.9)
        tv2 = TruthValue.from_single_observation(False)
        combined = revision(tv1, tv2)
        assert combined.confidence >= tv1.confidence  # More evidence = more confident

    def test_many_observations_high_confidence(self):
        tv = TruthValue.from_single_observation(True)
        for _ in range(99):
            tv = revision(tv, TruthValue.from_single_observation(True))
        assert tv.confidence > 0.95
        assert tv.frequency > 0.99


class TestDeduction:

    def test_chain_reduces_confidence(self):
        tv1 = TruthValue(0.9, 0.9)
        tv2 = TruthValue(1.0, 0.95)
        result = deduction(tv1, tv2)
        assert result.confidence < min(tv1.confidence, tv2.confidence)

    def test_perfect_premises(self):
        tv1 = TruthValue(1.0, 0.99)
        tv2 = TruthValue(1.0, 0.99)
        result = deduction(tv1, tv2)
        assert result.frequency > 0.95
        assert result.confidence > 0.9


class TestInductionAbduction:

    def test_induction_is_weak(self):
        tv1 = TruthValue(0.9, 0.8)
        tv2 = TruthValue(0.95, 0.85)
        result = induction(tv1, tv2)
        assert result.confidence < 0.5  # Weak!

    def test_abduction_is_weak(self):
        tv1 = TruthValue(0.9, 0.8)
        tv2 = TruthValue(0.95, 0.85)
        result = abduction(tv1, tv2)
        assert result.confidence < 0.5

    def test_induction_accumulates_through_revision(self):
        """Multiple weak inductions → strong belief through revision."""
        results = []
        for _ in range(5):
            ind = induction(
                TruthValue(0.9, 0.8),
                TruthValue(0.95, 0.85),
            )
            results.append(ind)

        combined = results[0]
        for r in results[1:]:
            combined = revision(combined, r)
        assert combined.confidence > results[0].confidence


class TestEvidenceStamp:

    def test_no_overlap(self):
        s1 = EvidenceStamp.new(1)
        s2 = EvidenceStamp.new(2)
        assert not s1.overlaps(s2)

    def test_overlap_detected(self):
        s1 = EvidenceStamp(frozenset([1, 2, 3]))
        s2 = EvidenceStamp(frozenset([3, 4, 5]))
        assert s1.overlaps(s2)  # Share source 3

    def test_merge(self):
        s1 = EvidenceStamp.new(1)
        s2 = EvidenceStamp.new(2)
        merged = s1.merge(s2)
        assert merged.sources == frozenset([1, 2])


class TestBelief:

    def test_revise_independent(self):
        b1 = Belief("sky is blue", TruthValue(1.0, 0.5),
                     EvidenceStamp.new(1), source="human")
        b2 = Belief("sky is blue", TruthValue(1.0, 0.5),
                     EvidenceStamp.new(2), source="self")
        revised = b1.revise_with(b2)
        assert revised is not None
        assert revised.truth.confidence > b1.truth.confidence

    def test_revise_overlapping_blocked(self):
        stamp = EvidenceStamp(frozenset([1, 2]))
        b1 = Belief("X", TruthValue(1.0, 0.5), stamp)
        b2 = Belief("X", TruthValue(0.5, 0.5), stamp)
        revised = b1.revise_with(b2)
        assert revised is None  # Blocked — same evidence

    def test_serialization(self):
        b = Belief("test", TruthValue(0.8, 0.6),
                   EvidenceStamp.new(42), subject="x", predicate="y")
        d = b.to_dict()
        b2 = Belief.from_dict(d)
        assert abs(b2.truth.frequency - 0.8) < 0.01
        assert abs(b2.truth.confidence - 0.6) < 0.01


class TestTemporalProjection:

    def test_recent_belief_strong(self):
        tv = TruthValue(0.9, 0.9)
        projected = temporal_projection(tv, time_distance=0.1)
        assert projected.confidence > 0.8

    def test_old_belief_weakens(self):
        tv = TruthValue(0.9, 0.9)
        projected = temporal_projection(tv, time_distance=5.0)
        assert projected.confidence < tv.confidence

    def test_frequency_unchanged(self):
        tv = TruthValue(0.9, 0.9)
        projected = temporal_projection(tv, time_distance=5.0)
        assert projected.frequency == tv.frequency


class TestLearningScenario:
    """Full learning scenario — the alligator problem."""

    def test_alligator_learning(self):
        """System learns about alligators through accumulated evidence."""
        evidence_id = 0

        def obs(positive: bool) -> Belief:
            nonlocal evidence_id
            evidence_id += 1
            return Belief(
                "alligators are green",
                TruthValue.from_single_observation(positive),
                EvidenceStamp.new(evidence_id),
                source="observation",
            )

        # 10 green alligators
        belief = obs(True)
        for _ in range(9):
            revised = belief.revise_with(obs(True))
            if revised:
                belief = revised

        assert belief.truth.frequency > 0.95
        assert belief.truth.confidence > 0.85

        # 1 brown alligator
        revised = belief.revise_with(obs(False))
        if revised:
            belief = revised

        # Frequency dropped, confidence still high
        assert belief.truth.frequency < 1.0
        assert belief.truth.confidence > 0.85
