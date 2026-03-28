"""
Focus System — filter noise, find relevance.

When the system has 900+ beliefs, it can't process everything.
This module scores and ranks information by relevance to the current query.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter


STOP_WORDS = frozenset({
    "is", "a", "an", "the", "what", "who", "where", "how",
    "when", "which", "why", "are", "was", "do", "does",
    "can", "could", "about", "tell", "me", "explain",
    "of", "in", "to", "for", "and", "or", "on", "at",
    "by", "from", "with", "as", "be", "been", "being",
})


@dataclass
class FocusResult:
    """Filtered, relevant information for a query."""
    query: str
    beliefs: list[tuple[str, float]]    # (statement, score)
    relations: list[tuple[str, float]]  # (relation_str, score)
    focus_concepts: list[str]
    noise_filtered: int

    def top_beliefs(self, n: int = 8) -> list[str]:
        return [s for s, _ in self.beliefs[:n]]

    def top_relations(self, n: int = 8) -> list[str]:
        return [s for s, _ in self.relations[:n]]


class FocusEngine:
    """Score and rank beliefs/relations by relevance."""

    def focus(self, query: str, beliefs: dict, relations,
              working_memory=None, max_results: int = 10) -> FocusResult:
        query_words = set(query.lower().split()) - STOP_WORDS
        if not query_words:
            query_words = set(query.lower().split()) - {"what", "is", "a", "the"}

        scored_beliefs = []
        for statement, belief in beliefs.items():
            score = 0.0
            if hasattr(belief, 'subject') and belief.subject:
                subj_words = set(belief.subject.lower().split()) - STOP_WORDS
                overlap = query_words & subj_words
                if overlap:
                    score += 1.0 * len(overlap) / max(len(subj_words), 1)
            if hasattr(belief, 'predicate') and belief.predicate:
                pred_words = set(belief.predicate.lower().split()) - STOP_WORDS
                if query_words & pred_words:
                    score += 0.3
            if hasattr(belief, 'truth'):
                score *= (0.5 + belief.truth.confidence * 0.5)
            if working_memory and hasattr(belief, 'subject') and working_memory.contains(belief.subject):
                score += 0.3
            if score > 0:
                scored_beliefs.append((statement, score))

        scored_beliefs.sort(key=lambda x: x[1], reverse=True)

        scored_relations = []
        if hasattr(relations, '_all'):
            for rel in relations._all:
                score = 0.0
                src = set(rel.source.lower().split()) - STOP_WORDS
                tgt = set(rel.target.lower().split()) - STOP_WORDS
                if query_words & src:
                    score += 0.8
                if query_words & tgt:
                    score += 0.5
                score *= rel.confidence
                if score > 0:
                    scored_relations.append((f"{rel.source} {rel.relation.value} {rel.target}", score))

        scored_relations.sort(key=lambda x: x[1], reverse=True)

        concept_counts: Counter = Counter()
        for stmt, _ in scored_beliefs[:max_results]:
            b = beliefs.get(stmt)
            if b and hasattr(b, 'subject') and b.subject:
                concept_counts[b.subject] += 1

        total = len(beliefs) + (len(relations._all) if hasattr(relations, '_all') else 0)
        returned = min(len(scored_beliefs), max_results) + min(len(scored_relations), max_results)

        return FocusResult(
            query=query,
            beliefs=scored_beliefs[:max_results],
            relations=scored_relations[:max_results],
            focus_concepts=[c for c, _ in concept_counts.most_common(5)],
            noise_filtered=max(0, total - returned),
        )
