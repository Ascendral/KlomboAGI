"""
Belief Deduplication — merge near-duplicate beliefs.

"gravity is a fundamental force" and "gravity is fundamental force of attraction"
are nearly the same. Keep the longer/more-specific one. Merge their confidence.

"energy is the capacity to do work" and "energy is capacity to do work or cause change"
— keep the more complete one.
"""

from __future__ import annotations

from collections import defaultdict


class BeliefDeduplicator:
    """Finds and merges near-duplicate beliefs."""

    def deduplicate(self, beliefs: dict) -> list[str]:
        """
        Find and remove near-duplicate beliefs.
        Keeps the longest/most-specific version.
        Returns list of removed statements.
        """
        # Group by subject
        by_subject: dict[str, list[tuple[str, object]]] = defaultdict(list)
        for stmt, belief in beliefs.items():
            if hasattr(belief, 'subject') and belief.subject:
                by_subject[belief.subject.lower()].append((stmt, belief))

        removed = []
        for subject, group in by_subject.items():
            if len(group) < 2:
                continue

            # Sort by predicate length (longest = most specific)
            group.sort(key=lambda x: len(x[1].predicate) if hasattr(x[1], 'predicate') and x[1].predicate else 0, reverse=True)

            # Check each pair for near-duplication
            keep = set()
            drop = set()
            for i, (stmt_a, belief_a) in enumerate(group):
                if stmt_a in drop:
                    continue
                pred_a = belief_a.predicate.lower() if hasattr(belief_a, 'predicate') and belief_a.predicate else ""
                if not pred_a:
                    continue

                for j in range(i + 1, len(group)):
                    stmt_b, belief_b = group[j]
                    if stmt_b in drop:
                        continue
                    pred_b = belief_b.predicate.lower() if hasattr(belief_b, 'predicate') and belief_b.predicate else ""
                    if not pred_b:
                        continue

                    # Check if one is a substring of the other
                    if pred_b in pred_a or pred_a in pred_b:
                        # Keep the longer one
                        if len(pred_a) >= len(pred_b):
                            drop.add(stmt_b)
                        else:
                            drop.add(stmt_a)
                        continue

                    # Check word overlap
                    words_a = set(pred_a.split())
                    words_b = set(pred_b.split())
                    if words_a and words_b:
                        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                        if overlap > 0.7:  # 70%+ overlap = near duplicate
                            if len(pred_a) >= len(pred_b):
                                drop.add(stmt_b)
                            else:
                                drop.add(stmt_a)

            for stmt in drop:
                if stmt in beliefs:
                    del beliefs[stmt]
                    removed.append(stmt)

        return removed


class AnswerQualityScorer:
    """
    Rate the system's own answers and track improvement over time.

    Scores based on:
    - Length: too short = bad, reasonable = good
    - Specificity: mentions the actual concept = good
    - Source diversity: uses beliefs + relations + reasoning = good
    - Confidence: hedging too much = bad
    """

    def __init__(self) -> None:
        self._scores: list[tuple[str, float]] = []

    def score(self, question: str, answer: str, beliefs_used: int = 0,
              relations_used: int = 0, systems_fired: int = 0) -> float:
        """Score an answer. 0-1."""
        s = 0.0

        # Length: too short = bad, 50-300 chars = good, too long = ok
        if len(answer) < 20:
            s += 0.0
        elif len(answer) < 50:
            s += 0.2
        elif len(answer) < 300:
            s += 0.4
        else:
            s += 0.3

        # Mentions "don't know" = low score
        if "don't know" in answer.lower() or "teach me" in answer.lower():
            s -= 0.3

        # Specificity: answer mentions words from the question
        stop = {"what", "is", "a", "an", "the", "how", "why", "does", "do"}
        q_words = set(question.lower().split()) - stop
        a_words = set(answer.lower().split())
        if q_words:
            specificity = len(q_words & a_words) / len(q_words)
            s += specificity * 0.3

        # Source diversity
        s += min(0.2, beliefs_used * 0.05)
        s += min(0.1, relations_used * 0.03)

        score = max(0.0, min(1.0, s))
        self._scores.append((question[:40], score))
        return score

    def average(self) -> float:
        if not self._scores:
            return 0.0
        return sum(s for _, s in self._scores) / len(self._scores)

    def trend(self) -> str:
        if len(self._scores) < 10:
            return "Not enough data."
        first = sum(s for _, s in self._scores[:len(self._scores)//2]) / (len(self._scores)//2)
        second = sum(s for _, s in self._scores[len(self._scores)//2:]) / (len(self._scores) - len(self._scores)//2)
        if second > first + 0.05:
            return f"Improving: {first:.0%} → {second:.0%}"
        elif second < first - 0.05:
            return f"Declining: {first:.0%} → {second:.0%}"
        return f"Stable: ~{(first+second)/2:.0%}"

    def report(self) -> str:
        lines = [f"Answer Quality ({len(self._scores)} scored):"]
        lines.append(f"  Average: {self.average():.0%}")
        lines.append(f"  Trend: {self.trend()}")
        if self._scores:
            lines.append(f"  Recent:")
            for q, s in self._scores[-5:]:
                bar = "█" * int(s * 10) + "░" * (10 - int(s * 10))
                lines.append(f"    [{bar}] {s:.0%} {q}")
        return "\n".join(lines)
