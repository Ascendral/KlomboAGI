"""
Belief Index — O(1) lookup by subject instead of O(n) linear scan.

Loading 10k beliefs is fine. Searching them 50 times per question
with linear scan = slow. Index by subject = instant.
"""

from __future__ import annotations

from collections import defaultdict


class BeliefIndex:
    """
    Fast index over beliefs by subject.

    Maintains a dict: subject → list of (statement, belief) pairs.
    Rebuild when beliefs change significantly.
    """

    def __init__(self) -> None:
        self._by_subject: dict[str, list[tuple[str, object]]] = defaultdict(list)
        self._size: int = 0

    def build(self, beliefs: dict) -> None:
        """Build the index from all beliefs."""
        self._by_subject.clear()
        for stmt, belief in beliefs.items():
            if hasattr(belief, 'subject') and belief.subject:
                self._by_subject[belief.subject.lower()].append((stmt, belief))
        self._size = len(beliefs)

    def get(self, subject: str) -> list[tuple[str, object]]:
        """Get all beliefs about a subject. O(1)."""
        return self._by_subject.get(subject.lower(), [])

    def search(self, query_words: set[str], max_results: int = 10) -> list[tuple[str, object]]:
        """Find beliefs matching any query word. Fast."""
        results = []
        for word in query_words:
            for stmt, belief in self._by_subject.get(word, []):
                if len(results) < max_results:
                    results.append((stmt, belief))
        return results

    def subjects(self) -> set[str]:
        """All indexed subjects."""
        return set(self._by_subject.keys())

    def needs_rebuild(self, current_size: int) -> bool:
        """Does the index need rebuilding?"""
        return abs(current_size - self._size) > 100
