"""
Auto-Refresh — re-read Wikipedia for low-confidence concepts.

If a concept has:
  - Low confidence (< 0.4)
  - Few facts (< 3 beliefs)
  - Been asked about but answered poorly

→ Automatically go re-read Wikipedia to strengthen knowledge.

This is the system MAINTAINING its own knowledge base.
"""

from __future__ import annotations


class AutoRefresher:
    """
    Identifies concepts that need refreshing and re-reads them.
    """

    CONFIDENCE_THRESHOLD = 0.4
    MIN_FACTS_THRESHOLD = 3

    def __init__(self, genesis) -> None:
        self.g = genesis

    def find_stale(self) -> list[tuple[str, str]]:
        """Find concepts that need refreshing. Returns (concept, reason)."""
        stale = []

        # Group beliefs by subject, find low-confidence ones
        from collections import defaultdict
        by_subject: dict[str, list] = defaultdict(list)
        for stmt, belief in self.g.base._beliefs.items():
            if hasattr(belief, 'subject') and belief.subject:
                by_subject[belief.subject].append(belief)

        for subject, beliefs_list in by_subject.items():
            if len(subject) < 3:
                continue

            # Average confidence
            confs = [b.truth.confidence for b in beliefs_list if hasattr(b, 'truth')]
            avg_conf = sum(confs) / len(confs) if confs else 0

            if avg_conf < self.CONFIDENCE_THRESHOLD and len(beliefs_list) < self.MIN_FACTS_THRESHOLD:
                stale.append((subject, f"low confidence ({avg_conf:.0%}) and few facts ({len(beliefs_list)})"))

        return stale[:20]

    def refresh(self, max_concepts: int = 5) -> list[str]:
        """Refresh the most stale concepts by re-reading Wikipedia."""
        stale = self.find_stale()
        refreshed = []

        for concept, reason in stale[:max_concepts]:
            try:
                result = self.g.read_and_learn(concept)
                if "Could not read" not in result:
                    refreshed.append(f"Refreshed {concept}: {reason}")
            except Exception:
                pass

        return refreshed

    def auto_maintain(self) -> str:
        """Run full maintenance: refresh stale, strengthen confirmed, clean garbage."""
        results = []

        # 1. Refresh stale concepts
        refreshed = self.refresh(max_concepts=3)
        results.extend(refreshed)

        # 2. Cross-confirm beliefs
        if hasattr(self.g, 'belief_strengthener'):
            confirmed = self.g.belief_strengthener.cross_confirm()
            if confirmed:
                results.append(f"Cross-confirmed {len(confirmed)} beliefs")

        # 3. Clean garbage
        cleaned = self.g.cleanup_memory()
        results.append(cleaned)

        return "\n".join(results) if results else "Nothing to maintain."
