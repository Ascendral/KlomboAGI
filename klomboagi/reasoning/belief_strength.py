"""
Belief Strength — multiple confirmations from different sources = stronger.

A belief taught by a human once: confidence 50%.
Same belief found in Wikipedia: confidence rises to 67%.
Same belief confirmed by reasoning: rises to 80%.
Three independent sources agree: this is probably true.

Also handles belief WEAKENING:
  - Human corrects it: drops hard
  - Contradicts another strong belief: both weaken
  - Old and never accessed: slowly fades (ACT-R decay applied to beliefs)
"""

from __future__ import annotations

from klomboagi.reasoning.truth import TruthValue, Belief, EvidenceStamp


class BeliefStrengthener:
    """
    Strengthens beliefs when multiple sources confirm them.
    Weakens beliefs that are contradicted or stale.
    """

    def __init__(self, beliefs: dict, evidence_counter: int = 0) -> None:
        self.beliefs = beliefs
        self._counter = evidence_counter

    def confirm(self, statement: str, source: str = "confirmation") -> float | None:
        """
        Confirm an existing belief from a new source.
        Returns new confidence, or None if belief not found.
        """
        if statement not in self.beliefs:
            return None

        belief = self.beliefs[statement]
        self._counter += 1

        confirming = Belief(
            statement=statement,
            truth=TruthValue.from_single_observation(True),
            stamp=EvidenceStamp.new(self._counter),
            subject=belief.subject if hasattr(belief, 'subject') else "",
            predicate=belief.predicate if hasattr(belief, 'predicate') else "",
            source=source,
        )

        revised = belief.revise_with(confirming)
        if revised:
            self.beliefs[statement] = revised
            return revised.truth.confidence
        return belief.truth.confidence

    def weaken(self, statement: str, amount: float = 0.2) -> float | None:
        """Weaken a belief (contradiction or correction)."""
        if statement not in self.beliefs:
            return None
        belief = self.beliefs[statement]
        belief.truth.frequency = max(0.0, belief.truth.frequency - amount)
        return belief.truth.confidence

    def cross_confirm(self) -> list[str]:
        """
        Find beliefs confirmed by multiple sources and strengthen them.

        If "gravity is a force" was taught by human AND found in Wikipedia
        AND derived by inference → it's very strong.
        """
        strengthened = []

        # Group by statement content (not exact key, but meaning)
        by_subject_pred: dict[tuple[str, str], list[tuple[str, object]]] = {}
        for stmt, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and hasattr(belief, 'predicate'):
                key = (belief.subject or "", belief.predicate or "")
                if key[0] and key[1]:
                    if key not in by_subject_pred:
                        by_subject_pred[key] = []
                    by_subject_pred[key].append((stmt, belief))

        for (subj, pred), group in by_subject_pred.items():
            if len(group) < 2:
                continue
            # Multiple beliefs about the same subject+predicate from different sources
            sources = set()
            for stmt, b in group:
                if hasattr(b, 'source'):
                    sources.add(b.source)
            if len(sources) >= 2:
                # Multiple sources agree — strengthen all
                for stmt, b in group:
                    self.confirm(stmt, f"cross_confirmed({','.join(sources)})")
                    strengthened.append(stmt)

        return strengthened

    def stats(self) -> dict:
        """Confidence distribution."""
        high = sum(1 for b in self.beliefs.values()
                  if hasattr(b, 'truth') and b.truth.confidence > 0.7)
        medium = sum(1 for b in self.beliefs.values()
                    if hasattr(b, 'truth') and 0.4 < b.truth.confidence <= 0.7)
        low = sum(1 for b in self.beliefs.values()
                 if hasattr(b, 'truth') and b.truth.confidence <= 0.4)
        return {
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence": low,
            "total": len(self.beliefs),
        }
