"""
Dual Process Theory — System 1 (fast) vs System 2 (slow).

From Kahneman's "Thinking Fast and Slow" + CLARION architecture.

System 1: Fast, automatic, uses compiled chunks and direct retrieval.
  - Fires instantly from cached knowledge
  - Low computational cost
  - Can be wrong (heuristic)

System 2: Slow, deliberate, uses full reasoning chains.
  - Fires the CognitionLoop, ReasoningEngine, inference
  - High computational cost
  - More likely correct

The key insight: System 1 fires FIRST. If its answer is confident
enough, System 2 never fires. If System 1 is uncertain or produces
a conflict, System 2 takes over.

This is how expertise works: experts use System 1 (intuition built
from compiled experience). Novices use System 2 (deliberate reasoning).
As KlomboAGI compiles more chunks, it shifts from System 2 to System 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DualProcessResult:
    """Result from dual process reasoning."""
    answer: str
    system_used: int          # 1 or 2
    system1_answer: str       # what System 1 produced
    system1_confidence: float
    system2_answer: str       # what System 2 produced (if used)
    system2_confidence: float
    reason: str               # why this system was chosen

    def to_dict(self) -> dict:
        return {
            "answer": self.answer[:100],
            "system_used": self.system_used,
            "s1_confidence": round(self.system1_confidence, 3),
            "s2_confidence": round(self.system2_confidence, 3),
            "reason": self.reason,
        }


class DualProcess:
    """
    System 1 (fast/compiled) vs System 2 (slow/deliberate).

    System 1 fires first. If confident, done.
    If not, System 2 takes over.
    """

    S1_THRESHOLD = 0.6  # System 1 confidence threshold to skip System 2

    def __init__(self, chunker, beliefs, relations, generator) -> None:
        self.chunker = chunker
        self.beliefs = beliefs
        self.relations = relations
        self.generator = generator

    def think(self, query: str, s2_func=None) -> DualProcessResult:
        """
        Think about a query using dual process.

        s2_func: optional callable for System 2 reasoning.
                 If not provided, System 2 is skipped.
        """
        # System 1: fast, automatic
        s1_answer, s1_conf = self._system1(query)

        # If System 1 is confident enough, use it
        if s1_conf >= self.S1_THRESHOLD:
            return DualProcessResult(
                answer=s1_answer,
                system_used=1,
                system1_answer=s1_answer,
                system1_confidence=s1_conf,
                system2_answer="",
                system2_confidence=0,
                reason=f"System 1 confident ({s1_conf:.0%}) — instant answer from compiled knowledge",
            )

        # System 2: slow, deliberate
        s2_answer = ""
        s2_conf = 0.0
        if s2_func:
            s2_answer = s2_func(query)
            s2_conf = 0.7  # System 2 is generally more reliable
        else:
            # Use generator as fallback System 2
            stop = {"is", "a", "an", "the", "what", "who", "how", "why", "where",
                    "when", "are", "was", "do", "does", "can", "about", "tell", "me"}
            terms = [w for w in query.lower().split() if w not in stop and len(w) > 2]
            for term in terms:
                exp = self.generator.explain(term)
                if exp.novel and exp.relations_used > 0:
                    s2_answer = exp.text
                    s2_conf = 0.5 + exp.relations_used * 0.05
                    break

        # Choose the better answer
        if s2_conf > s1_conf:
            return DualProcessResult(
                answer=s2_answer,
                system_used=2,
                system1_answer=s1_answer,
                system1_confidence=s1_conf,
                system2_answer=s2_answer,
                system2_confidence=s2_conf,
                reason=f"System 2 overrode System 1 ({s2_conf:.0%} vs {s1_conf:.0%})",
            )

        # System 1 wins by default if System 2 didn't produce anything better
        if s1_answer:
            return DualProcessResult(
                answer=s1_answer,
                system_used=1,
                system1_answer=s1_answer,
                system1_confidence=s1_conf,
                system2_answer=s2_answer,
                system2_confidence=s2_conf,
                reason=f"System 1 default ({s1_conf:.0%}) — System 2 didn't improve",
            )

        return DualProcessResult(
            answer=s2_answer or "I need to learn more about this.",
            system_used=2 if s2_answer else 0,
            system1_answer=s1_answer,
            system1_confidence=s1_conf,
            system2_answer=s2_answer,
            system2_confidence=s2_conf,
            reason="Neither system produced a confident answer",
        )

    def _system1(self, query: str) -> tuple[str, float]:
        """
        System 1: fast retrieval from chunks + direct beliefs.

        Returns (answer, confidence).
        """
        # Check compiled chunks first (fastest)
        chunks = self.chunker.lookup(query)
        if chunks:
            best = max(chunks, key=lambda c: c.confidence)
            return (f"{best.conclusion}.", best.confidence)

        # Direct belief lookup (fast)
        stop = {"is", "a", "an", "the", "what", "who", "how", "why"}
        terms = [w for w in query.lower().split() if w not in stop and len(w) > 2]
        for term in terms:
            for stmt, belief in self.beliefs.items():
                if hasattr(belief, 'subject') and belief.subject == term:
                    if belief.predicate and len(belief.predicate) < 80:
                        conf = belief.truth.confidence if hasattr(belief, 'truth') else 0.5
                        return (f"{belief.subject} is {belief.predicate}.", conf)

        return ("", 0.0)
