"""
Self-Testing — the system verifies its own beliefs.

Periodically scans beliefs and tests what it can:
- Mathematical claims get verified by the compute engine
- Contradictions between beliefs get flagged
- Low-confidence beliefs get marked for investigation
- Stale beliefs (old, untested) get marked for refresh

This is metacognition: the system THINKS about its own THINKING.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.reasoning.compute import ComputeEngine
from klomboagi.reasoning.truth import Belief


@dataclass
class TestResult:
    """Result of testing a belief."""
    statement: str
    verdict: str          # "verified", "contradicted", "untestable", "low_confidence", "stale"
    details: str = ""
    old_confidence: float = 0.0
    new_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "verdict": self.verdict,
            "details": self.details,
            "old_confidence": round(self.old_confidence, 3),
            "new_confidence": round(self.new_confidence, 3),
        }


@dataclass
class AuditReport:
    """Full report from a self-test audit."""
    total_beliefs: int
    tested: int
    verified: int
    contradicted: int
    low_confidence: int
    contradictions: list[tuple[str, str]]  # pairs of contradicting beliefs
    results: list[TestResult]

    def summary(self) -> str:
        lines = [
            f"Self-Test Audit: {self.total_beliefs} beliefs",
            f"  Tested: {self.tested}",
            f"  Verified: {self.verified}",
            f"  Contradicted: {self.contradicted}",
            f"  Low confidence: {self.low_confidence}",
        ]
        if self.contradictions:
            lines.append(f"\nContradictions found:")
            for a, b in self.contradictions[:10]:
                lines.append(f"  ! {a}")
                lines.append(f"    vs {b}")
        return "\n".join(lines)


class SelfTester:
    """
    Tests the system's own beliefs for consistency and correctness.
    """

    def __init__(self, compute: ComputeEngine) -> None:
        self.compute = compute

    def audit(self, beliefs: dict[str, Belief]) -> AuditReport:
        """
        Run a full audit of all beliefs.

        Tests:
        1. Mathematical claims against the compute engine
        2. Internal contradictions between beliefs
        3. Low-confidence beliefs that need more evidence
        """
        results = []
        verified = 0
        contradicted = 0
        low_confidence = 0
        contradictions = []

        belief_list = list(beliefs.values())

        for belief in belief_list:
            # Test 1: Can we verify computationally?
            comp_result = self.compute.verify_fact(belief.statement)
            if comp_result is not None:
                if comp_result.result is True:
                    results.append(TestResult(
                        belief.statement, "verified",
                        "Computationally confirmed",
                        belief.truth.confidence, belief.truth.confidence,
                    ))
                    verified += 1
                else:
                    results.append(TestResult(
                        belief.statement, "contradicted",
                        f"Computation says otherwise: {comp_result.steps}",
                        belief.truth.confidence, 0.0,
                    ))
                    contradicted += 1
                continue

            # Test 2: Low confidence?
            if belief.truth.confidence < 0.3:
                results.append(TestResult(
                    belief.statement, "low_confidence",
                    f"Only {belief.truth.confidence:.0%} confident",
                    belief.truth.confidence, belief.truth.confidence,
                ))
                low_confidence += 1

        # Test 3: Check for contradictions between beliefs
        for i, b1 in enumerate(belief_list):
            if not hasattr(b1, 'subject') or not b1.subject:
                continue
            for b2 in belief_list[i+1:]:
                if not hasattr(b2, 'subject') or not b2.subject:
                    continue
                if b1.subject == b2.subject and b1.predicate and b2.predicate:
                    # Same subject, different predicates — check if contradictory
                    if self._might_contradict(b1.predicate, b2.predicate):
                        contradictions.append((b1.statement, b2.statement))

        return AuditReport(
            total_beliefs=len(beliefs),
            tested=len(results),
            verified=verified,
            contradicted=contradicted,
            low_confidence=low_confidence,
            contradictions=contradictions,
            results=results,
        )

    def _might_contradict(self, pred_a: str, pred_b: str) -> bool:
        """Quick check if two predicates might contradict."""
        a = pred_a.lower().strip()
        b = pred_b.lower().strip()

        # Explicit negation
        if a.startswith("not ") and a[4:].strip() == b:
            return True
        if b.startswith("not ") and b[4:].strip() == a:
            return True

        return False
