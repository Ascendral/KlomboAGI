"""
Metacognition — thinking about thinking.

The system monitors its own reasoning and identifies:
- Confidence calibration: am I too confident or not confident enough?
- Knowledge gaps: what domains am I weakest in?
- Reasoning quality: are my recent answers getting better or worse?
- Learning rate: how fast am I acquiring new knowledge?
- Bias detection: am I over-relying on certain reasoning patterns?

This is what separates a database from a mind.
A database stores. A mind reflects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class CognitionMetrics:
    """Snapshot of how well the system is reasoning."""
    total_questions: int = 0
    answered_from_knowledge: int = 0
    answered_from_search: int = 0
    answered_from_hypothesis: int = 0
    unanswered: int = 0
    surprises: int = 0              # belief contradictions
    corrections_received: int = 0    # human said "wrong"
    hypotheses_formed: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_rejected: int = 0

    @property
    def answer_rate(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return (self.total_questions - self.unanswered) / self.total_questions

    @property
    def knowledge_ratio(self) -> float:
        """How often we answer from knowledge vs search."""
        answered = self.answered_from_knowledge + self.answered_from_search
        if answered == 0:
            return 0.0
        return self.answered_from_knowledge / answered

    @property
    def hypothesis_accuracy(self) -> float:
        tested = self.hypotheses_confirmed + self.hypotheses_rejected
        if tested == 0:
            return 0.0
        return self.hypotheses_confirmed / tested

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            "answer_rate": round(self.answer_rate, 3),
            "knowledge_ratio": round(self.knowledge_ratio, 3),
            "hypothesis_accuracy": round(self.hypothesis_accuracy, 3),
            "surprises": self.surprises,
            "corrections": self.corrections_received,
        }


@dataclass
class DomainAssessment:
    """How well the system knows a particular domain."""
    domain: str
    belief_count: int
    relation_count: int
    avg_confidence: float
    weakest_area: str
    strongest_area: str

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "beliefs": self.belief_count,
            "relations": self.relation_count,
            "avg_confidence": round(self.avg_confidence, 3),
            "weakest": self.weakest_area,
            "strongest": self.strongest_area,
        }


class MetacognitionEngine:
    """
    The system reflecting on its own cognitive state.

    Answers: How well am I doing? Where am I weak? What should I learn next?
    """

    def __init__(self) -> None:
        self.metrics = CognitionMetrics()
        self._reasoning_history: list[dict] = []  # last 100 reasoning events

    def record_question(self, source: str = "knowledge") -> None:
        """Record that a question was asked and how it was answered."""
        self.metrics.total_questions += 1
        if source == "knowledge":
            self.metrics.answered_from_knowledge += 1
        elif source == "search":
            self.metrics.answered_from_search += 1
        elif source == "hypothesis":
            self.metrics.answered_from_hypothesis += 1
            self.metrics.hypotheses_formed += 1
        elif source == "unanswered":
            self.metrics.unanswered += 1

    def record_surprise(self) -> None:
        self.metrics.surprises += 1

    def record_correction(self) -> None:
        self.metrics.corrections_received += 1

    def record_hypothesis_result(self, confirmed: bool) -> None:
        if confirmed:
            self.metrics.hypotheses_confirmed += 1
        else:
            self.metrics.hypotheses_rejected += 1

    def record_reasoning(self, event: dict) -> None:
        """Record a reasoning event for pattern analysis."""
        self._reasoning_history.append(event)
        if len(self._reasoning_history) > 100:
            self._reasoning_history = self._reasoning_history[-100:]

    def assess_domains(self, beliefs: dict, relations) -> list[DomainAssessment]:
        """Assess how well we know each domain."""
        from klomboagi.core.curriculum import CURRICULA

        assessments = []
        for domain, curriculum in CURRICULA.items():
            subjects = {subj.lower() for subj, _ in curriculum}
            domain_beliefs = {s: b for s, b in beliefs.items()
                            if hasattr(b, 'subject') and b.subject in subjects}

            if not domain_beliefs:
                continue

            confidences = [b.truth.confidence for b in domain_beliefs.values()]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            # Find weakest and strongest
            sorted_beliefs = sorted(domain_beliefs.values(),
                                   key=lambda b: b.truth.confidence)
            weakest = sorted_beliefs[0].subject if sorted_beliefs else ""
            strongest = sorted_beliefs[-1].subject if sorted_beliefs else ""

            # Count relations in this domain
            domain_rels = 0
            if hasattr(relations, 'query'):
                for subj in subjects:
                    domain_rels += len(relations.get_all_about(subj))

            assessments.append(DomainAssessment(
                domain=domain,
                belief_count=len(domain_beliefs),
                relation_count=domain_rels,
                avg_confidence=avg_conf,
                weakest_area=weakest,
                strongest_area=strongest,
            ))

        return sorted(assessments, key=lambda a: a.avg_confidence)

    def identify_learning_priorities(self, beliefs: dict, relations) -> list[str]:
        """What should the system learn next? Based on gaps and weakness."""
        priorities = []

        assessments = self.assess_domains(beliefs, relations)

        # Weak domains
        for a in assessments:
            if a.avg_confidence < 0.4:
                priorities.append(f"Study more {a.domain} (confidence: {a.avg_confidence:.0%})")
            if a.relation_count < 5:
                priorities.append(f"Learn relationships in {a.domain} (only {a.relation_count} relations)")

        # Answer rate
        if self.metrics.answer_rate < 0.7 and self.metrics.total_questions > 5:
            priorities.append(f"Improve answer rate (currently {self.metrics.answer_rate:.0%})")

        # Too many corrections
        if self.metrics.corrections_received > 3:
            priorities.append(f"Reduce errors ({self.metrics.corrections_received} corrections received)")

        if not priorities:
            priorities.append("No critical gaps detected — continue learning broadly")

        return priorities

    def reflect(self, beliefs: dict, relations) -> str:
        """
        Full metacognitive reflection — the system thinking about itself.
        """
        lines = ["Metacognitive Reflection:"]

        # Performance
        m = self.metrics
        lines.append(f"\n  Performance:")
        lines.append(f"    Questions answered: {m.total_questions - m.unanswered}/{m.total_questions}")
        lines.append(f"    From knowledge: {m.answered_from_knowledge} | From search: {m.answered_from_search}")
        lines.append(f"    Hypotheses: {m.hypotheses_formed} formed, "
                    f"{m.hypotheses_confirmed} confirmed, {m.hypotheses_rejected} rejected")
        lines.append(f"    Surprises: {m.surprises} | Corrections: {m.corrections_received}")

        # Knowledge state
        lines.append(f"\n  Knowledge state:")
        lines.append(f"    Total beliefs: {len(beliefs)}")
        if hasattr(relations, 'stats'):
            r = relations.stats()
            lines.append(f"    Total relations: {r['total_relations']} ({r.get('derived', 0)} inferred)")

        # Domain assessment
        assessments = self.assess_domains(beliefs, relations)
        if assessments:
            lines.append(f"\n  Domain assessment (weakest first):")
            for a in assessments[:5]:
                lines.append(f"    {a.domain:20s} {a.belief_count} beliefs, "
                           f"conf: {a.avg_confidence:.0%}")

        # Priorities
        priorities = self.identify_learning_priorities(beliefs, relations)
        if priorities:
            lines.append(f"\n  Learning priorities:")
            for p in priorities[:5]:
                lines.append(f"    → {p}")

        return "\n".join(lines)
