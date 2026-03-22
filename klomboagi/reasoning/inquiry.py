"""
Active Inquiry Engine — knowing what you don't know.

A real intelligence doesn't guess when uncertain. It identifies
the specific gap in its knowledge and formulates a question that
would fill that gap. Then it seeks the answer.

The algorithm:
1. Assess confidence in current knowledge
2. Identify the specific uncertainty (what exactly don't I know?)
3. Formulate a question that would resolve the uncertainty
4. Determine the best source for the answer
5. Ask, learn, update

This is metacognition — thinking about thinking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class KnowledgeGap:
    """A specific thing the system doesn't know but needs to."""
    id: str
    domain: str                 # What area: "entity", "relation", "process", "property"
    subject: str                # What it's about: "alligator", "authentication", "deployment"
    question: str               # The specific question
    why_needed: str             # Why knowing this matters for the current task
    confidence_without: float   # How confident we are WITHOUT this knowledge (0-1)
    confidence_with: float      # How confident we'd be WITH it (estimated)
    priority: float             # How urgently we need this (0-1)
    status: str = "open"        # open, asked, answered, stale
    answer: Any = None
    source: str | None = None
    created_at: str = ""
    resolved_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "subject": self.subject,
            "question": self.question,
            "why_needed": self.why_needed,
            "confidence_without": self.confidence_without,
            "confidence_with": self.confidence_with,
            "priority": self.priority,
            "status": self.status,
            "answer": self.answer,
            "source": self.source,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


class InquiryEngine:
    """
    Identifies what the system doesn't know and formulates questions.

    Not a chatbot asking clarifying questions.
    This is an internal process that monitors the system's own
    knowledge state and drives learning.
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self._gaps: list[KnowledgeGap] = []
        self._gap_counter = 0

    def load_gaps(self) -> list[dict]:
        return self.storage.load_json("knowledge_gaps", default=[])

    def save_gaps(self) -> None:
        self.storage.save_json("knowledge_gaps", [g.to_dict() for g in self._gaps])

    # ── Core: Assess what's missing ──

    def assess(self, context: dict) -> list[KnowledgeGap]:
        """
        Given a context (task, episode, world state), identify knowledge gaps.

        This is the metacognitive step: "What do I need to know
        that I don't currently know?"
        """
        gaps = []

        # 1. Check for unknown entities referenced in the task
        task_description = context.get("description", "")
        known_entities = set(context.get("known_entities", []))
        referenced_entities = set(context.get("referenced_entities", []))
        unknown_entities = referenced_entities - known_entities

        for entity in unknown_entities:
            gap = self._create_gap(
                domain="entity",
                subject=entity,
                question=f"What is '{entity}' and what are its properties?",
                why_needed=f"Referenced in task but not in knowledge base",
                confidence_without=0.3,
                confidence_with=0.8,
                priority=0.7,
            )
            gaps.append(gap)

        # 2. Check for missing causal links
        if context.get("expected_outcome") and context.get("proposed_action"):
            action = context["proposed_action"]
            outcome = context["expected_outcome"]
            causal_known = context.get("causal_link_known", False)

            if not causal_known:
                gap = self._create_gap(
                    domain="relation",
                    subject=f"{action} → {outcome}",
                    question=f"Does '{action}' reliably cause '{outcome}'? Under what conditions?",
                    why_needed="About to take action based on assumed causal link",
                    confidence_without=0.4,
                    confidence_with=0.85,
                    priority=0.9,  # High — about to act on this
                )
                gaps.append(gap)

        # 3. Check confidence on the overall task
        task_confidence = context.get("task_confidence", 1.0)
        if task_confidence < 0.5:
            gap = self._create_gap(
                domain="process",
                subject=task_description,
                question=f"What is the correct approach to: {task_description}?",
                why_needed=f"Overall confidence is only {task_confidence:.0%}",
                confidence_without=task_confidence,
                confidence_with=0.8,
                priority=1.0 - task_confidence,
            )
            gaps.append(gap)

        # 4. Check for contradictions in knowledge
        beliefs = context.get("beliefs", [])
        for i, belief_a in enumerate(beliefs):
            for belief_b in beliefs[i + 1:]:
                if self._contradicts(belief_a, belief_b):
                    gap = self._create_gap(
                        domain="relation",
                        subject=f"contradiction: {belief_a} vs {belief_b}",
                        question=f"Which is correct: '{belief_a}' or '{belief_b}'?",
                        why_needed="Contradictory beliefs detected — one must be wrong",
                        confidence_without=0.2,
                        confidence_with=0.9,
                        priority=0.95,  # Contradictions are dangerous
                    )
                    gaps.append(gap)

        self._gaps.extend(gaps)
        self.save_gaps()
        return gaps

    def _contradicts(self, a: str, b: str) -> bool:
        """Simple contradiction detection — same subject, opposite claims."""
        # This is a placeholder — real contradiction detection needs
        # semantic understanding. For now, look for negation patterns.
        a_lower = a.lower()
        b_lower = b.lower()

        # Check if one negates the other
        if f"not {a_lower}" in b_lower or f"not {b_lower}" in a_lower:
            return True
        if a_lower.startswith("no ") and b_lower.replace("no ", "", 1) == a_lower.replace("no ", "", 1):
            return True

        return False

    # ── Resolve a gap ──

    def resolve(self, gap_id: str, answer: Any, source: str) -> KnowledgeGap | None:
        """Record that a knowledge gap has been answered."""
        for gap in self._gaps:
            if gap.id == gap_id:
                gap.answer = answer
                gap.source = source
                gap.status = "answered"
                gap.resolved_at = utc_now()
                self.save_gaps()
                self.storage.event_log.append(
                    "inquiry.resolved",
                    {"gap_id": gap_id, "domain": gap.domain, "subject": gap.subject},
                )
                return gap
        return None

    # ── Prioritize: what to ask first ──

    def prioritize(self) -> list[KnowledgeGap]:
        """Return open gaps sorted by priority."""
        open_gaps = [g for g in self._gaps if g.status == "open"]
        open_gaps.sort(key=lambda g: g.priority, reverse=True)
        return open_gaps

    # ── Determine best source for answer ──

    def suggest_source(self, gap: KnowledgeGap) -> str:
        """
        Where should the system look for the answer?

        Options: memory, world_model, experiment, ask_user, web_search
        """
        if gap.domain == "entity":
            return "memory"  # Check if we've seen this entity before
        elif gap.domain == "relation":
            if "contradict" in gap.subject:
                return "experiment"  # Test which belief is correct
            return "world_model"  # Check causal model
        elif gap.domain == "process":
            return "memory"  # Check if we've done something similar
        elif gap.domain == "property":
            return "world_model"
        else:
            return "ask_user"  # When all else fails, ask

    def _create_gap(self, **kwargs) -> KnowledgeGap:
        self._gap_counter += 1
        gap_id = f"gap_{self._gap_counter}"
        return KnowledgeGap(id=gap_id, created_at=utc_now(), **kwargs)
