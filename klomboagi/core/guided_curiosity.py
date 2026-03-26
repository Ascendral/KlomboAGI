"""
Guided Curiosity Architecture — the core AGI algorithm.

Three algorithms working together:
1. LEARNING: how experience changes internal structure
2. MOTIVATION: why it seeks new structure (the "zest")
3. CONTROL: how it chooses actions under uncertainty

The drive function:
  next_focus = argmax(
    knowledge_gain +
    future_usefulness +
    teacher_priority +
    transfer_potential -
    risk -
    cost
  )

The loop:
  observe → infer → hypothesize → store → test → measure → reinforce/revise → question

Core structures:
  episode_memory — what happened
  teaching_memory — what the human was teaching
  concept_graph — abstractions the agent thinks are true
  procedure_library — how to do things
  anti_pattern_library — what to avoid
  curiosity_queue — what it knows it doesn't know
  teacher_model — what this human values and prioritizes
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import Counter


@dataclass
class CuriosityTarget:
    """Something the system wants to understand better."""
    concept: str
    uncertainty: float  # 0-1, how uncertain we are
    knowledge_gain: float  # expected information gain
    future_usefulness: float  # how useful this would be to learn
    teacher_priority: float  # how much the human cares about this
    transfer_potential: float  # would this help in other domains?
    risk: float  # cost of getting it wrong
    cost: float  # effort to investigate
    source: str  # "gap", "failure", "teacher", "curiosity"
    created_at: str = ""

    def drive_score(self) -> float:
        """The motivation score — should we investigate this?"""
        return (
            self.knowledge_gain * 2.0 +
            self.future_usefulness * 1.5 +
            self.teacher_priority * 3.0 +  # Teacher priority weighs most
            self.transfer_potential * 1.0 -
            self.risk * 2.0 -
            self.cost * 0.5
        )


@dataclass
class TeacherModel:
    """What we know about what the human values."""
    priorities: dict[str, float] = field(default_factory=dict)  # topic → importance
    preferences: list[tuple[str, str]] = field(default_factory=list)  # (preferred, over)
    corrections: list[str] = field(default_factory=list)  # things human corrected
    demonstrations: list[str] = field(default_factory=list)  # approaches human showed
    rejections: list[str] = field(default_factory=list)  # things human rejected
    patience: float = 1.0  # how patient the human is (decreases with repeated corrections)
    trust: float = 0.5  # how much the human trusts the system (increases with good work)

    def update_from_correction(self, topic: str) -> None:
        """Human corrected us — they care about this."""
        self.corrections.append(topic)
        self.priorities[topic] = min(1.0, self.priorities.get(topic, 0.5) + 0.1)
        self.patience = max(0.1, self.patience - 0.05)

    def update_from_success(self, topic: str) -> None:
        """We got it right — trust increases."""
        self.trust = min(1.0, self.trust + 0.05)
        self.patience = min(1.0, self.patience + 0.02)

    def update_from_demonstration(self, approach: str) -> None:
        """Human showed us how."""
        self.demonstrations.append(approach)

    def update_from_rejection(self, rejected: str) -> None:
        """Human rejected our output."""
        self.rejections.append(rejected)
        self.patience = max(0.1, self.patience - 0.1)

    def get_priority(self, topic: str) -> float:
        """How much does the human care about this topic?"""
        # Direct match
        if topic in self.priorities:
            return self.priorities[topic]
        # Keyword overlap with known priorities
        topic_words = set(topic.lower().split())
        best = 0.0
        for known, score in self.priorities.items():
            overlap = len(topic_words & set(known.lower().split()))
            if overlap > 0:
                best = max(best, score * overlap / max(len(topic_words), 1))
        return best


@dataclass
class Concept:
    """An abstraction the system believes is true."""
    name: str
    description: str
    confidence: float  # 0-1
    evidence_count: int = 0
    transfer_count: int = 0  # times this helped in a different domain
    last_tested: str = ""
    domains: list[str] = field(default_factory=list)


class GuidedCuriosityEngine:
    """
    The core AGI algorithm.

    Not a task executor. Not a benchmark solver.
    A system that actively seeks understanding, guided by its human.
    """

    def __init__(self) -> None:
        self.curiosity_queue: list[CuriosityTarget] = []
        self.teacher_model = TeacherModel()
        self.concepts: dict[str, Concept] = {}
        self.episode_count = 0

        # Callbacks
        self.on_investigate: Callable[[CuriosityTarget], Any] | None = None
        self.on_ask_teacher: Callable[[str], str] | None = None

    # === Motivation System ===

    def what_should_i_learn_next(self) -> CuriosityTarget | None:
        """Pick the highest-drive curiosity target."""
        active = [t for t in self.curiosity_queue if t.uncertainty > 0.1]
        if not active:
            return None
        active.sort(key=lambda t: t.drive_score(), reverse=True)
        return active[0]

    def notice_gap(self, concept: str, context: str = "", source: str = "gap") -> CuriosityTarget:
        """Notice something we don't understand."""
        # Check if already in queue
        for t in self.curiosity_queue:
            if t.concept == concept:
                t.uncertainty = min(1.0, t.uncertainty + 0.1)
                return t

        target = CuriosityTarget(
            concept=concept,
            uncertainty=0.8,
            knowledge_gain=0.5,
            future_usefulness=0.5,
            teacher_priority=self.teacher_model.get_priority(concept),
            transfer_potential=0.3,
            risk=0.1,
            cost=0.2,
            source=source,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.curiosity_queue.append(target)
        return target

    def notice_failure(self, task: str, error: str) -> CuriosityTarget:
        """A failure happened — we should learn from it."""
        target = self.notice_gap(
            f"why_{task}_failed",
            context=error,
            source="failure",
        )
        target.knowledge_gain = 0.8  # Failures are very informative
        target.future_usefulness = 0.7
        return target

    def notice_teacher_emphasis(self, topic: str, strength: float = 1.0) -> CuriosityTarget:
        """The human emphasized something — prioritize it."""
        self.teacher_model.priorities[topic] = min(1.0, strength)
        target = self.notice_gap(topic, source="teacher")
        target.teacher_priority = strength
        target.knowledge_gain = 0.7
        return target

    # === Learning System ===

    def learn_concept(self, name: str, description: str, domain: str,
                      confidence: float = 0.5) -> Concept:
        """Form or strengthen a concept."""
        if name in self.concepts:
            c = self.concepts[name]
            c.evidence_count += 1
            c.confidence = min(1.0, c.confidence + 0.1)
            if domain not in c.domains:
                c.domains.append(domain)
                c.transfer_count += 1
            return c

        concept = Concept(
            name=name,
            description=description,
            confidence=confidence,
            evidence_count=1,
            domains=[domain],
        )
        self.concepts[name] = concept

        # Reduce uncertainty on related curiosity targets
        for t in self.curiosity_queue:
            if name.lower() in t.concept.lower() or t.concept.lower() in name.lower():
                t.uncertainty = max(0.0, t.uncertainty - 0.3)

        return concept

    def learn_from_teacher(self, event_type: str, content: str, task: str = "") -> None:
        """Process a teaching event."""
        if event_type == "correction":
            self.teacher_model.update_from_correction(task)
            self.learn_concept(f"avoid_{content[:30]}", f"Don't: {content}", "general", 0.7)
        elif event_type == "demonstration":
            self.teacher_model.update_from_demonstration(content)
            self.learn_concept(f"approach_{content[:30]}", f"Do: {content}", "general", 0.6)
        elif event_type == "explanation":
            self.learn_concept(f"rule_{content[:30]}", content, "general", 0.8)
        elif event_type == "rejection":
            self.teacher_model.update_from_rejection(content)
            self.learn_concept(f"reject_{content[:30]}", f"Not acceptable: {content}", "general", 0.7)
        elif event_type == "success":
            self.teacher_model.update_from_success(task)

    def test_concept(self, name: str, succeeded: bool) -> None:
        """Test whether a concept holds up in practice."""
        if name in self.concepts:
            c = self.concepts[name]
            c.last_tested = time.strftime("%Y-%m-%dT%H:%M:%S")
            if succeeded:
                c.confidence = min(1.0, c.confidence + 0.1)
                c.evidence_count += 1
            else:
                c.confidence = max(0.0, c.confidence - 0.2)

    # === Control System ===

    def should_ask_teacher(self, task: str) -> bool:
        """Should we ask the human for help?"""
        # Ask if uncertainty is high AND teacher is patient
        relevant = self.get_relevant_concepts(task)
        if not relevant:
            return self.teacher_model.patience > 0.3  # Ask if teacher isn't annoyed

        avg_confidence = sum(c.confidence for c in relevant) / len(relevant)
        return avg_confidence < 0.4 and self.teacher_model.patience > 0.3

    def should_explore(self) -> bool:
        """Should we spend time exploring curiosity targets?"""
        high_uncertainty = [t for t in self.curiosity_queue if t.uncertainty > 0.5]
        return len(high_uncertainty) > 3  # Too many unknowns — explore

    def get_relevant_concepts(self, task: str) -> list[Concept]:
        """Find concepts relevant to a task."""
        words = set(task.lower().split())
        relevant = []
        for name, concept in self.concepts.items():
            concept_words = set(concept.description.lower().split())
            if len(words & concept_words) > 1:
                relevant.append(concept)
        return sorted(relevant, key=lambda c: c.confidence, reverse=True)

    # === The Main Loop ===

    def step(self, task: str | None = None) -> dict:
        """
        One step of the guided curiosity loop.

        If task is given: work on it, learn from the outcome.
        If no task: explore the highest-priority curiosity target.
        """
        self.episode_count += 1

        if task:
            # Working on a task
            relevant = self.get_relevant_concepts(task)
            return {
                "mode": "task",
                "task": task,
                "relevant_concepts": len(relevant),
                "should_ask": self.should_ask_teacher(task),
                "teacher_trust": self.teacher_model.trust,
                "teacher_patience": self.teacher_model.patience,
            }
        else:
            # Exploring
            target = self.what_should_i_learn_next()
            if target:
                return {
                    "mode": "explore",
                    "target": target.concept,
                    "drive_score": target.drive_score(),
                    "uncertainty": target.uncertainty,
                    "source": target.source,
                }
            return {"mode": "idle", "message": "Nothing to explore — waiting for task or teaching"}

    # === Stats ===

    def stats(self) -> dict:
        return {
            "concepts": len(self.concepts),
            "curiosity_targets": len(self.curiosity_queue),
            "high_uncertainty": len([t for t in self.curiosity_queue if t.uncertainty > 0.5]),
            "episodes": self.episode_count,
            "teacher_trust": self.teacher_model.trust,
            "teacher_patience": self.teacher_model.patience,
            "teacher_corrections": len(self.teacher_model.corrections),
            "avg_confidence": sum(c.confidence for c in self.concepts.values()) / len(self.concepts) if self.concepts else 0,
        }
