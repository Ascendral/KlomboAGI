"""
Human Apprenticeship Engine — learn from human teaching, not just evals.

This is the core insight: a human teaching the system is richer than
any eval suite or self-play loop. The system should capture:

1. What the human asked (task framing)
2. What the human changed (corrections)
3. What the human rejected (anti-preferences)
4. What the human emphasized (importance signals)
5. What tradeoff the human preferred (value learning)
6. Whether intervention decreased over time (learning proof)

The metric: after human teaching, does the agent need less correction
on similar future tasks?
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TeachingEvent:
    """One human teaching interaction."""
    event_id: str
    timestamp: str
    event_type: str  # demonstration, correction, explanation, rejection, preference, priority
    task_context: str  # what task was being worked on
    human_input: str  # what the human said/did
    system_state_before: str  # what the system was about to do
    system_state_after: str  # what it did after teaching
    lesson_extracted: str  # what reusable lesson was learned
    lesson_type: str  # rule, pattern, heuristic, value, constraint, anti_pattern
    generality: str  # specific (one task), domain (one domain), general (all domains)
    confidence: float = 0.5


@dataclass
class ApprenticeshipScore:
    """Measures whether human teaching is working."""
    total_teachings: int = 0
    lessons_extracted: int = 0
    lessons_reused: int = 0
    corrections_decreasing: bool = False  # Are corrections going down over time?
    transfer_observed: bool = False  # Did a lesson help on a different task?
    avg_interventions_early: float = 0.0  # Avg interventions in first N tasks
    avg_interventions_late: float = 0.0  # Avg interventions in last N tasks

    def learning_rate(self) -> float:
        """How much of human teaching turned into reusable lessons?"""
        return self.lessons_extracted / self.total_teachings if self.total_teachings > 0 else 0.0

    def reuse_rate(self) -> float:
        """How often are lessons actually used later?"""
        return self.lessons_reused / self.lessons_extracted if self.lessons_extracted > 0 else 0.0

    def intervention_trend(self) -> float:
        """Negative = improving (fewer interventions), positive = getting worse."""
        return self.avg_interventions_late - self.avg_interventions_early

    def summary(self) -> str:
        trend = "improving" if self.intervention_trend() < 0 else "stable" if self.intervention_trend() == 0 else "worsening"
        return (
            f"Apprenticeship: {self.lessons_extracted}/{self.total_teachings} lessons "
            f"({100*self.learning_rate():.0f}% extracted), "
            f"{self.lessons_reused} reused ({100*self.reuse_rate():.0f}%), "
            f"trend: {trend}"
        )


class ApprenticeshipEngine:
    """
    The human-teaching subsystem.

    Not a side feature. A CORE intelligence source.

    The loop:
    1. Human assigns task
    2. Agent attempts
    3. Human redirects/corrects/explains
    4. System extracts reusable lesson
    5. Lesson updates planning/retrieval/scoring
    6. Next similar task changes behavior
    7. Human intervention decreases over time
    """

    def __init__(self, store_dir: str = "datasets/apprenticeship"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[TeachingEvent] = []
        self._counter = 0
        self._load()

    def _load(self) -> None:
        """Load past teaching events."""
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.events.append(TeachingEvent(**data))
            except:
                pass

    def record_demonstration(self, task: str, human_action: str, lesson: str = "") -> TeachingEvent:
        """Human shows how to do something."""
        return self._record("demonstration", task, human_action, "", human_action,
                           lesson or f"When doing '{task[:50]}', the approach is: {human_action[:100]}",
                           "pattern", "domain")

    def record_correction(self, task: str, wrong_action: str, right_action: str,
                         explanation: str = "") -> TeachingEvent:
        """Human says 'no, do it this way instead'."""
        lesson = explanation or f"Don't {wrong_action[:50]}, instead {right_action[:50]}"
        return self._record("correction", task, right_action, wrong_action, right_action,
                           lesson, "anti_pattern", "domain")

    def record_explanation(self, task: str, explanation: str) -> TeachingEvent:
        """Human explains WHY something works a certain way."""
        return self._record("explanation", task, explanation, "", "",
                           explanation, "rule", "general")

    def record_rejection(self, task: str, rejected_output: str, reason: str = "") -> TeachingEvent:
        """Human rejects an output."""
        lesson = reason or f"Output like '{rejected_output[:50]}' is not acceptable"
        return self._record("rejection", task, reason, rejected_output, "",
                           lesson, "constraint", "domain")

    def record_preference(self, task: str, preferred: str, over: str, reason: str = "") -> TeachingEvent:
        """Human prefers one approach over another."""
        lesson = reason or f"Prefer '{preferred[:50]}' over '{over[:50]}'"
        return self._record("preference", task, preferred, over, preferred,
                           lesson, "value", "general")

    def record_priority(self, task: str, priority_signal: str) -> TeachingEvent:
        """Human signals what matters most."""
        return self._record("priority", task, priority_signal, "", "",
                           f"Priority: {priority_signal}", "heuristic", "general")

    def _record(self, event_type, task, human_input, before, after,
                lesson, lesson_type, generality) -> TeachingEvent:
        self._counter += 1
        event = TeachingEvent(
            event_id=f"teach_{self._counter}_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type=event_type,
            task_context=task,
            human_input=human_input,
            system_state_before=before,
            system_state_after=after,
            lesson_extracted=lesson,
            lesson_type=lesson_type,
            generality=generality,
        )
        self.events.append(event)
        self._save(event)
        return event

    def _save(self, event: TeachingEvent) -> None:
        path = self.store_dir / f"{event.event_id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(event), f, indent=2)

    def get_lessons(self, lesson_type: str | None = None, generality: str | None = None) -> list[TeachingEvent]:
        """Get lessons filtered by type and generality."""
        result = self.events
        if lesson_type:
            result = [e for e in result if e.lesson_type == lesson_type]
        if generality:
            result = [e for e in result if e.generality == generality]
        return result

    def get_relevant_lessons(self, task_description: str) -> list[TeachingEvent]:
        """Find lessons relevant to a task (by keyword overlap)."""
        words = set(task_description.lower().split())
        scored = []
        for event in self.events:
            lesson_words = set(event.lesson_extracted.lower().split())
            overlap = len(words & lesson_words)
            if overlap > 1:
                scored.append((overlap, event))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:5]]

    def get_score(self) -> ApprenticeshipScore:
        """Compute the apprenticeship score."""
        score = ApprenticeshipScore(
            total_teachings=len(self.events),
            lessons_extracted=len([e for e in self.events if e.lesson_extracted]),
        )
        # Count reuse (lessons that match other tasks)
        for e in self.events:
            for other in self.events:
                if e.event_id != other.event_id and e.lesson_type == other.lesson_type:
                    task_overlap = len(set(e.task_context.lower().split()) &
                                      set(other.task_context.lower().split()))
                    if task_overlap > 2:
                        score.lessons_reused += 1
                        break
        return score
