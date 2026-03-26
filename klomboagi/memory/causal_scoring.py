"""
Causal Memory Scoring — not "memory exists" but "memory changed the decision
and improved the result."

Every memory retrieval is logged with:
1. What was retrieved
2. Whether it changed the decision
3. Whether the changed decision improved the outcome

This is the difference between "we have memory" and "memory matters."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class MemoryEvent:
    """One memory retrieval event."""
    event_id: str
    timestamp: str
    task_id: str
    phase: str                          # which phase triggered retrieval
    query: str                          # what was searched for
    retrieved: list[str]                # what came back
    decision_before: str                # what the system was going to do
    decision_after: str                 # what it actually did
    decision_changed: bool              # did memory change the decision?
    outcome: str = ""                   # success/failure of the action
    outcome_improved: bool = False      # did the changed decision help?

    def was_useful(self) -> bool:
        return self.decision_changed and self.outcome_improved


@dataclass
class CausalMemoryScore:
    """Aggregate score for memory usefulness."""
    total_retrievals: int = 0
    decisions_changed: int = 0
    outcomes_improved: int = 0
    outcomes_worsened: int = 0
    no_effect: int = 0

    def usefulness_rate(self) -> float:
        return self.outcomes_improved / self.total_retrievals if self.total_retrievals > 0 else 0.0

    def change_rate(self) -> float:
        return self.decisions_changed / self.total_retrievals if self.total_retrievals > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Memory Score: {self.outcomes_improved}/{self.total_retrievals} useful "
            f"({100*self.usefulness_rate():.0f}%), "
            f"{self.decisions_changed} decisions changed, "
            f"{self.outcomes_worsened} worsened"
        )


class CausalMemoryTracker:
    """
    Tracks whether memory retrievals actually help.

    Usage:
        tracker = CausalMemoryTracker()

        # Before making a decision, record what you'd do without memory
        event = tracker.start_retrieval(task_id, phase, query, decision_before)

        # After retrieving memory, record what you actually do
        tracker.record_decision(event, decision_after, retrieved)

        # After seeing the outcome, record whether it helped
        tracker.record_outcome(event, outcome, improved)

        # Get the score
        score = tracker.get_score()
    """

    def __init__(self, store_path: str = "datasets/memory_events"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.events: list[MemoryEvent] = []
        self._counter = 0

    def start_retrieval(self, task_id: str, phase: str, query: str,
                        decision_before: str) -> MemoryEvent:
        """Start tracking a memory retrieval event."""
        self._counter += 1
        event = MemoryEvent(
            event_id=f"mem_{self._counter}_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_id=task_id,
            phase=phase,
            query=query,
            retrieved=[],
            decision_before=decision_before,
            decision_after=decision_before,  # Same until changed
            decision_changed=False,
        )
        self.events.append(event)
        return event

    def record_decision(self, event: MemoryEvent, decision_after: str,
                        retrieved: list[str]) -> None:
        """Record what the system decided after retrieving memory."""
        event.retrieved = retrieved
        event.decision_after = decision_after
        event.decision_changed = (decision_after != event.decision_before)

    def record_outcome(self, event: MemoryEvent, outcome: str,
                       improved: bool) -> None:
        """Record whether the memory-influenced decision helped."""
        event.outcome = outcome
        event.outcome_improved = improved
        self._save_event(event)

    def get_score(self) -> CausalMemoryScore:
        """Compute aggregate score."""
        score = CausalMemoryScore(total_retrievals=len(self.events))
        for e in self.events:
            if e.decision_changed:
                score.decisions_changed += 1
                if e.outcome_improved:
                    score.outcomes_improved += 1
                elif e.outcome and not e.outcome_improved:
                    score.outcomes_worsened += 1
            else:
                score.no_effect += 1
        return score

    def get_useful_memories(self) -> list[MemoryEvent]:
        """Get memories that actually helped."""
        return [e for e in self.events if e.was_useful()]

    def get_harmful_memories(self) -> list[MemoryEvent]:
        """Get memories that made things worse."""
        return [e for e in self.events if e.decision_changed and not e.outcome_improved and e.outcome]

    def _save_event(self, event: MemoryEvent) -> None:
        """Persist a completed memory event."""
        path = self.store_path / f"{event.event_id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(event), f, indent=2)

    def load_all(self) -> list[MemoryEvent]:
        """Load all persisted memory events."""
        events = []
        for path in sorted(self.store_path.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                events.append(MemoryEvent(**data))
            except:
                pass
        return events

    def lifetime_score(self) -> CausalMemoryScore:
        """Score across ALL persisted events, not just current session."""
        all_events = self.load_all()
        score = CausalMemoryScore(total_retrievals=len(all_events))
        for e in all_events:
            if e.decision_changed:
                score.decisions_changed += 1
                if e.outcome_improved:
                    score.outcomes_improved += 1
                elif e.outcome:
                    score.outcomes_worsened += 1
            else:
                score.no_effect += 1
        return score
