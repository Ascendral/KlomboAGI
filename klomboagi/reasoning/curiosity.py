"""
Curiosity Driver — "Let's find out."

Not "I don't know." Not "teach me." Not waiting.
When this system encounters something unknown, it GOES AND FINDS OUT.

The curiosity driver:
1. Monitors the knowledge graph for gaps
2. Prioritizes gaps by relevance to the current task
3. Decides which sense (tool) to use to fill each gap
4. Triggers the sense and processes the result
5. Updates the knowledge graph with what it learned

It's the difference between a passive student and a curious child.
The child doesn't wait to be taught. It picks things up, turns them
over, asks "what's this?", and figures it out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class GapPriority(Enum):
    """How urgently does this gap need to be filled?"""
    CRITICAL = "critical"       # Can't proceed without this
    HIGH = "high"               # Strongly related to current task
    MEDIUM = "medium"           # Related but not blocking
    LOW = "low"                 # Nice to know
    BACKGROUND = "background"   # Learn when idle


class SenseType(Enum):
    """What ability to use to fill a knowledge gap."""
    SEARCH = "search"           # Search the web
    READ = "read"               # Read a file or URL
    EXECUTE = "execute"         # Write and run code to test
    ASK_HUMAN = "ask_human"     # Last resort — ask the person


@dataclass
class KnowledgeGap:
    """Something the system doesn't know but wants to."""
    concept: str                            # What it doesn't know
    context: str                            # Why it needs to know
    priority: GapPriority = GapPriority.MEDIUM
    suggested_sense: SenseType = SenseType.SEARCH
    attempts: int = 0                       # How many times it tried to fill this
    max_attempts: int = 3
    resolved: bool = False
    resolution: str = ""                    # What it learned

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "context": self.context,
            "priority": self.priority.value,
            "suggested_sense": self.suggested_sense.value,
            "attempts": self.attempts,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class CuriosityEvent:
    """Something the curiosity driver decided to do."""
    action: str                 # "investigate", "learn", "verify", "explore"
    target: str                 # What concept
    sense_used: SenseType       # Which ability
    query: str                  # What it asked/searched/read
    result: str                 # What came back
    learned: bool               # Did it actually learn something?
    explanation: str             # What it thinks it learned

    def __repr__(self) -> str:
        status = "✓" if self.learned else "✗"
        return f"[{status}] {self.action}: {self.target} via {self.sense_used.value} → {self.explanation[:80]}"


class CuriosityDriver:
    """
    The engine of self-directed learning.

    Monitors gaps. Decides what to investigate. Uses senses to find out.
    Feeds results back to the knowledge graph.

    The human can:
    - Set priorities ("learn about Python first")
    - Block topics ("don't look up X")
    - Approve/reject what it learned ("no, that's wrong")

    But the default is: GO FIND OUT.
    """

    def __init__(self) -> None:
        self.gaps: list[KnowledgeGap] = []
        self.events: list[CuriosityEvent] = []
        self.blocked_topics: set[str] = set()
        self.priority_overrides: dict[str, GapPriority] = {}

        # Sense callbacks — wired by the system
        self.on_search: Callable[[str], str] | None = None
        self.on_read: Callable[[str], str] | None = None
        self.on_execute: Callable[[str], str] | None = None
        self.on_ask_human: Callable[[str], str] | None = None

    def notice_gap(self, concept: str, context: str = "",
                   priority: GapPriority | None = None) -> KnowledgeGap:
        """
        Notice that we don't know something.
        This is triggered by the reasoning engine when it hits an unknown.
        """
        # Don't duplicate
        for g in self.gaps:
            if g.concept == concept and not g.resolved:
                return g

        # Check for priority override from human
        if priority is None:
            priority = self.priority_overrides.get(concept, GapPriority.MEDIUM)

        # Decide which sense to use
        sense = self._choose_sense(concept, context)

        gap = KnowledgeGap(
            concept=concept,
            context=context,
            priority=priority,
            suggested_sense=sense,
        )
        self.gaps.append(gap)
        return gap

    def _choose_sense(self, concept: str, context: str) -> SenseType:
        """Decide which ability to use to fill a gap."""
        # If it looks like a coding concept, try executing
        code_indicators = {"function", "class", "module", "library", "syntax",
                           "error", "bug", "code", "program", "script",
                           "python", "javascript", "rust", "compile", "import"}
        concept_words = set(concept.lower().split())
        context_words = set(context.lower().split())
        all_words = concept_words | context_words

        if all_words & code_indicators:
            return SenseType.EXECUTE

        # If it looks like a file or URL, read it
        if concept.startswith(("/", "http", "www", "file:")):
            return SenseType.READ

        # Default: search
        return SenseType.SEARCH

    def get_next_gap(self) -> KnowledgeGap | None:
        """Get the highest priority unresolved gap."""
        unresolved = [g for g in self.gaps
                      if not g.resolved
                      and g.attempts < g.max_attempts
                      and g.concept not in self.blocked_topics]

        if not unresolved:
            return None

        # Sort by priority
        priority_order = {
            GapPriority.CRITICAL: 0,
            GapPriority.HIGH: 1,
            GapPriority.MEDIUM: 2,
            GapPriority.LOW: 3,
            GapPriority.BACKGROUND: 4,
        }
        unresolved.sort(key=lambda g: priority_order.get(g.priority, 5))
        return unresolved[0]

    def investigate(self, gap: KnowledgeGap | None = None) -> CuriosityEvent | None:
        """
        Investigate a knowledge gap. This is the "let's find out" moment.

        If no gap is provided, picks the highest priority one.
        Uses the appropriate sense to find information.
        Returns what it found (or didn't find).
        """
        if gap is None:
            gap = self.get_next_gap()
        if gap is None:
            return None

        gap.attempts += 1

        # Build the query based on sense type
        query = self._build_query(gap)

        # Use the sense
        result = self._use_sense(gap.suggested_sense, query)

        # Evaluate if we actually learned something
        learned = bool(result and len(result.strip()) > 10)

        if learned:
            gap.resolved = True
            gap.resolution = result

        event = CuriosityEvent(
            action="investigate",
            target=gap.concept,
            sense_used=gap.suggested_sense,
            query=query,
            result=result,
            learned=learned,
            explanation=result[:200] if learned else f"Could not find information about '{gap.concept}'",
        )
        self.events.append(event)

        # If this sense didn't work, try a different one next time
        if not learned and gap.attempts < gap.max_attempts:
            gap.suggested_sense = self._fallback_sense(gap.suggested_sense)

        return event

    def _build_query(self, gap: KnowledgeGap) -> str:
        """Build an appropriate query for the sense."""
        if gap.suggested_sense == SenseType.SEARCH:
            return f"what is {gap.concept}"
        elif gap.suggested_sense == SenseType.READ:
            return gap.concept  # Assume it's a path or URL
        elif gap.suggested_sense == SenseType.EXECUTE:
            return f"# Test understanding of {gap.concept}\nprint('{gap.concept}')"
        elif gap.suggested_sense == SenseType.ASK_HUMAN:
            if gap.context:
                return f"I'm trying to understand '{gap.concept}' in the context of {gap.context}. Can you help?"
            return f"What is '{gap.concept}'? I want to understand it."
        return gap.concept

    def _use_sense(self, sense: SenseType, query: str) -> str:
        """Use an ability to find information."""
        callbacks = {
            SenseType.SEARCH: self.on_search,
            SenseType.READ: self.on_read,
            SenseType.EXECUTE: self.on_execute,
            SenseType.ASK_HUMAN: self.on_ask_human,
        }
        callback = callbacks.get(sense)
        if callback:
            try:
                return callback(query)
            except Exception as e:
                return f"Error using {sense.value}: {e}"
        return f"No {sense.value} ability available"

    def _fallback_sense(self, failed: SenseType) -> SenseType:
        """If one sense didn't work, try another."""
        fallback_order = [SenseType.SEARCH, SenseType.READ, SenseType.EXECUTE, SenseType.ASK_HUMAN]
        try:
            idx = fallback_order.index(failed)
            return fallback_order[(idx + 1) % len(fallback_order)]
        except ValueError:
            return SenseType.ASK_HUMAN

    def explore(self, max_gaps: int = 5) -> list[CuriosityEvent]:
        """
        Explore mode — investigate multiple gaps autonomously.
        This is what happens when the system is idle.
        "I have free time. What don't I know that I should?"
        """
        events = []
        for _ in range(max_gaps):
            event = self.investigate()
            if event is None:
                break
            events.append(event)
        return events

    # ── Human controls ──

    def prioritize(self, concept: str, priority: GapPriority) -> None:
        """Human says: learn this first."""
        self.priority_overrides[concept] = priority
        # Update existing gap if present
        for g in self.gaps:
            if g.concept == concept:
                g.priority = priority

    def block(self, concept: str) -> None:
        """Human says: don't look this up."""
        self.blocked_topics.add(concept)

    def unblock(self, concept: str) -> None:
        """Human says: ok you can look at this now."""
        self.blocked_topics.discard(concept)

    def correct(self, concept: str, correction: str) -> None:
        """Human says: no, that's wrong. Here's what it actually is."""
        for g in self.gaps:
            if g.concept == concept:
                g.resolved = True
                g.resolution = correction

        event = CuriosityEvent(
            action="corrected",
            target=concept,
            sense_used=SenseType.ASK_HUMAN,
            query=f"Human corrected understanding of '{concept}'",
            result=correction,
            learned=True,
            explanation=f"Human correction: {correction}",
        )
        self.events.append(event)

    # ── Status ──

    def get_gaps(self, include_resolved: bool = False) -> list[KnowledgeGap]:
        if include_resolved:
            return self.gaps
        return [g for g in self.gaps if not g.resolved]

    def get_learning_history(self) -> list[CuriosityEvent]:
        return [e for e in self.events if e.learned]

    def stats(self) -> dict:
        total = len(self.gaps)
        resolved = len([g for g in self.gaps if g.resolved])
        return {
            "total_gaps_noticed": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "investigations": len(self.events),
            "successful_learnings": len([e for e in self.events if e.learned]),
            "blocked_topics": len(self.blocked_topics),
        }
