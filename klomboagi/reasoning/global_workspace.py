"""
Global Workspace — competitive broadcast to all subsystems.

From LIDA / Global Workspace Theory (Baars, 1988):

Multiple subsystems compete for attention. ONE winner gets
broadcast to ALL other subsystems simultaneously. This is
the bottleneck that makes cognition efficient.

Without it: every subsystem processes everything → chaos.
With it: only the most relevant signal gets broadcast →
         all subsystems respond to the same focus.

In KlomboAGI:
  Competitors: beliefs, relations, activation, working memory items,
               conflicts, curiosity gaps, inner state signals
  Broadcast: the winner gets sent to ALL reasoning systems at once
  Response: each system responds to the broadcast from its own perspective

This is the closest computational analog to consciousness —
a single unified signal that the whole system is aware of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """Types of signals that compete for the workspace."""
    BELIEF = "belief"
    RELATION = "relation"
    CONFLICT = "conflict"
    CURIOSITY = "curiosity"
    EMOTION = "emotion"
    PERCEPTION = "perception"
    GOAL = "goal"
    CHUNK = "chunk"


@dataclass
class WorkspaceSignal:
    """A signal competing for the global workspace."""
    content: str
    signal_type: SignalType
    salience: float        # 0-1, how important/urgent
    source: str            # which subsystem sent this
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.signal_type.value,
            "salience": round(self.salience, 3),
            "source": self.source,
        }


@dataclass
class BroadcastResult:
    """Result of a global broadcast."""
    signal: WorkspaceSignal
    responses: dict[str, str]  # subsystem → response
    cycle: int


class GlobalWorkspace:
    """
    Competitive broadcast — one signal, all subsystems.

    Each cognitive cycle:
    1. Subsystems submit signals with salience scores
    2. Highest salience wins the competition
    3. Winner is broadcast to ALL subsystems
    4. Each subsystem responds from its perspective

    This creates unified cognition from diverse subsystems.
    """

    def __init__(self) -> None:
        self._signals: list[WorkspaceSignal] = []
        self._broadcast_history: list[BroadcastResult] = []
        self._cycle: int = 0
        self._current_broadcast: WorkspaceSignal | None = None

    def submit(self, content: str, signal_type: SignalType,
               salience: float, source: str, **data) -> None:
        """Submit a signal to compete for the workspace."""
        self._signals.append(WorkspaceSignal(
            content=content, signal_type=signal_type,
            salience=min(1.0, max(0.0, salience)),
            source=source, data=data,
        ))

    def compete(self) -> WorkspaceSignal | None:
        """
        Run the competition. Highest salience wins.

        Conflict signals get a 1.5x boost (NARS principle).
        Novel signals (not recently broadcast) get a 1.2x boost.
        """
        if not self._signals:
            return None

        # Apply boosts
        for sig in self._signals:
            # Conflicts are urgent
            if sig.signal_type == SignalType.CONFLICT:
                sig.salience *= 1.5
            # Novelty boost — don't repeat the same broadcast
            recent = [b.signal.content for b in self._broadcast_history[-5:]]
            if sig.content not in recent:
                sig.salience *= 1.2

        # Winner takes all
        winner = max(self._signals, key=lambda s: s.salience)
        self._signals.clear()
        self._current_broadcast = winner
        return winner

    def broadcast(self, subsystems: dict[str, object] = None) -> BroadcastResult:
        """
        Broadcast the winning signal to all subsystems.

        Each subsystem can optionally respond.
        """
        self._cycle += 1
        winner = self._current_broadcast

        if not winner:
            winner = self.compete()
        if not winner:
            return BroadcastResult(
                signal=WorkspaceSignal("idle", SignalType.PERCEPTION, 0, "workspace"),
                responses={}, cycle=self._cycle,
            )

        result = BroadcastResult(
            signal=winner, responses={}, cycle=self._cycle,
        )
        self._broadcast_history.append(result)

        # Keep history bounded
        if len(self._broadcast_history) > 100:
            self._broadcast_history = self._broadcast_history[-100:]

        self._current_broadcast = None
        return result

    def current(self) -> WorkspaceSignal | None:
        """What's currently in the workspace?"""
        return self._current_broadcast

    def recent_broadcasts(self, n: int = 5) -> list[str]:
        """What was recently broadcast?"""
        return [b.signal.content for b in self._broadcast_history[-n:]]

    def stats(self) -> dict:
        return {
            "total_cycles": self._cycle,
            "pending_signals": len(self._signals),
            "broadcast_history": len(self._broadcast_history),
        }
