"""
Working Memory — the short-term focus buffer.

Long-term memory stores everything the system has ever learned.
Working memory holds what it's CURRENTLY thinking about.

Like a human's ~7 item capacity:
- Active concepts (what we're focused on right now)
- Active relations (connections in play)
- Recent context (last few exchanges)
- Current goal (what we're trying to accomplish)
- Attention weights (what's most relevant right now)

Working memory decays — items fade if not refreshed.
Items get promoted to long-term if they prove important.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class MemoryItem:
    """An item in working memory."""
    content: str
    item_type: str           # "concept", "relation", "context", "goal", "hypothesis"
    activation: float = 1.0  # 0-1, decays over time
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    source: str = ""         # what put this in working memory

    def access(self) -> None:
        """Refresh this item — it's being used."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.activation = min(1.0, self.activation + 0.2)

    def decay(self, rate: float = 0.1) -> None:
        """Item fades if not accessed."""
        self.activation = max(0.0, self.activation - rate)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.item_type,
            "activation": round(self.activation, 3),
            "access_count": self.access_count,
            "source": self.source,
        }


@dataclass
class Goal:
    """Something the system is trying to accomplish."""
    description: str
    priority: float = 0.5    # 0-1
    progress: float = 0.0    # 0-1
    subgoals: list[str] = field(default_factory=list)
    status: str = "active"   # active, completed, abandoned
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "priority": round(self.priority, 3),
            "progress": round(self.progress, 3),
            "subgoals": self.subgoals,
            "status": self.status,
        }


class WorkingMemory:
    """
    Short-term focus buffer with limited capacity.

    Holds the ~7 most relevant items. Decays over time.
    Items that persist get promoted to long-term memory.
    """

    CAPACITY = 7             # Miller's number
    DECAY_RATE = 0.05        # per tick
    PROMOTION_THRESHOLD = 5  # access count to promote

    def __init__(self) -> None:
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()
        self._goals: list[Goal] = []
        self._focus: str = ""            # current primary focus
        self._context: list[str] = []    # recent conversation context (last 5)

    def attend(self, content: str, item_type: str = "concept",
               source: str = "") -> MemoryItem:
        """
        Bring something into working memory.

        If already there, refresh it. If at capacity, evict the
        least activated item.
        """
        if content in self._items:
            item = self._items[content]
            item.access()
            # Move to end (most recent)
            self._items.move_to_end(content)
            return item

        # At capacity — evict lowest activation
        if len(self._items) >= self.CAPACITY:
            self._evict_weakest()

        item = MemoryItem(
            content=content,
            item_type=item_type,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            source=source,
        )
        self._items[content] = item
        return item

    def focus_on(self, concept: str) -> None:
        """Set the primary focus — what we're thinking about."""
        self._focus = concept
        self.attend(concept, "concept", "focus")

    def add_context(self, message: str) -> None:
        """Add recent conversation to context buffer."""
        self._context.append(message)
        if len(self._context) > 5:
            self._context = self._context[-5:]

    def set_goal(self, description: str, priority: float = 0.5) -> Goal:
        """Set a new goal."""
        goal = Goal(
            description=description,
            priority=priority,
            created_at=time.time(),
        )
        self._goals.append(goal)
        self.attend(description, "goal", "goal_system")
        return goal

    def complete_goal(self, description: str) -> None:
        """Mark a goal as completed."""
        for g in self._goals:
            if g.description == description:
                g.status = "completed"
                g.progress = 1.0

    def get_active_goals(self) -> list[Goal]:
        """Goals currently being pursued."""
        return [g for g in self._goals if g.status == "active"]

    def get_focus(self) -> str:
        """What are we currently focused on?"""
        return self._focus

    def get_context(self) -> list[str]:
        """Recent conversation context."""
        return self._context.copy()

    def get_active_items(self) -> list[MemoryItem]:
        """All items above minimum activation, sorted by activation."""
        active = [item for item in self._items.values() if item.activation > 0.1]
        return sorted(active, key=lambda i: i.activation, reverse=True)

    def get_promotable(self) -> list[MemoryItem]:
        """Items accessed enough to promote to long-term memory."""
        return [item for item in self._items.values()
                if item.access_count >= self.PROMOTION_THRESHOLD]

    def tick(self) -> list[str]:
        """
        Decay all items. Returns list of evicted items.

        Call this at the end of each thinking cycle.
        """
        evicted = []
        for key in list(self._items.keys()):
            self._items[key].decay(self.DECAY_RATE)
            if self._items[key].activation <= 0:
                evicted.append(key)
                del self._items[key]
        return evicted

    def contains(self, content: str) -> bool:
        return content in self._items

    def _evict_weakest(self) -> str | None:
        """Remove the least activated item."""
        if not self._items:
            return None
        weakest_key = min(self._items, key=lambda k: self._items[k].activation)
        del self._items[weakest_key]
        return weakest_key

    def stats(self) -> dict:
        return {
            "items": len(self._items),
            "capacity": self.CAPACITY,
            "focus": self._focus,
            "active_goals": len(self.get_active_goals()),
            "context_length": len(self._context),
            "items_detail": [i.to_dict() for i in self.get_active_items()],
        }

    def dump(self) -> str:
        """Human-readable working memory state."""
        lines = [f"Working Memory ({len(self._items)}/{self.CAPACITY}):"]
        if self._focus:
            lines.append(f"  Focus: {self._focus}")
        for item in self.get_active_items():
            bar = "█" * int(item.activation * 10) + "░" * (10 - int(item.activation * 10))
            lines.append(f"  [{bar}] {item.content} ({item.item_type}, x{item.access_count})")
        if self._goals:
            lines.append(f"\n  Goals:")
            for g in self.get_active_goals():
                lines.append(f"    {'→' if g.status == 'active' else '✓'} {g.description} ({g.progress:.0%})")
        return "\n".join(lines)
