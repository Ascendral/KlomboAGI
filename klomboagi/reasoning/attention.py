"""
Attention System — adapted from NARS Bag + Joscha Bach's motivation model.

The Bag (from NARS): A probabilistic priority queue where items are selected
with probability proportional to their priority. Bounded capacity — when full,
lowest priority items get evicted. This is how the system decides what to
think about.

Motivation (from MicroPsi/Bach): Drives create urgency. The system doesn't
just think about random things — it thinks about what MATTERS right now.
Drives: curiosity (need to know), competence (need to solve), coherence
(need to make sense), social (need to respond to human).

Combined: priorities = base_importance × drive_urgency × recency

From NARS: constant-time put/take/get via bucket array + hash map.
From Bach: priority influenced by motivational drives, not just access frequency.
From Hutter: prediction accuracy boosts importance — things you can predict
well become more useful tools for reasoning.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from enum import Enum


T = TypeVar('T')


# ── Drives (from Joscha Bach / MicroPsi) ──

class Drive(Enum):
    """What motivates the system to think about something."""
    CURIOSITY = "curiosity"        # Need to fill knowledge gaps
    COMPETENCE = "competence"      # Need to solve the current task
    COHERENCE = "coherence"        # Need to resolve contradictions
    SOCIAL = "social"              # Need to respond to the human
    PREDICTION = "prediction"      # Need to improve prediction accuracy


@dataclass
class DriveState:
    """Current urgency of each drive."""
    levels: dict[Drive, float] = field(default_factory=lambda: {
        Drive.CURIOSITY: 0.3,
        Drive.COMPETENCE: 0.0,
        Drive.COHERENCE: 0.2,
        Drive.SOCIAL: 0.0,
        Drive.PREDICTION: 0.1,
    })

    def boost(self, drive: Drive, amount: float = 0.2) -> None:
        """Increase a drive's urgency."""
        self.levels[drive] = min(1.0, self.levels.get(drive, 0.0) + amount)

    def satisfy(self, drive: Drive, amount: float = 0.3) -> None:
        """Reduce a drive's urgency (it's been addressed)."""
        self.levels[drive] = max(0.0, self.levels.get(drive, 0.0) - amount)

    def decay(self, rate: float = 0.05) -> None:
        """All drives decay slightly over time toward baseline."""
        for drive in self.levels:
            current = self.levels[drive]
            baseline = 0.1
            if current > baseline:
                self.levels[drive] = max(baseline, current - rate)

    def most_urgent(self) -> Drive:
        """What drive is screaming loudest?"""
        return max(self.levels, key=lambda d: self.levels[d])

    def urgency(self, drive: Drive) -> float:
        return self.levels.get(drive, 0.0)


# ── Bag Item ──

@dataclass
class BagItem(Generic[T]):
    """An item in the Bag with priority tracking."""
    key: str
    value: T
    priority: float = 0.5          # [0, 1] — selection probability weight
    durability: float = 0.5        # [0, 1] — resistance to priority decay
    quality: float = 0.5           # [0, 1] — intrinsic usefulness
    last_accessed: float = 0.0     # Timestamp
    access_count: int = 0
    creation_time: float = 0.0
    drive_relevance: dict[str, float] = field(default_factory=dict)  # drive → relevance

    def budget_string(self) -> str:
        return f"p={self.priority:.2f} d={self.durability:.2f} q={self.quality:.2f}"


# ── The Bag (from NARS) ──

class Bag(Generic[T]):
    """
    Probabilistic priority queue with bounded capacity.

    Items are selected with probability proportional to priority.
    When full, lowest priority items are evicted.
    Constant-time operations via bucket array + hash map.

    This is the core attention mechanism — what gets thought about.
    """

    def __init__(self, capacity: int = 1000, n_levels: int = 100) -> None:
        self.capacity = capacity
        self.n_levels = n_levels
        self._items: dict[str, BagItem[T]] = {}           # key → item (hash map)
        self._levels: list[list[str]] = [[] for _ in range(n_levels)]  # bucket array
        self._size = 0

    def put(self, key: str, value: T, priority: float = 0.5,
            durability: float = 0.5, quality: float = 0.5,
            drive_relevance: dict[str, float] | None = None) -> BagItem[T] | None:
        """
        Add or update an item. Returns evicted item if bag was full, else None.
        If key exists, merge and boost priority.
        """
        now = time.time()
        evicted = None

        if key in self._items:
            # Update existing — boost priority
            item = self._items[key]
            old_level = self._priority_to_level(item.priority)
            item.priority = min(1.0, item.priority + 0.1)  # Boost on revisit
            item.quality = max(item.quality, quality)
            item.last_accessed = now
            item.access_count += 1
            if drive_relevance:
                item.drive_relevance.update(drive_relevance)
            # Move to new level
            new_level = self._priority_to_level(item.priority)
            if old_level != new_level:
                if key in self._levels[old_level]:
                    self._levels[old_level].remove(key)
                self._levels[new_level].append(key)
            return None

        # New item
        if self._size >= self.capacity:
            evicted = self._evict_lowest()

        item = BagItem(
            key=key, value=value,
            priority=max(0.01, min(1.0, priority)),
            durability=durability, quality=quality,
            last_accessed=now, creation_time=now,
            drive_relevance=drive_relevance or {},
        )
        self._items[key] = item
        level = self._priority_to_level(item.priority)
        self._levels[level].append(key)
        self._size += 1
        return evicted

    def take(self) -> BagItem[T] | None:
        """
        Probabilistically select an item. Probability proportional to priority.
        Higher priority = more likely to be selected.
        Does NOT remove the item — just selects it.
        """
        if self._size == 0:
            return None

        # Weighted random selection across levels
        # Higher levels (higher priority) are more likely
        weighted_levels = []
        for i, level in enumerate(self._levels):
            if level:
                weight = (i + 1) ** 2  # Quadratic weighting — strong priority bias
                weighted_levels.append((i, weight))

        if not weighted_levels:
            return None

        total_weight = sum(w for _, w in weighted_levels)
        r = random.random() * total_weight
        cumulative = 0
        selected_level = weighted_levels[0][0]
        for level_idx, weight in weighted_levels:
            cumulative += weight
            if r <= cumulative:
                selected_level = level_idx
                break

        # Pick random item from selected level
        keys_in_level = self._levels[selected_level]
        if not keys_in_level:
            return None

        key = random.choice(keys_in_level)
        item = self._items.get(key)
        if item:
            item.last_accessed = time.time()
            item.access_count += 1
        return item

    def get(self, key: str) -> BagItem[T] | None:
        """Direct lookup by key."""
        item = self._items.get(key)
        if item:
            item.last_accessed = time.time()
            item.access_count += 1
        return item

    def remove(self, key: str) -> BagItem[T] | None:
        """Remove an item by key."""
        if key not in self._items:
            return None
        item = self._items.pop(key)
        level = self._priority_to_level(item.priority)
        if key in self._levels[level]:
            self._levels[level].remove(key)
        self._size -= 1
        return item

    def decay_all(self, rate: float = 0.01) -> None:
        """Decay all priorities slightly — unused items fade."""
        for key, item in list(self._items.items()):
            old_level = self._priority_to_level(item.priority)
            # Durability resists decay
            actual_decay = rate * (1 - item.durability)
            item.priority = max(0.01, item.priority - actual_decay)
            new_level = self._priority_to_level(item.priority)
            if old_level != new_level:
                if key in self._levels[old_level]:
                    self._levels[old_level].remove(key)
                self._levels[new_level].append(key)

    def _evict_lowest(self) -> BagItem[T] | None:
        """Remove the lowest priority item."""
        for level in self._levels:
            if level:
                key = level.pop(0)
                item = self._items.pop(key, None)
                if item:
                    self._size -= 1
                    return item
        return None

    def _priority_to_level(self, priority: float) -> int:
        """Map priority [0,1] to bucket level [0, n_levels-1]."""
        return min(self.n_levels - 1, max(0, int(priority * self.n_levels)))

    @property
    def size(self) -> int:
        return self._size

    def top(self, n: int = 10) -> list[BagItem[T]]:
        """Get the n highest priority items."""
        sorted_items = sorted(self._items.values(),
                              key=lambda x: x.priority, reverse=True)
        return sorted_items[:n]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return self._size


# ── Attention Controller ──

class AttentionController:
    """
    Decides what the system thinks about next.

    Combines:
    - NARS Bag (probabilistic priority selection)
    - Bach's drives (motivational urgency)
    - Hutter's prediction reward (accurate predictions boost priority)

    The human can override by boosting drive levels.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.concepts = Bag[dict](capacity=capacity)  # Concept memory
        self.tasks = Bag[dict](capacity=200)           # Active tasks
        self.drives = DriveState()

    def add_concept(self, name: str, data: dict,
                    priority: float = 0.5,
                    relevant_drives: list[Drive] | None = None) -> None:
        """Add a concept to attention."""
        drive_rel = {}
        if relevant_drives:
            for d in relevant_drives:
                drive_rel[d.value] = self.drives.urgency(d)

        self.concepts.put(name, data, priority=priority,
                          drive_relevance=drive_rel)

    def add_task(self, name: str, data: dict,
                 priority: float = 0.7) -> None:
        """Add a task (something to think about/do)."""
        self.tasks.put(name, data, priority=priority)

    def what_next(self) -> tuple[str, dict] | None:
        """
        What should the system think about next?
        Considers both concepts and tasks, weighted by drive urgency.
        """
        # Check if any drive is urgent
        urgent_drive = self.drives.most_urgent()
        urgency = self.drives.urgency(urgent_drive)

        # If high urgency, prefer tasks over concepts
        if urgency > 0.7 and self.tasks.size > 0:
            item = self.tasks.take()
            if item:
                return item.key, item.value

        # Otherwise, mix tasks and concepts
        if random.random() < 0.3 and self.tasks.size > 0:
            item = self.tasks.take()
        else:
            item = self.concepts.take()

        if item:
            return item.key, item.value
        return None

    def reward(self, concept_name: str, amount: float = 0.1) -> None:
        """
        Reward a concept — its reasoning was useful.
        From Hutter: accurate predictions get boosted.
        """
        item = self.concepts.get(concept_name)
        if item:
            item.priority = min(1.0, item.priority + amount)
            item.quality = min(1.0, item.quality + amount * 0.5)

    def penalize(self, concept_name: str, amount: float = 0.05) -> None:
        """Penalize — reasoning about this was wrong or unhelpful."""
        item = self.concepts.get(concept_name)
        if item:
            item.priority = max(0.01, item.priority - amount)

    def on_human_speaks(self) -> None:
        """Human said something — boost social drive."""
        self.drives.boost(Drive.SOCIAL, 0.5)

    def on_gap_found(self) -> None:
        """Knowledge gap found — boost curiosity."""
        self.drives.boost(Drive.CURIOSITY, 0.3)

    def on_contradiction(self) -> None:
        """Contradiction found — boost coherence."""
        self.drives.boost(Drive.COHERENCE, 0.4)

    def on_task_given(self) -> None:
        """Task/goal given — boost competence."""
        self.drives.boost(Drive.COMPETENCE, 0.5)

    def on_prediction_correct(self) -> None:
        """Prediction was right — boost prediction drive satisfaction."""
        self.drives.satisfy(Drive.PREDICTION, 0.2)

    def on_response_given(self) -> None:
        """Responded to human — satisfy social drive."""
        self.drives.satisfy(Drive.SOCIAL, 0.3)

    def tick(self) -> None:
        """Time passes — decay priorities and drives."""
        self.concepts.decay_all(rate=0.005)
        self.tasks.decay_all(rate=0.01)
        self.drives.decay()

    def stats(self) -> dict:
        return {
            "concepts": self.concepts.size,
            "tasks": self.tasks.size,
            "drives": {d.value: round(v, 2) for d, v in self.drives.levels.items()},
            "most_urgent": self.drives.most_urgent().value,
            "top_concepts": [(i.key, round(i.priority, 2)) for i in self.concepts.top(5)],
        }
