"""
Autonomous Curiosity Loop — KlomboAGI learns on its own when idle.

When nobody is talking to the brain, it picks something from its
curiosity gaps and goes to learn about it. Reads Wikipedia, stores
facts, connects to existing knowledge. Grows without being told to.

Runs as a background thread. Respects a configurable idle threshold
so it doesn't burn CPU during active conversations.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class CuriosityLoop:
    """Background thread that makes Klombo learn autonomously."""

    def __init__(
        self,
        genesis: "Genesis",
        idle_threshold: float = 60.0,    # seconds of no hear() before exploring
        explore_interval: float = 300.0,  # seconds between explorations
        max_per_session: int = 50,        # max topics to explore before stopping
    ):
        self.genesis = genesis
        self.idle_threshold = idle_threshold
        self.explore_interval = explore_interval
        self.max_per_session = max_per_session
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_hear_time = time.time()
        self._explorations = 0
        self._explored_topics: list[str] = []
        self.on_explore: callable | None = None  # callback(topic, result)

    def notify_activity(self) -> None:
        """Called when someone talks to the brain. Resets idle timer."""
        self._last_hear_time = time.time()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def status(self) -> dict:
        idle_seconds = time.time() - self._last_hear_time
        return {
            "running": self._running,
            "idle_seconds": round(idle_seconds, 1),
            "is_idle": idle_seconds > self.idle_threshold,
            "explorations": self._explorations,
            "recent_topics": self._explored_topics[-10:],
        }

    def _loop(self) -> None:
        while self._running:
            try:
                idle_seconds = time.time() - self._last_hear_time

                if idle_seconds > self.idle_threshold and self._explorations < self.max_per_session:
                    topic = self._pick_topic()
                    if topic:
                        self._explore(topic)

                time.sleep(self.explore_interval)
            except Exception:
                time.sleep(60)  # Back off on error

    def _pick_topic(self) -> str | None:
        """Pick the most interesting topic from curiosity gaps."""
        g = self.genesis

        # Priority 1: unresolved curiosity gaps
        for gap in g.base.curiosity.gaps:
            if not gap.resolved and gap.concept not in self._explored_topics:
                return gap.concept

        # Priority 2: concepts we know little about (few beliefs)
        if g.base.memory.concepts:
            thin = []
            for concept, data in g.base.memory.concepts.items():
                facts = data.get("facts", [])
                if 0 < len(facts) < 3 and concept not in self._explored_topics:
                    thin.append(concept)
            if thin:
                return thin[0]

        # Priority 3: relations with low confidence
        for rel in g.relations._relations[:20]:
            if hasattr(rel, 'confidence') and rel.confidence < 0.5:
                topic = getattr(rel, 'source', None) or getattr(rel, 'target', None)
                if topic and topic not in self._explored_topics:
                    return topic

        return None

    def _explore(self, topic: str) -> None:
        """Explore a topic — read and learn about it."""
        try:
            result = self.genesis._active_learn(topic)
            self._explorations += 1
            self._explored_topics.append(topic)

            if self.on_explore:
                self.on_explore(topic, result)

        except Exception:
            self._explored_topics.append(topic)  # Don't retry failed topics
