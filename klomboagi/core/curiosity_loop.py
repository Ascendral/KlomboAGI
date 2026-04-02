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
        """Pick the most interesting topic from reasoning gaps."""
        g = self.genesis

        # Priority 1: CoreReasoner gaps (real inference gaps)
        if hasattr(g, 'core_reasoner'):
            gaps = g.core_reasoner.find_gaps()
            for gap_q in gaps:
                # Extract the concept from "What is X?" style questions
                import re
                m = re.search(r"what (?:is|properties does) (\w[\w\s]*?)(?:\?|$)", gap_q.lower())
                if m:
                    topic = m.group(1).strip()
                    if topic not in self._explored_topics and len(topic) > 2:
                        return topic

        # Priority 2: unresolved curiosity gaps from conversation
        for gap in g.base.curiosity.gaps:
            if not gap.resolved and gap.concept not in self._explored_topics:
                return gap.concept

        # Priority 3: low-confidence derived facts to verify
        if hasattr(g, 'core_reasoner'):
            for f in g.core_reasoner.derived:
                if f.confidence < 0.5 and f.subject not in self._explored_topics:
                    return f.subject

        return None

    def _explore(self, topic: str) -> None:
        """Explore a topic — search, extract facts, feed to CoreReasoner."""
        try:
            g = self.genesis
            # Search for information
            raw = g.base.searcher.search(topic)
            if not raw or "Could not find" in raw:
                self._explored_topics.append(topic)
                return

            # Feed raw text to CoreReasoner's learn_from_text
            new_facts = []
            if hasattr(g, 'core_reasoner'):
                new_facts = g.core_reasoner.learn_from_text(raw)

            # Also feed to legacy system for backward compat
            g._active_learn(topic)

            self._explorations += 1
            self._explored_topics.append(topic)

            if self.on_explore:
                self.on_explore(topic, f"Learned {len(new_facts)} structured facts about {topic}")

        except Exception:
            self._explored_topics.append(topic)
