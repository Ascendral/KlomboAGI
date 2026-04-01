"""
Self-Improvement Loop — KlomboAGI identifies and fixes its own weaknesses.

Runs periodically (background thread or called from curiosity loop).
Analyzes:
  - Questions it couldn't answer → fills knowledge gaps
  - Low-confidence beliefs → seeks confirmation
  - Failed predictions → adjusts reasoning
  - Thin concepts → deepens understanding

This is what separates AGI from a chatbot: the system WANTS to be better
and takes action to make itself better.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class SelfImprovementLoop:
    """Background thread that makes the brain better at what it's bad at."""

    def __init__(
        self,
        genesis: "Genesis",
        interval: float = 600.0,  # 10 minutes between improvement cycles
        max_actions_per_cycle: int = 3,
    ):
        self.genesis = genesis
        self.interval = interval
        self.max_actions = max_actions_per_cycle
        self._running = False
        self._thread: threading.Thread | None = None
        self._improvements: list[dict] = []

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
        return {
            "running": self._running,
            "total_improvements": len(self._improvements),
            "recent": self._improvements[-5:],
        }

    def run_once(self) -> list[dict]:
        """Run one improvement cycle. Returns list of actions taken."""
        g = self.genesis
        actions = []

        # 1. Find unanswered questions from past conversations
        unanswered = g.conversation_memory.get_unanswered()
        for q in unanswered[:1]:
            try:
                result = g._active_learn(q)
                if "Could not find" not in result and "couldn't find" not in result.lower():
                    actions.append({
                        "type": "fill_gap",
                        "topic": q,
                        "result": "learned",
                        "time": time.strftime("%Y-%m-%d %H:%M"),
                    })
            except Exception:
                pass
            if len(actions) >= self.max_actions:
                break

        # 2. Strengthen low-confidence beliefs
        weak_beliefs = []
        for stmt, belief in g.base._beliefs.items():
            if hasattr(belief, 'truth') and hasattr(belief.truth, 'confidence'):
                if 0.3 < belief.truth.confidence < 0.6 and hasattr(belief, 'subject'):
                    weak_beliefs.append((stmt, belief))
        weak_beliefs.sort(key=lambda x: x[1].truth.confidence)

        for stmt, belief in weak_beliefs[:1]:
            if len(actions) >= self.max_actions:
                break
            try:
                result = g._active_learn(belief.subject)
                if "Could not find" not in result:
                    # Check if confidence improved
                    new_belief = g.base._beliefs.get(stmt)
                    if new_belief and new_belief.truth.confidence > belief.truth.confidence:
                        actions.append({
                            "type": "strengthen",
                            "belief": stmt,
                            "old_confidence": round(belief.truth.confidence, 3),
                            "new_confidence": round(new_belief.truth.confidence, 3),
                            "time": time.strftime("%Y-%m-%d %H:%M"),
                        })
            except Exception:
                pass

        # 3. Deepen thin concepts (< 3 facts known)
        thin_concepts = []
        for concept, data in g.base.memory.concepts.items():
            facts = data.get("facts", [])
            if 0 < len(facts) < 3:
                thin_concepts.append(concept)

        for concept in thin_concepts[:1]:
            if len(actions) >= self.max_actions:
                break
            try:
                before = len(g.base.memory.concepts.get(concept, {}).get("facts", []))
                result = g._active_learn(concept)
                after = len(g.base.memory.concepts.get(concept, {}).get("facts", []))
                if after > before:
                    actions.append({
                        "type": "deepen",
                        "concept": concept,
                        "facts_before": before,
                        "facts_after": after,
                        "time": time.strftime("%Y-%m-%d %H:%M"),
                    })
            except Exception:
                pass

        # 4. Check for belief consistency (look for contradictions in own beliefs)
        conflicts = g.conflict_detector.check(g.base._beliefs)
        for conflict in conflicts[:1]:
            if len(actions) >= self.max_actions:
                break
            actions.append({
                "type": "conflict_detected",
                "belief_a": conflict.belief_a,
                "belief_b": conflict.belief_b,
                "severity": conflict.severity,
                "time": time.strftime("%Y-%m-%d %H:%M"),
            })

        self._improvements.extend(actions)
        return actions

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_once()
            except Exception:
                pass
            time.sleep(self.interval)
