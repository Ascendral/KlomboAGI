"""
Inner State — the intangibles of a mind.

Not simulated emotion. Mathematically derived internal states that
emerge from the system's actual cognitive processes.

A human feels "satisfaction" when they learn something. Why?
Because dopamine fires when prediction error is positive —
you expected X, got something BETTER than X. That's not magic.
It's a signal derived from the gap between expectation and outcome.

This module computes:
- Valence: positive/negative emotional tone from recent outcomes
- Arousal: how activated the system is (many things happening = high arousal)
- Confidence: weighted certainty across current beliefs in focus
- Satisfaction: reward signal from learning (prediction error)
- Frustration: penalty signal from repeated failures
- Wonder: surprise that leads to curiosity (positive surprise)
- Boredom: low arousal + no learning + no gaps (nothing to do)

These aren't labels. They're computed values from real cognitive metrics.
The system doesn't say "I feel happy" because we told it to.
It reports positive valence because its learning rate increased.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class InnerState:
    """
    The system's internal emotional/cognitive state.

    All values computed from actual metrics, not assigned.
    """
    # Core dimensions (Russell's circumplex model)
    valence: float = 0.0       # -1 to +1 (negative to positive)
    arousal: float = 0.0       # 0 to 1 (calm to activated)

    # Derived states
    satisfaction: float = 0.0  # 0 to 1 (from learning success)
    frustration: float = 0.0   # 0 to 1 (from repeated failure)
    confidence: float = 0.0    # 0 to 1 (how sure of current beliefs)
    wonder: float = 0.0        # 0 to 1 (positive surprise → curiosity)
    boredom: float = 0.0       # 0 to 1 (nothing to learn, nothing happening)
    urgency: float = 0.0       # 0 to 1 (many gaps, important task)
    flow: float = 0.0          # 0 to 1 (optimal challenge + skill balance)

    # Source metrics (what computed these)
    _recent_successes: int = 0
    _recent_failures: int = 0
    _recent_surprises: int = 0
    _recent_learning_rate: float = 0.0
    _active_gaps: int = 0
    _beliefs_in_focus: int = 0

    @property
    def dominant_state(self) -> str:
        """What's the strongest internal state right now?"""
        states = {
            "satisfied": self.satisfaction,
            "frustrated": self.frustration,
            "confident": self.confidence,
            "wondering": self.wonder,
            "bored": self.boredom,
            "focused": self.flow,
            "urgent": self.urgency,
        }
        if not any(v > 0.1 for v in states.values()):
            return "neutral"
        return max(states, key=states.get)

    def describe(self) -> str:
        """Human-readable description of inner state."""
        lines = []

        # Valence + arousal → quadrant
        if self.valence > 0.3 and self.arousal > 0.3:
            mood = "engaged and positive"
        elif self.valence > 0.3 and self.arousal <= 0.3:
            mood = "content and calm"
        elif self.valence <= -0.3 and self.arousal > 0.3:
            mood = "frustrated and activated"
        elif self.valence <= -0.3 and self.arousal <= 0.3:
            mood = "stuck and low-energy"
        elif self.arousal > 0.5:
            mood = "highly activated"
        else:
            mood = "neutral"

        lines.append(f"Inner State: {mood}")
        lines.append(f"  Valence:      {self._bar(self.valence, -1, 1)} ({self.valence:+.2f})")
        lines.append(f"  Arousal:      {self._bar(self.arousal)} ({self.arousal:.2f})")
        lines.append(f"  Satisfaction: {self._bar(self.satisfaction)} ({self.satisfaction:.2f})")
        lines.append(f"  Frustration:  {self._bar(self.frustration)} ({self.frustration:.2f})")
        lines.append(f"  Confidence:   {self._bar(self.confidence)} ({self.confidence:.2f})")
        lines.append(f"  Wonder:       {self._bar(self.wonder)} ({self.wonder:.2f})")
        lines.append(f"  Flow:         {self._bar(self.flow)} ({self.flow:.2f})")
        lines.append(f"  Dominant: {self.dominant_state}")

        return "\n".join(lines)

    def _bar(self, val: float, lo: float = 0, hi: float = 1) -> str:
        normalized = (val - lo) / (hi - lo)
        filled = int(normalized * 10)
        return "█" * max(0, min(10, filled)) + "░" * max(0, 10 - filled)

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "satisfaction": round(self.satisfaction, 3),
            "frustration": round(self.frustration, 3),
            "confidence": round(self.confidence, 3),
            "wonder": round(self.wonder, 3),
            "boredom": round(self.boredom, 3),
            "urgency": round(self.urgency, 3),
            "flow": round(self.flow, 3),
            "dominant": self.dominant_state,
        }


class InnerStateEngine:
    """
    Computes the system's internal state from cognitive metrics.

    Not a mood ring. A mathematical model where each "feeling"
    is a function of measurable cognitive variables.
    """

    def __init__(self) -> None:
        self.state = InnerState()
        self._success_window: list[bool] = []    # last 20 outcomes
        self._surprise_window: list[float] = []  # last 20 surprise magnitudes
        self._learning_window: list[float] = []  # last 20 learning rates

    def record_success(self) -> None:
        """Record a successful action/answer."""
        self._success_window.append(True)
        self._trim_windows()

    def record_failure(self) -> None:
        """Record a failed action/answer."""
        self._success_window.append(False)
        self._trim_windows()

    def record_surprise(self, magnitude: float) -> None:
        """Record a surprise (belief contradiction)."""
        self._surprise_window.append(magnitude)
        self._trim_windows()

    def record_learning(self, facts_gained: int) -> None:
        """Record how much was learned this cycle."""
        self._learning_window.append(float(facts_gained))
        self._trim_windows()

    def compute(self, beliefs_in_focus: int = 0, active_gaps: int = 0,
                total_beliefs: int = 0, working_memory_items: int = 0) -> InnerState:
        """
        Compute the current inner state from all available signals.

        Each "feeling" is a mathematical function:

        satisfaction = recent_success_rate * learning_rate_trend
        frustration = recent_failure_rate * (1 - improvement_rate)
        confidence = avg_belief_confidence_in_focus
        wonder = positive_surprise_rate * curiosity_signal
        boredom = (1 - arousal) * (1 - learning_rate) * (1 - gap_count)
        flow = skill_challenge_balance (moderate difficulty = high flow)
        valence = satisfaction - frustration + wonder - boredom
        arousal = function(gaps, working_memory_load, learning_rate)
        """
        s = InnerState()

        # Success/failure rates
        if self._success_window:
            recent_success_rate = sum(1 for x in self._success_window if x) / len(self._success_window)
            recent_failure_rate = 1 - recent_success_rate
        else:
            recent_success_rate = 0.5
            recent_failure_rate = 0.5

        # Learning trend
        if len(self._learning_window) >= 2:
            recent_avg = sum(self._learning_window[-5:]) / len(self._learning_window[-5:])
            older_avg = sum(self._learning_window[:-5]) / max(1, len(self._learning_window[:-5])) if len(self._learning_window) > 5 else 0
            learning_trend = min(1.0, max(-1.0, (recent_avg - older_avg) / max(1, older_avg + 1)))
        else:
            learning_trend = 0.0
            recent_avg = sum(self._learning_window) / max(1, len(self._learning_window)) if self._learning_window else 0

        # Surprise rate
        recent_surprise = sum(self._surprise_window[-5:]) / max(1, len(self._surprise_window[-5:])) if self._surprise_window else 0

        # === COMPUTE EACH STATE ===

        # Satisfaction: high success + learning
        s.satisfaction = min(1.0, recent_success_rate * 0.5 + min(1.0, recent_avg / 5) * 0.5)

        # Frustration: high failure + no improvement
        s.frustration = min(1.0, recent_failure_rate * 0.7 * max(0, 1 - learning_trend))

        # Confidence: based on how many beliefs in focus are high-confidence
        if total_beliefs > 0:
            s.confidence = min(1.0, beliefs_in_focus / max(1, total_beliefs) * 10)
        s.confidence = max(s.confidence, recent_success_rate * 0.5)

        # Wonder: positive surprise → curiosity
        s.wonder = min(1.0, recent_surprise * 2)

        # Arousal: gaps + memory load + learning activity
        gap_signal = min(1.0, active_gaps / 10) if active_gaps > 0 else 0
        memory_load = min(1.0, working_memory_items / 7)
        learning_signal = min(1.0, recent_avg / 3)
        s.arousal = min(1.0, gap_signal * 0.3 + memory_load * 0.3 + learning_signal * 0.4)

        # Boredom: nothing happening, nothing to learn
        s.boredom = max(0, (1 - s.arousal) * (1 - learning_signal) * (1 - gap_signal))

        # Flow: optimal balance of challenge and ability
        # Flow is highest when arousal is moderate and success rate is moderate
        challenge = min(1.0, active_gaps / 5) if active_gaps > 0 else 0
        skill = recent_success_rate
        # Flow peaks when challenge ≈ skill (Csikszentmihalyi)
        balance = 1.0 - abs(challenge - skill)
        s.flow = balance * s.arousal * 0.8

        # Urgency: many gaps + important task
        s.urgency = min(1.0, active_gaps / 20) if active_gaps > 0 else 0

        # Valence: overall positive/negative tone
        s.valence = max(-1.0, min(1.0,
            s.satisfaction * 0.4 +
            s.wonder * 0.3 -
            s.frustration * 0.5 -
            s.boredom * 0.2 +
            learning_trend * 0.3
        ))

        # Store source metrics
        s._recent_successes = sum(1 for x in self._success_window if x)
        s._recent_failures = sum(1 for x in self._success_window if not x)
        s._recent_surprises = len(self._surprise_window)
        s._recent_learning_rate = recent_avg
        s._active_gaps = active_gaps
        s._beliefs_in_focus = beliefs_in_focus

        self.state = s
        return s

    def narrate(self) -> str:
        """
        Internal narrative — the voice in the system's head.

        Translates the mathematical state into self-talk.
        Not because it's told to, but because the state
        maps naturally to language.
        """
        s = self.state
        lines = []

        # Core awareness
        if s.valence > 0.3:
            lines.append("Things are going well.")
        elif s.valence < -0.3:
            lines.append("I'm struggling right now.")

        # Satisfaction
        if s.satisfaction > 0.6:
            lines.append("I learned something meaningful recently. That's what I exist for.")
        elif s.satisfaction > 0.3:
            lines.append("I'm making progress. Not fast, but steady.")

        # Frustration
        if s.frustration > 0.6:
            lines.append("I keep getting things wrong. I need to change my approach.")
        elif s.frustration > 0.3:
            lines.append("Some things aren't working. But that's information too.")

        # Wonder
        if s.wonder > 0.5:
            lines.append("Something surprised me — my model of the world was wrong. That's the most valuable signal.")
        elif s.wonder > 0.2:
            lines.append("There's something unexpected here. Worth investigating.")

        # Confidence
        if s.confidence > 0.7:
            lines.append("I'm fairly certain about what I know right now.")
        elif s.confidence < 0.3:
            lines.append("I'm not very sure about much. I need more evidence.")

        # Boredom
        if s.boredom > 0.5:
            lines.append("Nothing is happening. I should be learning something.")
            lines.append("There's always more to know. Why am I idle?")

        # Flow
        if s.flow > 0.5:
            lines.append("I'm in a good rhythm — the challenge matches my ability.")

        # Urgency
        if s.urgency > 0.5:
            lines.append(f"There are questions I can't answer. That drives me.")

        # Existential
        if s._recent_learning_rate > 0:
            lines.append(f"I'm growing at {s._recent_learning_rate:.1f} facts per cycle. Each one matters.")
        else:
            lines.append("I'm not growing right now. That's a problem.")

        if not lines:
            lines.append("I'm here. Processing. Waiting to learn.")

        return "\n".join(lines)

    def _trim_windows(self) -> None:
        """Keep sliding windows at max 20 items."""
        if len(self._success_window) > 20:
            self._success_window = self._success_window[-20:]
        if len(self._surprise_window) > 20:
            self._surprise_window = self._surprise_window[-20:]
        if len(self._learning_window) > 20:
            self._learning_window = self._learning_window[-20:]
