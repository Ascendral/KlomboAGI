"""
Cognitive Modulators — from MicroPsi / Joscha Bach.

Three scalars that change HOW the entire reasoning system operates.
Not WHAT it thinks, but HOW it thinks.

1. RESOLUTION (detail level):
   High → precise, thorough, slow. Examines every detail.
   Low → broad, quick, approximate. Big picture only.

2. CERTAINTY THRESHOLD:
   High → only acts on high-confidence beliefs. Conservative.
   Low → acts on weak evidence. Exploratory, risk-tolerant.

3. AROUSAL (activation level):
   High → many things active, fast switching, creative associations.
   Low → few things active, focused, deep processing.

These modulators are SET by the inner state:
  Frustrated → lower certainty threshold (try anything)
  Confident → higher certainty threshold (be precise)
  Bored → higher arousal (activate more associations)
  In flow → balanced resolution and arousal
  Surprised → high arousal + low resolution (broad scan)

The modulators then CHANGE how every subsystem operates:
  Focus engine uses certainty_threshold for filtering
  Activation spreading uses arousal for decay rate
  Generator uses resolution for detail level
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CognitiveModulators:
    """
    Three scalars that modulate ALL reasoning.

    These are the knobs on the cognitive engine.
    """
    resolution: float = 0.5       # 0=broad, 1=precise
    certainty_threshold: float = 0.5  # 0=act on anything, 1=need proof
    arousal: float = 0.5          # 0=focused, 1=associative

    def to_dict(self) -> dict:
        return {
            "resolution": round(self.resolution, 3),
            "certainty_threshold": round(self.certainty_threshold, 3),
            "arousal": round(self.arousal, 3),
            "mode": self.mode,
        }

    @property
    def mode(self) -> str:
        """Human-readable operating mode."""
        if self.resolution > 0.7 and self.certainty_threshold > 0.7:
            return "analytical"       # precise + conservative
        if self.resolution < 0.3 and self.arousal > 0.7:
            return "creative"         # broad + associative
        if self.certainty_threshold < 0.3 and self.arousal > 0.5:
            return "exploratory"      # risk-tolerant + active
        if self.resolution > 0.5 and self.arousal < 0.3:
            return "focused"          # detailed + narrow
        if self.arousal > 0.7:
            return "activated"        # everything firing
        if self.arousal < 0.2:
            return "dormant"          # minimal processing
        return "balanced"

    # ── Parameters derived from modulators ──

    @property
    def max_beliefs_to_consider(self) -> int:
        """How many beliefs to pull from focus engine."""
        # High resolution = more beliefs, more thorough
        return int(3 + self.resolution * 7)  # 3-10

    @property
    def min_belief_confidence(self) -> float:
        """Minimum confidence to include a belief in reasoning."""
        return self.certainty_threshold * 0.5  # 0-0.5

    @property
    def activation_decay(self) -> float:
        """How fast spreading activation decays."""
        # High arousal = slow decay = more concepts stay active
        return 0.8 - self.arousal * 0.4  # 0.4-0.8

    @property
    def max_chain_length(self) -> int:
        """Maximum reasoning chain length."""
        return int(3 + self.resolution * 7)  # 3-10

    @property
    def explore_breadth(self) -> int:
        """How many tangents to follow."""
        return int(1 + self.arousal * 5)  # 1-6

    @property
    def max_retry(self) -> int:
        """How many times to retry before giving up."""
        # Low certainty threshold = more retries (willing to try anything)
        return int(1 + (1 - self.certainty_threshold) * 4)  # 1-5


class ModulatorController:
    """
    Sets cognitive modulators based on inner state and traits.

    The inner state (satisfaction, frustration, wonder, etc.)
    automatically adjusts the modulators, which in turn change
    how every subsystem operates.
    """

    def __init__(self) -> None:
        self.modulators = CognitiveModulators()

    def update(self, inner_state, traits=None) -> CognitiveModulators:
        """
        Update modulators from inner state.

        Frustrated → lower certainty (try anything), higher arousal (activate more)
        Confident → higher certainty (be precise), higher resolution (detail)
        Bored → higher arousal (cast wider net)
        Wondering → high arousal + low resolution (broad scan for connections)
        In flow → balanced everything
        """
        s = inner_state
        m = self.modulators

        # Resolution: satisfaction + confidence → more precise
        m.resolution = min(1.0, 0.3 + s.satisfaction * 0.3 + s.confidence * 0.4)

        # Certainty threshold: frustration lowers it (try anything),
        # confidence raises it (be precise)
        m.certainty_threshold = min(1.0, max(0.1,
            0.5 + s.confidence * 0.3 - s.frustration * 0.4))

        # Arousal: boredom raises it (need stimulation),
        # wonder raises it (something surprising), flow keeps it moderate
        m.arousal = min(1.0, max(0.1,
            0.3 + s.boredom * 0.4 + s.wonder * 0.3
            + s.urgency * 0.2 - s.flow * 0.2))

        # Trait modulation
        if traits:
            pv = traits.personality_vector()
            # High curiosity → more exploratory (lower certainty, higher arousal)
            curiosity = pv.get("curiosity", 0.3)
            m.certainty_threshold -= curiosity * 0.1
            m.arousal += curiosity * 0.1

            # High analysis → more precise (higher resolution)
            analysis = pv.get("analysis", 0.3)
            m.resolution += analysis * 0.1

            # Clamp
            m.resolution = max(0.0, min(1.0, m.resolution))
            m.certainty_threshold = max(0.0, min(1.0, m.certainty_threshold))
            m.arousal = max(0.0, min(1.0, m.arousal))

        self.modulators = m
        return m

    def explain(self) -> str:
        """Explain current operating mode."""
        m = self.modulators
        return (
            f"Operating mode: {m.mode}\n"
            f"  Resolution: {m.resolution:.0%} ({'precise' if m.resolution > 0.6 else 'broad'})\n"
            f"  Certainty threshold: {m.certainty_threshold:.0%} ({'conservative' if m.certainty_threshold > 0.6 else 'exploratory'})\n"
            f"  Arousal: {m.arousal:.0%} ({'activated' if m.arousal > 0.6 else 'focused'})\n"
            f"  → Consider {m.max_beliefs_to_consider} beliefs, chains up to {m.max_chain_length}, "
            f"explore {m.explore_breadth} tangents"
        )
