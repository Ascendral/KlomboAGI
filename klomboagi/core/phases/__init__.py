"""
Cognitive phases — the hear() pipeline broken into discrete, testable steps.

Each phase is a class with run(ctx, genesis) -> ctx.
Context flows through the pipeline carrying state between phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis, Surprise


@dataclass
class HearContext:
    """State that flows through the hear() pipeline."""
    message: str
    resolved_message: str = ""
    intent: dict = field(default_factory=dict)
    surprise: "Surprise | None" = None
    response: str = ""
    trait_influence: object = None  # TraitInfluence
    query_terms: list[str] = field(default_factory=list)


class Phase:
    """Base class for pipeline phases."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        raise NotImplementedError


from klomboagi.core.phases.input_phase import InputPhase
from klomboagi.core.phases.surprise_phase import SurprisePhase
from klomboagi.core.phases.routing_phase import RoutingPhase
from klomboagi.core.phases.metacognitive_phase import MetacognitivePhase
from klomboagi.core.phases.dialog_phase import DialogPhase
from klomboagi.core.phases.learning_phase import LearningPhase
from klomboagi.core.phases.persistence_phase import PersistencePhase

__all__ = [
    "HearContext", "Phase",
    "InputPhase", "SurprisePhase", "RoutingPhase",
    "MetacognitivePhase", "DialogPhase", "LearningPhase", "PersistencePhase",
]
