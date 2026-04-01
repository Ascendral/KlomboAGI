"""Phase 2: Surprise detection — check for contradictions before learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class SurprisePhase(Phase):
    """Detect contradictions between new input and existing beliefs."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        ctx.surprise = g._check_surprise(ctx.intent)
        return ctx
