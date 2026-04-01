"""Phase 7: Persistence — cost tracking, save state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class PersistencePhase(Phase):
    """Record cycle cost and auto-save cognitive state to disk."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        g.cost_tracker.end("hear_cycle", success=True)
        g.save_state()
        return ctx
