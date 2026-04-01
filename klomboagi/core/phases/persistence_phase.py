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

        # Record question in conversation memory (for session summary)
        if ctx.intent.get("type") == "question":
            has_answer = len(ctx.response) > 20 and "don't know" not in ctx.response.lower()
            g.conversation_memory.record_question(ctx.resolved_message, answered=has_answer)

        # Save conversation memory every 10 turns
        if g.total_turns % 10 == 0:
            g.conversation_memory.save()

        return ctx
