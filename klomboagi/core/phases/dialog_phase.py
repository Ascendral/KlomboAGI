"""Phase 5: Dialog — emotional intel, follow-ups, conversation flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class DialogPhase(Phase):
    """Read human's emotional state and generate natural follow-ups."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        # Emotional Intelligence — read human's state
        emotional_reading = g.emotional_intel.read(ctx.message)

        # Dialog Flow — track conversation, generate follow-ups
        g.dialog_flow.update(ctx.message, ctx.response, ctx.intent["type"],
                             g.context.current_topic)
        followup = g.dialog_flow.get_followup()
        if followup and emotional_reading.primary.value not in ("frustrated", "impatient"):
            ctx.response += f"\n\n{followup}"

        return ctx
