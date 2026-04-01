"""Phase 1: Input processing — parse, resolve pronouns, set up working memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class InputPhase(Phase):
    """Parse intent, resolve pronouns, prime working memory and attention."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        g.total_turns += 1
        g.cost_tracker.start("hear_cycle")

        # Working memory
        g.working_memory.add_context(ctx.message)

        # Resolve pronouns
        ctx.resolved_message = g.context.resolve_pronoun(ctx.message)

        # Parse intent
        ctx.intent = g.base._parse_intent(ctx.resolved_message)

        # Conversation memory — track topics
        if g.context.current_topic:
            g.conversation_memory.record_topic(g.context.current_topic)

        # Working memory + ACT-R decay + attention economy
        for word in ctx.resolved_message.lower().split():
            if len(word) > 3 and word not in g.base.COMMON_WORDS:
                g.working_memory.attend(word, "concept", "input")
                g.memory_decay.access(word)
                g.attention_economy.allocate(word, 5.0)

        return ctx
