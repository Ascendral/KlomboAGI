"""Phase 3: Routing — dispatch to question, learn, correct, or statement handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class RoutingPhase(Phase):
    """Route message to the right handler based on intent type.

    Also consults traits and handles surprise response augmentation.
    """

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        # Consult traits
        ctx.trait_influence = g.traits.influence({
            "description": ctx.resolved_message,
            "known_entities": g.context.entities_mentioned,
        })

        # Check for system actions FIRST (before deep thinking)
        q_lower = ctx.resolved_message.lower().strip().rstrip("?")
        system_result = g._try_system_action(q_lower, ctx.resolved_message)
        if system_result is not None:
            ctx.response = system_result
            g.context.update(ctx.intent, ctx.resolved_message)
            return ctx

        # Check for system observation questions
        observe_patterns = ("how's the system", "hows the system", "how is the system",
                           "system health", "any anomalies", "any problems",
                           "what do you see", "what do you notice", "anything wrong",
                           "how are things", "how is everything", "how are you doing",
                           "what's happening", "whats happening", "status report",
                           "how is the machine", "machine status", "check the system")
        if any(p in q_lower for p in observe_patterns):
            ctx.response = g._system_observation_report()
            g.context.update(ctx.intent, ctx.resolved_message)
            return ctx

        # Route: deep think for questions, base system for everything else
        if ctx.intent["type"] == "question":
            g.metacognition.record_question("knowledge")
            ctx.response = g._think_deep(ctx.resolved_message, ctx.intent)
        elif ctx.intent["type"] == "command" and ctx.intent.get("command") == "learn":
            ctx.response = g._active_learn(ctx.intent.get("target", ""))
        elif ctx.intent["type"] == "correction":
            g.metacognition.record_correction()
            g.inner.record_failure()
            g.failure_memory.record(
                description=ctx.resolved_message,
                context=g.context.current_topic,
                approach="previous_answer",
                what_went_wrong="human corrected us",
            )
            ctx.response = g.base.hear(ctx.resolved_message)
        else:
            ctx.response = g.base.hear(ctx.resolved_message)
            g._extract_relations(ctx.resolved_message)

        # Update dialog context
        g.context.update(ctx.intent, ctx.resolved_message)

        # Handle surprise — append to response
        if ctx.surprise:
            g.total_surprises += 1
            ctx.response = g._handle_surprise(ctx.surprise, ctx.response)
            g.traits.record_outcome("accuracy", "verify", "self_check", True)
            g.inner.record_surprise(ctx.surprise.surprise_magnitude)

        # Record trait outcome
        if ctx.trait_influence and ctx.trait_influence.active_traits:
            for t_name in ctx.trait_influence.active_traits:
                trait = g.traits.get_trait(t_name)
                if trait:
                    trait.strengthen(0.01)

        # Check proactive curiosity
        proactive = g._check_proactive_curiosity()
        if proactive:
            g.total_proactive += 1
            ctx.response += f"\n\nBy the way — {proactive}"

        return ctx
