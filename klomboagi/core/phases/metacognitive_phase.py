"""Phase 4: Metacognitive — self-model, conflicts, workspace, modulators, inner state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext
from klomboagi.reasoning.global_workspace import SignalType

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class MetacognitivePhase(Phase):
    """Self-monitoring: working memory decay, self-model, conflict detection,
    global workspace broadcast, attention economy, modulators, inner state."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        # Working memory tick — decay unused items
        g.working_memory.tick()

        # Self-model snapshot
        gaps = len([ga for ga in g.base.curiosity.gaps if not ga.resolved])
        g.self_model.snapshot(
            g.base._beliefs, g.relations,
            g.base.memory.concepts, gaps)

        # NARS conflict check
        new_conflicts = g.conflict_detector.check(g.base._beliefs)
        for c in new_conflicts:
            g.workspace.submit(
                f"Conflict: {c.belief_a} vs {c.belief_b}",
                SignalType.CONFLICT, c.severity, "conflict_detector")

        # Global Workspace broadcast
        if g.context.current_topic:
            g.workspace.submit(
                g.context.current_topic, SignalType.PERCEPTION,
                0.5, "dialog_context")
        if g.inner.state.wonder > 0.3:
            g.workspace.submit(
                "surprise_detected", SignalType.EMOTION,
                g.inner.state.wonder, "inner_state")
        if g.inner.state.boredom > 0.5:
            g.workspace.submit(
                "bored_need_stimulation", SignalType.EMOTION,
                g.inner.state.boredom, "inner_state")

        broadcast = g.workspace.compete()
        g.workspace.broadcast()

        # Workspace winner influences working memory priority
        if broadcast and broadcast.signal_type == SignalType.CONFLICT:
            g.working_memory.focus_on(broadcast.content[:30])
        elif broadcast and broadcast.content:
            g.working_memory.attend(broadcast.content[:30], "broadcast", "workspace")

        # Attention economy tax
        g.attention_economy.tax()

        # Cognitive modulators
        g.modulator.update(g.inner.state, g.traits)

        # Inner state — compute how we "feel"
        g.inner.record_success()
        g.inner.compute(
            beliefs_in_focus=len(g.working_memory.get_active_items()),
            active_gaps=gaps,
            total_beliefs=len(g.base._beliefs),
            working_memory_items=len(g.working_memory._items),
        )

        return ctx
