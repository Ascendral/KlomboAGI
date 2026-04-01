"""Phase 6: Learning — meta-learning, strengthening, dedup, calibration, refresh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klomboagi.core.phases import Phase, HearContext

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class LearningPhase(Phase):
    """Periodic maintenance: meta-learning records, belief strengthening,
    deduplication, confidence calibration, auto-refresh of stale concepts."""

    def run(self, ctx: HearContext, g: "Genesis") -> HearContext:
        # Meta-Learning — track what learning approaches work
        if ctx.intent["type"] == "question":
            has_answer = len(ctx.response) > 20 and "don't know" not in ctx.response.lower()
            domain = g.context.current_topic or "general"
            g.meta_learner.record(
                method="conversation", domain=domain,
                duration_ms=0,
                facts_gained=1 if has_answer else 0,
                success=has_answer)

        # Belief Strengthening — periodically cross-confirm
        if g.total_turns % 10 == 0 and len(g.base._beliefs) > 20:
            g.belief_strengthener.cross_confirm()

        # Dedup — periodically clean near-duplicate beliefs
        if g.total_turns % 20 == 0 and len(g.base._beliefs) > 50:
            g.deduplicator.deduplicate(g.base._beliefs)

        # Confidence Calibration
        if g.total_turns % 15 == 0 and len(g.base._beliefs) > 30:
            has_answer = len(ctx.response) > 20 and "don't know" not in ctx.response.lower()
            raw_conf = 0.6 if has_answer else 0.2
            g.calibrator.record(raw_conf, has_answer)

        # Auto-Refresh — re-read stale concepts periodically
        if g.total_turns % 50 == 0 and g.total_turns > 0:
            try:
                g.refresher.refresh(max_concepts=2)
            except Exception:
                pass

        # Global inference — derive new beliefs from existing ones
        # Runs after teaches (when new facts are added) and periodically
        if ctx.intent.get("type") == "teach" or g.total_turns % 5 == 0:
            try:
                derived = g.inference_engine.run(max_derivations=50)
                if derived:
                    g.belief_index.build(g.base._beliefs)
            except Exception:
                pass

        return ctx
