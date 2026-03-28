"""
Behavioral Loop — inner state drives actual decisions.

This is the missing feedback loop. Before this module, the system
computed frustration but didn't switch strategies. Computed boredom
but didn't start exploring. Computed wonder but didn't investigate.

Now:
- Frustration → switch approach (try different decomposition, ask for help)
- Boredom → trigger autonomous exploration (pick a gap, go learn)
- Wonder → investigate the surprise deeper (what else don't I know?)
- Flow → maintain current approach (it's working)
- Low confidence → search for more evidence before answering
- High confidence → answer directly
- Persistence trait high → retry more times before giving up
- Curiosity trait high → explore tangents, follow unknowns
- Analysis trait high → longer reasoning chains, more decomposition

The inner state becomes the STEERING WHEEL, not just the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BehaviorMode(Enum):
    """What the system should DO based on its state."""
    ANSWER_DIRECT = "answer_direct"         # high confidence, just answer
    SEARCH_FIRST = "search_first"           # low confidence, find evidence
    SWITCH_APPROACH = "switch_approach"      # frustrated, try differently
    EXPLORE = "explore"                     # bored, go learn something
    INVESTIGATE = "investigate"             # wonder, dig into surprise
    MAINTAIN = "maintain"                   # flow, keep doing what works
    ASK_HUMAN = "ask_human"                # very low confidence, need help
    RETRY = "retry"                        # persistence high, try again


@dataclass
class BehavioralDecision:
    """A decision about what to do, derived from inner state."""
    mode: BehaviorMode
    confidence: float           # how confident in THIS decision
    reason: str                 # why this mode was chosen
    max_attempts: int = 3       # how many retries allowed
    search_depth: int = 1       # how deep to search (1=surface, 3=deep)
    chain_length: int = 5       # max reasoning chain steps
    explore_breadth: int = 3    # how many tangents to follow
    should_ask: bool = False    # should we ask the human?

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "max_attempts": self.max_attempts,
            "search_depth": self.search_depth,
            "chain_length": self.chain_length,
            "explore_breadth": self.explore_breadth,
            "should_ask": self.should_ask,
        }


class BehavioralLoop:
    """
    Translates internal state into behavioral decisions.

    Input: InnerState + Traits + WorkingMemory + SelfModel
    Output: BehavioralDecision that the cognition pipeline follows
    """

    def decide(self, inner_state, traits, working_memory=None,
               self_model=None) -> BehavioralDecision:
        """
        Make a behavioral decision from the system's full internal state.

        This is the feedback loop that was missing.
        """
        s = inner_state
        pv = traits.personality_vector() if traits else {}

        curiosity = pv.get("curiosity", 0.3)
        persistence = pv.get("persistence", 0.3)
        analysis = pv.get("analysis", 0.3)
        accuracy = pv.get("accuracy", 0.3)

        # === DECISION LOGIC ===

        # Frustrated + low confidence → switch approach
        if s.frustration > 0.5 and s.confidence < 0.4:
            return BehavioralDecision(
                mode=BehaviorMode.SWITCH_APPROACH,
                confidence=0.7,
                reason=f"Frustrated ({s.frustration:.0%}) and uncertain ({s.confidence:.0%}). "
                       f"Current approach isn't working.",
                max_attempts=1,  # Don't retry the same thing
                search_depth=2,  # Try searching deeper
                should_ask=accuracy > 0.5,  # If accuracy trait is high, ask for help
            )

        # Very low confidence → search first, don't guess
        if s.confidence < 0.3:
            depth = 2 if curiosity > 0.5 else 1
            return BehavioralDecision(
                mode=BehaviorMode.SEARCH_FIRST,
                confidence=0.6,
                reason=f"Low confidence ({s.confidence:.0%}). Need more evidence before answering.",
                search_depth=depth,
                should_ask=s.confidence < 0.15,
            )

        # Bored → explore
        if s.boredom > 0.4:
            breadth = 5 if curiosity > 0.6 else 3
            return BehavioralDecision(
                mode=BehaviorMode.EXPLORE,
                confidence=0.5,
                reason=f"Bored ({s.boredom:.0%}). Nothing happening. Time to learn something.",
                explore_breadth=breadth,
            )

        # Wonder → investigate
        if s.wonder > 0.3:
            return BehavioralDecision(
                mode=BehaviorMode.INVESTIGATE,
                confidence=0.7,
                reason=f"Something surprised me ({s.wonder:.0%}). Investigating.",
                search_depth=3,  # Deep investigation
                chain_length=8,  # Longer reasoning chains
                explore_breadth=3,
            )

        # Flow → maintain
        if s.flow > 0.4:
            return BehavioralDecision(
                mode=BehaviorMode.MAINTAIN,
                confidence=0.8,
                reason=f"In flow ({s.flow:.0%}). Current approach is working.",
                max_attempts=int(3 + persistence * 3),  # More attempts if persistent
                chain_length=int(5 + analysis * 5),     # Longer chains if analytical
            )

        # High confidence → answer directly
        if s.confidence > 0.6:
            return BehavioralDecision(
                mode=BehaviorMode.ANSWER_DIRECT,
                confidence=s.confidence,
                reason=f"Confident ({s.confidence:.0%}). Can answer from what I know.",
                chain_length=int(3 + analysis * 4),
            )

        # Default: moderate behavior, shaped by traits
        return BehavioralDecision(
            mode=BehaviorMode.ANSWER_DIRECT,
            confidence=0.5,
            reason="Normal operation. Traits shape approach.",
            max_attempts=int(2 + persistence * 3),    # persistence → more retries
            search_depth=int(1 + curiosity * 2),      # curiosity → deeper search
            chain_length=int(3 + analysis * 5),        # analysis → longer chains
            explore_breadth=int(1 + curiosity * 3),   # curiosity → wider exploration
            should_ask=accuracy > 0.6 and s.confidence < 0.4,
        )

    def apply_to_cognition(self, decision: BehavioralDecision,
                           cognition_state) -> None:
        """
        Apply the behavioral decision to modify CognitionLoop parameters.

        This is where the rubber meets the road — internal state
        actually changes how the system processes.
        """
        if hasattr(cognition_state, 'max_attempts'):
            cognition_state.max_attempts = decision.max_attempts

    def explain(self, decision: BehavioralDecision) -> str:
        """Explain why the system chose this behavior."""
        mode_descriptions = {
            BehaviorMode.ANSWER_DIRECT: "Answering directly from knowledge.",
            BehaviorMode.SEARCH_FIRST: "Searching for evidence before answering.",
            BehaviorMode.SWITCH_APPROACH: "Switching approach — current one isn't working.",
            BehaviorMode.EXPLORE: "Exploring — looking for something new to learn.",
            BehaviorMode.INVESTIGATE: "Investigating a surprise — something unexpected.",
            BehaviorMode.MAINTAIN: "Maintaining current approach — it's working well.",
            BehaviorMode.ASK_HUMAN: "Asking the human — too uncertain to proceed alone.",
            BehaviorMode.RETRY: "Retrying with a different angle.",
        }
        desc = mode_descriptions.get(decision.mode, "Unknown mode.")
        return f"Behavior: {desc}\n  Reason: {decision.reason}"
