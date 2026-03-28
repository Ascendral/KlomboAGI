"""
Expected Free Energy — principled explore vs exploit.

From Karl Friston's Active Inference framework.

The system should explore when uncertain, exploit when confident.
Not with ad-hoc heuristics — with a single mathematical function
that naturally balances both:

  G = E[information_gain] + E[pragmatic_value]

  information_gain: how much will I learn? (epistemic value)
  pragmatic_value: how much will it help my goals? (instrumental value)

When information_gain dominates → explore (I'm uncertain, learning helps)
When pragmatic_value dominates → exploit (I'm confident, act on what I know)

This REPLACES the ad-hoc curiosity/boredom/behavioral-loop decisions
with a single principled calculation.

Every possible action gets scored by G. Highest G wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ActionCandidate:
    """A possible action the system could take."""
    description: str
    action_type: str     # "answer", "search", "explore", "ask_human", "learn", "investigate"

    # Epistemic value — how much will I learn?
    uncertainty_before: float = 0.5   # current uncertainty about this domain
    expected_uncertainty_after: float = 0.3  # expected uncertainty after action

    # Pragmatic value — how much does this help my goals?
    goal_relevance: float = 0.5       # how relevant to current goal
    confidence_in_action: float = 0.5  # how likely to succeed

    @property
    def information_gain(self) -> float:
        """Expected reduction in uncertainty (epistemic value)."""
        return max(0, self.uncertainty_before - self.expected_uncertainty_after)

    @property
    def pragmatic_value(self) -> float:
        """Expected value toward goals."""
        return self.goal_relevance * self.confidence_in_action

    @property
    def expected_free_energy(self) -> float:
        """
        G = information_gain + pragmatic_value

        Negative free energy = good (we want to MINIMIZE surprise).
        But we maximize G for action selection.
        """
        return self.information_gain + self.pragmatic_value

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "type": self.action_type,
            "info_gain": round(self.information_gain, 3),
            "pragmatic": round(self.pragmatic_value, 3),
            "G": round(self.expected_free_energy, 3),
        }


class FreeEnergyMinimizer:
    """
    Selects actions by minimizing expected free energy.

    Given the current state, generates candidate actions,
    scores each by G, and returns the best one.
    """

    def __init__(self) -> None:
        self._history: list[dict] = []

    def select_action(self, context: dict) -> ActionCandidate:
        """
        Given the current context, select the best action.

        Context should contain:
        - query: what was asked
        - confidence: how confident are we (0-1)
        - has_beliefs: do we have relevant beliefs
        - has_relations: do we have relevant relations
        - active_gaps: number of knowledge gaps
        - goal: current goal if any
        """
        candidates = self._generate_candidates(context)

        # Score and sort by expected free energy
        candidates.sort(key=lambda c: c.expected_free_energy, reverse=True)

        best = candidates[0] if candidates else ActionCandidate(
            description="default", action_type="answer",
        )

        # Record decision
        self._history.append({
            "context": str(context)[:100],
            "selected": best.to_dict(),
            "candidates": len(candidates),
        })

        return best

    def _generate_candidates(self, ctx: dict) -> list[ActionCandidate]:
        """Generate possible actions for the current context."""
        candidates = []
        confidence = ctx.get("confidence", 0.5)
        has_beliefs = ctx.get("has_beliefs", False)
        has_relations = ctx.get("has_relations", False)
        active_gaps = ctx.get("active_gaps", 0)

        # Answer directly — good when confident
        if has_beliefs:
            candidates.append(ActionCandidate(
                description="Answer from existing knowledge",
                action_type="answer",
                uncertainty_before=1 - confidence,
                expected_uncertainty_after=1 - confidence,  # no new learning
                goal_relevance=0.8,
                confidence_in_action=confidence,
            ))

        # Search for more info — good when uncertain
        candidates.append(ActionCandidate(
            description="Search for information",
            action_type="search",
            uncertainty_before=1 - confidence,
            expected_uncertainty_after=max(0.1, (1 - confidence) * 0.5),  # search reduces uncertainty
            goal_relevance=0.6,
            confidence_in_action=0.4,
        ))

        # Explore related topics — good when many gaps
        if active_gaps > 2:
            candidates.append(ActionCandidate(
                description="Explore knowledge gaps",
                action_type="explore",
                uncertainty_before=0.8,
                expected_uncertainty_after=0.4,
                goal_relevance=0.3,
                confidence_in_action=0.5,
            ))

        # Ask the human — good when very uncertain AND gaps are critical
        if confidence < 0.3:
            candidates.append(ActionCandidate(
                description="Ask the human for help",
                action_type="ask_human",
                uncertainty_before=1 - confidence,
                expected_uncertainty_after=0.1,  # human answer = very informative
                goal_relevance=0.9,
                confidence_in_action=0.8,
            ))

        # Investigate deeper — good when we have SOME knowledge but not enough
        if has_beliefs and confidence < 0.6:
            candidates.append(ActionCandidate(
                description="Investigate deeper with reasoning",
                action_type="investigate",
                uncertainty_before=1 - confidence,
                expected_uncertainty_after=max(0.1, (1 - confidence) * 0.6),
                goal_relevance=0.7,
                confidence_in_action=0.5,
            ))

        return candidates

    def explain_last(self) -> str:
        """Explain the last action selection."""
        if not self._history:
            return "No actions taken yet."
        last = self._history[-1]
        sel = last["selected"]
        return (
            f"Selected: {sel['description']} (G={sel['G']:.2f})\n"
            f"  Information gain: {sel['info_gain']:.2f}\n"
            f"  Pragmatic value: {sel['pragmatic']:.2f}\n"
            f"  From {last['candidates']} candidates"
        )
