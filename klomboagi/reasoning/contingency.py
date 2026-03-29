"""
Contingency Planning — what's plan B?

Real intelligence doesn't just plan. It plans for FAILURE.
  "If step 3 fails, what do I do?"
  "If search returns nothing, try first principles."
  "If approach A costs too much, switch to approach B."

Each plan step gets a fallback. The system picks fallbacks
based on cost tracker data (what's cheapest that works).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContingencyStep:
    """A plan step with a fallback."""
    primary: str            # what to try first
    fallback: str           # what to try if primary fails
    fallback_trigger: str   # condition that triggers fallback
    primary_cost_est: float = 0.0  # estimated cost in ms
    fallback_cost_est: float = 0.0


@dataclass
class ContingencyPlan:
    """A plan where every step has a backup."""
    goal: str
    steps: list[ContingencyStep] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"Plan for: {self.goal}"]
        for i, step in enumerate(self.steps):
            lines.append(f"  Step {i+1}: {step.primary}")
            lines.append(f"    If fails ({step.fallback_trigger}): {step.fallback}")
        return "\n".join(lines)


class ContingencyPlanner:
    """
    Creates plans with fallbacks at every step.

    Uses cost tracker to pick fallbacks that are cheap and effective.
    """

    def __init__(self, cost_tracker=None) -> None:
        self.cost_tracker = cost_tracker

    def plan(self, goal: str, approaches: list[str] = None) -> ContingencyPlan:
        """Create a contingency plan for a goal."""
        plan = ContingencyPlan(goal=goal)

        # Default approaches with fallbacks
        default_steps = [
            ContingencyStep(
                primary="Check compiled chunks (instant recall)",
                fallback="Check direct beliefs",
                fallback_trigger="no chunk matches",
            ),
            ContingencyStep(
                primary="Search beliefs for relevant facts",
                fallback="Reason from first principles",
                fallback_trigger="no relevant beliefs found",
            ),
            ContingencyStep(
                primary="Generate explanation from relations",
                fallback="Search Wikipedia for information",
                fallback_trigger="no relations found for concept",
            ),
            ContingencyStep(
                primary="Construct answer from fragments",
                fallback="Admit unknown and offer to learn",
                fallback_trigger="confidence below 30%",
            ),
        ]

        # If cost tracker available, optimize fallback order
        if self.cost_tracker:
            cheapest = self.cost_tracker.cheapest_successful()
            if cheapest:
                default_steps.insert(0, ContingencyStep(
                    primary=f"Try {cheapest} (historically cheapest)",
                    fallback=default_steps[0].primary,
                    fallback_trigger="fails or too slow",
                ))

        plan.steps = default_steps
        return plan

    def plan_for_question(self, question: str) -> ContingencyPlan:
        """Create a specific contingency plan for answering a question."""
        plan = ContingencyPlan(goal=f"Answer: {question}")

        # Determine question complexity
        words = question.lower().split()
        is_complex = len(words) > 8 or "how" in words or "why" in words

        if is_complex:
            plan.steps = [
                ContingencyStep(
                    primary="Decompose into sub-questions",
                    fallback="Answer the whole question directly",
                    fallback_trigger="decomposition produces nothing useful",
                ),
                ContingencyStep(
                    primary="Answer each sub-question",
                    fallback="Answer what we can, mark gaps",
                    fallback_trigger="some sub-questions unanswerable",
                ),
                ContingencyStep(
                    primary="Synthesize sub-answers into coherent response",
                    fallback="Return partial answer with caveats",
                    fallback_trigger="sub-answers don't combine cleanly",
                ),
            ]
        else:
            plan.steps = [
                ContingencyStep(
                    primary="Direct belief lookup",
                    fallback="Constructive memory reconstruction",
                    fallback_trigger="no direct belief",
                ),
                ContingencyStep(
                    primary="Generate from relations",
                    fallback="First principles reasoning",
                    fallback_trigger="no relations found",
                ),
                ContingencyStep(
                    primary="Return confident answer",
                    fallback="Search and learn, then retry",
                    fallback_trigger="confidence below threshold",
                ),
            ]

        return plan
