"""
Multi-Step Problem Solver — plan, execute steps, verify each one.

Real problems aren't answered in one shot. They require:
  1. Understand what's being asked
  2. Break into steps
  3. Execute each step using the right system
  4. Verify the result of each step
  5. If a step fails, replan
  6. Combine step results into final answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SolveStep:
    """One step in a multi-step solution."""
    description: str
    step_type: str       # "lookup", "compute", "reason", "search", "connect"
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    confidence: float = 0.0
    depends_on: list[int] = field(default_factory=list)


@dataclass
class Solution:
    """A multi-step solution."""
    question: str
    steps: list[SolveStep] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0

    def explain(self) -> str:
        lines = []
        for i, step in enumerate(self.steps):
            icon = "✓" if step.status == StepStatus.COMPLETED else "✗" if step.status == StepStatus.FAILED else "○"
            lines.append(f"  {icon} {step.description}")
            if step.result:
                lines.append(f"    → {step.result[:70]}")
        if self.answer:
            lines.append(f"\n{self.answer}")
        return "\n".join(lines)


class MultiStepSolver:
    """Plan and execute multi-step solutions."""

    def __init__(self, genesis) -> None:
        self.g = genesis

    def solve(self, question: str) -> Solution:
        """Plan, execute, combine."""
        sol = self._plan(question)
        for step in sol.steps:
            if step.status == StepStatus.PENDING:
                self._execute(step, sol)
        sol.answer = self._combine(sol)
        sol.confidence = self._score(sol)
        return sol

    def _plan(self, question: str) -> Solution:
        sol = Solution(question=question)
        stop = {"what", "is", "a", "an", "the", "how", "why", "can", "does",
                "do", "much", "many", "would", "could", "about", "you", "your",
                "think", "know", "tell", "me"}
        terms = [w.lower().strip("?.,!") for w in question.split()
                 if w.lower().strip("?.,!") not in stop and len(w) > 2]

        for term in terms[:4]:
            sol.steps.append(SolveStep(
                description=f"What is {term}?", step_type="lookup"))

        if len(terms) >= 2:
            sol.steps.append(SolveStep(
                description=f"How do {terms[0]} and {terms[-1]} connect?",
                step_type="connect"))

        if any(w in question.lower() for w in ["how much", "how many", "calculate"]):
            sol.steps.append(SolveStep(
                description="Calculate", step_type="compute"))

        sol.steps.append(SolveStep(
            description="Combine into answer", step_type="combine"))
        return sol

    def _execute(self, step: SolveStep, sol: Solution) -> None:
        try:
            if step.step_type == "lookup":
                concept = step.description.replace("What is ", "").rstrip("?")
                for stmt, b in self.g.base._beliefs.items():
                    if hasattr(b, 'subject') and b.subject == concept.lower():
                        if b.predicate and len(b.predicate) < 80:
                            step.result = f"{concept} is {b.predicate}"
                            step.confidence = b.truth.confidence if hasattr(b, 'truth') else 0.5
                            step.status = StepStatus.COMPLETED
                            return
                step.result = f"Unknown: {concept}"
                step.confidence = 0.1
                step.status = StepStatus.COMPLETED

            elif step.step_type == "connect":
                parts = step.description.replace("How do ", "").replace(" connect?", "").split(" and ")
                if len(parts) == 2 and hasattr(self.g, 'relations'):
                    path = self.g.relations.find_path(parts[0].strip(), parts[1].strip())
                    if path:
                        chain = " → ".join(f"{r.target}" for r in path)
                        step.result = f"{parts[0]} connects to {parts[1]}: {chain}"
                        step.confidence = 0.6
                    else:
                        step.result = f"No direct connection found"
                        step.confidence = 0.2
                step.status = StepStatus.COMPLETED

            elif step.step_type == "compute":
                result = self.g.compute.compute(sol.question)
                if result.success:
                    step.result = str(result.result)
                    step.confidence = 0.9
                step.status = StepStatus.COMPLETED

            elif step.step_type == "combine":
                done = [s for s in sol.steps if s.status == StepStatus.COMPLETED
                       and s.result and s.confidence > 0.3 and s.step_type != "combine"]
                step.result = ". ".join(s.result for s in done)
                step.confidence = sum(s.confidence for s in done) / max(1, len(done))
                step.status = StepStatus.COMPLETED

        except Exception:
            step.status = StepStatus.FAILED

    def _combine(self, sol: Solution) -> str:
        combine = next((s for s in sol.steps if s.step_type == "combine" and s.result), None)
        return combine.result if combine else "Could not solve."

    def _score(self, sol: Solution) -> float:
        done = [s for s in sol.steps if s.status == StepStatus.COMPLETED]
        return sum(s.confidence for s in done) / max(1, len(done)) if done else 0.0
