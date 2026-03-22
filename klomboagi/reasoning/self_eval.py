"""
Self-Evaluation Engine — the system that checks its own thinking.

A real intelligence doesn't just produce an answer. It:
1. Produces a candidate answer
2. Checks if the answer makes sense
3. Looks for contradictions
4. Tries a DIFFERENT approach if the first one fails
5. Knows when it's confident vs when it's guessing

The alligator lesson: I gave the "obvious" answer (longer).
A human thought deeper and found a better answer (greener — 2D > 1D).
The self-evaluator should have caught that the obvious answer
wasn't the only valid structural interpretation.

This is not a confidence score bolted on after the fact.
This is an active process that runs DURING reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from klomboagi.utils.time import utc_now


@dataclass
class ReasoningAttempt:
    """One attempt at answering/solving something."""
    id: int
    approach: str               # Description of the approach taken
    answer: Any                 # The candidate answer
    reasoning_chain: list[str]  # Steps of reasoning
    confidence: float           # Self-assessed confidence
    assumptions: list[str]      # What assumptions were made
    alternatives_considered: int  # How many other approaches were considered
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "approach": self.approach,
            "answer": self.answer,
            "reasoning_chain": self.reasoning_chain,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "alternatives_considered": self.alternatives_considered,
            "created_at": self.created_at,
        }


@dataclass
class EvaluationResult:
    """Result of self-evaluating a reasoning attempt."""
    attempt_id: int
    passed: bool
    issues: list[str]           # Problems found
    suggestions: list[str]      # How to improve
    should_retry: bool          # Should we try a different approach?
    retry_hint: str             # What to try differently

    def to_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "passed": self.passed,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "should_retry": self.should_retry,
            "retry_hint": self.retry_hint,
        }


class SelfEvaluator:
    """
    Evaluates the system's own reasoning and triggers re-thinking.

    Core checks:
    1. Consistency — does the answer contradict known facts?
    2. Completeness — did we consider all relevant factors?
    3. Assumptions — are we making unwarranted assumptions?
    4. Alternatives — did we consider other interpretations?
    5. Dimensionality — are we looking at this from enough angles?
    """

    def __init__(self) -> None:
        self.attempts: list[ReasoningAttempt] = []
        self.evaluations: list[EvaluationResult] = []
        self._attempt_counter = 0
        self.max_retries = 3

    def attempt(
        self,
        approach: str,
        answer: Any,
        reasoning_chain: list[str],
        assumptions: list[str] | None = None,
    ) -> ReasoningAttempt:
        """Record a reasoning attempt."""
        self._attempt_counter += 1
        attempt = ReasoningAttempt(
            id=self._attempt_counter,
            approach=approach,
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence=self._assess_confidence(reasoning_chain, assumptions or []),
            assumptions=assumptions or [],
            alternatives_considered=len(self.attempts),
            created_at=utc_now(),
        )
        self.attempts.append(attempt)
        return attempt

    def evaluate(self, attempt: ReasoningAttempt, known_facts: list[str] | None = None) -> EvaluationResult:
        """
        Evaluate a reasoning attempt. This is where thinking about thinking happens.
        """
        issues = []
        suggestions = []

        # Check 1: Consistency with known facts
        if known_facts:
            contradictions = self._check_contradictions(attempt, known_facts)
            if contradictions:
                issues.extend(contradictions)
                suggestions.append("Re-examine answer against known facts")

        # Check 2: Reasoning chain completeness
        chain_issues = self._check_chain(attempt.reasoning_chain)
        if chain_issues:
            issues.extend(chain_issues)
            suggestions.append("Fill gaps in reasoning chain")

        # Check 3: Assumption audit
        assumption_issues = self._check_assumptions(attempt.assumptions)
        if assumption_issues:
            issues.extend(assumption_issues)
            suggestions.append("Question assumptions and try without them")

        # Check 4: Single-perspective bias
        if attempt.alternatives_considered == 0:
            issues.append("No alternative approaches were considered")
            suggestions.append("Try at least one different approach before committing")

        # Check 5: Dimensionality check — are we looking at this from enough angles?
        if len(attempt.reasoning_chain) < 2:
            issues.append("Reasoning chain too short — may be jumping to conclusions")
            suggestions.append("Break down the problem into more steps")

        # Check 6: Confidence calibration
        if attempt.confidence > 0.8 and attempt.alternatives_considered == 0:
            issues.append("High confidence with no alternatives considered — likely overconfident")
            suggestions.append("Reduce confidence or explore alternatives")

        # Determine if we should retry
        should_retry = len(issues) > 0 and len(self.attempts) < self.max_retries
        retry_hint = ""
        if should_retry:
            if "contradiction" in str(issues).lower():
                retry_hint = "reverse_assumption"
            elif "alternative" in str(issues).lower():
                retry_hint = "different_perspective"
            elif "gap" in str(issues).lower():
                retry_hint = "add_detail"
            elif "overconfident" in str(issues).lower():
                retry_hint = "consider_uncertainty"
            else:
                retry_hint = "restructure"

        result = EvaluationResult(
            attempt_id=attempt.id,
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            should_retry=should_retry,
            retry_hint=retry_hint,
        )
        self.evaluations.append(result)
        return result

    def nudge(self, direction: str) -> str:
        """
        Human nudge — shift reasoning approach.

        This is the interface where a human says "think about it differently"
        and the system actually changes HOW it's decomposing the problem.

        Nudge types:
        - "reverse" — flip the assumption
        - "dimensional" — consider more dimensions
        - "simplify" — strip away complexity
        - "perspective" — look from the other side
        - "literal" — take it literally, not abstractly
        - "abstract" — take it abstractly, not literally
        - "question" — what am I not asking?
        """
        nudge_strategies = {
            "reverse": "Flip your main assumption. If you assumed X, try not-X.",
            "dimensional": "You're looking at this in too few dimensions. What properties are you ignoring? Consider surface area, volume, time, not just the obvious axis.",
            "simplify": "Strip away everything except the core structure. What's the simplest version of this problem?",
            "perspective": "Look at this from the other entity's perspective. What would THEY say?",
            "literal": "Stop abstracting. Take the question literally. What does each word actually mean?",
            "abstract": "Stop being literal. What's the structural pattern underneath?",
            "question": "You're trying to answer. Stop. What question should you be asking instead?",
            "both": "Both answers might be correct. Under what conditions is each one true?",
        }

        strategy = nudge_strategies.get(direction, f"Shift approach: {direction}")

        # Clear confidence on the current best attempt
        if self.attempts:
            self.attempts[-1].confidence *= 0.5  # Cut confidence — human says we're wrong

        return strategy

    def resolve(self) -> dict:
        """
        After attempts and evaluations, produce the best answer with full context.

        Not just the answer — the entire reasoning audit trail.
        """
        if not self.attempts:
            return {"answer": None, "reason": "No attempts made"}

        # Find the best attempt (highest confidence that passed evaluation)
        best = None
        for attempt in reversed(self.attempts):
            eval_result = next(
                (e for e in self.evaluations if e.attempt_id == attempt.id),
                None
            )
            if eval_result and eval_result.passed:
                best = attempt
                break

        if best is None:
            # No attempt passed — return the last one with caveats
            best = self.attempts[-1]

        return {
            "answer": best.answer,
            "confidence": best.confidence,
            "approach": best.approach,
            "reasoning": best.reasoning_chain,
            "assumptions": best.assumptions,
            "attempts_made": len(self.attempts),
            "issues_found": sum(len(e.issues) for e in self.evaluations),
            "all_attempts": [a.to_dict() for a in self.attempts],
            "all_evaluations": [e.to_dict() for e in self.evaluations],
        }

    # ── Internal checks ──

    def _assess_confidence(self, chain: list[str], assumptions: list[str]) -> float:
        """Estimate confidence based on reasoning quality."""
        confidence = 0.5  # Start neutral

        # Longer reasoning chain = more thorough (up to a point)
        chain_bonus = min(len(chain) / 5, 0.2)
        confidence += chain_bonus

        # More assumptions = less confident
        assumption_penalty = min(len(assumptions) * 0.1, 0.3)
        confidence -= assumption_penalty

        # Previous failures reduce confidence
        failed_evals = sum(1 for e in self.evaluations if not e.passed)
        failure_penalty = min(failed_evals * 0.15, 0.3)
        confidence -= failure_penalty

        return max(0.05, min(0.95, confidence))

    def _check_contradictions(self, attempt: ReasoningAttempt, known_facts: list[str]) -> list[str]:
        """Check if the answer contradicts known facts."""
        issues = []
        answer_str = str(attempt.answer).lower()
        for fact in known_facts:
            fact_lower = fact.lower()
            # Simple negation check
            if f"not {answer_str}" in fact_lower or f"no {answer_str}" in fact_lower:
                issues.append(f"Answer may contradict known fact: '{fact}'")
        return issues

    def _check_chain(self, chain: list[str]) -> list[str]:
        """Check reasoning chain for gaps."""
        issues = []
        if len(chain) == 0:
            issues.append("Empty reasoning chain — no justification for answer")
        elif len(chain) == 1:
            issues.append("Single-step reasoning — likely jumping to conclusion")

        # Check for "because" or reasoning connectors
        has_justification = any(
            word in step.lower()
            for step in chain
            for word in ["because", "therefore", "since", "due to", "implies", "causes"]
        )
        if not has_justification and len(chain) > 0:
            issues.append("No causal justification in reasoning chain")

        return issues

    def _check_assumptions(self, assumptions: list[str]) -> list[str]:
        """Flag potentially dangerous assumptions."""
        issues = []
        dangerous_words = ["always", "never", "all", "none", "must", "impossible", "obvious", "clearly"]
        for assumption in assumptions:
            for word in dangerous_words:
                if word in assumption.lower():
                    issues.append(f"Strong assumption detected: '{assumption}' (contains '{word}')")
                    break
        return issues
