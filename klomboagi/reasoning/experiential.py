"""
Experiential Learning — learn by trying, failing, and adjusting.

Not learning from being TOLD. Learning from ATTEMPTING.

The cycle:
  1. ATTEMPT — try to answer/solve something
  2. EVALUATE — check if the attempt was correct
  3. ANALYZE — if wrong, figure out WHY
  4. ADJUST — change the approach based on analysis
  5. RETRY — try again with the adjusted approach
  6. STORE — save what worked as a new pattern

This is how humans learn:
  - Try to ride a bike → fall → analyze (leaned too far) → adjust → try again
  - Try to solve an equation → get wrong answer → check work → find error → fix
  - Try to explain something → listener confused → rephrase → listener understands

For KlomboAGI:
  - Try to answer "what is consciousness?" → weak answer → analyze (no deep knowledge)
    → search for more → try again with new knowledge → better answer → store the approach
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Attempt:
    """One attempt at solving/answering something."""
    question: str
    approach: str          # how we tried to answer
    answer: str            # what we produced
    confidence: float      # how confident we were
    timestamp: float = 0.0


@dataclass
class Evaluation:
    """Assessment of an attempt."""
    attempt: Attempt
    succeeded: bool
    feedback: str          # what was wrong or right
    error_type: str = ""   # "no_knowledge", "wrong_fact", "shallow", "off_topic"


@dataclass
class Lesson:
    """What was learned from an attempt cycle."""
    question: str
    failed_approach: str
    successful_approach: str
    insight: str           # what we learned
    confidence: float
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question[:60],
            "failed": self.failed_approach[:40],
            "succeeded": self.successful_approach[:40],
            "insight": self.insight[:80],
            "confidence": round(self.confidence, 3),
        }


class ExperientialLearner:
    """
    Learn from trying, failing, analyzing, and retrying.

    Maintains a library of lessons learned from past attempts.
    Before answering, checks if we've faced a similar question
    and applies the learned approach.
    """

    MAX_RETRIES = 3

    def __init__(self) -> None:
        self.lessons: list[Lesson] = []
        self.attempts: list[Attempt] = []
        self._current_question: str = ""

    def attempt(self, question: str, approach: str, answer: str,
                confidence: float) -> Attempt:
        """Record an attempt at answering something."""
        att = Attempt(
            question=question, approach=approach,
            answer=answer, confidence=confidence,
            timestamp=time.time(),
        )
        self.attempts.append(att)
        self._current_question = question
        return att

    def evaluate(self, attempt: Attempt, succeeded: bool,
                 feedback: str = "", error_type: str = "") -> Evaluation:
        """Evaluate whether an attempt succeeded."""
        return Evaluation(
            attempt=attempt, succeeded=succeeded,
            feedback=feedback, error_type=error_type,
        )

    def learn_from_failure(self, evaluation: Evaluation,
                           new_approach: str, new_answer: str) -> Lesson | None:
        """
        After a failed attempt, record what we learned.

        The lesson: "For questions like X, approach A failed because Y.
        Approach B worked because Z."
        """
        if evaluation.succeeded:
            return None  # Nothing to learn from success... yet

        lesson = Lesson(
            question=evaluation.attempt.question,
            failed_approach=evaluation.attempt.approach,
            successful_approach=new_approach,
            insight=f"'{evaluation.attempt.approach}' failed ({evaluation.error_type}). "
                    f"'{new_approach}' worked better.",
            confidence=0.6,
            created_at=time.time(),
        )
        self.lessons.append(lesson)
        return lesson

    def learn_from_success(self, attempt: Attempt) -> Lesson:
        """Record what worked so we can repeat it."""
        lesson = Lesson(
            question=attempt.question,
            failed_approach="",
            successful_approach=attempt.approach,
            insight=f"'{attempt.approach}' worked for '{attempt.question[:30]}...'",
            confidence=attempt.confidence,
            created_at=time.time(),
        )
        self.lessons.append(lesson)
        return lesson

    def suggest_approach(self, question: str) -> str | None:
        """
        Before attempting, check if we've learned anything
        from similar past questions.
        """
        q_words = set(question.lower().split())
        best_match = None
        best_overlap = 0

        for lesson in self.lessons:
            lesson_words = set(lesson.question.lower().split())
            overlap = len(q_words & lesson_words)
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_match = lesson

        if best_match:
            return best_match.successful_approach
        return None

    def try_and_learn(self, question: str, attempt_func, evaluate_func,
                      adjust_func=None) -> dict:
        """
        Full experiential learning cycle:
        try → evaluate → if failed: analyze, adjust, retry → store lesson

        attempt_func(question, approach) → answer
        evaluate_func(answer) → (succeeded, feedback, error_type)
        adjust_func(approach, error_type) → new_approach
        """
        # Check for past lessons
        suggested = self.suggest_approach(question)
        approach = suggested or "default"

        for retry in range(self.MAX_RETRIES):
            # ATTEMPT
            answer = attempt_func(question, approach)
            att = self.attempt(question, approach, answer, 0.5)

            # EVALUATE
            succeeded, feedback, error_type = evaluate_func(answer)
            evaluation = self.evaluate(att, succeeded, feedback, error_type)

            if succeeded:
                self.learn_from_success(att)
                return {
                    "answer": answer,
                    "attempts": retry + 1,
                    "approach": approach,
                    "succeeded": True,
                }

            # ANALYZE + ADJUST
            if adjust_func:
                approach = adjust_func(approach, error_type)
            else:
                # Default adjustment strategies
                if error_type == "no_knowledge":
                    approach = "search_then_answer"
                elif error_type == "shallow":
                    approach = "deep_reasoning"
                elif error_type == "wrong_fact":
                    approach = "verify_then_answer"
                else:
                    approach = f"retry_{retry}"

        # All retries failed
        return {
            "answer": answer,
            "attempts": self.MAX_RETRIES,
            "approach": approach,
            "succeeded": False,
        }

    def stats(self) -> dict:
        return {
            "total_attempts": len(self.attempts),
            "lessons_learned": len(self.lessons),
            "success_rate": (
                sum(1 for l in self.lessons if l.successful_approach)
                / max(1, len(self.lessons))
            ),
        }
