"""
Failure Memory — learn from mistakes, don't repeat them.

Every failure is data. The system records:
- What it tried
- What went wrong
- What it should have done instead
- How to detect similar situations

This creates anti-patterns — approaches the system learns to AVOID.
More valuable than success patterns because failures are rarer and
more informative (Taleb's via negativa).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Failure:
    """A recorded failure with analysis."""
    description: str           # what happened
    context: str               # what were we trying to do
    approach_used: str         # what approach we took
    what_went_wrong: str       # why it failed
    better_approach: str = ""  # what we should have done (learned later)
    timestamp: str = ""
    times_repeated: int = 1    # how many times we've made this mistake
    pattern: str = ""          # generalized pattern to avoid

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "context": self.context,
            "approach": self.approach_used,
            "what_went_wrong": self.what_went_wrong,
            "better_approach": self.better_approach,
            "times_repeated": self.times_repeated,
            "pattern": self.pattern,
        }


class FailureMemory:
    """
    Stores and learns from failures.

    Not just a log — an active learning system that:
    1. Detects when we're about to repeat a mistake
    2. Suggests alternative approaches
    3. Generalizes failures into avoidable patterns
    """

    def __init__(self) -> None:
        self.failures: list[Failure] = []
        self.patterns: dict[str, str] = {}  # pattern → what to do instead

    def record(self, description: str, context: str,
               approach: str, what_went_wrong: str) -> Failure:
        """Record a failure."""
        # Check if this is a repeated mistake
        for existing in self.failures:
            if (existing.approach_used == approach and
                    self._similar(existing.context, context)):
                existing.times_repeated += 1
                if existing.times_repeated >= 3 and not existing.pattern:
                    existing.pattern = f"avoid_{approach}_when_{self._key(context)}"
                    self.patterns[existing.pattern] = f"Don't use '{approach}' in this context"
                return existing

        failure = Failure(
            description=description,
            context=context,
            approach_used=approach,
            what_went_wrong=what_went_wrong,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.failures.append(failure)
        return failure

    def record_correction(self, failure_desc: str, better_approach: str) -> None:
        """When the human corrects us, record what we should have done."""
        for f in reversed(self.failures):
            if self._similar(f.description, failure_desc):
                f.better_approach = better_approach
                return

    def check_approach(self, context: str, approach: str) -> str | None:
        """
        Before taking an action, check if it matches a known failure pattern.

        Returns warning message if this approach has failed before, None if safe.
        """
        for failure in self.failures:
            if (failure.approach_used == approach and
                    self._similar(failure.context, context)):
                if failure.better_approach:
                    return (f"Warning: '{approach}' failed before in similar context. "
                           f"Better approach: {failure.better_approach}")
                return f"Warning: '{approach}' failed before ({failure.what_went_wrong})"
        return None

    def get_anti_patterns(self) -> list[str]:
        """Return all learned anti-patterns."""
        return list(self.patterns.keys())

    def worst_mistakes(self, n: int = 5) -> list[Failure]:
        """Most repeated failures."""
        return sorted(self.failures, key=lambda f: f.times_repeated, reverse=True)[:n]

    def stats(self) -> dict:
        return {
            "total_failures": len(self.failures),
            "anti_patterns": len(self.patterns),
            "repeated_mistakes": sum(1 for f in self.failures if f.times_repeated > 1),
            "corrected": sum(1 for f in self.failures if f.better_approach),
        }

    def _similar(self, a: str, b: str) -> bool:
        """Quick similarity check — shared words."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
        return overlap > 0.3

    def _key(self, context: str) -> str:
        """Generate a short key from context."""
        words = context.lower().split()[:3]
        return "_".join(words)
