"""
Cost Tracker — economics teaches better than warnings.

Every operation has a cost: time, tokens, API calls, memory.
The system tracks these costs per task type. When a pattern
is expensive AND fails, the next attempt sees that cost and
chooses a cheaper approach.

"This approach failed 3 times and cost 47 seconds" →
the system optimizes for both correctness AND efficiency.

Cost-aware decision making:
  - If search costs 5s and direct answer costs 0.01s, prefer direct answer
  - If approach X failed 3 times at 10s each = 30s wasted, try approach Y
  - If learning a topic takes 60s but answers 10 future questions, good ROI
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CostEntry:
    """Cost of a single operation."""
    operation: str          # "search", "reason", "infer", "read_wiki", "answer"
    duration_ms: float
    success: bool
    tokens_used: int = 0    # if LLM was involved
    facts_gained: int = 0   # what we got for the cost

    @property
    def efficiency(self) -> float:
        """Facts gained per second. Higher = more efficient."""
        if self.duration_ms <= 0:
            return 0.0
        return self.facts_gained / (self.duration_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 1),
            "success": self.success,
            "tokens": self.tokens_used,
            "facts_gained": self.facts_gained,
            "efficiency": round(self.efficiency, 2),
        }


class CostTracker:
    """
    Tracks resource costs and learns to optimize.

    Every operation recorded. Patterns emerge:
    - "search" averages 3s, succeeds 70%
    - "wiki_read" averages 8s, gains 15 facts
    - "first_principles" averages 0.5s, succeeds 40%
    - "hypothesis" averages 0.1s, succeeds 30%

    The system uses these stats to pick the cheapest approach
    that's likely to succeed.
    """

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []
        self._by_operation: dict[str, list[CostEntry]] = defaultdict(list)
        self._active_timer: dict[str, float] = {}

    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self._active_timer[operation] = time.time()

    def end(self, operation: str, success: bool = True,
            tokens: int = 0, facts_gained: int = 0) -> CostEntry:
        """End timing and record the cost."""
        start_time = self._active_timer.pop(operation, time.time())
        duration = (time.time() - start_time) * 1000  # ms

        entry = CostEntry(
            operation=operation,
            duration_ms=duration,
            success=success,
            tokens_used=tokens,
            facts_gained=facts_gained,
        )
        self._entries.append(entry)
        self._by_operation[operation].append(entry)

        # Keep bounded
        if len(self._entries) > 1000:
            self._entries = self._entries[-1000:]

        return entry

    def record(self, operation: str, duration_ms: float, success: bool,
               tokens: int = 0, facts_gained: int = 0) -> CostEntry:
        """Record a cost directly (without timer)."""
        entry = CostEntry(
            operation=operation, duration_ms=duration_ms,
            success=success, tokens_used=tokens, facts_gained=facts_gained,
        )
        self._entries.append(entry)
        self._by_operation[operation].append(entry)
        return entry

    def avg_cost(self, operation: str) -> float:
        """Average cost in ms for an operation type."""
        entries = self._by_operation.get(operation, [])
        if not entries:
            return 0.0
        return sum(e.duration_ms for e in entries) / len(entries)

    def success_rate(self, operation: str) -> float:
        """Success rate for an operation type."""
        entries = self._by_operation.get(operation, [])
        if not entries:
            return 0.0
        return sum(1 for e in entries if e.success) / len(entries)

    def total_wasted(self, operation: str) -> float:
        """Total time wasted on failed attempts (ms)."""
        entries = self._by_operation.get(operation, [])
        return sum(e.duration_ms for e in entries if not e.success)

    def cheapest_successful(self) -> str | None:
        """Which operation type is cheapest AND succeeds?"""
        best = None
        best_score = 0

        for op, entries in self._by_operation.items():
            if not entries:
                continue
            success_entries = [e for e in entries if e.success]
            if not success_entries:
                continue
            avg_time = sum(e.duration_ms for e in success_entries) / len(success_entries)
            success_pct = len(success_entries) / len(entries)
            # Score = success_rate / avg_time (higher = better)
            score = success_pct / max(1, avg_time)
            if score > best_score:
                best_score = score
                best = op

        return best

    def recommend_approach(self, options: list[str]) -> str:
        """Given options, recommend the most cost-effective one."""
        scores = {}
        for op in options:
            sr = self.success_rate(op)
            avg = self.avg_cost(op)
            # Score = success_rate / (1 + avg_cost_seconds)
            scores[op] = sr / (1 + avg / 1000)

        if not scores:
            return options[0] if options else "default"

        return max(scores, key=scores.get)

    def report(self) -> str:
        """Cost efficiency report."""
        lines = [f"Cost Tracker ({len(self._entries)} operations recorded)"]

        for op in sorted(self._by_operation.keys()):
            entries = self._by_operation[op]
            avg = self.avg_cost(op)
            sr = self.success_rate(op)
            wasted = self.total_wasted(op)
            total_facts = sum(e.facts_gained for e in entries)
            lines.append(
                f"  {op:25s} avg: {avg:>7.0f}ms | "
                f"success: {sr:>4.0%} | "
                f"wasted: {wasted:>7.0f}ms | "
                f"facts: {total_facts}")

        cheapest = self.cheapest_successful()
        if cheapest:
            lines.append(f"\n  Most cost-effective: {cheapest}")

        return "\n".join(lines)
