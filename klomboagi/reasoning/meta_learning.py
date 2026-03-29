"""
Meta-Learning — learn HOW to learn faster.

Not learning facts. Learning STRATEGIES for learning.

Tracks which learning approaches work best:
  - "Reading Wikipedia about physics → 15 facts in 8 seconds"
  - "Reading Wikipedia about philosophy → 2 facts in 8 seconds"
  - "Direct teaching from human → 1 fact in 0.01 seconds"
  - "Active learning via search → 3 facts in 5 seconds"

Over time, the system learns:
  - Physics articles are information-dense → read more of those
  - Philosophy articles are hard to extract from → ask human instead
  - Direct teaching is fastest per-fact → prioritize human interaction
  - Search is good for filling specific gaps

The meta-learner adjusts learning STRATEGY based on past results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LearningAttemptRecord:
    """Record of one learning attempt."""
    method: str          # "wiki_read", "human_teach", "search", "active_learn", "infer"
    domain: str          # what domain was being learned
    duration_ms: float
    facts_gained: int
    relations_gained: int
    success: bool

    @property
    def efficiency(self) -> float:
        """Facts per second."""
        if self.duration_ms <= 0:
            return 0.0
        return self.facts_gained / (self.duration_ms / 1000)


@dataclass
class LearningStrategy:
    """An optimal learning strategy for a domain."""
    domain: str
    best_method: str
    avg_efficiency: float       # facts per second
    total_attempts: int
    recommendation: str


class MetaLearner:
    """
    Learns which learning methods work best for which domains.

    Tracks: method × domain → efficiency. Picks optimal strategy.
    """

    def __init__(self) -> None:
        self._records: list[LearningAttemptRecord] = []
        self._by_method_domain: dict[tuple[str, str], list[LearningAttemptRecord]] = defaultdict(list)

    def record(self, method: str, domain: str, duration_ms: float,
               facts_gained: int, relations_gained: int = 0,
               success: bool = True) -> None:
        """Record a learning attempt."""
        rec = LearningAttemptRecord(
            method=method, domain=domain, duration_ms=duration_ms,
            facts_gained=facts_gained, relations_gained=relations_gained,
            success=success,
        )
        self._records.append(rec)
        self._by_method_domain[(method, domain)].append(rec)

    def best_method_for(self, domain: str) -> str:
        """What learning method works best for this domain?"""
        methods: dict[str, float] = {}

        for (method, d), records in self._by_method_domain.items():
            if d != domain:
                continue
            successful = [r for r in records if r.success and r.facts_gained > 0]
            if successful:
                avg_eff = sum(r.efficiency for r in successful) / len(successful)
                methods[method] = avg_eff

        if not methods:
            return "wiki_read"  # Default

        return max(methods, key=methods.get)

    def optimal_strategy(self, domain: str) -> LearningStrategy:
        """Get the optimal learning strategy for a domain."""
        best = self.best_method_for(domain)
        records = self._by_method_domain.get((best, domain), [])
        successful = [r for r in records if r.success]
        avg_eff = sum(r.efficiency for r in successful) / max(1, len(successful))

        return LearningStrategy(
            domain=domain,
            best_method=best,
            avg_efficiency=avg_eff,
            total_attempts=len(records),
            recommendation=f"For {domain}, use {best} "
                         f"({avg_eff:.1f} facts/sec from {len(records)} attempts)",
        )

    def learning_rate_trend(self) -> str:
        """Is learning getting faster or slower?"""
        if len(self._records) < 10:
            return "Not enough data."

        first_half = self._records[:len(self._records)//2]
        second_half = self._records[len(self._records)//2:]

        eff1 = sum(r.efficiency for r in first_half if r.success) / max(1, len(first_half))
        eff2 = sum(r.efficiency for r in second_half if r.success) / max(1, len(second_half))

        if eff2 > eff1 * 1.2:
            return f"Getting faster: {eff1:.1f} → {eff2:.1f} facts/sec"
        elif eff2 < eff1 * 0.8:
            return f"Getting slower: {eff1:.1f} → {eff2:.1f} facts/sec"
        return f"Stable: ~{(eff1+eff2)/2:.1f} facts/sec"

    def report(self) -> str:
        """Full meta-learning report."""
        if not self._records:
            return "No learning attempts recorded yet."

        lines = [f"Meta-Learning Report ({len(self._records)} attempts):"]
        lines.append(f"  Trend: {self.learning_rate_trend()}")

        # Per-method stats
        method_stats: dict[str, list] = defaultdict(list)
        for r in self._records:
            method_stats[r.method].append(r)

        lines.append(f"\n  Method efficiency:")
        for method, records in sorted(method_stats.items()):
            successful = [r for r in records if r.success]
            if successful:
                avg = sum(r.efficiency for r in successful) / len(successful)
                lines.append(f"    {method:20s} {avg:>6.1f} facts/sec ({len(records)} attempts)")

        # Per-domain best method
        domains = set(r.domain for r in self._records)
        if domains:
            lines.append(f"\n  Best method per domain:")
            for domain in sorted(domains):
                strategy = self.optimal_strategy(domain)
                lines.append(f"    {domain:20s} → {strategy.best_method}")

        return "\n".join(lines)
