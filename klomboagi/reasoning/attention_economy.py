"""
Attention Economy — from OpenCog's ECAN.

Attention is ZERO-SUM. There's a fixed budget. When concept A
gains attention, concept B MUST lose it. This prevents "everything
is important" — forces the system to actually prioritize.

The budget circulates like money:
  - Active concepts spend attention (STI = Short Term Importance)
  - Concepts that contribute to successful reasoning earn attention
  - Concepts below a threshold get evicted from active memory
  - Total attention in the system stays constant

This is the missing piece that makes focus REAL, not advisory.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class AttentionUnit:
    """A concept with its attention allocation."""
    name: str
    sti: float = 0.0       # Short Term Importance (active attention)
    lti: float = 0.0       # Long Term Importance (how often useful historically)
    vlti: bool = False      # Very Long Term Importance (never evict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sti": round(self.sti, 3),
            "lti": round(self.lti, 3),
            "vlti": self.vlti,
        }


class AttentionEconomy:
    """
    Zero-sum attention budget.

    Total STI in the system = BUDGET (constant).
    Giving attention to one thing takes it from everything else.
    """

    BUDGET = 100.0          # Total attention available
    EVICTION_THRESHOLD = 1.0  # Below this = evicted from active set
    RENT = 0.5              # Cost per cycle to stay in active set

    def __init__(self) -> None:
        self._units: dict[str, AttentionUnit] = {}
        self._total_sti: float = 0.0

    def allocate(self, concept: str, amount: float) -> None:
        """
        Give attention to a concept. Takes from the pool.
        If pool is empty, steals from lowest-STI concepts.
        """
        if concept not in self._units:
            self._units[concept] = AttentionUnit(name=concept)

        # How much can we give?
        available = self.BUDGET - self._total_sti
        if amount > available:
            # Need to steal from others
            self._steal(amount - available, exclude=concept)

        actual = min(amount, self.BUDGET - self._total_sti + 0.01)
        self._units[concept].sti += actual
        self._total_sti += actual

    def reward(self, concept: str, amount: float = 2.0) -> None:
        """Reward a concept that contributed to successful reasoning."""
        if concept in self._units:
            self._units[concept].lti += amount * 0.1  # LTI grows slowly
            self.allocate(concept, amount)

    def tax(self) -> list[str]:
        """
        Charge rent on all active concepts. Evict those below threshold.
        Call once per cognitive cycle.

        Returns list of evicted concepts.
        """
        evicted = []
        for name in list(self._units.keys()):
            unit = self._units[name]
            if unit.vlti:
                continue  # Protected
            unit.sti -= self.RENT
            self._total_sti -= self.RENT
            if unit.sti < self.EVICTION_THRESHOLD:
                evicted.append(name)
                self._total_sti -= max(0, unit.sti)
                del self._units[name]

        self._total_sti = max(0, self._total_sti)
        return evicted

    def top(self, n: int = 10) -> list[AttentionUnit]:
        """Highest attention concepts."""
        return sorted(self._units.values(), key=lambda u: u.sti, reverse=True)[:n]

    def is_active(self, concept: str) -> bool:
        """Is this concept currently in the active attention set?"""
        return concept in self._units and self._units[concept].sti > self.EVICTION_THRESHOLD

    def _steal(self, amount: float, exclude: str = "") -> None:
        """Steal attention from lowest-STI concepts."""
        targets = sorted(
            [(n, u) for n, u in self._units.items() if n != exclude and not u.vlti],
            key=lambda x: x[1].sti,
        )
        stolen = 0.0
        for name, unit in targets:
            take = min(unit.sti, amount - stolen)
            unit.sti -= take
            self._total_sti -= take
            stolen += take
            if stolen >= amount:
                break

    def stats(self) -> dict:
        return {
            "active_concepts": len(self._units),
            "total_sti": round(self._total_sti, 1),
            "budget": self.BUDGET,
            "utilization": round(self._total_sti / self.BUDGET, 2),
        }
