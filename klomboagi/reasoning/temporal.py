"""
Temporal Reasoning — understanding time, sequence, and duration.

The system needs to understand:
  "X happened before Y"
  "First A, then B, then C"
  "X takes longer than Y"
  "Newton discovered gravity before Einstein developed relativity"

Stores temporal relations as a partial order (DAG).
Can answer: "what came first?", "what happens after X?", "what's the sequence?"
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemporalFact:
    """A fact with temporal information."""
    event: str
    year: int | None = None
    era: str = ""              # "ancient", "medieval", "modern", "contemporary"
    before: list[str] = field(default_factory=list)  # events this happened before
    after: list[str] = field(default_factory=list)    # events this happened after
    duration: str = ""         # "instant", "seconds", "minutes", "hours", "years", "centuries"

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "year": self.year,
            "era": self.era,
            "before": self.before,
            "after": self.after,
            "duration": self.duration,
        }


class TemporalEngine:
    """
    Manages temporal knowledge — when things happen and in what order.
    """

    def __init__(self) -> None:
        self.facts: dict[str, TemporalFact] = {}

    def record(self, event: str, year: int | None = None,
               before: list[str] | None = None,
               after: list[str] | None = None,
               duration: str = "") -> TemporalFact:
        """Record a temporal fact."""
        if event in self.facts:
            f = self.facts[event]
            if year is not None:
                f.year = year
            if before:
                f.before.extend(b for b in before if b not in f.before)
            if after:
                f.after.extend(a for a in after if a not in f.after)
            if duration:
                f.duration = duration
        else:
            era = ""
            if year is not None:
                if year < 0:
                    era = "ancient"
                elif year < 500:
                    era = "classical"
                elif year < 1500:
                    era = "medieval"
                elif year < 1800:
                    era = "early modern"
                elif year < 1950:
                    era = "modern"
                else:
                    era = "contemporary"

            f = TemporalFact(
                event=event, year=year, era=era,
                before=before or [], after=after or [],
                duration=duration,
            )
            self.facts[event] = f

        # Maintain consistency — if A before B, then B after A
        for b in (before or []):
            if b in self.facts:
                if event not in self.facts[b].after:
                    self.facts[b].after.append(event)
        for a in (after or []):
            if a in self.facts:
                if event not in self.facts[a].before:
                    self.facts[a].before.append(event)

        return f

    def what_came_first(self, a: str, b: str) -> str:
        """Determine temporal order of two events."""
        fact_a = self.facts.get(a)
        fact_b = self.facts.get(b)

        if not fact_a and not fact_b:
            return f"I don't know when either {a} or {b} happened."
        if not fact_a:
            return f"I don't know when {a} happened."
        if not fact_b:
            return f"I don't know when {b} happened."

        # Check explicit ordering
        if b in fact_a.before:
            return f"{a} happened before {b}."
        if b in fact_a.after:
            return f"{b} happened before {a}."

        # Check years
        if fact_a.year is not None and fact_b.year is not None:
            if fact_a.year < fact_b.year:
                return f"{a} ({fact_a.year}) happened before {b} ({fact_b.year})."
            elif fact_b.year < fact_a.year:
                return f"{b} ({fact_b.year}) happened before {a} ({fact_a.year})."
            else:
                return f"{a} and {b} both happened around {fact_a.year}."

        return f"I can't determine which came first."

    def sequence(self, events: list[str]) -> list[str]:
        """Order a list of events chronologically."""
        known = [(e, self.facts[e]) for e in events if e in self.facts]

        # Sort by year if available, otherwise by partial order
        with_years = [(e, f) for e, f in known if f.year is not None]
        without_years = [(e, f) for e, f in known if f.year is None]

        with_years.sort(key=lambda x: x[1].year)
        ordered = [e for e, _ in with_years]

        # Add events without years based on before/after
        for event, fact in without_years:
            inserted = False
            for i, existing in enumerate(ordered):
                if existing in fact.before:
                    ordered.insert(i, event)
                    inserted = True
                    break
            if not inserted:
                ordered.append(event)

        return ordered

    def timeline(self, events: list[str] | None = None) -> str:
        """Generate a timeline of events."""
        if events is None:
            events = list(self.facts.keys())

        ordered = self.sequence(events)
        lines = ["Timeline:"]
        for event in ordered:
            fact = self.facts.get(event)
            if fact and fact.year:
                lines.append(f"  {fact.year:>6d}  {event}")
            else:
                lines.append(f"      ?  {event}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "total_events": len(self.facts),
            "with_years": sum(1 for f in self.facts.values() if f.year),
            "eras": dict(sorted({f.era: 0 for f in self.facts.values() if f.era}.items())),
        }
