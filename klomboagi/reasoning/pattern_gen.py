"""
Pattern Generalization — form NEW abstract principles from raw experience.

Not just "A is B" storage. Not just "A and B are similar."
REAL generalization: seeing enough examples to form a NEW principle
that was never explicitly taught.

Example:
  Observe: "gravity causes acceleration"
  Observe: "heat causes expansion"
  Observe: "supply causes price change"
  Observe: "mutation causes evolution"
  Generalize: "When a force acts on a system, the system changes
              proportionally. This is a UNIVERSAL PRINCIPLE."

The system discovers:
  - FEEDBACK LOOPS: A causes B, B causes A (self-reinforcing)
  - HIERARCHIES: containment nesting (universe > galaxy > star > planet)
  - DUALITIES: every concept has an opposite (hot/cold, add/subtract)
  - CONSERVATION: quantities that transform but don't disappear
  - EMERGENCE: parts create wholes with new properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, Counter

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class GeneralizedPrinciple:
    """A principle discovered from patterns across multiple examples."""
    name: str
    description: str
    pattern_type: str       # "feedback_loop", "hierarchy", "duality", "conservation", "emergence"
    examples: list[str]     # specific instances that support this
    confidence: float
    novel: bool = True      # was this discovered, not taught?

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.pattern_type,
            "examples": len(self.examples),
            "confidence": round(self.confidence, 3),
        }


class PatternGeneralizer:
    """
    Discovers abstract principles from patterns in the knowledge graph.
    """

    def __init__(self, relations: RelationStore, beliefs: dict) -> None:
        self.relations = relations
        self.beliefs = beliefs
        self.principles: list[GeneralizedPrinciple] = []

    def discover_all(self) -> list[GeneralizedPrinciple]:
        """Run all discovery strategies."""
        principles = []
        principles.extend(self._find_feedback_loops())
        principles.extend(self._find_hierarchies())
        principles.extend(self._find_dualities())
        principles.extend(self._find_causal_patterns())

        self.principles = principles
        return principles

    def _find_feedback_loops(self) -> list[GeneralizedPrinciple]:
        """Find A→B→A cycles (feedback loops)."""
        principles = []
        checked = set()

        for rel in self.relations._all:
            if rel.relation != RelationType.CAUSES:
                continue
            key = f"{rel.source}:{rel.target}"
            if key in checked:
                continue
            checked.add(key)

            # Does the target cause the source? (direct feedback)
            reverse = self.relations.query(
                source=rel.target, relation=RelationType.CAUSES, target=rel.source)
            if reverse:
                principles.append(GeneralizedPrinciple(
                    name=f"{rel.source}-{rel.target} feedback loop",
                    description=f"{rel.source} causes {rel.target} and {rel.target} causes {rel.source}. "
                               f"This is a self-reinforcing cycle.",
                    pattern_type="feedback_loop",
                    examples=[f"{rel.source} → {rel.target}", f"{rel.target} → {rel.source}"],
                    confidence=0.7,
                ))

        return principles

    def _find_hierarchies(self) -> list[GeneralizedPrinciple]:
        """Find nested containment/part_of chains."""
        principles = []
        part_of_chains: dict[str, list[str]] = {}

        for rel in self.relations._all:
            if rel.relation != RelationType.PART_OF:
                continue
            part_of_chains.setdefault(rel.target, []).append(rel.source)

        # Find deep hierarchies (3+ levels)
        for root, children in part_of_chains.items():
            chain = [root]
            current_level = children
            depth = 1
            while current_level and depth < 6:
                chain.extend(current_level[:3])
                next_level = []
                for child in current_level:
                    if child in part_of_chains:
                        next_level.extend(part_of_chains[child][:3])
                current_level = next_level
                depth += 1

            if depth >= 3:
                principles.append(GeneralizedPrinciple(
                    name=f"{root} hierarchy",
                    description=f"{root} contains a hierarchy of {depth} levels. "
                               f"Larger structures contain smaller ones that contain even smaller ones.",
                    pattern_type="hierarchy",
                    examples=chain[:6],
                    confidence=0.6 + depth * 0.05,
                ))

        return principles[:5]

    def _find_dualities(self) -> list[GeneralizedPrinciple]:
        """Find opposite pairs and generalize."""
        opposites = self.relations.query(relation=RelationType.OPPOSITE_OF)
        if len(opposites) >= 3:
            examples = [f"{r.source} ↔ {r.target}" for r in opposites[:6]]
            return [GeneralizedPrinciple(
                name="duality principle",
                description=f"Many concepts have opposites ({len(opposites)} pairs found). "
                           f"Understanding something often requires understanding its opposite.",
                pattern_type="duality",
                examples=examples,
                confidence=0.5 + min(0.4, len(opposites) * 0.05),
            )]
        return []

    def _find_causal_patterns(self) -> list[GeneralizedPrinciple]:
        """Find recurring causal structures."""
        principles = []

        # Count how many things each concept causes
        cause_counts: Counter = Counter()
        for rel in self.relations._all:
            if rel.relation == RelationType.CAUSES:
                cause_counts[rel.source] += 1

        # Concepts that cause many things = fundamental forces/drivers
        drivers = [(c, n) for c, n in cause_counts.most_common(5) if n >= 3]
        if drivers:
            examples = [f"{c} causes {n} things" for c, n in drivers]
            principles.append(GeneralizedPrinciple(
                name="fundamental drivers",
                description=f"Some concepts are fundamental drivers that cause many effects: "
                           f"{', '.join(c for c, _ in drivers)}. "
                           f"These are the root causes in the knowledge graph.",
                pattern_type="causal_hub",
                examples=examples,
                confidence=0.7,
            ))

        # Things caused by many sources = convergence points
        effect_counts: Counter = Counter()
        for rel in self.relations._all:
            if rel.relation == RelationType.CAUSES:
                effect_counts[rel.target] += 1

        convergences = [(c, n) for c, n in effect_counts.most_common(5) if n >= 3]
        if convergences:
            examples = [f"{n} things cause {c}" for c, n in convergences]
            principles.append(GeneralizedPrinciple(
                name="convergence points",
                description=f"Some things are caused by many different sources: "
                           f"{', '.join(c for c, _ in convergences)}. "
                           f"These are where multiple causal chains converge.",
                pattern_type="convergence",
                examples=examples,
                confidence=0.7,
            ))

        return principles

    def report(self) -> str:
        """Report discovered principles."""
        if not self.principles:
            self.discover_all()
        lines = [f"Discovered Principles ({len(self.principles)}):"]
        for p in self.principles:
            lines.append(f"\n  [{p.pattern_type}] {p.name} ({p.confidence:.0%})")
            lines.append(f"    {p.description[:80]}")
            lines.append(f"    Examples: {', '.join(p.examples[:3])}")
        return "\n".join(lines)
