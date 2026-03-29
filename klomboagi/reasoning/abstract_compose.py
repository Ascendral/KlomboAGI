"""
Abstract Composition — invent new abstractions on the fly.

Not just "find existing patterns." CREATE NEW ONES.

When the system encounters multiple concepts that share structure
but no existing abstraction covers them, it should INVENT one.

Example:
  Observe: gravity pulls things together
  Observe: magnetism pulls things together
  Observe: love pulls people together
  No existing abstraction covers all three.
  INVENT: "attractive force" — something that draws entities closer.
  Apply: social media is also an attractive force (draws attention).

The difference from concept formation:
  Concept formation: "things that cause X" (grouping by shared target)
  Abstract composition: "attractive force" (NEW CONCEPT that didn't exist)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class InventedAbstraction:
    """A new abstraction the system invented."""
    name: str
    definition: str
    instances: list[str]      # concrete things that are examples
    structural_pattern: str   # what they share
    confidence: float
    useful_for: list[str] = field(default_factory=list)  # what problems this helps solve

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "definition": self.definition,
            "instances": self.instances,
            "pattern": self.structural_pattern,
            "confidence": round(self.confidence, 3),
        }


class AbstractComposer:
    """
    Invents new abstractions by finding uncovered structural patterns.
    """

    def __init__(self, relations: RelationStore, beliefs: dict) -> None:
        self.relations = relations
        self.beliefs = beliefs
        self.invented: list[InventedAbstraction] = []

    def compose(self) -> list[InventedAbstraction]:
        """Find structural patterns and invent abstractions for them."""
        new_abstractions = []

        # Strategy 1: Same verb, different subjects → name the pattern
        new_abstractions.extend(self._abstract_by_shared_action())

        # Strategy 2: Same targets from different relations → convergence abstraction
        new_abstractions.extend(self._abstract_by_convergence())

        # Strategy 3: Symmetric pairs → invent the relationship category
        new_abstractions.extend(self._abstract_by_symmetry())

        self.invented.extend(new_abstractions)
        return new_abstractions

    def _abstract_by_shared_action(self) -> list[InventedAbstraction]:
        """Things that DO the same thing → name what they share."""
        abstractions = []

        # Group by (relation_type, target)
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rel in self.relations._all:
            key = (rel.relation.value, rel.target)
            groups[key].append(rel.source)

        for (rel_type, target), sources in groups.items():
            if len(sources) >= 3:
                # Name this pattern
                name = f"{target}_{rel_type}rs"
                definition = (f"Things that {rel_type} {target}: "
                            f"{', '.join(sources[:5])}. They share "
                            f"the ability to {rel_type} {target}.")

                existing_names = {a.name for a in self.invented}
                if name not in existing_names:
                    abstractions.append(InventedAbstraction(
                        name=name,
                        definition=definition,
                        instances=sources[:5],
                        structural_pattern=f"X {rel_type} {target}",
                        confidence=0.4 + len(sources) * 0.05,
                    ))

        return abstractions[:10]

    def _abstract_by_convergence(self) -> list[InventedAbstraction]:
        """Things that are targets of many different relations → important concept."""
        abstractions = []

        target_counts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for rel in self.relations._all:
            target_counts[rel.target][rel.relation.value].append(rel.source)

        for target, rel_types in target_counts.items():
            if len(rel_types) >= 3:  # Connected by 3+ different relation types
                all_sources = []
                for sources in rel_types.values():
                    all_sources.extend(sources[:2])
                definition = (
                    f"{target} is a convergence point — "
                    f"connected by {len(rel_types)} different relationship types: "
                    f"{', '.join(rel_types.keys())}."
                )
                abstractions.append(InventedAbstraction(
                    name=f"{target}_nexus",
                    definition=definition,
                    instances=all_sources[:5],
                    structural_pattern=f"multiple_relations → {target}",
                    confidence=0.5 + len(rel_types) * 0.05,
                ))

        return abstractions[:5]

    def _abstract_by_symmetry(self) -> list[InventedAbstraction]:
        """Find symmetric patterns and name them."""
        abstractions = []

        # Find concepts that both cause and are caused by something
        for rel in self.relations._all:
            if rel.relation != RelationType.CAUSES:
                continue
            reverse = self.relations.query(
                source=rel.target, relation=RelationType.CAUSES, target=rel.source)
            if reverse:
                name = f"{rel.source}_{rel.target}_cycle"
                existing = {a.name for a in self.invented + abstractions}
                if name not in existing:
                    abstractions.append(InventedAbstraction(
                        name=name,
                        definition=(f"{rel.source} and {rel.target} form a feedback cycle — "
                                  f"each causes the other."),
                        instances=[rel.source, rel.target],
                        structural_pattern="A causes B AND B causes A",
                        confidence=0.6,
                    ))

        return abstractions[:5]

    def apply_abstraction(self, abstraction_name: str, new_instance: str) -> str:
        """Apply an existing abstraction to a new instance."""
        for a in self.invented:
            if a.name == abstraction_name:
                a.instances.append(new_instance)
                return (f"Applied '{a.name}' to {new_instance}. "
                       f"This means {new_instance} shares the pattern: {a.structural_pattern}")
        return f"Abstraction '{abstraction_name}' not found."

    def report(self) -> str:
        if not self.invented:
            self.compose()
        lines = [f"Invented Abstractions ({len(self.invented)}):"]
        for a in self.invented[:10]:
            lines.append(f"\n  {a.name} ({a.confidence:.0%})")
            lines.append(f"    {a.definition[:80]}")
            lines.append(f"    Instances: {', '.join(a.instances[:4])}")
        return "\n".join(lines)
