"""
Concept Formation — the system creates NEW abstract concepts.

Instead of only learning "X is Y" from teaching, the system notices
patterns across its beliefs and forms its own categories:

  "red is a color, blue is a color, green is a color"
  → forms concept: "primary color group" (shares parent "color")

  "gravity causes force, force causes acceleration"
  → forms concept: "gravitational chain" (linked causal sequence)

  "physics uses mathematics, economics uses mathematics, CS uses mathematics"
  → forms concept: "mathematical sciences" (shared dependency)

This is how real minds form abstractions — not by being told,
but by noticing what's invariant across experiences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class FormedConcept:
    """A concept the system created on its own."""
    name: str
    description: str
    members: list[str]           # concepts that belong to this group
    basis: str                   # what pattern formed this ("shared_parent", "shared_target", "causal_chain")
    shared_property: str         # what the members have in common
    confidence: float = 0.5
    evidence_count: int = 0      # how many members support this

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "members": self.members,
            "basis": self.basis,
            "shared_property": self.shared_property,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
        }


class ConceptFormation:
    """
    Forms new abstract concepts by detecting patterns in beliefs and relations.

    Strategies:
    1. Shared parent: many things are X → "the X group"
    2. Shared target: many things cause/use/require X → "X dependents"
    3. Shared source: X causes/uses many things → "X products"
    4. Causal chains: A→B→C→D → "the A-D chain"
    5. Cluster: concepts frequently mentioned together → "concept cluster"
    """

    MIN_GROUP_SIZE = 3  # Need at least 3 members to form a concept

    def __init__(self, relations: RelationStore,
                 beliefs: dict | None = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}
        self.formed: list[FormedConcept] = []

    def scan(self) -> list[FormedConcept]:
        """
        Scan all knowledge and form new concepts.

        Returns list of newly formed concepts.
        """
        new_concepts = []

        # Strategy 1: Shared parent — things that are all "X"
        new_concepts.extend(self._find_shared_parents())

        # Strategy 2: Shared target — things that all cause/use/require the same thing
        new_concepts.extend(self._find_shared_targets())

        # Strategy 3: Shared source — things all caused/enabled by the same thing
        new_concepts.extend(self._find_shared_sources())

        # Strategy 4: Causal chains
        new_concepts.extend(self._find_causal_chains())

        # Deduplicate by member set
        seen_members = set()
        unique = []
        for c in new_concepts:
            key = frozenset(c.members)
            if key not in seen_members:
                seen_members.add(key)
                unique.append(c)

        self.formed.extend(unique)
        return unique

    def _find_shared_parents(self) -> list[FormedConcept]:
        """Find groups of beliefs with the same predicate."""
        # Group beliefs by predicate
        predicate_groups: dict[str, list[str]] = defaultdict(list)
        for statement, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and hasattr(belief, 'predicate'):
                if belief.predicate and belief.subject:
                    predicate_groups[belief.predicate].append(belief.subject)

        concepts = []
        for predicate, subjects in predicate_groups.items():
            if len(subjects) >= self.MIN_GROUP_SIZE:
                name = f"{predicate} group"
                concepts.append(FormedConcept(
                    name=name,
                    description=f"Things that are {predicate}: {', '.join(subjects[:5])}",
                    members=subjects,
                    basis="shared_parent",
                    shared_property=predicate,
                    confidence=min(0.9, 0.3 + len(subjects) * 0.1),
                    evidence_count=len(subjects),
                ))

        return concepts

    def _find_shared_targets(self) -> list[FormedConcept]:
        """Find concepts that all relate to the same target."""
        # Group by (relation_type, target)
        target_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rel in self.relations._all:
            key = (rel.relation.value, rel.target)
            target_groups[key].append(rel.source)

        concepts = []
        for (rel_type, target), sources in target_groups.items():
            if len(sources) >= self.MIN_GROUP_SIZE:
                name = f"things that {rel_type} {target}"
                concepts.append(FormedConcept(
                    name=name,
                    description=f"Concepts that {rel_type} {target}: {', '.join(sources[:5])}",
                    members=sources,
                    basis="shared_target",
                    shared_property=f"{rel_type} {target}",
                    confidence=min(0.9, 0.3 + len(sources) * 0.1),
                    evidence_count=len(sources),
                ))

        return concepts

    def _find_shared_sources(self) -> list[FormedConcept]:
        """Find concepts all produced/caused by the same source."""
        source_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rel in self.relations._all:
            key = (rel.relation.value, rel.source)
            source_groups[key].append(rel.target)

        concepts = []
        for (rel_type, source), targets in source_groups.items():
            if len(targets) >= self.MIN_GROUP_SIZE:
                name = f"things {source} {rel_type}"
                concepts.append(FormedConcept(
                    name=name,
                    description=f"Things that {source} {rel_type}: {', '.join(targets[:5])}",
                    members=targets,
                    basis="shared_source",
                    shared_property=f"{source} {rel_type}",
                    confidence=min(0.9, 0.3 + len(targets) * 0.1),
                    evidence_count=len(targets),
                ))

        return concepts

    def _find_causal_chains(self) -> list[FormedConcept]:
        """Find long causal chains: A→B→C→D."""
        concepts = []

        # Build causal adjacency
        causal_forward: dict[str, list[str]] = defaultdict(list)
        for rel in self.relations._all:
            if rel.relation == RelationType.CAUSES:
                causal_forward[rel.source].append(rel.target)

        # Find chains of length >= 3
        for start in causal_forward:
            chain = self._trace_chain(start, causal_forward, max_length=6)
            if len(chain) >= 3:
                name = f"{chain[0]}-to-{chain[-1]} chain"
                concepts.append(FormedConcept(
                    name=name,
                    description=f"Causal chain: {' → '.join(chain)}",
                    members=chain,
                    basis="causal_chain",
                    shared_property="causal sequence",
                    confidence=0.6,
                    evidence_count=len(chain),
                ))

        return concepts

    def _trace_chain(self, start: str, adj: dict, max_length: int = 6) -> list[str]:
        """Trace the longest chain from start."""
        chain = [start]
        current = start
        visited = {start}
        while len(chain) < max_length:
            nexts = [n for n in adj.get(current, []) if n not in visited]
            if not nexts:
                break
            current = nexts[0]
            visited.add(current)
            chain.append(current)
        return chain

    def report(self) -> str:
        """Report all formed concepts."""
        if not self.formed:
            return "No concepts formed yet. Need more knowledge."

        lines = [f"Formed Concepts ({len(self.formed)}):"]
        for c in self.formed:
            lines.append(f"\n  {c.name} ({c.basis}, {c.evidence_count} members)")
            lines.append(f"    {c.description}")
        return "\n".join(lines)
