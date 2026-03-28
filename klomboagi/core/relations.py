"""
Relation System — multi-directional reasoning across the knowledge graph.

The base knowledge graph only stores "is_a" relationships.
This module adds richer relation types that enable spherical reasoning:
every concept connects to others in MULTIPLE directions.

Relation types:
  is_a       — taxonomy (dog is_a animal)
  causes     — causation (force causes acceleration)
  requires   — dependency (division requires non-zero denominator)
  part_of    — composition (a proton is part_of an atom)
  uses       — application (physics uses mathematics)
  opposite_of — antonymy (hot is opposite_of cold)
  enables    — capability (electricity enables computation)
  measures   — quantification (temperature measures kinetic energy)
  example_of — instantiation (earth is example_of planet)
  analogous_to — structural similarity (a cell is analogous_to a factory)

Each relation is bidirectional in reasoning:
  "force causes acceleration" also means "acceleration is caused by force"

The InferenceEngine scans all relations to derive new knowledge:
  if A causes B and B causes C → A indirectly causes C
  if A is_a B and B requires C → A also requires C
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RelationType(Enum):
    IS_A = "is_a"
    CAUSES = "causes"
    REQUIRES = "requires"
    PART_OF = "part_of"
    USES = "uses"
    OPPOSITE_OF = "opposite_of"
    ENABLES = "enables"
    MEASURES = "measures"
    EXAMPLE_OF = "example_of"
    ANALOGOUS_TO = "analogous_to"


# Inverse relations — for bidirectional traversal
INVERSE_RELATIONS: dict[RelationType, str] = {
    RelationType.IS_A: "has_subtype",
    RelationType.CAUSES: "caused_by",
    RelationType.REQUIRES: "required_by",
    RelationType.PART_OF: "has_part",
    RelationType.USES: "used_by",
    RelationType.OPPOSITE_OF: "opposite_of",  # symmetric
    RelationType.ENABLES: "enabled_by",
    RelationType.MEASURES: "measured_by",
    RelationType.EXAMPLE_OF: "has_example",
    RelationType.ANALOGOUS_TO: "analogous_to",  # symmetric
}

# Which relations are transitive (A→B, B→C implies A→C)
TRANSITIVE_RELATIONS = {
    RelationType.IS_A,       # dog is_a mammal, mammal is_a animal → dog is_a animal
    RelationType.CAUSES,     # heat causes expansion, expansion causes cracking → heat causes cracking
    RelationType.PART_OF,    # mitochondria part_of cell, cell part_of organism → mitochondria part_of organism
    RelationType.REQUIRES,   # calculus requires algebra, algebra requires arithmetic
}


@dataclass
class Relation:
    """A directed relationship between two concepts."""
    source: str
    relation: RelationType
    target: str
    confidence: float = 0.5     # NARS-compatible
    source_domain: str = ""     # which curriculum taught this
    derived: bool = False       # was this inferred, not taught?

    def inverse(self) -> str:
        """The inverse relation label."""
        return INVERSE_RELATIONS.get(self.relation, f"inverse_{self.relation.value}")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "relation": self.relation.value,
            "target": self.target,
            "confidence": round(self.confidence, 3),
            "source_domain": self.source_domain,
            "derived": self.derived,
        }

    def __repr__(self) -> str:
        return f"{self.source} --{self.relation.value}--> {self.target}"


@dataclass
class InferenceResult:
    """A new fact derived from existing relations."""
    relation: Relation
    chain: list[Relation]      # the relations that led to this
    explanation: str

    def to_dict(self) -> dict:
        return {
            "derived": self.relation.to_dict(),
            "chain_length": len(self.chain),
            "explanation": self.explanation,
        }


class RelationStore:
    """
    Stores and queries relations between concepts.

    Supports forward traversal (what does X cause?),
    backward traversal (what causes X?), and
    transitive inference (if A→B→C then A→C).
    """

    def __init__(self) -> None:
        # Forward index: source → list of relations
        self._forward: dict[str, list[Relation]] = {}
        # Backward index: target → list of relations
        self._backward: dict[str, list[Relation]] = {}
        # All relations for scanning
        self._all: list[Relation] = []

    def add(self, source: str, relation: RelationType, target: str,
            confidence: float = 0.5, domain: str = "") -> Relation:
        """Add a relation. Deduplicates."""
        # Check for duplicate
        for existing in self._forward.get(source, []):
            if existing.relation == relation and existing.target == target:
                existing.confidence = max(existing.confidence, confidence)
                return existing

        rel = Relation(
            source=source, relation=relation, target=target,
            confidence=confidence, source_domain=domain,
        )
        self._forward.setdefault(source, []).append(rel)
        self._backward.setdefault(target, []).append(rel)
        self._all.append(rel)
        return rel

    def get_forward(self, source: str, relation: RelationType | None = None) -> list[Relation]:
        """What does source relate to? Optionally filter by relation type."""
        rels = self._forward.get(source, [])
        if relation:
            return [r for r in rels if r.relation == relation]
        return rels

    def get_backward(self, target: str, relation: RelationType | None = None) -> list[Relation]:
        """What relates to target? (inverse traversal)"""
        rels = self._backward.get(target, [])
        if relation:
            return [r for r in rels if r.relation == relation]
        return rels

    def get_all_about(self, concept: str) -> list[Relation]:
        """Everything we know about a concept — forward AND backward."""
        forward = self._forward.get(concept, [])
        backward = self._backward.get(concept, [])
        return forward + backward

    def query(self, source: str | None = None, relation: RelationType | None = None,
              target: str | None = None) -> list[Relation]:
        """Flexible query: any combination of source, relation, target."""
        results = self._all
        if source:
            results = [r for r in results if r.source == source]
        if relation:
            results = [r for r in results if r.relation == relation]
        if target:
            results = [r for r in results if r.target == target]
        return results

    def infer_transitive(self, max_depth: int = 3) -> list[InferenceResult]:
        """
        Derive new facts through transitive chains.

        If A causes B and B causes C → A indirectly causes C.
        If A is_a B and B is_a C → A is_a C.

        This is the global inference sweep — it finds ALL derivable chains.
        """
        inferred: list[InferenceResult] = []
        seen: set[str] = set()

        for rel_type in TRANSITIVE_RELATIONS:
            # Build adjacency for this relation type
            edges: dict[str, list[Relation]] = {}
            for r in self._all:
                if r.relation == rel_type:
                    edges.setdefault(r.source, []).append(r)

            # BFS from each source
            for start in list(edges.keys()):
                visited: set[str] = {start}
                queue: list[tuple[str, list[Relation]]] = []

                for r in edges.get(start, []):
                    queue.append((r.target, [r]))

                while queue:
                    current, chain = queue.pop(0)
                    if len(chain) > max_depth:
                        continue
                    if current in visited:
                        continue
                    visited.add(current)

                    # Check if this is a NEW derivation (not already known)
                    key = f"{start}:{rel_type.value}:{current}"
                    if key not in seen and len(chain) > 1:
                        existing = self.query(source=start, relation=rel_type, target=current)
                        if not existing:
                            seen.add(key)
                            # Confidence decreases with chain length
                            conf = min(r.confidence for r in chain) * (0.8 ** (len(chain) - 1))
                            derived_rel = Relation(
                                source=start, relation=rel_type, target=current,
                                confidence=round(conf, 3), derived=True,
                            )
                            path = " → ".join(r.target for r in chain)
                            explanation = (
                                f"{start} {rel_type.value} {current} "
                                f"(via: {start} → {path}, confidence: {conf:.0%})"
                            )
                            inferred.append(InferenceResult(
                                relation=derived_rel,
                                chain=chain,
                                explanation=explanation,
                            ))

                    # Continue BFS
                    for r in edges.get(current, []):
                        if r.target not in visited:
                            queue.append((r.target, chain + [r]))

        return inferred

    def infer_cross_relation(self) -> list[InferenceResult]:
        """
        Cross-relation inference:
        If A is_a B and B requires C → A requires C
        If A is_a B and B causes C → A causes C
        """
        inferred: list[InferenceResult] = []
        seen: set[str] = set()

        # Collect all is_a chains first
        is_a_map: dict[str, set[str]] = {}
        for r in self._all:
            if r.relation == RelationType.IS_A:
                is_a_map.setdefault(r.source, set()).add(r.target)

        # For each concept that IS something, inherit the parent's relations
        inheritable = {
            RelationType.CAUSES, RelationType.REQUIRES,
            RelationType.USES, RelationType.ENABLES,
        }

        for child, parents in is_a_map.items():
            for parent in parents:
                for parent_rel in self._forward.get(parent, []):
                    if parent_rel.relation in inheritable:
                        key = f"{child}:{parent_rel.relation.value}:{parent_rel.target}"
                        if key not in seen:
                            existing = self.query(
                                source=child, relation=parent_rel.relation,
                                target=parent_rel.target,
                            )
                            if not existing:
                                seen.add(key)
                                conf = parent_rel.confidence * 0.7
                                derived = Relation(
                                    source=child,
                                    relation=parent_rel.relation,
                                    target=parent_rel.target,
                                    confidence=round(conf, 3),
                                    derived=True,
                                )
                                explanation = (
                                    f"{child} {parent_rel.relation.value} {parent_rel.target} "
                                    f"(inherited: {child} is_a {parent}, "
                                    f"{parent} {parent_rel.relation.value} {parent_rel.target})"
                                )
                                inferred.append(InferenceResult(
                                    relation=derived, chain=[],
                                    explanation=explanation,
                                ))

        return inferred

    def run_inference(self, max_depth: int = 3) -> list[InferenceResult]:
        """
        Run all inference types and store results.

        Returns list of newly derived facts.
        """
        all_inferred: list[InferenceResult] = []

        # Transitive inference
        transitive = self.infer_transitive(max_depth=max_depth)
        all_inferred.extend(transitive)

        # Cross-relation inheritance
        cross = self.infer_cross_relation()
        all_inferred.extend(cross)

        # Store derived relations
        for result in all_inferred:
            r = result.relation
            self.add(r.source, r.relation, r.target, r.confidence, "inference")

        return all_inferred

    def stats(self) -> dict:
        """Statistics about the relation store."""
        by_type: dict[str, int] = {}
        derived_count = 0
        for r in self._all:
            by_type[r.relation.value] = by_type.get(r.relation.value, 0) + 1
            if r.derived:
                derived_count += 1

        return {
            "total_relations": len(self._all),
            "unique_concepts": len(set(
                r.source for r in self._all
            ) | set(
                r.target for r in self._all
            )),
            "by_type": by_type,
            "derived": derived_count,
            "taught": len(self._all) - derived_count,
        }
