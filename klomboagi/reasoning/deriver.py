"""
Property Deriver — derives facts about properties from first principles.

This is the actual reasoning piece. Not a lookup table.
Given a property like "green", it works through what that property IS
by chaining derivations until it hits something measurable.

The key insight: every property ultimately exists in some space.
The engine discovers WHICH space by asking "where does this property
exist?" and decomposing the answer.

This module builds a KNOWLEDGE GRAPH of concepts and their relationships.
When it encounters a new property, it traverses the graph to derive
dimensional signatures instead of looking them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Concept:
    """A node in the knowledge graph."""
    name: str
    is_a: list[str] = field(default_factory=list)       # parent concepts
    has_property: list[str] = field(default_factory=list)  # properties this concept has
    exists_in: str = ""                                   # what space it occupies
    composed_of: list[str] = field(default_factory=list)  # what dimensions compose it
    dimension_count: int = -1                             # derived, not assigned

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_a": self.is_a,
            "has_property": self.has_property,
            "exists_in": self.exists_in,
            "composed_of": self.composed_of,
            "dimension_count": self.dimension_count,
        }


@dataclass
class Derivation:
    """One step of property derivation."""
    from_concept: str
    relation: str       # "is_a", "exists_in", "composed_of", "therefore"
    to_concept: str
    explanation: str

    def __repr__(self) -> str:
        return f"{self.from_concept} --{self.relation}--> {self.to_concept}"


class KnowledgeGraph:
    """
    A graph of concepts and their relationships.

    This is NOT a static database. The graph grows as the system
    encounters new concepts and derives relationships between them.

    Seed knowledge is minimal — just enough to bootstrap reasoning
    about spatial concepts. Everything else is derived.
    """

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self._seed()

    def _seed(self) -> None:
        """
        Minimal seed knowledge — spatial primitives only.
        These are the axioms. Everything else is derived.
        """
        # Spatial primitives — these are definitional
        self.add("point", dimension_count=0, exists_in="nothing")
        self.add("line", dimension_count=1, composed_of=["length"], exists_in="1d_space")
        self.add("surface", dimension_count=2, composed_of=["length", "width"], exists_in="2d_space")
        self.add("volume", dimension_count=3, composed_of=["length", "width", "depth"], exists_in="3d_space")

        # What things exist in — spatial relationships
        self.add("scalar", dimension_count=0, is_a=["point"])
        self.add("axis", dimension_count=1, is_a=["line"])
        self.add("area", dimension_count=2, is_a=["surface"])
        self.add("body", dimension_count=3, is_a=["volume"])

        # How properties attach to objects
        self.add("visual_property", exists_in="surface", is_a=["property"])
        self.add("linear_property", exists_in="axis", is_a=["property"])
        self.add("scalar_property", exists_in="point", is_a=["property"])
        self.add("volumetric_property", exists_in="volume", is_a=["property"])
        self.add("temporal_property", exists_in="axis", is_a=["property"])
        self.add("waveform_property", exists_in="axis", is_a=["property"])

    def add(self, name: str, **kwargs) -> Concept:
        """Add or update a concept."""
        if name in self.concepts:
            c = self.concepts[name]
            for k, v in kwargs.items():
                if hasattr(c, k):
                    if isinstance(getattr(c, k), list) and isinstance(v, list):
                        existing = getattr(c, k)
                        for item in v:
                            if item not in existing:
                                existing.append(item)
                    else:
                        setattr(c, k, v)
        else:
            c = Concept(name=name)
            for k, v in kwargs.items():
                if hasattr(c, k):
                    setattr(c, k, v)
            self.concepts[name] = c
        return c

    def get(self, name: str) -> Concept | None:
        return self.concepts.get(name)

    def find_dimension(self, concept_name: str, visited: set | None = None) -> int:
        """
        Derive the dimensional count of a concept by traversing the graph.
        This is the core derivation — not a lookup.
        """
        if visited is None:
            visited = set()
        if concept_name in visited:
            return -1  # Cycle detection
        visited.add(concept_name)

        concept = self.get(concept_name)
        if concept is None:
            return -1

        # If we already know, return it
        if concept.dimension_count >= 0:
            return concept.dimension_count

        # Try to derive from "exists_in"
        if concept.exists_in:
            space_dims = self.find_dimension(concept.exists_in, visited)
            if space_dims >= 0:
                concept.dimension_count = space_dims
                return space_dims

        # Try to derive from "is_a" (inheritance)
        for parent in concept.is_a:
            parent_dims = self.find_dimension(parent, visited)
            if parent_dims >= 0:
                concept.dimension_count = parent_dims
                return parent_dims

        # Try to derive from "composed_of" (count independent axes)
        if concept.composed_of:
            concept.dimension_count = len(concept.composed_of)
            return concept.dimension_count

        return -1


class PropertyDeriver:
    """
    Derives properties of concepts from first principles.

    Instead of looking up "color = 2D" in a table, it reasons:

    "green" → is it a known concept? no
    → what kind of thing is "green"? → a color
    → "color" → is_a visual_property → exists_in surface
    → "surface" → composed_of [length, width] → dimension_count = 2
    → therefore "green" has dimension_count = 2

    Each step is a traversal of the knowledge graph.
    New concepts learned along the way are added to the graph.
    """

    def __init__(self, graph: KnowledgeGraph | None = None) -> None:
        self.graph = graph or KnowledgeGraph()
        self.derivation_trace: list[Derivation] = []

    def teach(self, concept: str, **relationships) -> None:
        """
        Teach the system about a concept.
        This is how humans nudge the system's knowledge.

        Example:
            deriver.teach("color", is_a=["visual_property"])
            deriver.teach("temperature", exists_in="scalar")
        """
        self.graph.add(concept, **relationships)

    def derive_dimensions(self, property_name: str) -> tuple[int, list[Derivation]]:
        """
        Derive the dimensional signature of a property.
        Returns (dimension_count, derivation_trace).

        The trace shows HOW the engine arrived at its answer —
        every step of the reasoning chain.
        """
        self.derivation_trace = []

        # Step 1: Is this concept already in the graph?
        concept = self.graph.get(property_name)
        if concept and concept.dimension_count >= 0:
            self.derivation_trace.append(Derivation(
                property_name, "known_as", f"{concept.dimension_count}D",
                f"'{property_name}' is already known to be {concept.dimension_count}D"
            ))
            return concept.dimension_count, self.derivation_trace

        # Step 2: Try to find it through relationships
        if concept:
            dims = self._trace_derivation(property_name)
            if dims >= 0:
                return dims, self.derivation_trace

        # Step 3: We don't know this concept — return unknown
        self.derivation_trace.append(Derivation(
            property_name, "unknown", "?",
            f"'{property_name}' is not in the knowledge graph and cannot be derived"
        ))
        return -1, self.derivation_trace

    def _trace_derivation(self, name: str, visited: set | None = None) -> int:
        """Trace through the graph recording each reasoning step."""
        if visited is None:
            visited = set()
        if name in visited:
            return -1
        visited.add(name)

        concept = self.graph.get(name)
        if not concept:
            return -1

        if concept.dimension_count >= 0:
            self.derivation_trace.append(Derivation(
                name, "is", f"{concept.dimension_count}D",
                f"'{name}' is defined as {concept.dimension_count}D"
            ))
            return concept.dimension_count

        # Try exists_in
        if concept.exists_in:
            self.derivation_trace.append(Derivation(
                name, "exists_in", concept.exists_in,
                f"'{name}' exists in '{concept.exists_in}'"
            ))
            dims = self._trace_derivation(concept.exists_in, visited)
            if dims >= 0:
                concept.dimension_count = dims
                self.derivation_trace.append(Derivation(
                    name, "therefore", f"{dims}D",
                    f"'{name}' inherits {dims}D from '{concept.exists_in}'"
                ))
                return dims

        # Try is_a
        for parent in concept.is_a:
            self.derivation_trace.append(Derivation(
                name, "is_a", parent,
                f"'{name}' is a kind of '{parent}'"
            ))
            dims = self._trace_derivation(parent, visited)
            if dims >= 0:
                concept.dimension_count = dims
                self.derivation_trace.append(Derivation(
                    name, "therefore", f"{dims}D",
                    f"'{name}' inherits {dims}D from '{parent}'"
                ))
                return dims

        # Try composed_of
        if concept.composed_of:
            dims = len(concept.composed_of)
            concept.dimension_count = dims
            self.derivation_trace.append(Derivation(
                name, "composed_of", str(concept.composed_of),
                f"'{name}' is composed of {dims} independent axes: {concept.composed_of}"
            ))
            return dims

        return -1

    def explain(self) -> str:
        """Show the derivation chain in human-readable form."""
        if not self.derivation_trace:
            return "No derivation performed."
        lines = ["Derivation chain:"]
        for i, d in enumerate(self.derivation_trace):
            lines.append(f"  {i + 1}. {d.from_concept} --{d.relation}--> {d.to_concept}")
            lines.append(f"     {d.explanation}")
        return "\n".join(lines)

    def derive_and_compare(self, prop_a: str, prop_b: str) -> dict:
        """
        Derive dimensional signatures for two properties and compare them.
        Returns a full reasoning report.
        """
        dims_a, trace_a = self.derive_dimensions(prop_a)
        dims_b, trace_b = self.derive_dimensions(prop_b)

        comparison = {
            "property_a": {"name": prop_a, "dimensions": dims_a, "trace": [repr(d) for d in trace_a]},
            "property_b": {"name": prop_b, "dimensions": dims_b, "trace": [repr(d) for d in trace_b]},
            "comparable": dims_a >= 0 and dims_b >= 0,
        }

        if dims_a >= 0 and dims_b >= 0:
            if dims_a > dims_b:
                comparison["result"] = f"'{prop_a}' ({dims_a}D) occupies more dimensional space than '{prop_b}' ({dims_b}D)"
                comparison["winner"] = prop_a
            elif dims_b > dims_a:
                comparison["result"] = f"'{prop_b}' ({dims_b}D) occupies more dimensional space than '{prop_a}' ({dims_a}D)"
                comparison["winner"] = prop_b
            else:
                comparison["result"] = f"Both '{prop_a}' and '{prop_b}' are {dims_a}D — compare by magnitude"
                comparison["winner"] = None
        else:
            unknowns = []
            if dims_a < 0:
                unknowns.append(prop_a)
            if dims_b < 0:
                unknowns.append(prop_b)
            comparison["result"] = f"Cannot compare — unknown dimensions for: {', '.join(unknowns)}"
            comparison["winner"] = None
            comparison["needs_teaching"] = unknowns

        return comparison
