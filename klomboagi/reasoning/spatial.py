"""
Spatial Reasoning — think about positions, shapes, containment, distance.

A mind needs to reason about space:
  "Is Japan closer to China or to Brazil?"
  "Can a elephant fit through a doorway?"
  "If I'm facing north and turn right, which direction am I facing?"

Not visual processing — CONCEPTUAL spatial reasoning.
Uses stored knowledge about sizes, positions, containment.

Spatial relations:
  contains: a city contains buildings
  inside: a nucleus is inside a cell
  near: the moon is near earth
  larger_than: the sun is larger than earth
  above/below: the atmosphere is above the surface
  between: earth is between venus and mars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SpatialRelation(Enum):
    CONTAINS = "contains"
    INSIDE = "inside"
    NEAR = "near"
    FAR = "far"
    LARGER = "larger_than"
    SMALLER = "smaller_than"
    ABOVE = "above"
    BELOW = "below"
    BETWEEN = "between"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ADJACENT = "adjacent"


@dataclass
class SpatialFact:
    """A fact about spatial relationships."""
    subject: str
    relation: SpatialRelation
    object: str
    magnitude: float = 0.0  # for size comparisons, distances

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation.value,
            "object": self.object,
            "magnitude": self.magnitude,
        }


class SpatialReasoner:
    """
    Reasons about spatial relationships between concepts.

    Can answer:
    - Containment: "Is X inside Y?" (cell inside organism)
    - Size comparison: "Is X larger than Y?"
    - Transitivity: if A contains B and B contains C, then A contains C
    - Relative position: "What is between X and Y?"
    """

    def __init__(self) -> None:
        self.facts: list[SpatialFact] = []
        self._by_subject: dict[str, list[SpatialFact]] = {}
        self._seed_spatial_knowledge()

    def _seed_spatial_knowledge(self) -> None:
        """Seed with basic spatial knowledge."""
        seeds = [
            # Size hierarchy
            ("the universe", SpatialRelation.CONTAINS, "galaxy"),
            ("galaxy", SpatialRelation.CONTAINS, "solar system"),
            ("solar system", SpatialRelation.CONTAINS, "planet"),
            ("planet", SpatialRelation.CONTAINS, "continent"),
            ("continent", SpatialRelation.CONTAINS, "country"),
            ("country", SpatialRelation.CONTAINS, "city"),
            ("city", SpatialRelation.CONTAINS, "building"),
            ("building", SpatialRelation.CONTAINS, "room"),

            # Biology hierarchy
            ("organism", SpatialRelation.CONTAINS, "organ"),
            ("organ", SpatialRelation.CONTAINS, "tissue"),
            ("tissue", SpatialRelation.CONTAINS, "cell"),
            ("cell", SpatialRelation.CONTAINS, "organelle"),
            ("organelle", SpatialRelation.CONTAINS, "molecule"),
            ("molecule", SpatialRelation.CONTAINS, "atom"),
            ("atom", SpatialRelation.CONTAINS, "nucleus"),
            ("nucleus", SpatialRelation.CONTAINS, "proton"),
            ("nucleus", SpatialRelation.CONTAINS, "neutron"),

            # Solar system
            ("sun", SpatialRelation.LARGER, "jupiter"),
            ("jupiter", SpatialRelation.LARGER, "earth"),
            ("earth", SpatialRelation.LARGER, "moon"),
            ("earth", SpatialRelation.NEAR, "moon"),

            # Earth
            ("atmosphere", SpatialRelation.ABOVE, "surface"),
            ("surface", SpatialRelation.ABOVE, "mantle"),
            ("mantle", SpatialRelation.ABOVE, "core"),
        ]

        for subj, rel, obj in seeds:
            self.add(subj, rel, obj)

    def add(self, subject: str, relation: SpatialRelation,
            obj: str, magnitude: float = 0.0) -> None:
        """Add a spatial fact."""
        fact = SpatialFact(subject=subject, relation=relation,
                          object=obj, magnitude=magnitude)
        self.facts.append(fact)
        self._by_subject.setdefault(subject.lower(), []).append(fact)

        # Auto-add inverse
        inverse_map = {
            SpatialRelation.CONTAINS: SpatialRelation.INSIDE,
            SpatialRelation.INSIDE: SpatialRelation.CONTAINS,
            SpatialRelation.LARGER: SpatialRelation.SMALLER,
            SpatialRelation.SMALLER: SpatialRelation.LARGER,
            SpatialRelation.ABOVE: SpatialRelation.BELOW,
            SpatialRelation.BELOW: SpatialRelation.ABOVE,
        }
        if relation in inverse_map:
            inv = SpatialFact(subject=obj, relation=inverse_map[relation],
                            object=subject, magnitude=magnitude)
            self.facts.append(inv)
            self._by_subject.setdefault(obj.lower(), []).append(inv)

    def query(self, subject: str, relation: SpatialRelation = None) -> list[SpatialFact]:
        """Find spatial facts about a subject."""
        facts = self._by_subject.get(subject.lower(), [])
        if relation:
            return [f for f in facts if f.relation == relation]
        return facts

    def contains(self, container: str, thing: str, max_depth: int = 6) -> bool:
        """Does container contain thing? (transitive)"""
        visited = set()
        queue = [container.lower()]
        for _ in range(max_depth):
            if not queue:
                break
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for fact in self._by_subject.get(current, []):
                if fact.relation == SpatialRelation.CONTAINS:
                    if fact.object.lower() == thing.lower():
                        return True
                    queue.append(fact.object.lower())
        return False

    def larger_than(self, a: str, b: str) -> bool | None:
        """Is A larger than B? Returns None if unknown."""
        # Direct
        for f in self._by_subject.get(a.lower(), []):
            if f.relation == SpatialRelation.LARGER and f.object.lower() == b.lower():
                return True
            if f.relation == SpatialRelation.SMALLER and f.object.lower() == b.lower():
                return False

        # Transitive through containment — container is larger than contents
        if self.contains(a, b):
            return True
        if self.contains(b, a):
            return False

        return None

    def reason(self, question: str) -> str:
        """Answer a spatial reasoning question."""
        q = question.lower().strip().rstrip("?")

        # "Is X inside Y?" / "Does Y contain X?"
        if "inside" in q or "contain" in q or "in " in q:
            words = q.split()
            # Find the two concepts
            concepts = [w for w in words if len(w) > 2
                       and w not in {"is", "does", "inside", "contain", "contains",
                                     "the", "a", "an", "in", "what"}]
            if len(concepts) >= 2:
                if self.contains(concepts[-1], concepts[0]):
                    return f"Yes, {concepts[0]} is inside {concepts[-1]}."
                if self.contains(concepts[0], concepts[-1]):
                    return f"No, it's the other way — {concepts[0]} contains {concepts[-1]}."
                return f"I don't know the spatial relationship between {concepts[0]} and {concepts[-1]}."

        # "Is X larger than Y?"
        if "larger" in q or "bigger" in q or "smaller" in q:
            concepts = [w for w in q.split() if len(w) > 2
                       and w not in {"is", "than", "larger", "bigger", "smaller",
                                     "the", "a", "an", "which"}]
            if len(concepts) >= 2:
                result = self.larger_than(concepts[0], concepts[-1])
                if result is True:
                    return f"Yes, {concepts[0]} is larger than {concepts[-1]}."
                elif result is False:
                    return f"No, {concepts[-1]} is larger than {concepts[0]}."
                return f"I can't determine the size relationship."

        # "What contains X?" / "What is X inside?"
        if "what contains" in q or "what is" in q and "inside" in q:
            concepts = [w for w in q.split() if len(w) > 3
                       and w not in {"what", "contains", "inside", "the"}]
            if concepts:
                containers = []
                for fact in self.facts:
                    if (fact.relation == SpatialRelation.CONTAINS
                            and fact.object.lower() == concepts[0].lower()):
                        containers.append(fact.subject)
                if containers:
                    return f"{concepts[0]} is inside {', '.join(containers)}."

        return "I need more spatial information to answer this."
