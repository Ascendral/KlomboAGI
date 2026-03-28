"""
Explanation Synthesizer — turn raw facts into coherent explanations.

Instead of dumping a list of beliefs, compose a paragraph that
EXPLAINS the concept using everything the system knows.

This is the difference between:
  "gravity is a fundamental force" + "gravity causes acceleration"
and:
  "Gravity is a fundamental force of attraction between objects with
   mass. It causes acceleration, which in turn creates velocity.
   Physics uses geometry to describe how gravity curves space."

No LLM. Constructs explanations by ordering facts logically.
"""

from __future__ import annotations

from klomboagi.core.relations import RelationStore, RelationType
from klomboagi.reasoning.truth import Belief


class Synthesizer:
    """Compose coherent explanations from raw knowledge."""

    # Order in which relation types should appear in an explanation
    RELATION_ORDER = [
        RelationType.IS_A,
        RelationType.PART_OF,
        RelationType.CAUSES,
        RelationType.ENABLES,
        RelationType.REQUIRES,
        RelationType.USES,
        RelationType.MEASURES,
        RelationType.OPPOSITE_OF,
        RelationType.EXAMPLE_OF,
        RelationType.ANALOGOUS_TO,
    ]

    RELATION_TEMPLATES = {
        RelationType.IS_A: "{source} is a type of {target}",
        RelationType.CAUSES: "{source} causes {target}",
        RelationType.REQUIRES: "{source} requires {target}",
        RelationType.PART_OF: "{source} is part of {target}",
        RelationType.USES: "{source} uses {target}",
        RelationType.OPPOSITE_OF: "The opposite of {source} is {target}",
        RelationType.ENABLES: "{source} enables {target}",
        RelationType.MEASURES: "{source} measures {target}",
        RelationType.EXAMPLE_OF: "{source} is an example of {target}",
        RelationType.ANALOGOUS_TO: "{source} is analogous to {target}",
    }

    def __init__(self, relations: RelationStore,
                 beliefs: dict | None = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}

    def explain(self, concept: str) -> str:
        """
        Generate a coherent explanation of a concept.

        Combines beliefs (what it IS) with relations (how it CONNECTS)
        into a flowing paragraph.
        """
        sentences = []

        # 1. Core definition from beliefs
        definition = self._get_definition(concept)
        if definition:
            sentences.append(definition)

        # 2. Relations, ordered by type
        for rel_type in self.RELATION_ORDER:
            forward = self.relations.get_forward(concept, rel_type)
            backward = self.relations.get_backward(concept, rel_type)

            for r in forward[:2]:
                template = self.RELATION_TEMPLATES.get(r.relation, "{source} relates to {target}")
                sentences.append(template.format(source=self._cap(concept), target=r.target))

            # Backward relations phrased differently
            if backward and rel_type == RelationType.CAUSES:
                causes = [r.source for r in backward[:3]]
                if causes:
                    sentences.append(f"It is caused by {self._join_list(causes)}")
            elif backward and rel_type == RelationType.PART_OF:
                parts = [r.source for r in backward[:5]]
                if parts:
                    sentences.append(f"It contains {self._join_list(parts)}")
            elif backward and rel_type == RelationType.USES:
                users = [r.source for r in backward[:3]]
                if users:
                    sentences.append(f"It is used by {self._join_list(users)}")

        if not sentences:
            return ""

        # 3. Combine into paragraph
        return ". ".join(sentences) + "."

    def _get_definition(self, concept: str) -> str:
        """Find the core 'X is Y' definition from beliefs."""
        concept_lower = concept.lower()
        for statement, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                # Use the highest-confidence definition
                pred = belief.predicate
                if pred and len(pred) > 3 and len(pred) < 100:
                    return f"{self._cap(concept)} is {pred}"
        return ""

    def _cap(self, s: str) -> str:
        """Capitalize first letter."""
        return s[0].upper() + s[1:] if s else s

    def _join_list(self, items: list[str]) -> str:
        """Join list with commas and 'and'."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]
