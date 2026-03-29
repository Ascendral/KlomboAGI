"""
Explanation Generator — construct novel sentences from knowledge.

The difference between retrieval and understanding:
  Retrieval: "gravity is a fundamental force" (stored fact)
  Generation: "Because gravity causes force, and force causes acceleration,
              anything with mass will accelerate toward other masses.
              This is why objects fall." (constructed from relations)

The generator chains beliefs + relations + causal links to build
explanations the system has NEVER seen as text. It constructs them
from first principles using the knowledge graph.

This is language PRODUCTION, not reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.core.relations import RelationStore, RelationType


# Templates for turning relations into natural language
RELATION_TEMPLATES = {
    RelationType.IS_A: [
        "{subject} is a type of {object}",
        "{subject} is {object}",
    ],
    RelationType.CAUSES: [
        "{subject} causes {object}",
        "because of {subject}, {object} occurs",
        "when {subject} happens, it leads to {object}",
    ],
    RelationType.REQUIRES: [
        "{subject} requires {object}",
        "{subject} cannot happen without {object}",
        "for {subject} to work, {object} is needed",
    ],
    RelationType.PART_OF: [
        "{subject} is part of {object}",
        "{object} contains {subject}",
        "{subject} is a component of {object}",
    ],
    RelationType.USES: [
        "{subject} uses {object}",
        "{subject} relies on {object}",
        "{object} is a tool used by {subject}",
    ],
    RelationType.ENABLES: [
        "{subject} enables {object}",
        "{subject} makes {object} possible",
        "without {subject}, {object} could not happen",
    ],
    RelationType.OPPOSITE_OF: [
        "{subject} is the opposite of {object}",
        "{subject} and {object} are opposites",
    ],
    RelationType.MEASURES: [
        "{subject} measures {object}",
        "{subject} is a way to quantify {object}",
    ],
}

# Connective words for chaining sentences
CONNECTIVES = {
    "causal": ["Therefore", "As a result", "Consequently", "This means that"],
    "additive": ["Additionally", "Furthermore", "Also", "Moreover"],
    "contrastive": ["However", "On the other hand", "In contrast", "But"],
    "elaborative": ["Specifically", "In particular", "For example", "That is"],
    "temporal": ["Then", "Subsequently", "After that", "Next"],
}


@dataclass
class Explanation:
    """A generated explanation constructed from knowledge."""
    topic: str
    sentences: list[str]
    facts_used: int
    relations_used: int
    novel: bool = True   # True if this explanation was CONSTRUCTED, not retrieved

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "text": self.text,
            "sentences": self.sentences,
            "facts_used": self.facts_used,
            "relations_used": self.relations_used,
            "novel": self.novel,
        }


class ExplanationGenerator:
    """
    Constructs explanations by chaining knowledge.

    Not retrieval. Construction.
    """

    def __init__(self, relations: RelationStore,
                 beliefs: dict | None = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}

    def explain(self, concept: str, depth: int = 3) -> Explanation:
        """
        Generate a multi-sentence explanation of a concept.

        Chains: definition → properties → causes/effects → connections → significance.
        """
        sentences = []
        facts_used = 0
        relations_used = 0

        # 1. Definition — what IS it?
        definition = self._get_definition(concept)
        if definition:
            sentences.append(definition)
            facts_used += 1

        # 2. What does it cause? (forward causal)
        effects = self.relations.get_forward(concept, RelationType.CAUSES)
        if effects:
            effect_names = [e.target for e in effects[:3]]
            if len(effect_names) == 1:
                sentences.append(f"{self._cap(concept)} causes {effect_names[0]}.")
            elif len(effect_names) > 1:
                listed = ", ".join(effect_names[:-1]) + f" and {effect_names[-1]}"
                sentences.append(f"{self._cap(concept)} causes {listed}.")
            relations_used += len(effects[:3])

            # Chain one deeper — what do the effects cause?
            if depth > 1:
                for eff in effects[:2]:
                    deeper = self.relations.get_forward(eff.target, RelationType.CAUSES)
                    if deeper:
                        d = deeper[0]
                        sentences.append(
                            f"This means that {concept} indirectly leads to {d.target}, "
                            f"because {eff.target} causes {d.target}.")
                        relations_used += 1
                        break

        # 3. What causes it? (backward causal)
        causes = self.relations.get_backward(concept, RelationType.CAUSES)
        if causes:
            cause_names = [c.source for c in causes[:3]]
            if len(cause_names) == 1:
                sentences.append(f"{self._cap(concept)} is caused by {cause_names[0]}.")
            else:
                listed = ", ".join(cause_names[:-1]) + f" and {cause_names[-1]}"
                sentences.append(f"{self._cap(concept)} is caused by {listed}.")
            relations_used += len(causes[:3])

        # 4. What does it use?
        uses = self.relations.get_forward(concept, RelationType.USES)
        if uses:
            use_names = [u.target for u in uses[:3]]
            listed = ", ".join(use_names[:-1]) + f" and {use_names[-1]}" if len(use_names) > 1 else use_names[0]
            sentences.append(f"{self._cap(concept)} uses {listed}.")
            relations_used += len(uses[:3])

        # 5. What is it part of / what are its parts?
        parent = self.relations.get_forward(concept, RelationType.PART_OF)
        if parent:
            sentences.append(f"{self._cap(concept)} is part of {parent[0].target}.")
            relations_used += 1

        children = self.relations.get_backward(concept, RelationType.PART_OF)
        if children:
            child_names = [c.source for c in children[:5]]
            listed = ", ".join(child_names[:-1]) + f" and {child_names[-1]}" if len(child_names) > 1 else child_names[0]
            sentences.append(f"{self._cap(concept)} contains {listed}.")
            relations_used += len(children[:5])

        # 6. What enables it / what does it enable?
        enables = self.relations.get_forward(concept, RelationType.ENABLES)
        if enables:
            sentences.append(f"{self._cap(concept)} enables {enables[0].target}.")
            relations_used += 1

        enabled_by = self.relations.get_backward(concept, RelationType.ENABLES)
        if enabled_by:
            sentences.append(f"{self._cap(concept)} is made possible by {enabled_by[0].source}.")
            relations_used += 1

        # 7. Opposite?
        opposites = self.relations.get_forward(concept, RelationType.OPPOSITE_OF)
        if not opposites:
            opposites = self.relations.get_backward(concept, RelationType.OPPOSITE_OF)
            if opposites:
                sentences.append(f"The opposite of {concept} is {opposites[0].source}.")
                relations_used += 1
        elif opposites:
            sentences.append(f"The opposite of {concept} is {opposites[0].target}.")
            relations_used += 1

        # 8. Who uses it?
        used_by = self.relations.get_backward(concept, RelationType.USES)
        if used_by:
            user_names = [u.source for u in used_by[:3]]
            listed = ", ".join(user_names[:-1]) + f" and {user_names[-1]}" if len(user_names) > 1 else user_names[0]
            sentences.append(f"{self._cap(concept)} is used by {listed}.")
            relations_used += 1

        # 9. Additional beliefs
        concept_lower = concept.lower()
        for statement, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                if belief.predicate and len(belief.predicate) > 5:
                    fact_sentence = f"{self._cap(concept)} is {belief.predicate}."
                    if fact_sentence not in sentences and len(sentences) < 8:
                        sentences.append(fact_sentence)
                        facts_used += 1

        if not sentences:
            sentences.append(f"I don't know enough about {concept} to explain it.")

        return Explanation(
            topic=concept,
            sentences=sentences,
            facts_used=facts_used,
            relations_used=relations_used,
            novel=relations_used > 0,
        )

    def compare(self, concept_a: str, concept_b: str) -> str:
        """
        Generate a comparison between two concepts.

        Finds shared properties, differences, and connections.
        """
        lines = []

        # Get explanations for both
        exp_a = self.explain(concept_a, depth=1)
        exp_b = self.explain(concept_b, depth=1)

        # Shared relations
        rels_a = {(r.relation, r.target) for r in self.relations.get_forward(concept_a)}
        rels_b = {(r.relation, r.target) for r in self.relations.get_forward(concept_b)}
        shared = rels_a & rels_b

        if shared:
            lines.append(f"Both {concept_a} and {concept_b} share:")
            for rel, target in list(shared)[:3]:
                lines.append(f"  {rel.value} {target}")

        # Unique to each
        only_a = rels_a - rels_b
        only_b = rels_b - rels_a

        if only_a:
            lines.append(f"\nOnly {concept_a}:")
            for rel, target in list(only_a)[:3]:
                lines.append(f"  {rel.value} {target}")

        if only_b:
            lines.append(f"\nOnly {concept_b}:")
            for rel, target in list(only_b)[:3]:
                lines.append(f"  {rel.value} {target}")

        # Connection between them
        path = self.relations.find_path(concept_a, concept_b)
        if path:
            chain = " → ".join(f"{r.target}" for r in path)
            lines.append(f"\nConnection: {concept_a} → {chain}")

        if not lines:
            lines.append(f"I don't know enough to compare {concept_a} and {concept_b}.")

        return "\n".join(lines)

    def _get_definition(self, concept: str) -> str:
        """Get the core definition from beliefs."""
        concept_lower = concept.lower()
        for statement, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                pred = belief.predicate
                if pred and len(pred) > 5 and len(pred) < 100:
                    return f"{self._cap(concept)} is {pred}."
        return ""

    def _cap(self, s: str) -> str:
        return s[0].upper() + s[1:] if s else s
