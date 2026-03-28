"""
Constructive Memory — reconstruct, don't retrieve.

From Joscha Bach / cognitive science of memory.

Human memory doesn't work like a database. You don't "retrieve"
a memory — you RECONSTRUCT it from fragments using current context.
That's why memories change over time. That's why you remember
things differently depending on your mood and what you're thinking about.

For KlomboAGI:
  Instead of exact retrieval (find the belief that matches),
  RECONSTRUCT an understanding from:
  1. Direct beliefs about the concept
  2. Inherited properties from parent categories
  3. Inferred properties from related concepts
  4. Current context (what we're talking about)
  5. Activation levels (what's recently/frequently accessed)

The reconstruction IS the understanding — not a copy of stored data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReconstructedMemory:
    """A memory reconstructed from fragments, not retrieved verbatim."""
    concept: str
    fragments: list[str]        # pieces used in reconstruction
    sources: list[str]          # where each fragment came from
    reconstruction: str         # the assembled understanding
    confidence: float           # how confident in the reconstruction
    context_influence: float    # how much current context shaped it (0-1)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "fragments": len(self.fragments),
            "reconstruction": self.reconstruction[:200],
            "confidence": round(self.confidence, 3),
            "context_influence": round(self.context_influence, 3),
        }


class ConstructiveMemory:
    """
    Reconstruct understanding from fragments + context.
    Not retrieval. Construction.
    """

    def __init__(self, beliefs: dict, relations, activation_decay=None,
                 working_memory=None) -> None:
        self.beliefs = beliefs
        self.relations = relations
        self.decay = activation_decay
        self.working_memory = working_memory

    def reconstruct(self, concept: str) -> ReconstructedMemory:
        """
        Reconstruct understanding of a concept from all available sources.
        """
        fragments = []
        sources = []
        concept_lower = concept.lower()

        # 1. Direct beliefs
        for stmt, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                if belief.predicate and len(belief.predicate) < 80:
                    weight = belief.truth.confidence if hasattr(belief, 'truth') else 0.5
                    # Boost if recently accessed (ACT-R)
                    if self.decay and self.decay.retrievable(concept_lower):
                        weight *= 1.3
                    fragments.append((f"{concept} is {belief.predicate}", weight))
                    sources.append("direct_belief")

        # 2. Inherited from parents (is_a chain)
        if hasattr(self.relations, 'get_forward'):
            from klomboagi.core.relations import RelationType
            parents = self.relations.get_forward(concept_lower, RelationType.IS_A)
            for p in parents[:3]:
                # What do we know about the parent?
                for stmt, belief in self.beliefs.items():
                    if hasattr(belief, 'subject') and belief.subject == p.target:
                        if belief.predicate and len(belief.predicate) < 80:
                            fragments.append(
                                (f"As a {p.target}, {concept} {belief.predicate}", 0.3))
                            sources.append(f"inherited_from_{p.target}")

        # 3. Relations — what it causes, uses, is part of
        if hasattr(self.relations, 'get_forward'):
            for rel in self.relations.get_forward(concept_lower)[:5]:
                fragments.append(
                    (f"{concept} {rel.relation.value} {rel.target}", rel.confidence * 0.5))
                sources.append(f"relation_{rel.relation.value}")

            for rel in self.relations.get_backward(concept_lower)[:3]:
                fragments.append(
                    (f"{rel.source} {rel.relation.value} {concept}", rel.confidence * 0.3))
                sources.append(f"inverse_relation")

        # 4. Context influence — what's in working memory right now
        context_influence = 0.0
        if self.working_memory:
            active = self.working_memory.get_active_items()
            for item in active[:5]:
                # If something in working memory is related, boost it
                for stmt, belief in self.beliefs.items():
                    if (hasattr(belief, 'subject') and belief.subject == item.content
                            and hasattr(belief, 'predicate') and concept_lower in str(belief.predicate).lower()):
                        fragments.append(
                            (f"In current context: {item.content} relates to {concept}", 0.2))
                        sources.append("context")
                        context_influence += 0.1

        # 5. Assemble reconstruction — highest weight fragments first
        fragments.sort(key=lambda x: x[1], reverse=True)

        if not fragments:
            return ReconstructedMemory(
                concept=concept, fragments=[], sources=[],
                reconstruction=f"I have no understanding of {concept}.",
                confidence=0.0, context_influence=0.0,
            )

        # Build natural language reconstruction
        sentences = []
        total_weight = 0.0
        for text, weight in fragments[:6]:  # Top 6 fragments
            sentences.append(text)
            total_weight += weight

        confidence = min(0.95, total_weight / max(len(sentences), 1))
        reconstruction = ". ".join(sentences) + "."

        return ReconstructedMemory(
            concept=concept,
            fragments=[t for t, _ in fragments[:6]],
            sources=sources[:6],
            reconstruction=reconstruction,
            confidence=confidence,
            context_influence=min(1.0, context_influence),
        )
