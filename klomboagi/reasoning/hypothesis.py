"""
Hypothesis Engine — reason from what you know to guess what you don't.

When asked something it doesn't directly know, the system should:
1. Find related facts it DOES know
2. Chain them together logically
3. Form a hypothesis with explicit uncertainty
4. Explain its reasoning

This is the difference between "I don't know" and "Based on what I know,
I think X because Y and Z, but I'm not sure."

No LLM. Pure structural reasoning from the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.core.relations import RelationStore, RelationType
from klomboagi.reasoning.truth import TruthValue


@dataclass
class Hypothesis:
    """A guess formed from existing knowledge."""
    claim: str
    confidence: float              # 0-1, how confident
    supporting_facts: list[str]    # evidence FOR
    chain: list[str]               # reasoning chain
    assumptions: list[str]         # what we assumed
    alternative: str = ""          # another possibility

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "confidence": round(self.confidence, 3),
            "supporting_facts": self.supporting_facts,
            "chain": self.chain,
            "assumptions": self.assumptions,
            "alternative": self.alternative,
        }

    def explain(self) -> str:
        """Human-readable explanation of the hypothesis."""
        lines = []
        conf_word = "very likely" if self.confidence > 0.7 else \
                    "likely" if self.confidence > 0.5 else \
                    "possibly" if self.confidence > 0.3 else "uncertain"

        lines.append(f"I think: {self.claim} ({conf_word}, {self.confidence:.0%})")

        if self.chain:
            lines.append("My reasoning:")
            for step in self.chain:
                lines.append(f"  {step}")

        if self.supporting_facts:
            lines.append(f"Based on: {'; '.join(self.supporting_facts[:5])}")

        if self.assumptions:
            lines.append(f"I'm assuming: {'; '.join(self.assumptions)}")

        if self.alternative:
            lines.append(f"But it could also be: {self.alternative}")

        return "\n".join(lines)


class HypothesisEngine:
    """
    Forms hypotheses by chaining existing knowledge.

    Strategies:
    1. Inheritance: if A is_a B and B has property P, maybe A has P too
    2. Causal: if A causes B and B causes C, maybe A indirectly causes C
    3. Analogy: if A is similar to B and B has property P, maybe A does too
    4. Part-whole: if A is part_of B and B has property P, maybe A has P too
    """

    def __init__(self, relations: RelationStore,
                 beliefs: dict | None = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}

    def hypothesize(self, question: str, query_concepts: list[str]) -> Hypothesis | None:
        """
        Form a hypothesis about a question from existing knowledge.

        Returns None if we can't form any reasonable hypothesis.
        """
        hypotheses = []

        for concept in query_concepts:
            # Strategy 1: Inheritance — what do parent concepts tell us?
            h = self._try_inheritance(concept, question)
            if h:
                hypotheses.append(h)

            # Strategy 2: Causal chain — what can we derive through causation?
            h = self._try_causal(concept, question)
            if h:
                hypotheses.append(h)

            # Strategy 3: Part-whole — what does the containing system tell us?
            h = self._try_part_whole(concept, question)
            if h:
                hypotheses.append(h)

        if not hypotheses:
            return None

        # Return highest confidence hypothesis
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[0]

    def _try_inheritance(self, concept: str, question: str) -> Hypothesis | None:
        """If X is_a Y, and we know things about Y, apply them to X."""
        parents = self.relations.get_forward(concept, RelationType.IS_A)
        if not parents:
            # Check beliefs for is_a
            for stmt, belief in self.beliefs.items():
                if hasattr(belief, 'subject') and belief.subject == concept and belief.predicate:
                    # concept is_a predicate — check what predicate knows
                    parent = belief.predicate
                    parent_rels = self.relations.get_forward(parent)
                    if parent_rels:
                        parents_from_beliefs = [type('R', (), {'target': parent})]
                        for pr in parent_rels[:3]:
                            return Hypothesis(
                                claim=f"{concept} probably {pr.relation.value} {pr.target}",
                                confidence=0.4 * belief.truth.confidence,
                                supporting_facts=[f"{concept} is {parent}", f"{parent} {pr.relation.value} {pr.target}"],
                                chain=[
                                    f"{concept} is {parent}",
                                    f"{parent} {pr.relation.value} {pr.target}",
                                    f"Therefore {concept} might {pr.relation.value} {pr.target}",
                                ],
                                assumptions=[f"{concept} inherits properties from {parent}"],
                            )
            return None

        for parent_rel in parents:
            parent = parent_rel.target
            # What do we know about the parent?
            parent_relations = self.relations.get_forward(parent)
            for pr in parent_relations[:3]:
                return Hypothesis(
                    claim=f"{concept} probably {pr.relation.value} {pr.target}",
                    confidence=0.5 * parent_rel.confidence,
                    supporting_facts=[f"{concept} is_a {parent}", f"{parent} {pr.relation.value} {pr.target}"],
                    chain=[
                        f"{concept} is_a {parent}",
                        f"{parent} {pr.relation.value} {pr.target}",
                        f"Therefore {concept} probably {pr.relation.value} {pr.target}",
                    ],
                    assumptions=[f"Properties of {parent} apply to {concept}"],
                )
        return None

    def _try_causal(self, concept: str, question: str) -> Hypothesis | None:
        """Chain causal relations to form hypothesis."""
        # What does this concept cause?
        effects = self.relations.get_forward(concept, RelationType.CAUSES)
        if effects:
            # Chain one more level
            for eff in effects[:2]:
                deeper = self.relations.get_forward(eff.target, RelationType.CAUSES)
                if deeper:
                    for d in deeper[:1]:
                        return Hypothesis(
                            claim=f"{concept} indirectly causes {d.target}",
                            confidence=eff.confidence * d.confidence * 0.7,
                            supporting_facts=[
                                f"{concept} causes {eff.target}",
                                f"{eff.target} causes {d.target}",
                            ],
                            chain=[
                                f"{concept} causes {eff.target}",
                                f"{eff.target} causes {d.target}",
                                f"Therefore {concept} indirectly causes {d.target}",
                            ],
                            assumptions=["Causal chains are transitive"],
                        )

        # What causes this concept?
        causes = self.relations.get_backward(concept, RelationType.CAUSES)
        if causes:
            for c in causes[:2]:
                deeper = self.relations.get_backward(c.source, RelationType.CAUSES)
                if deeper:
                    for d in deeper[:1]:
                        return Hypothesis(
                            claim=f"{d.source} indirectly causes {concept}",
                            confidence=c.confidence * d.confidence * 0.7,
                            supporting_facts=[
                                f"{d.source} causes {c.source}",
                                f"{c.source} causes {concept}",
                            ],
                            chain=[
                                f"{d.source} causes {c.source}",
                                f"{c.source} causes {concept}",
                                f"Therefore {d.source} indirectly causes {concept}",
                            ],
                            assumptions=["Causal chains are transitive"],
                        )
        return None

    def _try_part_whole(self, concept: str, question: str) -> Hypothesis | None:
        """If X is part_of Y, what does Y tell us about X?"""
        wholes = self.relations.get_forward(concept, RelationType.PART_OF)
        for whole_rel in wholes[:2]:
            whole = whole_rel.target
            whole_rels = self.relations.get_forward(whole)
            for wr in whole_rels[:2]:
                if wr.relation != RelationType.PART_OF:  # Don't recurse part_of
                    return Hypothesis(
                        claim=f"{concept} is connected to {wr.target} through {whole}",
                        confidence=0.3 * whole_rel.confidence,
                        supporting_facts=[
                            f"{concept} is part of {whole}",
                            f"{whole} {wr.relation.value} {wr.target}",
                        ],
                        chain=[
                            f"{concept} is part of {whole}",
                            f"{whole} {wr.relation.value} {wr.target}",
                            f"As part of {whole}, {concept} is connected to {wr.target}",
                        ],
                        assumptions=[f"Parts of {whole} share its relationships"],
                    )
        return None
