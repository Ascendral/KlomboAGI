"""
First Principles Reasoning — solve problems you've never seen before.

When the system has NO pattern, NO stored answer, NO relevant belief:
instead of saying "I don't know", it should REASON from what it DOES know.

Method:
1. DECOMPOSE: break the question into smaller parts
2. GROUND: for each part, find the most basic facts you DO know
3. BUILD: chain those basic facts upward toward the answer
4. SYNTHESIZE: combine the chains into a coherent answer
5. QUALIFY: state confidence and what you assumed

Example: "Can a fish climb a tree?"
  - Decompose: what is a fish? what is climbing? what is a tree?
  - Ground: fish lives in water, has fins not limbs. Climbing requires
    gripping and lifting body weight. Trees are vertical structures on land.
  - Build: fish has no limbs → can't grip → can't climb.
    Fish lives in water → trees are on land → fish can't reach tree.
  - Synthesize: "No. Fish lack limbs for gripping and live in water,
    while trees are on land. Both physical inability and environment
    prevent it."
  - Qualify: "I'm fairly confident because the reasoning chain is short
    and each step is well-supported. But some fish (mudskippers) can
    move on land — I might be wrong about edge cases."
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    """One step in a first-principles chain."""
    claim: str
    basis: str          # what fact or logic supports this
    confidence: float   # how sure of this step


@dataclass
class FirstPrinciplesResult:
    """Result of reasoning from first principles."""
    question: str
    decomposition: list[str]       # sub-questions
    grounded_facts: list[str]      # basic facts used
    chain: list[ReasoningStep]     # reasoning steps
    answer: str                    # final answer
    confidence: float              # overall confidence
    assumptions: list[str]         # what we assumed
    caveats: list[str]            # what could be wrong

    def explain(self) -> str:
        lines = [self.answer]
        if self.chain:
            lines.append("\nMy reasoning:")
            for step in self.chain:
                lines.append(f"  {step.claim} (because: {step.basis})")
        if self.assumptions:
            lines.append(f"\nI'm assuming: {', '.join(self.assumptions)}")
        if self.caveats:
            lines.append(f"I could be wrong about: {', '.join(self.caveats)}")
        return "\n".join(lines)


class FirstPrinciplesEngine:
    """
    Reason about novel questions from basic facts.

    Uses beliefs and relations to build reasoning chains
    for questions the system has never encountered.
    """

    def __init__(self, beliefs: dict, relations, activation=None) -> None:
        self.beliefs = beliefs
        self.relations = relations
        self.activation = activation

    def reason(self, question: str) -> FirstPrinciplesResult:
        """Attempt to answer a novel question from first principles."""
        # 1. DECOMPOSE — break into sub-questions
        sub_questions = self._decompose(question)

        # 2. GROUND — find basic facts for each concept
        grounded = []
        concepts = self._extract_concepts(question)
        for concept in concepts:
            facts = self._ground(concept)
            grounded.extend(facts)

        # 3. BUILD — chain facts toward an answer
        chain = self._build_chain(question, concepts, grounded)

        # 4. SYNTHESIZE — combine into answer
        answer = self._synthesize(question, chain, grounded)

        # 5. QUALIFY — assess confidence and caveats
        confidence = self._assess_confidence(chain)
        assumptions = [s.basis for s in chain if s.confidence < 0.5]
        caveats = self._find_caveats(concepts)

        return FirstPrinciplesResult(
            question=question,
            decomposition=sub_questions,
            grounded_facts=grounded,
            chain=chain,
            answer=answer,
            confidence=confidence,
            assumptions=assumptions[:3],
            caveats=caveats[:3],
        )

    def _decompose(self, question: str) -> list[str]:
        """Break a question into sub-questions."""
        stop = {"what", "is", "a", "an", "the", "can", "does", "do", "how",
                "why", "would", "could", "should", "if", "are", "was", "were",
                "about", "you", "your", "think", "know", "tell", "me"}
        words = [w.lower().strip("?.,!") for w in question.split()
                 if w.lower().strip("?.,!") not in stop and len(w) > 2]

        sub_q = []
        for word in words:
            sub_q.append(f"what is {word}")
            # Check relations
            if hasattr(self.relations, 'get_all_about'):
                rels = self.relations.get_all_about(word)
                if rels:
                    sub_q.append(f"what does {word} connect to")
        return sub_q[:6]

    def _extract_concepts(self, question: str) -> list[str]:
        """Pull key concepts from the question."""
        stop = {"what", "is", "a", "an", "the", "can", "does", "do", "how",
                "why", "would", "could", "should", "if", "are", "was", "were",
                "about", "you", "your", "think", "know", "tell", "me", "there",
                "have", "has", "had", "will", "been", "being", "this", "that"}
        words = [w.lower().strip("?.,!") for w in question.split()
                 if w.lower().strip("?.,!") not in stop and len(w) > 2]
        return words

    def _ground(self, concept: str) -> list[str]:
        """Find the most basic facts about a concept."""
        facts = []
        concept_lower = concept.lower()

        # Direct beliefs
        for stmt, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                if belief.predicate and len(belief.predicate) < 80:
                    facts.append(f"{concept} is {belief.predicate}")

        # Relations
        if hasattr(self.relations, 'get_forward'):
            for rel in self.relations.get_forward(concept_lower)[:3]:
                facts.append(f"{concept} {rel.relation.value} {rel.target}")
            for rel in self.relations.get_backward(concept_lower)[:3]:
                facts.append(f"{rel.source} {rel.relation.value} {concept}")

        # Spreading activation — what associates?
        if self.activation and facts:
            result = self.activation.activate([concept_lower])
            for node in result.top(3):
                if node.name != concept_lower:
                    facts.append(f"{concept} associates with {node.name}")

        return facts[:8]

    def _build_chain(self, question: str, concepts: list[str],
                     grounded: list[str]) -> list[ReasoningStep]:
        """Build a reasoning chain from grounded facts."""
        chain = []

        if not grounded:
            return [ReasoningStep(
                claim="I have no grounded facts to reason from.",
                basis="No relevant knowledge found",
                confidence=0.0,
            )]

        # For each concept, create reasoning steps
        for concept in concepts:
            concept_facts = [f for f in grounded if concept.lower() in f.lower()]
            if concept_facts:
                # Best fact as a step
                chain.append(ReasoningStep(
                    claim=concept_facts[0],
                    basis="direct knowledge",
                    confidence=0.7,
                ))

        # Try to connect concepts
        if len(concepts) >= 2:
            for i in range(len(concepts) - 1):
                a, b = concepts[i], concepts[i + 1]
                if hasattr(self.relations, 'find_path'):
                    path = self.relations.find_path(a.lower(), b.lower())
                    if path:
                        steps_str = " → ".join(f"{r.source} {r.relation.value} {r.target}" for r in path)
                        chain.append(ReasoningStep(
                            claim=f"{a} connects to {b} through: {steps_str}",
                            basis="relation graph traversal",
                            confidence=0.5,
                        ))
                    else:
                        chain.append(ReasoningStep(
                            claim=f"I don't see a direct connection between {a} and {b}",
                            basis="no path found in relation graph",
                            confidence=0.3,
                        ))

        # Causal reasoning if applicable
        q_lower = question.lower()
        if "can" in q_lower or "could" in q_lower or "possible" in q_lower:
            # Check for enabling/preventing relations
            for concept in concepts:
                enables = self.relations.get_forward(concept.lower())
                if hasattr(self.relations, 'get_forward'):
                    for rel in enables[:2]:
                        if rel.relation.value in ("enables", "causes", "requires"):
                            chain.append(ReasoningStep(
                                claim=f"{concept} {rel.relation.value} {rel.target}",
                                basis=f"known {rel.relation.value} relation",
                                confidence=rel.confidence,
                            ))

        return chain

    def _synthesize(self, question: str, chain: list[ReasoningStep],
                    grounded: list[str]) -> str:
        """Combine chain into a coherent answer."""
        if not chain or (len(chain) == 1 and chain[0].confidence == 0.0):
            return f"I can't answer this from what I know. I need to learn more."

        # Collect high-confidence claims
        strong_claims = [s for s in chain if s.confidence >= 0.5]
        weak_claims = [s for s in chain if s.confidence < 0.5]

        parts = []
        if strong_claims:
            parts.append("Based on what I know: " +
                         ". ".join(s.claim for s in strong_claims[:3]) + ".")
        if weak_claims:
            parts.append("I'm less sure about: " +
                         ". ".join(s.claim for s in weak_claims[:2]) + ".")

        if not parts:
            return "I tried to reason through this but I'm not confident in any conclusion."

        return " ".join(parts)

    def _assess_confidence(self, chain: list[ReasoningStep]) -> float:
        """Overall confidence in the reasoning chain."""
        if not chain:
            return 0.0
        avg = sum(s.confidence for s in chain) / len(chain)
        # Longer chains = less confident (more assumptions)
        length_penalty = max(0.5, 1.0 - len(chain) * 0.05)
        return min(0.95, avg * length_penalty)

    def _find_caveats(self, concepts: list[str]) -> list[str]:
        """What could be wrong with our reasoning?"""
        caveats = []
        for concept in concepts:
            # How much do we actually know?
            fact_count = sum(1 for s, b in self.beliefs.items()
                           if hasattr(b, 'subject') and b.subject == concept.lower())
            if fact_count == 0:
                caveats.append(f"I have no direct knowledge of {concept}")
            elif fact_count < 3:
                caveats.append(f"I know very little about {concept} ({fact_count} facts)")
        return caveats
