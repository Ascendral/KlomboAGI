"""
Unified Reasoning Loop -- orchestrates the Seven Pillars.

Given a problem, this loop:
1. DECOMPOSES it into parts
2. COMPARES to known concepts
3. ABSTRACTS patterns from similar past problems
4. TRANSFERS knowledge from related domains
5. INQUIRES about what's missing
6. Builds a CAUSAL MODEL if relevant
7. SELF-EVALUATES the answer before returning it

This is the full reasoning cycle. Not a pipeline that passes
data through -- each step can loop back to previous steps
if the evaluation fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.reasoning.core_reasoner import CoreReasoner, Rel
from klomboagi.reasoning.seven_pillars import (
    decompose, compare, abstract, transfer, inquire,
    build_causal_model, predict_effects, self_evaluate,
    Decomposition, Comparison, Abstraction, Transfer as TransferResult,
    KnowledgeGap, CausalChain, Evaluation,
)


@dataclass
class ReasoningTrace:
    """Full trace of a reasoning cycle."""
    problem: str
    decomposition: Decomposition | None = None
    comparisons: list[Comparison] = field(default_factory=list)
    abstraction: Abstraction | None = None
    transfers: list[TransferResult] = field(default_factory=list)
    gaps: list[KnowledgeGap] = field(default_factory=list)
    causal_chain: CausalChain | None = None
    evaluation: Evaluation | None = None
    answer: str = ""
    confidence: float = 0.0
    iterations: int = 0
    honest: bool = True  # True = we're confident, False = we're guessing


class ReasoningLoop:
    """Orchestrates the Seven Pillars into a complete reasoning cycle."""

    def __init__(self, reasoner: CoreReasoner, max_iterations: int = 3):
        self.reasoner = reasoner
        self.max_iterations = max_iterations

    def think(self, problem: str) -> ReasoningTrace:
        """Think about a problem using all seven pillars.

        Returns a full trace of the reasoning process.
        """
        trace = ReasoningTrace(problem=problem)

        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1

            # 1. DECOMPOSE -- what are we working with?
            trace.decomposition = decompose(self.reasoner, problem)

            # 2. COMPARE -- do any parts relate to things we know?
            known_parts = [p for p in trace.decomposition.parts
                          if p not in trace.decomposition.unknowns]
            if len(known_parts) >= 2:
                trace.comparisons = []
                for i, a in enumerate(known_parts[:4]):
                    for b in known_parts[i+1:4]:
                        comp = compare(self.reasoner, a, b)
                        if comp.similarity > 0:
                            trace.comparisons.append(comp)

            # 3. ABSTRACT -- if we have multiple known things, find the pattern
            if len(known_parts) >= 2:
                trace.abstraction = abstract(self.reasoner, known_parts[:5])

            # 4. TRANSFER -- if we know about some parts but not others,
            #    transfer knowledge from known to unknown
            if known_parts and trace.decomposition.unknowns:
                trace.transfers = []
                for unknown in trace.decomposition.unknowns[:2]:
                    best_known = max(known_parts,
                                     key=lambda k: compare(self.reasoner, k, unknown).similarity)
                    t = transfer(self.reasoner, best_known, unknown)
                    if t.confidence > 0:
                        trace.transfers.append(t)

            # 5. INQUIRY -- what don't we know that we need?
            trace.gaps = inquire(self.reasoner)

            # 6. CAUSAL MODEL -- if the problem is about why/how
            problem_lower = problem.lower()
            if any(w in problem_lower for w in ("why", "cause", "effect", "happen", "because")):
                # Extract the effect from the problem
                for part in trace.decomposition.parts:
                    chain = build_causal_model(self.reasoner, part)
                    if chain:
                        trace.causal_chain = chain
                        break

            # 7. Generate answer from what we've gathered
            answer = self._synthesize_answer(trace)
            trace.answer = answer

            # 8. SELF-EVALUATE -- is our answer good?
            if answer and "don't know" not in answer.lower():
                trace.evaluation = self._evaluate_answer(trace)
                trace.confidence = trace.evaluation.confidence

                if trace.evaluation.is_supported:
                    trace.honest = True
                    return trace  # Good answer, return it

                # Answer not well-supported -- try again with more knowledge
                if iteration < self.max_iterations - 1:
                    # Try to fill the most important gap
                    if trace.gaps:
                        # The gap-filling would happen here if we had search capability
                        pass
                    continue  # Try another iteration
            else:
                trace.honest = False
                trace.confidence = 0.0

        return trace

    def _synthesize_answer(self, trace: ReasoningTrace) -> str:
        """Build an answer from the reasoning trace."""
        parts = []

        # Direct knowledge
        d = trace.decomposition
        if d:
            for part in d.parts:
                if part in d.unknowns:
                    continue
                result = self.reasoner.ask(part, Rel.IS_A, "?")
                if result.known:
                    parts.append(result.answer)
                    break

        # Causal chain
        if trace.causal_chain:
            chain = trace.causal_chain
            chain_str = " -> ".join(f"{c}" for c, _ in chain.chain)
            parts.append(f"Causal chain: {chain_str} -> {chain.final_effect}")

        # Comparisons
        if trace.comparisons:
            best = max(trace.comparisons, key=lambda c: c.similarity)
            if best.similarity > 0.3:
                parts.append(f"{best.a} and {best.b} are {best.similarity:.0%} similar "
                           f"(shared: {', '.join(best.shared_properties[:3])})")

        # Abstraction
        if trace.abstraction and trace.abstraction.confidence > 0:
            a = trace.abstraction
            parts.append(f"Pattern '{a.pattern_name}': "
                        f"common properties = {', '.join(a.common_properties[:3])}")

        # Transfers
        for t in trace.transfers:
            if t.confidence > 0:
                parts.append(f"By analogy with {t.source}, {t.target} may have: "
                           f"{', '.join(t.transferred_properties[:3])}")

        if not parts:
            return "I don't know enough to answer this."

        return "\n".join(parts)

    def _evaluate_answer(self, trace: ReasoningTrace) -> Evaluation:
        """Evaluate the answer by checking each claim."""
        # Extract the first factual claim from the answer
        answer = trace.answer
        if not answer:
            return Evaluation("", False, [], [], 0.0, ["No answer"], [])

        # Try to evaluate the first line as a claim
        first_line = answer.split("\n")[0]

        # If it's a direct reasoner answer, check it
        if "is_a:" in first_line:
            # Parse "X is_a: Y, Z"
            import re
            m = re.match(r'(\w+)\s+is_a:\s*(\w+)', first_line)
            if m:
                return self_evaluate(self.reasoner, f"{m.group(1)} is a {m.group(2)}")

        # If it's a causal claim
        if "Causal chain:" in first_line:
            if trace.causal_chain:
                return Evaluation(
                    first_line, True,
                    [f"Chain traced with {trace.causal_chain.confidence:.0%} confidence"],
                    [], trace.causal_chain.confidence, [], [])

        # Default: check if we have any supporting knowledge
        for part in (trace.decomposition.parts if trace.decomposition else []):
            result = self.reasoner.ask(part, Rel.IS_A, "?")
            if result.known:
                return Evaluation(first_line, True, [result.answer],
                                 [], result.confidence, [], [])

        return Evaluation(first_line, False, [], [], 0.0,
                         ["Could not verify"], [])
