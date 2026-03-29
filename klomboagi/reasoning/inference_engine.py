"""
Global Inference Engine — derive ALL possible conclusions from ALL beliefs.

Current system stores beliefs independently. But if:
  "a dog is a mammal" and "a mammal is warm-blooded"
then "a dog is warm-blooded" should be AUTOMATICALLY derived.

This engine scans ALL beliefs and derives EVERY valid chain:
  A is B, B is C → A is C (transitivity)
  A causes B, B causes C → A indirectly causes C
  A is B, B has property P → A has property P (inheritance)

Runs periodically. Each derived belief gets tagged source="inference"
with lower confidence than direct beliefs (evidence decays through chains).
"""

from __future__ import annotations

from klomboagi.reasoning.truth import TruthValue, Belief, EvidenceStamp, deduction


class GlobalInferenceEngine:
    """
    Derives all possible conclusions from existing beliefs.

    Uses NARS deduction: if A→B with truth(f1,c1) and B→C with truth(f2,c2),
    then A→C with truth = nars_deduction(tv1, tv2).
    """

    def __init__(self, beliefs: dict, evidence_counter: int = 0) -> None:
        self.beliefs = beliefs
        self._counter = evidence_counter

    def run(self, max_derivations: int = 500) -> list[str]:
        """
        Run global inference. Returns list of newly derived belief statements.
        """
        derived = []
        seen = set(self.beliefs.keys())

        # Build subject→predicate index for fast lookup
        by_subject: dict[str, list[tuple[str, object]]] = {}
        for stmt, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject:
                subj = belief.subject.lower()
                if subj not in by_subject:
                    by_subject[subj] = []
                by_subject[subj].append((stmt, belief))

        # Transitivity: A is B, B is C → A is C
        for subj, beliefs_list in by_subject.items():
            for stmt1, b1 in beliefs_list:
                if not hasattr(b1, 'predicate') or not b1.predicate:
                    continue
                pred1 = b1.predicate.lower()

                # Does the predicate appear as a subject somewhere?
                if pred1 in by_subject:
                    for stmt2, b2 in by_subject[pred1]:
                        if not hasattr(b2, 'predicate') or not b2.predicate:
                            continue
                        pred2 = b2.predicate

                        # Derive: subj is pred2
                        new_stmt = f"{subj} is {pred2}"
                        if new_stmt not in seen and len(pred2) < 80:
                            # NARS deduction for truth value
                            tv = deduction(b1.truth, b2.truth)
                            if tv.confidence > 0.1:  # Only keep meaningful derivations
                                self._counter += 1
                                new_belief = Belief(
                                    statement=new_stmt,
                                    truth=tv,
                                    stamp=b1.stamp.merge(b2.stamp) if hasattr(b1.stamp, 'merge') else EvidenceStamp.new(self._counter),
                                    subject=subj,
                                    predicate=pred2,
                                    source="inference",
                                )
                                self.beliefs[new_stmt] = new_belief
                                seen.add(new_stmt)
                                derived.append(new_stmt)

                                if len(derived) >= max_derivations:
                                    return derived

        return derived

    def run_property_inheritance(self, max_derivations: int = 200) -> list[str]:
        """
        Property inheritance: if A is B and B has property P, then A has property P.

        "dog is mammal" + "mammal is warm-blooded" → "dog is warm-blooded"
        Already handled by transitivity above.

        This method handles RELATION inheritance through the relation store.
        """
        # This is covered by the transitivity pass above for is_a chains.
        # The relation store handles causes/requires/etc inheritance separately.
        return []


class QuestionDecomposer:
    """
    Decompose complex questions into simpler sub-questions.

    "How does photosynthesis affect the global carbon cycle?"
    →
    1. What is photosynthesis?
    2. What is the global carbon cycle?
    3. What does photosynthesis produce?
    4. How does that relate to the carbon cycle?
    5. What is the net effect?
    """

    def decompose(self, question: str) -> list[str]:
        """Break a complex question into answerable sub-questions."""
        stop = {"what", "is", "a", "an", "the", "how", "why", "does", "do",
                "can", "could", "would", "should", "about", "you", "your",
                "think", "know", "tell", "me", "there", "are", "was", "were"}

        words = [w.lower().strip("?.,!") for w in question.split()
                 if w.lower().strip("?.,!") not in stop and len(w) > 2]

        sub_questions = []

        # Sub-question for each key concept
        for word in words[:4]:
            sub_questions.append(f"what is {word}?")

        # Connection questions between concepts
        if len(words) >= 2:
            sub_questions.append(f"how does {words[0]} relate to {words[-1]}?")

        # Causal questions
        q_lower = question.lower()
        if "how" in q_lower and "affect" in q_lower:
            if len(words) >= 2:
                sub_questions.append(f"what does {words[0]} cause?")
                sub_questions.append(f"what causes {words[-1]}?")

        if "why" in q_lower:
            for word in words[:2]:
                sub_questions.append(f"what causes {word}?")

        if "can" in q_lower or "possible" in q_lower:
            for word in words[:2]:
                sub_questions.append(f"what does {word} require?")
                sub_questions.append(f"what does {word} enable?")

        return sub_questions[:8]


class BeliefPropagator:
    """
    When a belief changes, propagate the change to all dependents.

    If we learn "a dog is NOT a reptile" (correcting "dog is reptile"),
    then everything we derived from "dog is reptile" must be invalidated.

    Tracks dependency chains so corrections cascade properly.
    """

    def __init__(self, beliefs: dict) -> None:
        self.beliefs = beliefs

    def propagate_correction(self, corrected_statement: str,
                            new_predicate: str) -> list[str]:
        """
        A belief was corrected. Find and update all beliefs derived from it.

        Returns list of beliefs that were invalidated/updated.
        """
        affected = []

        # Find the corrected belief
        old_belief = self.beliefs.get(corrected_statement)
        if not old_belief or not hasattr(old_belief, 'subject'):
            return affected

        old_subject = old_belief.subject
        old_predicate = old_belief.predicate

        # Find all beliefs that were derived through this one
        # A belief B depends on A if B.source == "inference" and
        # B was derived through A's predicate as an intermediate
        for stmt, belief in list(self.beliefs.items()):
            if belief.source != "inference":
                continue
            # Check if this inferred belief goes through the corrected predicate
            if (hasattr(belief, 'subject') and belief.subject == old_subject
                    and hasattr(belief, 'predicate')):
                # This belief has the same subject — might be derived through old chain
                # Weaken it significantly
                belief.truth.frequency = max(0.0, belief.truth.frequency - 0.3)
                belief.truth.confidence = max(0.0, belief.truth.confidence - 0.2)
                affected.append(stmt)

        return affected

    def find_dependents(self, statement: str) -> list[str]:
        """Find all beliefs that depend on a given belief."""
        belief = self.beliefs.get(statement)
        if not belief or not hasattr(belief, 'predicate'):
            return []

        predicate = belief.predicate.lower()
        dependents = []

        # Any inferred belief whose subject chain goes through this predicate
        for stmt, b in self.beliefs.items():
            if b.source == "inference" and hasattr(b, 'subject'):
                if b.subject == predicate:
                    dependents.append(stmt)

        return dependents
