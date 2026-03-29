"""
Causal Simulation — run mental simulations forward in time.

"If I heat water, what happens?"
  Step 1: heat → temperature increases
  Step 2: temperature increases → molecules move faster
  Step 3: at 100°C → water boils
  Step 4: boiling → water becomes steam
  Step 5: steam → volume expands

The system runs the causal graph FORWARD, simulating
what would happen step by step. Each step has confidence.
The simulation stops when confidence drops too low or
no more causal links exist.

This is mental rehearsal — imagining the future without acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class SimulationStep:
    """One step in a causal simulation."""
    event: str
    caused_by: str
    confidence: float
    step_number: int


@dataclass
class Simulation:
    """A forward causal simulation."""
    trigger: str                    # what starts the chain
    steps: list[SimulationStep] = field(default_factory=list)
    final_state: str = ""
    total_confidence: float = 1.0

    def explain(self) -> str:
        lines = [f"If {self.trigger}:"]
        for step in self.steps:
            indent = "  " * min(step.step_number, 4)
            conf = f" ({step.confidence:.0%})" if step.confidence < 0.9 else ""
            lines.append(f"  {indent}→ {step.event}{conf}")
        if self.final_state:
            lines.append(f"\n  Final outcome: {self.final_state}")
        lines.append(f"  Overall confidence: {self.total_confidence:.0%}")
        return "\n".join(lines)


class CausalSimulator:
    """
    Simulates causal chains forward from a trigger event.

    Given "heat water", traces: heat → temperature → molecular motion
    → phase change → steam → expansion.
    """

    MIN_CONFIDENCE = 0.1
    MAX_STEPS = 8

    def __init__(self, relations: RelationStore, beliefs: dict = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}

    def simulate(self, trigger: str) -> Simulation:
        """Run a forward causal simulation from a trigger."""
        sim = Simulation(trigger=trigger)
        visited = {trigger.lower()}
        current = trigger.lower()
        confidence = 1.0

        for step_num in range(self.MAX_STEPS):
            # Find what this causes
            effects = self.relations.get_forward(current, RelationType.CAUSES)

            # Also check what it enables
            if not effects:
                effects = self.relations.get_forward(current, RelationType.ENABLES)

            if not effects:
                break

            for effect in effects:
                if effect.target.lower() in visited:
                    continue
                visited.add(effect.target.lower())

                step_conf = confidence * effect.confidence
                if step_conf < self.MIN_CONFIDENCE:
                    continue

                sim.steps.append(SimulationStep(
                    event=effect.target,
                    caused_by=current,
                    confidence=round(step_conf, 3),
                    step_number=step_num + 1,
                ))

                # Continue chain from the first effect
                current = effect.target.lower()
                confidence = step_conf
                break
            else:
                break  # No unvisited effects

        if sim.steps:
            sim.final_state = sim.steps[-1].event
            sim.total_confidence = sim.steps[-1].confidence

        return sim

    def simulate_multiple(self, trigger: str) -> list[Simulation]:
        """
        Run multiple branching simulations — follow ALL causal paths.
        """
        sims = []
        effects = self.relations.get_forward(trigger.lower(), RelationType.CAUSES)

        if not effects:
            # Single simulation with no branches
            return [self.simulate(trigger)]

        for effect in effects[:4]:  # Max 4 branches
            branch_sim = Simulation(trigger=trigger)
            self._trace_branch(effect.target, trigger, effect.confidence,
                             1, set(), branch_sim)
            if branch_sim.steps:
                branch_sim.final_state = branch_sim.steps[-1].event
                branch_sim.total_confidence = branch_sim.steps[-1].confidence
                sims.append(branch_sim)

        return sims if sims else [self.simulate(trigger)]

    def _trace_branch(self, current: str, caused_by: str, confidence: float,
                      depth: int, visited: set, sim: Simulation) -> None:
        """Trace one branch of a causal simulation."""
        if depth > self.MAX_STEPS or confidence < self.MIN_CONFIDENCE:
            return
        if current.lower() in visited:
            return
        visited.add(current.lower())

        sim.steps.append(SimulationStep(
            event=current, caused_by=caused_by,
            confidence=round(confidence, 3), step_number=depth,
        ))

        effects = self.relations.get_forward(current.lower(), RelationType.CAUSES)
        for effect in effects[:1]:  # Follow strongest path
            if effect.target.lower() not in visited:
                self._trace_branch(
                    effect.target, current,
                    confidence * effect.confidence,
                    depth + 1, visited, sim)


class ConfidenceCalibrator:
    """
    Track whether confidence scores are CALIBRATED.

    If the system says it's 70% confident, is it right 70% of the time?
    Tracks predictions vs outcomes and adjusts confidence scaling.
    """

    def __init__(self) -> None:
        self._predictions: list[tuple[float, bool]] = []  # (confidence, was_correct)
        self._calibration_offset: float = 0.0  # adjustment factor

    def record(self, confidence: float, was_correct: bool) -> None:
        """Record a prediction and whether it was correct."""
        self._predictions.append((confidence, was_correct))
        if len(self._predictions) > 500:
            self._predictions = self._predictions[-500:]
        self._recalibrate()

    def calibrate(self, raw_confidence: float) -> float:
        """Adjust a raw confidence score based on calibration history."""
        adjusted = raw_confidence + self._calibration_offset
        return max(0.01, min(0.99, adjusted))

    def _recalibrate(self) -> None:
        """Compute calibration offset from prediction history."""
        if len(self._predictions) < 10:
            return

        # Bucket predictions by confidence level
        buckets: dict[int, list[bool]] = {}
        for conf, correct in self._predictions:
            bucket = int(conf * 10)  # 0-9
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(correct)

        # Compare predicted confidence to actual accuracy
        offsets = []
        for bucket, outcomes in buckets.items():
            if len(outcomes) < 3:
                continue
            predicted = bucket / 10.0 + 0.05  # center of bucket
            actual = sum(outcomes) / len(outcomes)
            offsets.append(actual - predicted)

        if offsets:
            self._calibration_offset = sum(offsets) / len(offsets)

    def report(self) -> str:
        """Calibration report."""
        if not self._predictions:
            return "No predictions recorded yet."

        total = len(self._predictions)
        correct = sum(1 for _, c in self._predictions if c)
        accuracy = correct / total

        lines = [
            f"Confidence Calibration ({total} predictions):",
            f"  Overall accuracy: {accuracy:.0%}",
            f"  Calibration offset: {self._calibration_offset:+.2f}",
        ]

        # Per-bucket analysis
        buckets: dict[int, list[bool]] = {}
        for conf, correct_val in self._predictions:
            bucket = int(conf * 10)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(correct_val)

        for bucket in sorted(buckets.keys()):
            outcomes = buckets[bucket]
            if len(outcomes) >= 3:
                predicted = f"{bucket*10}-{bucket*10+9}%"
                actual = sum(outcomes) / len(outcomes)
                lines.append(f"  {predicted:>8s} predicted → {actual:.0%} actual ({len(outcomes)} samples)")

        return "\n".join(lines)


class SemanticSimilarity:
    """
    Find related concepts without exact keyword match.

    "car" and "automobile" should be recognized as the same thing.
    "happy" and "joyful" should be connected.

    Method: concepts that share the same predicates or appear in
    similar relation structures are semantically similar.
    """

    def __init__(self, beliefs: dict, relations: RelationStore) -> None:
        self.beliefs = beliefs
        self.relations = relations

    def similar_to(self, concept: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Find concepts semantically similar to the given one.

        Similarity = shared predicates + shared relation structure.
        """
        concept_lower = concept.lower()
        scores: dict[str, float] = {}

        # Get concept's predicates
        my_predicates = set()
        for stmt, belief in self.beliefs.items():
            if hasattr(belief, 'subject') and belief.subject == concept_lower:
                if belief.predicate:
                    my_predicates.add(belief.predicate.lower())

        # Get concept's relation targets
        my_targets = set()
        my_rel_types = set()
        for rel in self.relations.get_forward(concept_lower):
            my_targets.add(rel.target.lower())
            my_rel_types.add(rel.relation.value)
        for rel in self.relations.get_backward(concept_lower):
            my_targets.add(rel.source.lower())
            my_rel_types.add(rel.relation.value)

        # Score every other concept by similarity
        checked = set()
        for stmt, belief in self.beliefs.items():
            if not hasattr(belief, 'subject') or not belief.subject:
                continue
            other = belief.subject.lower()
            if other == concept_lower or other in checked:
                continue
            checked.add(other)

            score = 0.0

            # Shared predicates
            other_predicates = set()
            for s2, b2 in self.beliefs.items():
                if hasattr(b2, 'subject') and b2.subject == other and b2.predicate:
                    other_predicates.add(b2.predicate.lower())

            shared_preds = my_predicates & other_predicates
            if shared_preds:
                score += len(shared_preds) * 0.3

            # Same IS_A parent (siblings)
            if my_predicates & other_predicates:
                score += 0.2

            # Shared relation structure
            other_rel_types = set()
            for rel in self.relations.get_forward(other):
                other_rel_types.add(rel.relation.value)
            shared_rels = my_rel_types & other_rel_types
            if shared_rels:
                score += len(shared_rels) * 0.2

            # Shared relation targets
            other_targets = set()
            for rel in self.relations.get_forward(other):
                other_targets.add(rel.target.lower())
            shared_targets = my_targets & other_targets
            if shared_targets:
                score += len(shared_targets) * 0.15

            if score > 0.1:
                scores[other] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def are_similar(self, a: str, b: str) -> float:
        """How similar are two concepts? Returns 0-1."""
        similar = self.similar_to(a, top_n=50)
        for concept, score in similar:
            if concept == b.lower():
                return min(1.0, score)
        return 0.0
