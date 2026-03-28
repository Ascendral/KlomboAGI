"""
Counterfactual Reasoning — "what if X were different?"

Real intelligence reasons about hypotheticals:
  "What if gravity were twice as strong?"
  "What would happen without friction?"
  "If energy couldn't be created or destroyed, what follows?"

The method:
1. Parse the counterfactual: what's being changed?
2. Find all downstream effects through the relation graph
3. Propagate the change through causal chains
4. Report what would be different

This uses the relation graph as a causal model:
  if X causes Y and we remove X → Y also goes away
  if X causes Y and we double X → Y increases
  if X requires Y and we remove Y → X also goes away
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class CounterfactualEffect:
    """One downstream effect of a counterfactual change."""
    concept: str
    effect: str          # "removed", "increased", "decreased", "unchanged", "unknown"
    reason: str          # why this effect happens
    chain_length: int    # how many hops from the original change
    confidence: float    # decreases with chain length


@dataclass
class CounterfactualResult:
    """Full analysis of a 'what if' scenario."""
    scenario: str
    changed_concept: str
    change_type: str              # "removed", "doubled", "halved", "reversed", "absent"
    effects: list[CounterfactualEffect]
    total_affected: int

    def explain(self) -> str:
        lines = [f"What if: {self.scenario}"]
        lines.append(f"Change: {self.changed_concept} → {self.change_type}")
        lines.append(f"Affected concepts: {self.total_affected}")

        if self.effects:
            lines.append(f"\nDownstream effects:")
            for eff in self.effects:
                indent = "  " * min(eff.chain_length, 4)
                conf = f"({eff.confidence:.0%})" if eff.confidence < 1.0 else ""
                lines.append(f"  {indent}{eff.concept}: {eff.effect} {conf}")
                lines.append(f"  {indent}  because: {eff.reason}")
        else:
            lines.append("\nNo downstream effects found.")

        return "\n".join(lines)


class CounterfactualEngine:
    """
    Reasons about hypothetical changes to the world.

    Uses the relation graph to propagate effects:
    - CAUSES: if cause removed → effect removed
    - REQUIRES: if requirement removed → dependent removed
    - ENABLES: if enabler removed → enabled thing can't happen
    - PART_OF: if whole changes → parts change too
    """

    def __init__(self, relations: RelationStore) -> None:
        self.relations = relations

    def what_if(self, scenario: str) -> CounterfactualResult:
        """
        Analyze a 'what if' scenario.

        Parses natural language: "what if there were no gravity?"
        Propagates through causal/dependency chains.
        """
        parsed = self._parse_scenario(scenario)
        if not parsed:
            return CounterfactualResult(
                scenario=scenario, changed_concept="unknown",
                change_type="unknown", effects=[], total_affected=0,
            )

        concept, change_type = parsed
        effects = self._propagate(concept, change_type)

        return CounterfactualResult(
            scenario=scenario,
            changed_concept=concept,
            change_type=change_type,
            effects=effects,
            total_affected=len(effects),
        )

    def _parse_scenario(self, scenario: str) -> tuple[str, str] | None:
        """Parse 'what if X?' into (concept, change_type)."""
        s = scenario.lower().strip().rstrip("?")

        patterns = [
            # Removal: "no X", "without X", "X didn't exist", "X were removed"
            (r"what if there (?:were|was) no (.+)", "removed"),
            (r"what if (.+?) (?:didn't|did not) exist", "removed"),
            (r"what if (.+?) (?:were|was) removed", "removed"),
            (r"what (?:would happen )?without (.+)", "removed"),
            (r"without (.+?) what", "removed"),
            (r"if there (?:were|was) no (.+)", "removed"),

            # Increase: "X doubled", "X were twice as strong"
            (r"what if (.+?) (?:doubled|were twice)", "doubled"),
            (r"what if (.+?) (?:increased|were stronger)", "increased"),

            # Decrease: "X halved", "X were weaker"
            (r"what if (.+?) (?:halved|were half)", "halved"),
            (r"what if (.+?) (?:decreased|were weaker)", "decreased"),

            # Reversal: "X were reversed", "X were opposite"
            (r"what if (.+?) (?:were|was) reversed", "reversed"),
            (r"what if (.+?) (?:were|was) the opposite", "reversed"),
        ]

        for pattern, change_type in patterns:
            m = re.match(pattern, s)
            if m:
                concept = m.group(1).strip()
                return (concept, change_type)

        return None

    def _propagate(self, concept: str, change_type: str,
                   max_depth: int = 4) -> list[CounterfactualEffect]:
        """Propagate a change through the relation graph."""
        effects: list[CounterfactualEffect] = []
        visited: set[str] = {concept}

        # BFS propagation
        queue: list[tuple[str, str, str, int, float]] = []
        # (affected_concept, effect, reason, depth, confidence)

        # Direct effects from forward relations
        for rel in self.relations.get_forward(concept):
            effect = self._determine_effect(rel.relation, change_type)
            if effect != "unchanged":
                queue.append((
                    rel.target, effect,
                    f"{concept} {rel.relation.value} {rel.target}",
                    1, rel.confidence,
                ))

        # Direct effects from backward relations (things that depend on concept)
        for rel in self.relations.get_backward(concept):
            if rel.relation in (RelationType.REQUIRES, RelationType.USES):
                # If something requires/uses the changed concept
                effect = "disrupted" if change_type == "removed" else "affected"
                queue.append((
                    rel.source, effect,
                    f"{rel.source} {rel.relation.value} {concept}",
                    1, rel.confidence * 0.8,
                ))
            elif rel.relation == RelationType.PART_OF:
                # Part of the changed concept → also affected
                queue.append((
                    rel.source, change_type,
                    f"{rel.source} is part of {concept}",
                    1, rel.confidence * 0.7,
                ))

        while queue:
            affected, effect, reason, depth, confidence = queue.pop(0)
            if depth > max_depth or affected in visited:
                continue
            visited.add(affected)

            effects.append(CounterfactualEffect(
                concept=affected,
                effect=effect,
                reason=reason,
                chain_length=depth,
                confidence=round(confidence, 3),
            ))

            # Continue propagating — effects cascade
            if effect in ("removed", "disrupted"):
                for rel in self.relations.get_forward(affected):
                    if rel.target not in visited:
                        child_effect = self._determine_effect(rel.relation, "removed")
                        if child_effect != "unchanged":
                            queue.append((
                                rel.target, child_effect,
                                f"{affected} {rel.relation.value} {rel.target} (cascading from {concept})",
                                depth + 1, confidence * 0.7,
                            ))

        # Sort by chain length then confidence
        effects.sort(key=lambda e: (e.chain_length, -e.confidence))
        return effects

    def _determine_effect(self, relation: RelationType, change_type: str) -> str:
        """Given a relation type and change, what happens to the target?"""
        if change_type == "removed":
            effects = {
                RelationType.CAUSES: "removed",       # no cause → no effect
                RelationType.ENABLES: "disabled",      # no enabler → can't happen
                RelationType.REQUIRES: "unchanged",    # requirement removal doesn't cascade forward
                RelationType.USES: "unchanged",        # tool removal doesn't destroy the user
                RelationType.PART_OF: "incomplete",    # missing part → whole is incomplete
                RelationType.MEASURES: "unmeasurable",
            }
        elif change_type in ("doubled", "increased"):
            effects = {
                RelationType.CAUSES: "increased",
                RelationType.ENABLES: "enhanced",
                RelationType.REQUIRES: "unchanged",
                RelationType.USES: "unchanged",
                RelationType.PART_OF: "increased",
                RelationType.MEASURES: "higher_reading",
            }
        elif change_type in ("halved", "decreased"):
            effects = {
                RelationType.CAUSES: "decreased",
                RelationType.ENABLES: "weakened",
                RelationType.PART_OF: "decreased",
                RelationType.MEASURES: "lower_reading",
            }
        elif change_type == "reversed":
            effects = {
                RelationType.CAUSES: "reversed",
                RelationType.ENABLES: "inverted",
                RelationType.PART_OF: "reversed",
            }
        else:
            return "unknown"

        return effects.get(relation, "unchanged")
