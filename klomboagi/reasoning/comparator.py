"""
Structural Comparator — the bridge between memory and thinking.

This is the component that turns KlomboAGI from a memory system
into a thinking system. It compares ANY two things structurally —
not by content similarity, but by the ROLES their parts play.

A bug fix in Python and a bug fix in Go are structurally identical:
  identify problem → locate source → modify → verify

A recipe and a deployment pipeline are structurally identical:
  gather ingredients → combine in order → apply transformation → verify result

The comparator doesn't know what Python, Go, recipes, or deployments are.
It sees: sequence of roles, dependencies between steps, transformations,
and outcomes. The STRUCTURE is what transfers.

This sits between:
- Raw episodes (what happened)
- Abstraction engine (what pattern is this)
- Causal model (what causes what)
- Transfer (apply pattern to new domain)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klomboagi.reasoning.abstraction import AbstractionEngine, StructuralElement


@dataclass
class ComparisonResult:
    """The result of comparing two structures."""
    similarity: float           # 0.0 to 1.0 — overall structural similarity
    shared_roles: list[str]     # Roles that appear in both
    unique_to_a: list[str]      # Roles only in A
    unique_to_b: list[str]      # Roles only in B
    alignments: list[dict]      # Detailed element-to-element mapping
    structural_type: str        # "identical", "analogous", "partial", "unrelated"
    transferable: list[str]     # What could transfer from A to B
    gaps: list[str]             # What B is missing that A has

    def to_dict(self) -> dict:
        return {
            "similarity": self.similarity,
            "shared_roles": self.shared_roles,
            "unique_to_a": self.unique_to_a,
            "unique_to_b": self.unique_to_b,
            "alignments": self.alignments,
            "structural_type": self.structural_type,
            "transferable": self.transferable,
            "gaps": self.gaps,
        }


@dataclass
class AnalogicalMapping:
    """A mapping between two different domains based on structural similarity."""
    source_domain: str
    target_domain: str
    mappings: list[dict]        # [{source_role: X, target_role: Y, confidence: Z}, ...]
    strength: float             # How strong is this analogy
    predictions: list[dict]     # What the analogy predicts about the target

    def to_dict(self) -> dict:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "mappings": self.mappings,
            "strength": self.strength,
            "predictions": self.predictions,
        }


class StructuralComparator:
    """
    Compares structures to find deep similarities and enable transfer.

    Three operations:
    1. compare() — find structural similarities between two things
    2. analogize() — map structure from one domain to another
    3. transfer() — apply knowledge from source to target via analogy

    This is how a child who knows how to stack blocks can learn
    to stack plates — the structure (stack, balance, order-matters)
    transfers even though the objects are completely different.
    """

    def __init__(self, abstraction_engine: AbstractionEngine) -> None:
        self.abstraction = abstraction_engine

    # ── Compare: find structural similarity ──

    def compare(self, a: dict, b: dict) -> ComparisonResult:
        """
        Compare two episodes/structures and determine how similar they are.

        Not content similarity (that's just string matching).
        STRUCTURAL similarity: do the parts play the same roles?
        """
        elements_a = self.abstraction.decompose(a)
        elements_b = self.abstraction.decompose(b)
        aligned = self.abstraction.align(elements_a, elements_b)

        # Classify each alignment
        shared_roles = []
        unique_to_a = []
        unique_to_b = []
        alignments = []

        for ea, eb in aligned:
            if ea is not None and eb is not None:
                shared_roles.append(ea.role)
                score = self._element_similarity(ea, eb)
                alignments.append({
                    "role": ea.role,
                    "a_value": ea.value,
                    "b_value": eb.value,
                    "a_type": ea.type_tag,
                    "b_type": eb.type_tag,
                    "similarity": score,
                    "same_structure": ea.signature() == eb.signature(),
                    "same_value": ea.value == eb.value,
                })
            elif ea is not None:
                unique_to_a.append(ea.role)
            else:
                unique_to_b.append(eb.role)

        # Calculate overall similarity
        total_elements = len(elements_a) + len(elements_b)
        if total_elements == 0:
            similarity = 0.0
        else:
            # Structural similarity weighted more than value similarity
            struct_matches = sum(1 for a in alignments if a["same_structure"])
            value_matches = sum(1 for a in alignments if a["same_value"])
            similarity = (
                (struct_matches * 0.7 + value_matches * 0.3) * 2 / total_elements
            )
            similarity = min(1.0, similarity)

        # Classify the relationship
        if similarity > 0.8:
            structural_type = "identical"
        elif similarity > 0.5:
            structural_type = "analogous"
        elif similarity > 0.2:
            structural_type = "partial"
        else:
            structural_type = "unrelated"

        # What can transfer from A to B?
        transferable = []
        gaps = []
        for role in unique_to_a:
            transferable.append(f"{role} (exists in A, missing in B)")
        for role in unique_to_b:
            gaps.append(f"{role} (exists in B, not in A)")

        # Also check: where A has more detail than B in shared roles
        for alignment in alignments:
            if alignment["a_value"] is not None and alignment["b_value"] is None:
                transferable.append(f"{alignment['role']} value from A could fill B")

        return ComparisonResult(
            similarity=similarity,
            shared_roles=shared_roles,
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
            alignments=alignments,
            structural_type=structural_type,
            transferable=transferable,
            gaps=gaps,
        )

    # ── Analogize: map structure across domains ──

    def analogize(self, source: dict, target: dict) -> AnalogicalMapping:
        """
        Create an analogical mapping between two different domains.

        "Debugging code is LIKE diagnosing a patient":
        - bug → symptom
        - source code → patient history
        - fix → treatment
        - test → follow-up

        The mapping lets you apply debugging strategies to diagnosis
        and vice versa, because the STRUCTURE is the same.
        """
        comparison = self.compare(source, target)

        source_domain = source.get("domain", source.get("description", "source"))
        target_domain = target.get("domain", target.get("description", "target"))

        mappings = []
        for alignment in comparison.alignments:
            if alignment["same_structure"]:
                mappings.append({
                    "source_role": alignment["role"],
                    "source_value": alignment["a_value"],
                    "target_role": alignment["role"],
                    "target_value": alignment["b_value"],
                    "confidence": alignment["similarity"],
                    "type": "direct" if alignment["same_value"] else "analogical",
                })

        # Generate predictions: what the analogy tells us about the target
        predictions = []
        for role in comparison.unique_to_a:
            # Source has something target doesn't — predict target needs it
            source_elements = self.abstraction.decompose(source)
            for el in source_elements:
                if el.role == role:
                    predictions.append({
                        "prediction": f"Target likely needs a '{role}' element",
                        "source_evidence": f"Source has {role}={el.value}",
                        "confidence": comparison.similarity * 0.6,
                    })

        return AnalogicalMapping(
            source_domain=str(source_domain),
            target_domain=str(target_domain),
            mappings=mappings,
            strength=comparison.similarity,
            predictions=predictions,
        )

    # ── Transfer: apply knowledge from one domain to another ──

    def transfer(self, source_episode: dict, target_context: dict) -> dict:
        """
        Apply what was learned in the source to the target.

        This is the holy grail: genuine transfer of understanding
        across domains. Not copy-paste. Not retrieval. Structural transfer.

        "I fixed a bug by reading the code, finding the error, and testing the fix.
        Now I have a broken recipe. I should: read the recipe, find the error,
        and test the fix."
        """
        analogy = self.analogize(source_episode, target_context)

        if analogy.strength < 0.2:
            return {
                "transfer_possible": False,
                "reason": "Structures too different for meaningful transfer",
                "similarity": analogy.strength,
            }

        # Build transfer plan: map source actions to target domain
        source_actions = source_episode.get("actions", source_episode.get("steps", []))
        transferred_actions = []

        for action in source_actions:
            if isinstance(action, dict):
                action_type = action.get("type", action.get("action", "unknown"))
                action_target = action.get("target", "")
            else:
                action_type = str(action)
                action_target = ""

            # Find the analogical mapping for this action's target
            mapped_target = action_target  # Default: keep same
            for mapping in analogy.mappings:
                if mapping["source_value"] == action_target:
                    mapped_target = mapping["target_value"]
                    break

            transferred_actions.append({
                "action": action_type,
                "target": mapped_target,
                "transferred_from": action_target,
                "confidence": analogy.strength,
            })

        return {
            "transfer_possible": True,
            "analogy_strength": analogy.strength,
            "source_domain": analogy.source_domain,
            "target_domain": analogy.target_domain,
            "transferred_actions": transferred_actions,
            "predictions": analogy.predictions,
            "gaps": [f"May need adaptation: {p['prediction']}" for p in analogy.predictions],
        }

    # ── Multi-compare: find the most similar past experience ──

    def find_most_similar(self, target: dict, candidates: list[dict]) -> list[tuple[dict, ComparisonResult]]:
        """
        Given a target situation and a list of past experiences,
        find which past experiences are most structurally similar.

        This is how memory becomes useful:
        "I've never seen this exact problem before, but I've seen
        something with the same STRUCTURE."
        """
        results = []
        for candidate in candidates:
            comparison = self.compare(candidate, target)
            results.append((candidate, comparison))

        results.sort(key=lambda x: x[1].similarity, reverse=True)
        return results

    # ── Dimensional comparison (the alligator insight) ──

    def compare_properties(self, entity: str, properties: list[dict]) -> list[dict]:
        """
        Compare properties of an entity along their dimensions.

        Each property has:
        - value (the measurement)
        - dimension (what axis: length, color, weight, etc.)
        - dimensionality (1D=line, 2D=surface, 3D=volume)
        - distinctiveness (how unusual is this value for this type of thing?)

        The alligator insight: green is 2D (covers surface), length is 1D.
        2D > 1D, so "greener" covers more of what the alligator IS.
        """
        analyzed = []
        for prop in properties:
            name = prop.get("name", "unknown")
            value = prop.get("value", "unknown")
            dimension = prop.get("dimension", "unknown")
            dimensionality = prop.get("dimensionality", 1)  # Default 1D

            analyzed.append({
                "entity": entity,
                "property": name,
                "value": value,
                "dimension": dimension,
                "dimensionality": dimensionality,
                "coverage": self._estimate_coverage(dimensionality),
                "reasoning": self._dimensional_reasoning(name, dimensionality),
            })

        # Sort by coverage (higher dimensionality = more coverage)
        analyzed.sort(key=lambda x: x["coverage"], reverse=True)
        return analyzed

    def _estimate_coverage(self, dimensionality: int) -> float:
        """How much of the entity does this property cover?"""
        # 1D = line (length, height) → covers a single axis
        # 2D = surface (color, texture) → covers all visible surface
        # 3D = volume (weight, density) → covers the whole thing
        coverage_map = {1: 0.33, 2: 0.67, 3: 1.0}
        return coverage_map.get(dimensionality, 0.1)

    def _dimensional_reasoning(self, property_name: str, dimensionality: int) -> str:
        dim_names = {1: "linear (1D)", 2: "surface (2D)", 3: "volumetric (3D)"}
        dim_name = dim_names.get(dimensionality, f"{dimensionality}D")
        return f"'{property_name}' is a {dim_name} property"

    def _element_similarity(self, a: StructuralElement, b: StructuralElement) -> float:
        """Detailed similarity between two elements."""
        score = 0.0
        if a.role == b.role:
            score += 0.4
        if a.type_tag == b.type_tag:
            score += 0.3
        if a.value == b.value:
            score += 0.2
        if len(a.children) == len(b.children):
            score += 0.1
        return score
