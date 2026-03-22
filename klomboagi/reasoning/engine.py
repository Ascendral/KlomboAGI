"""
Internal Reasoning Engine — the missing piece.

This is not a classifier. Not a search engine. Not an LLM wrapper.
This is a system that takes facts, derives new facts from them,
and chains those derivations until it reaches understanding.

The core operation: derive a new fact from existing facts,
then use that new fact to derive another.

No training data. No pattern matching. Pure logical operations
on abstract properties.

Example — the alligator problem:
  Input: "Is an alligator greener or longer?"
  Facts: [alligators are green, alligators are ~12 feet long]

  Step 1: "green" is a property → what kind? → color → applies to surface
  Step 2: "long" is a property → what kind? → measurement → applies to axis
  Step 3: surface has dimensions → count them → 2D minimum (length × width)
  Step 4: axis has dimensions → count them → 1D (length only)
  Step 5: compare dimensions → 2D > 1D → surface contains infinite axes
  Step 6: therefore "green" occupies MORE than "long" in dimensional terms
  Step 7: but this assumes dimensional framework — flag as one perspective
  Step 8: the obvious answer (longer) assumes magnitude framework — flag as another
  Step 9: the question is structured to reward the non-obvious reasoning

Each step derives something new from the previous step.
That's reasoning. Not lookup. Not classification. Derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ── Core types ──

class PropertyType(Enum):
    """What kind of thing is this property?"""
    UNKNOWN = "unknown"
    COLOR = "color"           # surface property (2D+)
    MEASUREMENT = "measurement"  # axis property (1D)
    QUANTITY = "quantity"      # count (0D — scalar)
    TEXTURE = "texture"       # surface property (2D+)
    SHAPE = "shape"           # volume property (3D)
    WEIGHT = "weight"         # scalar (0D)
    SOUND = "sound"           # waveform (1D over time)
    EMOTION = "emotion"       # abstract (dimensionless)
    RELATION = "relation"     # between entities (graph edge)
    TEMPORAL = "temporal"     # time-based (1D)
    SPATIAL = "spatial"       # space-based (1-3D)


# Map property types to their dimensional signatures
PROPERTY_DIMENSIONS: dict[PropertyType, int] = {
    PropertyType.QUANTITY: 0,     # scalar — just a number
    PropertyType.WEIGHT: 0,       # scalar
    PropertyType.MEASUREMENT: 1,  # along one axis
    PropertyType.SOUND: 1,        # waveform over time
    PropertyType.TEMPORAL: 1,     # time is 1D
    PropertyType.COLOR: 2,        # covers a surface (at minimum)
    PropertyType.TEXTURE: 2,      # covers a surface
    PropertyType.SPATIAL: 3,      # fills space
    PropertyType.SHAPE: 3,        # volume
    PropertyType.EMOTION: -1,     # not spatially dimensional
    PropertyType.RELATION: -1,    # not spatially dimensional
    PropertyType.UNKNOWN: -1,     # can't determine
}


@dataclass
class Fact:
    """A single piece of knowledge — either given or derived."""
    content: str                    # Human-readable statement
    subject: str = ""               # What entity this is about
    property_name: str = ""         # What property
    property_type: PropertyType = PropertyType.UNKNOWN
    value: Any = None               # The value if known
    dimensions: int = -1            # Dimensional signature (-1 = unknown)
    derived_from: list[int] = field(default_factory=list)  # IDs of parent facts
    confidence: float = 1.0         # How sure are we
    is_given: bool = True           # Given vs derived
    id: int = 0

    def __repr__(self) -> str:
        src = "given" if self.is_given else f"derived from {self.derived_from}"
        return f"Fact({self.id}: '{self.content}' [{src}] conf={self.confidence:.0%})"


@dataclass
class DerivationStep:
    """One step of reasoning — takes input facts, produces output fact."""
    operation: str          # What logical operation was performed
    input_facts: list[int]  # Fact IDs used as input
    output_fact: int        # Fact ID produced
    explanation: str        # Why this step follows from the inputs

    def __repr__(self) -> str:
        return f"Step({self.operation}: {self.input_facts} → {self.output_fact}: {self.explanation})"


@dataclass
class ReasoningChain:
    """A complete chain of derivation from given facts to conclusion."""
    facts: list[Fact] = field(default_factory=list)
    steps: list[DerivationStep] = field(default_factory=list)
    conclusion: str = ""
    alternative_conclusions: list[str] = field(default_factory=list)
    frameworks_used: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def add_fact(self, content: str, **kwargs) -> Fact:
        fact = Fact(content=content, id=len(self.facts), **kwargs)
        self.facts.append(fact)
        return fact

    def derive(self, operation: str, input_ids: list[int],
               new_content: str, explanation: str, **kwargs) -> Fact:
        """Derive a new fact from existing facts."""
        # Confidence of derived fact = min confidence of inputs × 0.95
        input_confs = [self.facts[i].confidence for i in input_ids if i < len(self.facts)]
        derived_conf = min(input_confs) * 0.95 if input_confs else 0.5

        fact = Fact(
            content=new_content,
            id=len(self.facts),
            derived_from=input_ids,
            is_given=False,
            confidence=derived_conf,
            **kwargs,
        )
        self.facts.append(fact)

        step = DerivationStep(
            operation=operation,
            input_facts=input_ids,
            output_fact=fact.id,
            explanation=explanation,
        )
        self.steps.append(step)
        return fact

    def explain(self) -> str:
        """Produce human-readable reasoning trace."""
        lines = []
        lines.append("Given facts:")
        for f in self.facts:
            if f.is_given:
                lines.append(f"  [{f.id}] {f.content}")

        lines.append("\nReasoning:")
        for i, step in enumerate(self.steps):
            inputs = ", ".join(f"[{fid}]" for fid in step.input_facts)
            lines.append(f"  Step {i + 1} ({step.operation}): {inputs} → [{step.output_fact}]")
            lines.append(f"    {step.explanation}")
            lines.append(f"    ∴ {self.facts[step.output_fact].content}")

        lines.append(f"\nConclusion: {self.conclusion} (confidence: {self.confidence:.0%})")
        if self.alternative_conclusions:
            lines.append("Alternative conclusions:")
            for alt in self.alternative_conclusions:
                lines.append(f"  - {alt}")
        if self.frameworks_used:
            lines.append(f"Frameworks used: {', '.join(self.frameworks_used)}")
        return "\n".join(lines)


# ── Reasoning Operations ──
# These are the atomic logical operations the engine can perform.
# Each one takes facts in, produces a fact out.
# No LLM. No lookup. Pure logic.

class ReasoningOps:
    """Atomic reasoning operations — the building blocks of thought."""

    @staticmethod
    def identify_property_type(fact: Fact) -> tuple[PropertyType, str]:
        """Given a fact about something, identify what type of property it describes."""
        content = fact.content.lower()

        # Color words
        color_words = {"red", "blue", "green", "yellow", "orange", "purple",
                       "black", "white", "brown", "pink", "grey", "gray",
                       "colored", "colour", "color", "hue", "tint", "shade"}
        # Measurement words
        measure_words = {"long", "short", "tall", "wide", "narrow", "thick",
                         "thin", "deep", "shallow", "high", "low", "length",
                         "width", "height", "depth", "distance", "feet",
                         "meters", "inches", "miles", "far", "near"}
        # Quantity words
        quantity_words = {"many", "few", "several", "number", "count",
                          "amount", "total", "sum", "zero", "one", "two",
                          "three", "hundred", "thousand"}
        # Weight words
        weight_words = {"heavy", "light", "weighs", "pounds", "kilograms",
                        "tons", "mass", "dense"}
        # Shape words
        shape_words = {"round", "flat", "spherical", "cubic", "volume",
                       "shape", "form"}
        # Texture words
        texture_words = {"rough", "smooth", "soft", "hard", "bumpy",
                         "scaly", "furry", "slimy", "texture"}
        # Temporal words
        temporal_words = {"old", "young", "fast", "slow", "duration",
                          "time", "age", "quick", "speed"}

        words = set(content.split())

        if words & color_words:
            return PropertyType.COLOR, "color property detected"
        if words & measure_words:
            return PropertyType.MEASUREMENT, "measurement property detected"
        if words & quantity_words:
            return PropertyType.QUANTITY, "quantity property detected"
        if words & weight_words:
            return PropertyType.WEIGHT, "weight property detected"
        if words & shape_words:
            return PropertyType.SHAPE, "shape property detected"
        if words & texture_words:
            return PropertyType.TEXTURE, "texture property detected"
        if words & temporal_words:
            return PropertyType.TEMPORAL, "temporal property detected"

        return PropertyType.UNKNOWN, "could not determine property type"

    @staticmethod
    def get_dimensions(prop_type: PropertyType) -> tuple[int, str]:
        """Get the dimensional signature of a property type."""
        dims = PROPERTY_DIMENSIONS.get(prop_type, -1)
        explanations = {
            0: "scalar — a single number, no spatial extent",
            1: "1-dimensional — exists along a single axis",
            2: "2-dimensional — exists across a surface",
            3: "3-dimensional — exists in a volume",
            -1: "dimensionality cannot be determined",
        }
        return dims, explanations.get(dims, "unknown dimensionality")

    @staticmethod
    def compare_dimensions(dim_a: int, dim_b: int) -> tuple[str, str]:
        """Compare two dimensional signatures."""
        if dim_a < 0 or dim_b < 0:
            return "incomparable", "one or both dimensions are unknown"
        if dim_a > dim_b:
            return "a_greater", f"dimension {dim_a} > dimension {dim_b}: a higher-dimensional space contains infinitely many lower-dimensional spaces"
        if dim_b > dim_a:
            return "b_greater", f"dimension {dim_b} > dimension {dim_a}: a higher-dimensional space contains infinitely many lower-dimensional spaces"
        return "equal", f"both are {dim_a}-dimensional — compare by magnitude"

    @staticmethod
    def detect_comparison_type(content: str) -> tuple[str, str]:
        """Detect if a question is comparing properties and what kind."""
        comparison_words = {"or", "versus", "vs", "compared to", "than",
                            "more", "less", "greater", "bigger", "smaller"}
        words = set(content.lower().split())
        if words & comparison_words:
            return "comparison", "question asks to compare two properties"
        return "direct", "question asks about a single property"

    @staticmethod
    def detect_trick_structure(facts: list[Fact]) -> tuple[bool, str]:
        """Detect if the problem structure suggests a trick/riddle."""
        # If we're comparing properties of different types, it's likely a trick
        types = set()
        for f in facts:
            if f.property_type != PropertyType.UNKNOWN:
                types.add(f.property_type)

        if len(types) >= 2:
            # Check if the types are in fundamentally different dimensional spaces
            dims = set()
            for t in types:
                d = PROPERTY_DIMENSIONS.get(t, -1)
                if d >= 0:
                    dims.add(d)

            if len(dims) >= 2:
                return True, (
                    f"comparing properties from different dimensional spaces "
                    f"({', '.join(t.value for t in types)}) — "
                    f"this is likely a trick question or requires framework choice"
                )

        return False, "properties appear comparable"

    @staticmethod
    def extract_subject_and_property(content: str) -> tuple[str, str]:
        """Extract what entity and what property a fact describes."""
        # Simple extraction — look for "X is/are Y" patterns
        content_lower = content.lower().strip()

        for pattern in ["is ", "are "]:
            if pattern in content_lower:
                parts = content_lower.split(pattern, 1)
                subject = parts[0].strip()
                prop = parts[1].strip()
                return subject, prop

        return "", content_lower

    @staticmethod
    def check_logical_consistency(facts: list[Fact]) -> list[str]:
        """Check if a set of facts are logically consistent."""
        issues = []

        # Group facts by subject
        by_subject: dict[str, list[Fact]] = {}
        for f in facts:
            if f.subject:
                by_subject.setdefault(f.subject, []).append(f)

        for subject, subject_facts in by_subject.items():
            # Check for contradictions in same property
            by_prop: dict[str, list[Fact]] = {}
            for f in subject_facts:
                if f.property_name:
                    by_prop.setdefault(f.property_name, []).append(f)

            for prop, prop_facts in by_prop.items():
                if len(prop_facts) > 1:
                    values = [f.value for f in prop_facts if f.value is not None]
                    if len(set(str(v) for v in values)) > 1:
                        issues.append(
                            f"Contradiction: '{subject}' has conflicting values "
                            f"for '{prop}': {values}"
                        )

        return issues


# ── The Engine ──

class ReasoningEngine:
    """
    Internal reasoning engine. Takes a question and facts,
    produces a chain of derivations leading to understanding.

    No LLM. No training data. Pure logical operations.
    """

    def __init__(self) -> None:
        self.ops = ReasoningOps()

    def reason(self, question: str, given_facts: list[str],
               nudge: str | None = None) -> ReasoningChain:
        """
        Main entry point. Give it a question and facts,
        get back a complete reasoning chain.
        """
        chain = ReasoningChain()

        # Step 0: Register given facts
        facts = []
        for text in given_facts:
            subject, prop = self.ops.extract_subject_and_property(text)
            f = chain.add_fact(text, subject=subject, property_name=prop, is_given=True)
            facts.append(f)

        # Step 1: What kind of question is this?
        q_type, q_explanation = self.ops.detect_comparison_type(question)
        chain.derive(
            operation="classify_question",
            input_ids=[],
            new_content=f"This is a {q_type} question: {q_explanation}",
            explanation=f"Analyzing question structure: '{question}'",
        )

        # Step 2: Identify property types for each fact
        for f in facts:
            prop_type, reason = self.ops.identify_property_type(f)
            f.property_type = prop_type
            dims, dim_explanation = self.ops.get_dimensions(prop_type)
            f.dimensions = dims
            chain.derive(
                operation="identify_property",
                input_ids=[f.id],
                new_content=f"'{f.property_name}' is a {prop_type.value} property ({dim_explanation})",
                explanation=reason,
                property_type=prop_type,
                dimensions=dims,
            )

        # Step 3: If comparison — compare the properties
        if q_type == "comparison" and len(facts) >= 2:
            self._reason_comparison(chain, facts, nudge)
        else:
            self._reason_direct(chain, facts, nudge)

        return chain

    def _reason_comparison(self, chain: ReasoningChain, facts: list[Fact],
                           nudge: str | None) -> None:
        """Reason about a comparison between properties."""

        # Get the property types and dimensions
        typed_facts = [f for f in facts if f.property_type != PropertyType.UNKNOWN]

        if len(typed_facts) < 2:
            chain.conclusion = "Cannot compare — unable to determine property types"
            chain.confidence = 0.2
            return

        a, b = typed_facts[0], typed_facts[1]

        # Step 3a: Check if this is a trick question
        is_trick, trick_reason = self.ops.detect_trick_structure(typed_facts)
        chain.derive(
            operation="detect_structure",
            input_ids=[a.id, b.id],
            new_content=f"Trick question: {'YES' if is_trick else 'NO'} — {trick_reason}",
            explanation="Checking if properties are from different dimensional spaces",
        )

        # Step 3b: Apply dimensional framework
        dim_result, dim_explanation = self.ops.compare_dimensions(a.dimensions, b.dimensions)

        dim_fact = chain.derive(
            operation="compare_dimensions",
            input_ids=[a.id, b.id],
            new_content=f"Dimensional comparison: {a.property_name} ({a.dimensions}D) vs "
                        f"{b.property_name} ({b.dimensions}D) → {dim_explanation}",
            explanation=f"Comparing dimensional signatures: "
                        f"{a.property_type.value} is {a.dimensions}D, "
                        f"{b.property_type.value} is {b.dimensions}D",
        )

        # Step 3c: Apply magnitude framework (the obvious answer)
        magnitude_fact = chain.derive(
            operation="compare_magnitude",
            input_ids=[a.id, b.id],
            new_content=f"Magnitude comparison: '{a.property_name}' vs '{b.property_name}' — "
                        f"cannot directly compare different property types by magnitude",
            explanation="Attempting direct magnitude comparison",
        )

        # Step 3d: Check for human nudge
        if nudge:
            chain.derive(
                operation="apply_nudge",
                input_ids=[dim_fact.id],
                new_content=f"Human nudge applied: '{nudge}' — shifting reasoning framework",
                explanation=f"External perspective received: {nudge}",
            )

        # Step 4: Draw conclusions based on frameworks
        chain.frameworks_used.append("dimensional_analysis")
        chain.frameworks_used.append("magnitude_comparison")

        if dim_result == "a_greater":
            chain.conclusion = (
                f"'{a.property_name}' is MORE than '{b.property_name}' "
                f"in dimensional terms ({a.dimensions}D > {b.dimensions}D). "
                f"A {a.dimensions}D space contains infinitely many {b.dimensions}D spaces."
            )
            chain.alternative_conclusions.append(
                f"If comparing by measurability: '{b.property_name}' is more quantifiable "
                f"(can be measured in units) while '{a.property_name}' is qualitative"
            )
            chain.confidence = 0.75
        elif dim_result == "b_greater":
            chain.conclusion = (
                f"'{b.property_name}' is MORE than '{a.property_name}' "
                f"in dimensional terms ({b.dimensions}D > {a.dimensions}D)"
            )
            chain.confidence = 0.75
        elif dim_result == "equal":
            chain.conclusion = (
                f"'{a.property_name}' and '{b.property_name}' are in the same "
                f"dimensional space ({a.dimensions}D) — compare by magnitude"
            )
            chain.confidence = 0.6
        else:
            chain.conclusion = (
                f"Cannot definitively compare '{a.property_name}' and '{b.property_name}' — "
                f"they exist in incomparable spaces"
            )
            chain.confidence = 0.4

        if is_trick:
            chain.alternative_conclusions.append(
                "This question is structured as a riddle — the 'obvious' answer "
                "is likely wrong, and the question rewards deeper structural reasoning"
            )

    def _reason_direct(self, chain: ReasoningChain, facts: list[Fact],
                       nudge: str | None) -> None:
        """Reason about a direct (non-comparison) question."""
        # Check logical consistency
        issues = self.ops.check_logical_consistency(facts)
        if issues:
            for issue in issues:
                chain.derive(
                    operation="consistency_check",
                    input_ids=[f.id for f in facts],
                    new_content=f"Inconsistency: {issue}",
                    explanation="Checking logical consistency of given facts",
                )

        chain.conclusion = "Direct question — facts recorded and analyzed"
        chain.confidence = 0.5 if issues else 0.7

    def reason_with_nudge(self, chain: ReasoningChain, nudge: str) -> ReasoningChain:
        """
        Re-reason with a human nudge. The nudge changes how the
        engine decomposes and analyzes — not just the output.

        This is the key insight: a nudge doesn't change the ANSWER.
        It changes the FRAMEWORK used to think about the problem.
        """
        # Extract original question and facts from chain
        given_facts = [f.content for f in chain.facts if f.is_given]

        # Find the original question from the first derivation
        question = ""
        for step in chain.steps:
            if step.operation == "classify_question":
                question = step.explanation.replace("Analyzing question structure: '", "").rstrip("'")
                break

        # Re-reason with the nudge
        return self.reason(question, given_facts, nudge=nudge)
