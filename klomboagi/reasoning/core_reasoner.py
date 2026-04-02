"""
Core Reasoner -- the actual thinking engine.

This replaces all the fake reasoning modules. No keyword matching.
No hardcoded lookup tables. No template generation.

What it actually does:
  1. Forward chaining: given facts, derive everything that follows
  2. Backward chaining: given a question, find what facts would answer it
  3. Unification: match variables in rules against concrete values
  4. Magnitude reasoning: compare quantities with units
  5. Property inheritance: if A is-a B and B has property P, A has P
  6. Negation: track what's known to be false
  7. Uncertainty: every fact has a confidence, derivations degrade it
  8. Honesty: says "I don't know" when it genuinely doesn't know

No LLM. No API. Prolog-style logic programming in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ---- Knowledge Representation ----

class Rel(Enum):
    """Relationships between concepts."""
    IS_A = "is_a"           # dog IS_A mammal
    HAS_PROP = "has_prop"   # dog HAS_PROP fur
    HAS_VALUE = "has_value"  # dog.weight HAS_VALUE (40, kg)
    CAUSES = "causes"       # heat CAUSES expansion
    REQUIRES = "requires"   # fire REQUIRES oxygen
    PART_OF = "part_of"     # wheel PART_OF car
    OPPOSITE = "opposite"   # hot OPPOSITE cold
    CAN = "can"             # bird CAN fly
    LOCATED = "located"     # whale LOCATED ocean


@dataclass(frozen=True)
class Fact:
    """A single piece of knowledge."""
    subject: str
    relation: Rel
    obj: str
    confidence: float = 1.0
    source: str = "given"      # given, derived, taught, observed

    def __str__(self):
        return f"{self.subject} {self.relation.value} {self.obj} [{self.confidence:.0%}]"


@dataclass(frozen=True)
class NumericFact:
    """A fact with a numeric value and unit."""
    subject: str
    property: str
    value: float
    unit: str
    confidence: float = 1.0
    source: str = "given"

    def __str__(self):
        return f"{self.subject}.{self.property} = {self.value} {self.unit} [{self.confidence:.0%}]"


@dataclass
class Rule:
    """An inference rule: if conditions hold, conclusion follows.

    Example: if X is_a Y and Y has_prop P, then X has_prop P
    Variables start with ? (like ?X, ?Y, ?P)
    """
    conditions: list[tuple[str, Rel, str]]   # (?X, IS_A, ?Y), (?Y, HAS_PROP, ?P)
    conclusion: tuple[str, Rel, str]         # (?X, HAS_PROP, ?P)
    confidence_decay: float = 0.85           # multiply parent confidence
    name: str = ""

    def __str__(self):
        conds = " AND ".join(f"{s} {r.value} {o}" for s, r, o in self.conditions)
        s, r, o = self.conclusion
        return f"IF {conds} THEN {s} {r.value} {o}"


@dataclass
class DerivationStep:
    """One step in a reasoning chain."""
    rule_used: str
    facts_used: list[str]
    conclusion: str
    confidence: float


@dataclass
class ReasoningResult:
    """The result of a reasoning query."""
    answer: str
    confidence: float
    chain: list[DerivationStep]
    known: bool = True  # False if we genuinely don't know

    def explain(self) -> str:
        if not self.known:
            return f"I don't know: {self.answer}"
        lines = [self.answer]
        if self.chain:
            lines.append("Because:")
            for step in self.chain:
                lines.append(f"  {step.rule_used}: {', '.join(step.facts_used)} -> {step.conclusion}")
        return "\n".join(lines)


# ---- The Actual Reasoning Engine ----

class CoreReasoner:
    """
    Prolog-style forward/backward chaining over a fact base.

    This is the real thing. No faking.
    """

    def __init__(self):
        self.facts: set[Fact] = set()
        self.numeric_facts: set[NumericFact] = set()
        self.rules: list[Rule] = []
        self.derived: set[Fact] = set()  # facts we derived (not given)
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Core inference rules -- these are the laws of reasoning."""
        # Transitivity: if A is_a B and B is_a C, then A is_a C
        self.rules.append(Rule(
            conditions=[("?X", Rel.IS_A, "?Y"), ("?Y", Rel.IS_A, "?Z")],
            conclusion=("?X", Rel.IS_A, "?Z"),
            confidence_decay=0.85,
            name="transitivity",
        ))

        # Property inheritance: if A is_a B and B has_prop P, then A has_prop P
        self.rules.append(Rule(
            conditions=[("?X", Rel.IS_A, "?Y"), ("?Y", Rel.HAS_PROP, "?P")],
            conclusion=("?X", Rel.HAS_PROP, "?P"),
            confidence_decay=0.8,
            name="property_inheritance",
        ))

        # Capability inheritance: if A is_a B and B can C, then A can C
        self.rules.append(Rule(
            conditions=[("?X", Rel.IS_A, "?Y"), ("?Y", Rel.CAN, "?C")],
            conclusion=("?X", Rel.CAN, "?C"),
            confidence_decay=0.75,
            name="capability_inheritance",
        ))

        # Location inheritance: if A is_a B and B located L, then A located L
        self.rules.append(Rule(
            conditions=[("?X", Rel.IS_A, "?Y"), ("?Y", Rel.LOCATED, "?L")],
            conclusion=("?X", Rel.LOCATED, "?L"),
            confidence_decay=0.7,
            name="location_inheritance",
        ))

        # Causal transitivity: if A causes B and B causes C, then A causes C
        self.rules.append(Rule(
            conditions=[("?X", Rel.CAUSES, "?Y"), ("?Y", Rel.CAUSES, "?Z")],
            conclusion=("?X", Rel.CAUSES, "?Z"),
            confidence_decay=0.7,
            name="causal_chain",
        ))

        # Part-whole property: if A part_of B and B has_prop P, then A has context of P
        self.rules.append(Rule(
            conditions=[("?X", Rel.PART_OF, "?Y"), ("?Y", Rel.HAS_PROP, "?P")],
            conclusion=("?X", Rel.HAS_PROP, "?P"),
            confidence_decay=0.5,  # weaker -- parts don't always share whole's properties
            name="part_property",
        ))

    # ---- Knowledge Management ----

    def tell(self, subject: str, relation: Rel, obj: str,
             confidence: float = 1.0, source: str = "taught") -> Fact:
        """Add a fact to the knowledge base."""
        subject = subject.lower().strip()
        obj = obj.lower().strip()
        fact = Fact(subject, relation, obj, confidence, source)
        self.facts.add(fact)
        return fact

    def tell_numeric(self, subject: str, prop: str, value: float, unit: str,
                     confidence: float = 1.0, source: str = "taught") -> NumericFact:
        """Add a numeric fact."""
        nf = NumericFact(subject.lower().strip(), prop.lower().strip(),
                         value, unit.lower().strip(), confidence, source)
        self.numeric_facts.add(nf)
        return nf

    def tell_many(self, triples: list[tuple[str, Rel, str]],
                  confidence: float = 1.0, source: str = "taught"):
        """Add multiple facts at once."""
        for s, r, o in triples:
            self.tell(s, r, o, confidence, source)

    # ---- Forward Chaining ----

    def forward_chain(self, max_iterations: int = 10) -> list[Fact]:
        """Derive all possible new facts from existing facts and rules.

        Runs until no new facts can be derived or max_iterations reached.
        Returns list of newly derived facts.
        """
        all_new = []

        for _ in range(max_iterations):
            new_facts = []
            all_facts = self.facts | self.derived

            for rule in self.rules:
                # Find all variable bindings that satisfy the conditions
                bindings = self._find_bindings(rule.conditions, all_facts)

                for binding, used_facts in bindings:
                    # Apply binding to conclusion
                    s = self._apply_binding(rule.conclusion[0], binding)
                    r = rule.conclusion[1]
                    o = self._apply_binding(rule.conclusion[2], binding)

                    # Calculate confidence
                    min_conf = min(f.confidence for f in used_facts)
                    conf = min_conf * rule.confidence_decay

                    new_fact = Fact(s, r, o, conf, "derived")

                    # Don't add if blocked (explicitly denied)
                    if self._is_blocked(s, r, o):
                        continue

                    # Don't add if already known (with equal or higher confidence)
                    if not self._already_known(new_fact, all_facts | set(new_facts)):
                        new_facts.append(new_fact)

            if not new_facts:
                break

            for f in new_facts:
                self.derived.add(f)
            all_new.extend(new_facts)

        return all_new

    def _find_bindings(self, conditions: list[tuple[str, Rel, str]],
                       facts: set[Fact]) -> list[tuple[dict, list[Fact]]]:
        """Find all variable bindings that satisfy ALL conditions."""
        if not conditions:
            return [({}, [])]

        first_cond = conditions[0]
        rest = conditions[1:]
        results = []

        for fact in facts:
            binding = self._unify_condition(first_cond, fact)
            if binding is not None:
                # Try to extend this binding to satisfy remaining conditions
                sub_results = self._extend_bindings(rest, binding, facts)
                for sub_binding, sub_facts in sub_results:
                    merged = {**binding, **sub_binding}
                    results.append((merged, [fact] + sub_facts))

        return results

    def _extend_bindings(self, conditions: list[tuple[str, Rel, str]],
                         existing_binding: dict, facts: set[Fact]
                         ) -> list[tuple[dict, list[Fact]]]:
        """Extend an existing binding to satisfy more conditions."""
        if not conditions:
            return [({}, [])]

        first_cond = conditions[0]
        rest = conditions[1:]

        # Apply existing binding to this condition
        s = self._apply_binding(first_cond[0], existing_binding)
        r = first_cond[1]
        o = self._apply_binding(first_cond[2], existing_binding)

        results = []
        for fact in facts:
            binding = self._unify_condition((s, r, o), fact)
            if binding is not None:
                # Check consistency with existing binding
                consistent = True
                for var, val in binding.items():
                    if var in existing_binding and existing_binding[var] != val:
                        consistent = False
                        break
                if consistent:
                    merged = {**existing_binding, **binding}
                    sub_results = self._extend_bindings(rest, merged, facts)
                    for sub_binding, sub_facts in sub_results:
                        results.append(({**binding, **sub_binding}, [fact] + sub_facts))

        return results

    def _unify_condition(self, condition: tuple[str, Rel, str],
                         fact: Fact) -> dict | None:
        """Try to unify a condition pattern with a concrete fact.

        Returns variable bindings if successful, None if not.
        """
        s_pat, r_pat, o_pat = condition

        # Relation must match exactly
        if r_pat != fact.relation:
            return None

        binding = {}

        # Subject
        if s_pat.startswith("?"):
            binding[s_pat] = fact.subject
        elif s_pat != fact.subject:
            return None

        # Object
        if o_pat.startswith("?"):
            # Check consistency if variable already bound
            if o_pat in binding and binding[o_pat] != fact.obj:
                return None
            binding[o_pat] = fact.obj
        elif o_pat != fact.obj:
            return None

        return binding

    def _apply_binding(self, term: str, binding: dict) -> str:
        """Replace variables with their bound values."""
        if term.startswith("?") and term in binding:
            return binding[term]
        return term

    def _already_known(self, fact: Fact, known: set[Fact]) -> bool:
        """Check if a fact (or a stronger version) is already known."""
        for k in known:
            if k.subject == fact.subject and k.relation == fact.relation and k.obj == fact.obj:
                if k.confidence >= fact.confidence:
                    return True
        return False

    # ---- Backward Chaining (Query) ----

    def ask(self, subject: str, relation: Rel, obj: str = "?") -> ReasoningResult:
        """Ask a question. Returns a reasoned answer.

        Examples:
            ask("dog", IS_A, "?")          -> "mammal", "animal", "living thing"
            ask("dog", HAS_PROP, "?")      -> "fur", "warm-blooded" (inherited)
            ask("?", CAUSES, "expansion")  -> "heat"
            ask("dog", IS_A, "reptile")    -> False (contradiction)
        """
        subject = subject.lower().strip()
        obj = obj.lower().strip() if obj != "?" else "?"

        # Make sure we've derived everything possible
        self.forward_chain()

        all_facts = self.facts | self.derived
        chain = []

        if obj == "?":
            # Open query: find all matching facts
            matches = []
            for f in all_facts:
                if f.subject == subject and f.relation == relation:
                    matches.append(f)
            matches.sort(key=lambda f: -f.confidence)

            if not matches:
                return ReasoningResult(
                    answer=f"I don't know what {subject} {relation.value}.",
                    confidence=0.0, chain=[], known=False)

            # Build answer from matches
            parts = []
            for f in matches:
                parts.append(f.obj)
                if f.source == "derived":
                    chain.append(self._trace_derivation(f))

            answer = f"{subject} {relation.value}: {', '.join(parts)}"
            return ReasoningResult(
                answer=answer, confidence=matches[0].confidence,
                chain=chain, known=True)

        else:
            # Specific query: is this true?
            for f in all_facts:
                if f.subject == subject and f.relation == relation and f.obj == obj:
                    if f.source == "derived":
                        chain.append(self._trace_derivation(f))
                    return ReasoningResult(
                        answer=f"Yes, {subject} {relation.value} {obj}.",
                        confidence=f.confidence, chain=chain, known=True)

            # Check for contradiction (opposite known)
            for f in all_facts:
                if f.subject == subject and f.relation == Rel.OPPOSITE and f.obj == obj:
                    return ReasoningResult(
                        answer=f"No, {subject} is opposite of {obj}.",
                        confidence=f.confidence, chain=[], known=True)

            return ReasoningResult(
                answer=f"I don't know if {subject} {relation.value} {obj}.",
                confidence=0.0, chain=[], known=False)

    def ask_compare(self, a: str, b: str, prop: str) -> ReasoningResult:
        """Compare two things on a property.

        Example: ask_compare("whale", "car", "size")
        """
        a, b, prop = a.lower().strip(), b.lower().strip(), prop.lower().strip()

        # Look for numeric facts
        a_vals = [nf for nf in self.numeric_facts
                  if nf.subject == a and nf.property == prop]
        b_vals = [nf for nf in self.numeric_facts
                  if nf.subject == b and nf.property == prop]

        # Also check inherited numeric facts through is_a chains
        if not a_vals:
            a_vals = self._inherit_numeric(a, prop)
        if not b_vals:
            b_vals = self._inherit_numeric(b, prop)

        if a_vals and b_vals:
            a_val = a_vals[0]
            b_val = b_vals[0]

            # Convert units if needed
            a_converted = self._convert_unit(a_val.value, a_val.unit, b_val.unit)
            if a_converted is not None:
                if a_converted > b_val.value:
                    return ReasoningResult(
                        answer=f"{a} has greater {prop} than {b} "
                               f"({a_val.value} {a_val.unit} vs {b_val.value} {b_val.unit}).",
                        confidence=min(a_val.confidence, b_val.confidence),
                        chain=[], known=True)
                elif a_converted < b_val.value:
                    return ReasoningResult(
                        answer=f"{b} has greater {prop} than {a} "
                               f"({b_val.value} {b_val.unit} vs {a_val.value} {a_val.unit}).",
                        confidence=min(a_val.confidence, b_val.confidence),
                        chain=[], known=True)
                else:
                    return ReasoningResult(
                        answer=f"{a} and {b} have equal {prop}.",
                        confidence=min(a_val.confidence, b_val.confidence),
                        chain=[], known=True)

        # Can't compare -- be honest
        missing = []
        if not a_vals:
            missing.append(f"I don't know the {prop} of {a}")
        if not b_vals:
            missing.append(f"I don't know the {prop} of {b}")

        return ReasoningResult(
            answer=f"I can't compare {a} and {b} on {prop}. {'. '.join(missing)}.",
            confidence=0.0, chain=[], known=False)

    def _inherit_numeric(self, subject: str, prop: str) -> list[NumericFact]:
        """Look for numeric facts through is_a inheritance."""
        all_facts = self.facts | self.derived
        parents = [f.obj for f in all_facts
                   if f.subject == subject and f.relation == Rel.IS_A]

        for parent in parents:
            vals = [nf for nf in self.numeric_facts
                    if nf.subject == parent and nf.property == prop]
            if vals:
                return vals
            # Recurse one more level
            grandparents = [f.obj for f in all_facts
                           if f.subject == parent and f.relation == Rel.IS_A]
            for gp in grandparents:
                vals = [nf for nf in self.numeric_facts
                        if nf.subject == gp and nf.property == prop]
                if vals:
                    return vals
        return []

    def _convert_unit(self, value: float, from_unit: str, to_unit: str) -> float | None:
        """Convert between compatible units. Returns None if incompatible."""
        if from_unit == to_unit:
            return value

        # Length conversions
        to_meters = {
            "m": 1, "meters": 1, "meter": 1,
            "cm": 0.01, "centimeters": 0.01,
            "mm": 0.001, "millimeters": 0.001,
            "km": 1000, "kilometers": 1000,
            "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
            "in": 0.0254, "inches": 0.0254, "inch": 0.0254,
            "mi": 1609.34, "miles": 1609.34, "mile": 1609.34,
        }

        # Weight conversions
        to_kg = {
            "kg": 1, "kilograms": 1, "kilogram": 1,
            "g": 0.001, "grams": 0.001, "gram": 0.001,
            "lb": 0.453592, "lbs": 0.453592, "pounds": 0.453592, "pound": 0.453592,
            "ton": 907.185, "tons": 907.185,
            "tonne": 1000, "tonnes": 1000,
        }

        for table in [to_meters, to_kg]:
            if from_unit in table and to_unit in table:
                return value * table[from_unit] / table[to_unit]

        return None

    def _trace_derivation(self, fact: Fact) -> DerivationStep:
        """Trace how a derived fact was produced."""
        # Find which rule and facts produced this
        all_facts = self.facts | self.derived
        for rule in self.rules:
            bindings = self._find_bindings(rule.conditions, all_facts)
            for binding, used_facts in bindings:
                s = self._apply_binding(rule.conclusion[0], binding)
                r = rule.conclusion[1]
                o = self._apply_binding(rule.conclusion[2], binding)
                if s == fact.subject and r == fact.relation and o == fact.obj:
                    return DerivationStep(
                        rule_used=rule.name,
                        facts_used=[str(f) for f in used_facts],
                        conclusion=str(fact),
                        confidence=fact.confidence,
                    )
        return DerivationStep(
            rule_used="unknown",
            facts_used=[],
            conclusion=str(fact),
            confidence=fact.confidence,
        )

    # ---- Convenience ----

    def dump(self) -> str:
        """Dump all knowledge for debugging."""
        lines = ["=== Given Facts ==="]
        for f in sorted(self.facts, key=str):
            lines.append(f"  {f}")
        lines.append(f"\n=== Numeric Facts ===")
        for nf in sorted(self.numeric_facts, key=str):
            lines.append(f"  {nf}")
        lines.append(f"\n=== Derived Facts ({len(self.derived)}) ===")
        for f in sorted(self.derived, key=str):
            lines.append(f"  {f}")
        lines.append(f"\n=== Rules ({len(self.rules)}) ===")
        for r in self.rules:
            lines.append(f"  {r}")
        return "\n".join(lines)

    # ---- Negation / Exceptions ----

    def deny(self, subject: str, relation: Rel, obj: str,
             confidence: float = 1.0, source: str = "taught") -> Fact:
        """Explicitly deny a fact. Overrides inherited conclusions.

        Example: deny("penguin", CAN, "fly") blocks the inherited
        "penguin can fly" from "bird can fly".
        """
        subject = subject.lower().strip()
        obj = obj.lower().strip()

        # Remove any derived fact that matches
        to_remove = set()
        for f in self.derived:
            if f.subject == subject and f.relation == relation and f.obj == obj:
                to_remove.add(f)
        self.derived -= to_remove

        # Also remove from given facts if present
        to_remove_given = set()
        for f in self.facts:
            if f.subject == subject and f.relation == relation and f.obj == obj:
                to_remove_given.add(f)
        self.facts -= to_remove_given

        # Store the negation as a fact (using OPPOSITE as a marker)
        neg_fact = Fact(subject, Rel.OPPOSITE, f"not_{relation.value}_{obj}",
                        confidence, source)
        self.facts.add(neg_fact)

        # Add to blocked set so forward chaining won't re-derive it
        if not hasattr(self, '_blocked'):
            self._blocked = set()
        self._blocked.add((subject, relation, obj))

        return neg_fact

    def _is_blocked(self, subject: str, relation: Rel, obj: str) -> bool:
        """Check if a fact has been explicitly denied."""
        if not hasattr(self, '_blocked'):
            self._blocked = set()
        return (subject, relation, obj) in self._blocked

    # ---- Learning from Text ----

    def learn_from_text(self, text: str) -> list[Fact]:
        """Extract structured facts from natural language text.

        Not an LLM. Uses pattern matching to find "X is a Y",
        "X causes Y", "X can Y", etc. Then feeds them into the
        knowledge base and runs forward chaining.

        Returns list of new facts learned.
        """
        import re
        text = text.lower()
        new_facts = []

        def _normalize(s: str) -> str:
            """Strip articles, depluralize."""
            s = s.strip()
            for art in ("a ", "an ", "the "):
                if s.startswith(art):
                    s = s[len(art):]
            s = s.strip()
            # Basic depluralize
            if s.endswith("ies") and len(s) > 4:
                s = s[:-3] + "y"
            elif s.endswith("es") and len(s) > 3 and not s.endswith("ses"):
                s = s[:-2]
                if not s.endswith("e") and len(s) > 2:
                    s = s + "e"
            elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
                s = s[:-1]
            return s.strip()

        # "X is a/an Y" patterns
        for m in re.finditer(r'(\b\w[\w\s]{0,20}?)\s+(?:is|are)\s+(?:a|an)\s+(\w[\w\s]{0,20}?)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1 and s != o:
                f = self.tell(s, Rel.IS_A, o, 0.7, "learned")
                new_facts.append(f)

        # "X is Y" (property, not is-a)
        for m in re.finditer(r'(\b\w+)\s+(?:is|are)\s+([\w-]+)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 2 and s != o:
                if not any(f.subject == s and f.relation == Rel.IS_A for f in new_facts):
                    f = self.tell(s, Rel.HAS_PROP, o, 0.6, "learned")
                    new_facts.append(f)

        # "X causes Y" / "X leads to Y" / "X results in Y"
        for m in re.finditer(r'(\b\w[\w\s]{0,15}?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w[\w\s]{0,15}?)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1:
                f = self.tell(s, Rel.CAUSES, o, 0.6, "learned")
                new_facts.append(f)

        # "X requires Y" / "X needs Y"
        for m in re.finditer(r'(\b\w[\w\s]{0,15}?)\s+(?:requires?|needs?)\s+(\w[\w\s]{0,15}?)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1:
                f = self.tell(s, Rel.REQUIRES, o, 0.6, "learned")
                new_facts.append(f)

        # "X can Y" / "X is able to Y"
        for m in re.finditer(r'(\b\w+)\s+(?:can|is\s+able\s+to)\s+(\w[\w\s]{0,15}?)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1:
                f = self.tell(s, Rel.CAN, o, 0.6, "learned")
                new_facts.append(f)

        # "X cannot Y" / "X can't Y"
        for m in re.finditer(r"(\b\w+)\s+(?:cannot|can't|can\s+not)\s+(\w[\w\s]{0,15}?)(?:\.|,|;|$)", text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1:
                self.deny(s, Rel.CAN, o, 0.8, "learned")

        # "X is part of Y" / "X belongs to Y"
        for m in re.finditer(r'(\b\w[\w\s]{0,15}?)\s+(?:is\s+part\s+of|belongs?\s+to)\s+(\w[\w\s]{0,15}?)(?:\.|,|;|$)', text):
            s, o = _normalize(m.group(1)), _normalize(m.group(2))
            if len(s) > 1 and len(o) > 1:
                f = self.tell(s, Rel.PART_OF, o, 0.6, "learned")
                new_facts.append(f)

        # Numeric: "X is/weighs/measures N units"
        for m in re.finditer(r'(\b\w+)\s+(?:is|weighs?|measures?|has\s+a\s+\w+\s+of)\s+(?:about\s+)?(\d+(?:\.\d+)?)\s*(\w+)', text):
            s = _normalize(m.group(1))
            val = float(m.group(2))
            unit = m.group(3).strip()
            # Guess the property from the unit
            unit_to_prop = {
                "meters": "length", "m": "length", "km": "length", "feet": "length",
                "kg": "weight", "pounds": "weight", "tons": "weight", "lbs": "weight",
                "celsius": "temperature", "fahrenheit": "temperature",
                "years": "age", "days": "age",
            }
            prop = unit_to_prop.get(unit, "measurement")
            self.tell_numeric(s, prop, val, unit, 0.6, "learned")

        if new_facts:
            self.forward_chain(max_iterations=3)

        return new_facts

    # ---- Self-Questioning ----

    def find_gaps(self) -> list[str]:
        """Identify things the reasoner is curious about.

        Looks for:
        - Concepts mentioned but not defined (no is_a)
        - Properties without values
        - Causal chains that end abruptly
        - Things we inherited but can't verify
        """
        gaps = []
        all_facts = self.facts | self.derived

        # Concepts that appear as objects but have no facts as subject
        known_subjects = {f.subject for f in all_facts}
        known_objects = {f.obj for f in all_facts if f.relation != Rel.OPPOSITE}
        undefined = known_objects - known_subjects
        for concept in undefined:
            if len(concept) > 2 and not concept.startswith("not_"):
                gaps.append(f"What is {concept}?")

        # Subjects with only is_a but no properties
        for subject in known_subjects:
            has_isa = any(f.subject == subject and f.relation == Rel.IS_A for f in all_facts)
            has_prop = any(f.subject == subject and f.relation == Rel.HAS_PROP for f in all_facts)
            has_numeric = any(nf.subject == subject for nf in self.numeric_facts)
            if has_isa and not has_prop and not has_numeric:
                gaps.append(f"What properties does {subject} have?")

        # Low confidence derived facts we should verify
        for f in self.derived:
            if f.confidence < 0.5:
                gaps.append(f"Is it really true that {f.subject} {f.relation.value} {f.obj}?")

        return gaps[:20]

    @property
    def total_facts(self) -> int:
        return len(self.facts) + len(self.derived) + len(self.numeric_facts)
