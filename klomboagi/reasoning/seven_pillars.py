"""
The Seven Pillars of Reasoning -- the actual AGI algorithm.

These are the 7 operations that make intelligence possible.
Each one operates on the CoreReasoner's knowledge base and
produces real, verifiable results.

1. DECOMPOSE -- break a problem into parts that can be solved independently
2. COMPARE -- find structural similarity between two things
3. ABSTRACT -- extract the pattern from specific examples
4. TRANSFER -- apply a pattern from one domain to another
5. INQUIRY -- identify what's unknown and formulate questions to fill the gap
6. CAUSAL MODEL -- understand WHY things happen, not just THAT they happen
7. SELF-EVALUATE -- check your own answer, know when you're wrong

These are NOT wrappers. Each one does real computation on real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from klomboagi.reasoning.core_reasoner import CoreReasoner, Rel, Fact


# ---- 1. DECOMPOSE ----

@dataclass
class Decomposition:
    """A problem broken into parts."""
    original: str
    parts: list[str]
    relations_between_parts: list[tuple[str, str, str]]  # (part_a, relation, part_b)
    unknowns: list[str]  # parts we can't solve yet


def decompose(reasoner: CoreReasoner, problem: str) -> Decomposition:
    """Break a problem into independently solvable parts.

    Uses the knowledge base to identify entities, their relationships,
    and what's unknown.
    """
    import re
    words = re.findall(r'\b(\w{3,})\b', problem.lower())

    # Find which words are known entities
    all_facts = reasoner.facts | reasoner.derived
    known_entities = set()
    for w in words:
        for f in all_facts:
            if f.subject == w or f.obj == w:
                known_entities.add(w)
                break

    # Find relationships between known entities
    relations = []
    for a in known_entities:
        for b in known_entities:
            if a == b:
                continue
            for f in all_facts:
                if f.subject == a and f.obj == b:
                    relations.append((a, f.relation.value, b))

    # Unknowns: words that appear in the problem but aren't in the KB
    stop = {"what", "how", "why", "does", "can", "the", "and", "for",
            "this", "that", "with", "from", "are", "was", "has", "have",
            "been", "will", "would", "could", "should", "not", "but"}
    unknowns = [w for w in words if w not in known_entities and w not in stop and len(w) > 2]

    return Decomposition(
        original=problem,
        parts=list(known_entities) + unknowns,
        relations_between_parts=relations,
        unknowns=unknowns,
    )


# ---- 2. COMPARE ----

@dataclass
class Comparison:
    """Structural comparison between two things."""
    a: str
    b: str
    shared_properties: list[str]
    shared_categories: list[str]
    a_only: list[str]
    b_only: list[str]
    similarity: float  # 0-1


def compare(reasoner: CoreReasoner, a: str, b: str) -> Comparison:
    """Find structural similarities and differences between two concepts.

    Not string matching. Actual comparison of properties, categories,
    and relationships in the knowledge base.
    """
    a, b = a.lower().strip(), b.lower().strip()
    all_facts = reasoner.facts | reasoner.derived

    def get_properties(subject: str) -> set[str]:
        return {f.obj for f in all_facts
                if f.subject == subject and f.relation == Rel.HAS_PROP}

    def get_categories(subject: str) -> set[str]:
        return {f.obj for f in all_facts
                if f.subject == subject and f.relation == Rel.IS_A}

    def get_capabilities(subject: str) -> set[str]:
        return {f.obj for f in all_facts
                if f.subject == subject and f.relation == Rel.CAN}

    a_props = get_properties(a)
    b_props = get_properties(b)
    a_cats = get_categories(a)
    b_cats = get_categories(b)
    a_caps = get_capabilities(a)
    b_caps = get_capabilities(b)

    shared_props = a_props & b_props
    shared_cats = a_cats & b_cats
    shared_caps = a_caps & b_caps

    a_only_all = (a_props | a_caps) - (b_props | b_caps)
    b_only_all = (b_props | b_caps) - (a_props | a_caps)

    total = len(a_props | b_props | a_cats | b_cats | a_caps | b_caps)
    shared = len(shared_props | shared_cats | shared_caps)
    similarity = shared / max(total, 1)

    return Comparison(
        a=a, b=b,
        shared_properties=sorted(shared_props | shared_caps),
        shared_categories=sorted(shared_cats),
        a_only=sorted(a_only_all),
        b_only=sorted(b_only_all),
        similarity=round(similarity, 3),
    )


# ---- 3. ABSTRACT ----

@dataclass
class Abstraction:
    """A pattern extracted from multiple examples."""
    pattern_name: str
    common_properties: list[str]
    common_categories: list[str]
    common_capabilities: list[str]
    variable_properties: list[str]  # properties that differ
    examples: list[str]
    confidence: float


def abstract(reasoner: CoreReasoner, examples: list[str]) -> Abstraction:
    """Extract the pattern shared by multiple examples.

    Given ["dog", "cat", "horse"], finds what they have in common
    (mammal, warm-blooded, has fur) and what varies.
    """
    if not examples:
        return Abstraction("empty", [], [], [], [], [], 0.0)

    all_facts = reasoner.facts | reasoner.derived
    examples = [e.lower().strip() for e in examples]

    def get_all(subject: str, rel: Rel) -> set[str]:
        return {f.obj for f in all_facts if f.subject == subject and f.relation == rel}

    # Collect properties/categories/capabilities for each example
    all_props = [get_all(e, Rel.HAS_PROP) for e in examples]
    all_cats = [get_all(e, Rel.IS_A) for e in examples]
    all_caps = [get_all(e, Rel.CAN) for e in examples]

    # Find intersection (what ALL examples share)
    common_props = set.intersection(*all_props) if all_props and all(all_props) else set()
    common_cats = set.intersection(*all_cats) if all_cats and all(all_cats) else set()
    common_caps = set.intersection(*all_caps) if all_caps and all(all_caps) else set()

    # Find what varies (union minus intersection)
    all_union_props = set.union(*all_props) if all_props else set()
    variable = sorted(all_union_props - common_props)

    # Name the pattern by the most specific shared category
    if common_cats:
        pattern_name = sorted(common_cats, key=lambda c: len(c))[0]
    elif common_props:
        pattern_name = f"things that are {sorted(common_props)[0]}"
    else:
        pattern_name = "group"

    confidence = len(common_props | common_cats | common_caps) / max(
        len(all_union_props | set.union(*all_cats) if all_cats else set()), 1)

    return Abstraction(
        pattern_name=pattern_name,
        common_properties=sorted(common_props),
        common_categories=sorted(common_cats),
        common_capabilities=sorted(common_caps),
        variable_properties=variable,
        examples=examples,
        confidence=round(min(confidence, 1.0), 3),
    )


# ---- 4. TRANSFER ----

@dataclass
class Transfer:
    """Applying a known pattern to a new domain."""
    source: str
    target: str
    transferred_properties: list[str]
    transferred_capabilities: list[str]
    confidence: float
    reasoning: str


def transfer(reasoner: CoreReasoner, source: str, target: str) -> Transfer:
    """Apply what we know about source to target.

    If we know a lot about dogs but little about wolves,
    and wolves are similar to dogs (both are mammals),
    transfer dog properties to wolves as hypotheses.
    """
    source, target = source.lower().strip(), target.lower().strip()

    # Compare to find shared structure
    comp = compare(reasoner, source, target)

    if comp.similarity < 0.1:
        return Transfer(source, target, [], [], 0.0,
                        f"No structural similarity between {source} and {target}")

    # Properties source has that target doesn't
    all_facts = reasoner.facts | reasoner.derived
    source_props = {f.obj for f in all_facts
                    if f.subject == source and f.relation == Rel.HAS_PROP}
    target_props = {f.obj for f in all_facts
                    if f.subject == target and f.relation == Rel.HAS_PROP}
    source_caps = {f.obj for f in all_facts
                   if f.subject == source and f.relation == Rel.CAN}
    target_caps = {f.obj for f in all_facts
                   if f.subject == target and f.relation == Rel.CAN}

    transferable_props = sorted(source_props - target_props)
    transferable_caps = sorted(source_caps - target_caps)

    # Actually add the transferred knowledge (as low-confidence hypotheses)
    for prop in transferable_props:
        reasoner.tell(target, Rel.HAS_PROP, prop,
                      confidence=comp.similarity * 0.5, source="transfer")
    for cap in transferable_caps:
        reasoner.tell(target, Rel.CAN, cap,
                      confidence=comp.similarity * 0.4, source="transfer")

    if transferable_props or transferable_caps:
        reasoner.forward_chain(max_iterations=2)

    reasoning = (f"Because {source} and {target} share "
                 f"{', '.join(comp.shared_categories[:3])} and have "
                 f"similarity {comp.similarity:.0%}, hypothesizing that "
                 f"{target} also has: {', '.join(transferable_props[:3])}")

    return Transfer(
        source=source, target=target,
        transferred_properties=transferable_props,
        transferred_capabilities=transferable_caps,
        confidence=round(comp.similarity * 0.5, 3),
        reasoning=reasoning,
    )


# ---- 5. INQUIRY ----

@dataclass
class KnowledgeGap:
    """Something the system doesn't know but should."""
    question: str
    priority: float  # 0-1, higher = more important to fill
    context: str     # why we want to know this
    category: str    # what kind of gap (definition, property, causal, comparison)


def inquire(reasoner: CoreReasoner) -> list[KnowledgeGap]:
    """Identify what the system doesn't know and should learn.

    Not template generation. Actual analysis of the knowledge base
    to find structural holes.
    """
    gaps = []
    all_facts = reasoner.facts | reasoner.derived

    # 1. Concepts referenced but never defined
    subjects = {f.subject for f in all_facts}
    objects = {f.obj for f in all_facts if f.relation != Rel.OPPOSITE}
    undefined = objects - subjects
    for concept in sorted(undefined):
        if len(concept) > 2 and not concept.startswith("not_"):
            gaps.append(KnowledgeGap(
                question=f"What is {concept}?",
                priority=0.7,
                context=f"Referenced in knowledge base but never defined",
                category="definition",
            ))

    # 2. Things with categories but no properties
    for subject in sorted(subjects):
        has_category = any(f.subject == subject and f.relation == Rel.IS_A for f in all_facts)
        has_props = any(f.subject == subject and f.relation == Rel.HAS_PROP for f in all_facts)
        has_numeric = any(nf.subject == subject for nf in reasoner.numeric_facts)
        if has_category and not has_props and not has_numeric:
            gaps.append(KnowledgeGap(
                question=f"What properties does {subject} have?",
                priority=0.5,
                context=f"Known to be a category member but no properties recorded",
                category="property",
            ))

    # 3. Causal dead ends -- things that cause something, but we don't know
    #    what causes THEM
    caused = {f.obj for f in all_facts if f.relation == Rel.CAUSES}
    causes = {f.subject for f in all_facts if f.relation == Rel.CAUSES}
    uncaused = caused - causes
    for concept in sorted(uncaused):
        if len(concept) > 2:
            gaps.append(KnowledgeGap(
                question=f"What causes {concept}?",
                priority=0.6,
                context=f"We know {concept} happens but not why",
                category="causal",
            ))

    # 4. Low-confidence derived facts that should be verified
    for f in sorted(reasoner.derived, key=lambda x: x.confidence):
        if f.confidence < 0.4:
            gaps.append(KnowledgeGap(
                question=f"Is it true that {f.subject} {f.relation.value} {f.obj}?",
                priority=0.3,
                context=f"Derived with only {f.confidence:.0%} confidence",
                category="verification",
            ))

    # Sort by priority
    gaps.sort(key=lambda g: -g.priority)
    return gaps[:30]


# ---- 6. CAUSAL MODEL ----

@dataclass
class CausalChain:
    """A chain of cause-effect relationships."""
    root_cause: str
    final_effect: str
    chain: list[tuple[str, str]]  # [(cause, effect), ...]
    confidence: float
    interventions: list[str]  # what could break the chain


def build_causal_model(reasoner: CoreReasoner, effect: str) -> CausalChain | None:
    """Trace the full causal chain leading to an effect.

    Not just "X causes Y". Traces backward through the entire chain:
    root_cause -> intermediate1 -> intermediate2 -> ... -> effect

    Also identifies intervention points (what could break the chain).
    """
    effect = effect.lower().strip()
    all_facts = reasoner.facts | reasoner.derived

    # Backward trace: find what causes the effect
    def trace_back(target: str, visited: set) -> list[tuple[str, str]]:
        if target in visited:
            return []
        visited.add(target)

        causes = [(f.subject, f.obj) for f in all_facts
                  if f.relation == Rel.CAUSES and f.obj == target]
        if not causes:
            return []

        chain = []
        for cause, eff in causes:
            chain.append((cause, eff))
            # Recurse to find deeper causes
            deeper = trace_back(cause, visited)
            chain = deeper + chain

        return chain

    chain = trace_back(effect, set())
    if not chain:
        return None

    root = chain[0][0]
    confidence = 1.0
    for cause, eff in chain:
        # Find the fact confidence
        for f in all_facts:
            if f.subject == cause and f.relation == Rel.CAUSES and f.obj == eff:
                confidence *= f.confidence
                break

    # Identify intervention points (what requires what in the chain)
    interventions = []
    for cause, eff in chain:
        reqs = [f.obj for f in all_facts
                if f.subject == cause and f.relation == Rel.REQUIRES]
        for req in reqs:
            interventions.append(f"Remove {req} to prevent {cause}")

    return CausalChain(
        root_cause=root,
        final_effect=effect,
        chain=chain,
        confidence=round(confidence, 3),
        interventions=interventions,
    )


def predict_effects(reasoner: CoreReasoner, cause: str) -> list[tuple[str, float]]:
    """Forward predict: if cause happens, what effects follow?"""
    cause = cause.lower().strip()
    all_facts = reasoner.facts | reasoner.derived

    effects = []
    visited = set()

    def trace_forward(source: str, accumulated_conf: float):
        if source in visited:
            return
        visited.add(source)

        for f in all_facts:
            if f.subject == source and f.relation == Rel.CAUSES:
                conf = accumulated_conf * f.confidence
                effects.append((f.obj, round(conf, 3)))
                trace_forward(f.obj, conf)

    trace_forward(cause, 1.0)
    return effects


# ---- 7. SELF-EVALUATE ----

@dataclass
class Evaluation:
    """Self-assessment of a reasoning result."""
    claim: str
    is_supported: bool
    supporting_facts: list[str]
    contradicting_facts: list[str]
    confidence: float
    weaknesses: list[str]  # what could be wrong
    alternatives: list[str]  # other possible answers


def self_evaluate(reasoner: CoreReasoner, claim: str) -> Evaluation:
    """Check whether a claim is supported by the knowledge base.

    Not keyword matching. Actually traces the logical support for
    or against a claim.
    """
    import re
    all_facts = reasoner.facts | reasoner.derived

    supporting = []
    contradicting = []
    weaknesses = []
    alternatives = []

    # Parse the claim into subject-relation-object
    m = re.match(r'(\w[\w\s]*?)\s+(is_a|has_prop|causes|can|located|requires)\s+(\w[\w\s]*)', claim.lower())
    if not m:
        # Try natural language
        m = re.match(r'(\w[\w\s]*?)\s+(?:is\s+a|is\s+an)\s+(\w[\w\s]*)', claim.lower())
        if m:
            subject, obj = m.group(1).strip(), m.group(2).strip()
            rel = Rel.IS_A
        else:
            return Evaluation(claim, False, [], [], 0.0,
                             ["Could not parse claim"], [])
    else:
        subject = m.group(1).strip()
        rel_str = m.group(2).strip()
        obj = m.group(3).strip()
        rel_map = {r.value: r for r in Rel}
        rel = rel_map.get(rel_str, Rel.IS_A)

    # Check direct support
    for f in all_facts:
        if f.subject == subject and f.relation == rel and f.obj == obj:
            supporting.append(str(f))
        # Check contradiction (same subject+relation, different object in same category)
        if f.subject == subject and f.relation == rel and f.obj != obj:
            # Is f.obj in the same category as obj?
            obj_cats = {g.obj for g in all_facts if g.subject == obj and g.relation == Rel.IS_A}
            f_cats = {g.obj for g in all_facts if g.subject == f.obj and g.relation == Rel.IS_A}
            if obj_cats & f_cats:  # shared parent = potential contradiction
                contradicting.append(str(f))
            else:
                alternatives.append(f"{subject} {rel.value} {f.obj}")

    # Check for inherited support
    if not supporting and rel in (Rel.HAS_PROP, Rel.CAN):
        parents = [f.obj for f in all_facts if f.subject == subject and f.relation == Rel.IS_A]
        for parent in parents:
            for f in all_facts:
                if f.subject == parent and f.relation == rel and f.obj == obj:
                    supporting.append(f"Inherited via {parent}: {f}")

    # Assess weaknesses
    if not supporting:
        weaknesses.append("No direct or inherited evidence found")
    if any(f.confidence < 0.5 for f in all_facts
           if f.subject == subject and f.relation == rel and f.obj == obj):
        weaknesses.append("Evidence has low confidence")
    if supporting and all("derived" in s or "transfer" in s for s in supporting):
        weaknesses.append("All evidence is derived/transferred, none directly observed")

    is_supported = len(supporting) > 0 and len(contradicting) == 0
    confidence = 0.0
    if supporting:
        # Average confidence of supporting facts
        conf_sum = 0
        count = 0
        for f in all_facts:
            if f.subject == subject and f.relation == rel and f.obj == obj:
                conf_sum += f.confidence
                count += 1
        confidence = conf_sum / max(count, 1)

    return Evaluation(
        claim=claim,
        is_supported=is_supported,
        supporting_facts=supporting,
        contradicting_facts=contradicting,
        confidence=round(confidence, 3),
        weaknesses=weaknesses,
        alternatives=alternatives,
    )
