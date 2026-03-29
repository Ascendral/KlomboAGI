"""
Deep Transfer — apply techniques across domains autonomously.

Not just "physics uses math" (explicit relation).
REAL transfer: "This debugging technique from software engineering
could work for diagnosing medical symptoms because both involve
systematic elimination of hypotheses."

Method:
1. STRUCTURAL MAPPING — find structural parallels between domains
2. PRINCIPLE EXTRACTION — extract the abstract principle from one domain
3. APPLICATION — apply the principle to the target domain
4. VALIDATION — check if the transfer makes sense

Example:
  Source: "In economics, supply and demand reach equilibrium"
  Target: "In ecology, predator and prey populations balance"
  Transfer: "Both are negative feedback systems that self-regulate.
            The principle: opposing forces create stable balance."

This uses the AbstractionEngine + StructuralComparator from the
CognitionLoop — they were built for exactly this purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TransferMapping:
    """A structural parallel between two domains."""
    source_domain: str
    target_domain: str
    source_concept: str
    target_concept: str
    shared_structure: str       # what's the same
    principle: str              # the abstract principle
    confidence: float
    novel: bool = True          # was this discovered, not taught?

    def explain(self) -> str:
        return (
            f"{self.source_concept} ({self.source_domain}) parallels "
            f"{self.target_concept} ({self.target_domain}). "
            f"{self.principle}"
        )

    def to_dict(self) -> dict:
        return {
            "source": f"{self.source_domain}/{self.source_concept}",
            "target": f"{self.target_domain}/{self.target_concept}",
            "principle": self.principle,
            "confidence": round(self.confidence, 3),
        }


class DeepTransferEngine:
    """
    Find and apply structural parallels across domains.

    Scans the relation graph for patterns that repeat across
    different domains — those patterns ARE the transferable principles.
    """

    def __init__(self, relations, beliefs: dict) -> None:
        self.relations = relations
        self.beliefs = beliefs
        self.discovered_transfers: list[TransferMapping] = []

    def scan_all(self) -> list[TransferMapping]:
        """
        Scan the entire knowledge base for cross-domain structural parallels.

        Finds: same relation pattern in different domains.
        """
        transfers = []

        # Group relations by structure: (relation_type, number_of_targets)
        # Two concepts with the same structural signature in different domains = transfer
        structural_signatures: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for rel in self.relations._all:
            # Get domain for source
            source_domain = self._get_domain(rel.source)
            sig = f"{rel.relation.value}"
            structural_signatures[sig].append((rel.source, source_domain))

        # Find cross-domain matches
        for sig, concepts in structural_signatures.items():
            # Group by domain
            by_domain: dict[str, list[str]] = defaultdict(list)
            for concept, domain in concepts:
                if domain:
                    by_domain[domain].append(concept)

            # Cross-domain pairs
            domains = list(by_domain.keys())
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    d1, d2 = domains[i], domains[j]
                    for c1 in by_domain[d1][:3]:
                        for c2 in by_domain[d2][:3]:
                            # Check structural similarity
                            parallel = self._check_structural_parallel(c1, c2, sig)
                            if parallel:
                                transfers.append(parallel)

        # Deduplicate
        seen = set()
        unique = []
        for t in transfers:
            key = f"{t.source_concept}:{t.target_concept}"
            if key not in seen:
                seen.add(key)
                unique.append(t)

        self.discovered_transfers = unique
        return unique

    def transfer(self, source_concept: str, target_domain: str) -> list[TransferMapping]:
        """
        Given a concept from one domain, find what transfers to another domain.

        "What from physics applies to economics?"
        """
        results = []
        source_domain = self._get_domain(source_concept)

        # Get all relations from source concept
        source_rels = self.relations.get_forward(source_concept)
        source_back = self.relations.get_backward(source_concept)

        # Find target domain concepts with similar structure
        for rel in source_rels:
            # Find things in target domain with same relation type
            for target_rel in self.relations._all:
                if (target_rel.relation == rel.relation
                        and self._get_domain(target_rel.source) == target_domain
                        and target_rel.source != rel.source):
                    principle = self._extract_principle(rel, target_rel)
                    results.append(TransferMapping(
                        source_domain=source_domain or "unknown",
                        target_domain=target_domain,
                        source_concept=rel.source,
                        target_concept=target_rel.source,
                        shared_structure=rel.relation.value,
                        principle=principle,
                        confidence=min(rel.confidence, target_rel.confidence) * 0.7,
                    ))

        return results[:10]

    def apply_to_question(self, question: str, concepts: list[str]) -> str | None:
        """
        Try to answer a question by transferring knowledge from another domain.

        If we don't know about X directly, but X is structurally similar
        to Y which we DO know about, transfer what we know about Y.
        """
        for concept in concepts:
            # Check if we have direct knowledge
            has_direct = any(
                hasattr(b, 'subject') and b.subject == concept.lower()
                for b in self.beliefs.values()
            )
            if has_direct:
                continue  # Don't need transfer, we have direct knowledge

            # Find structural parallels from domains we DO know
            for other_concept in list(self.beliefs.keys())[:100]:
                b = self.beliefs[other_concept]
                if not hasattr(b, 'subject'):
                    continue
                subj = b.subject
                if subj == concept.lower():
                    continue

                # Check if this concept has similar relations
                parallel = self._find_parallel_for(concept.lower(), subj)
                if parallel:
                    return (
                        f"I don't know {concept} directly, but it might be similar to "
                        f"{parallel.source_concept}. {parallel.principle}"
                    )

        return None

    def _check_structural_parallel(self, c1: str, c2: str,
                                    relation_type: str) -> TransferMapping | None:
        """Check if two concepts from different domains share structure."""
        d1 = self._get_domain(c1)
        d2 = self._get_domain(c2)
        if not d1 or not d2 or d1 == d2:
            return None

        # Get their full relation profiles
        rels1 = {r.relation.value for r in self.relations.get_forward(c1)}
        rels2 = {r.relation.value for r in self.relations.get_forward(c2)}

        shared = rels1 & rels2
        if len(shared) >= 2:  # At least 2 shared relation types = structural parallel
            principle = f"Both {c1} and {c2} share: {', '.join(shared)}"
            return TransferMapping(
                source_domain=d1, target_domain=d2,
                source_concept=c1, target_concept=c2,
                shared_structure=", ".join(shared),
                principle=principle,
                confidence=0.4 + len(shared) * 0.1,
            )
        return None

    def _find_parallel_for(self, unknown: str, known: str) -> TransferMapping | None:
        """Find if known concept is structurally parallel to unknown."""
        known_rels = self.relations.get_forward(known)
        unknown_rels = self.relations.get_forward(unknown)

        if not known_rels:
            return None

        known_types = {r.relation.value for r in known_rels}
        unknown_types = {r.relation.value for r in unknown_rels}

        shared = known_types & unknown_types
        if shared:
            return TransferMapping(
                source_domain=self._get_domain(known) or "known",
                target_domain=self._get_domain(unknown) or "unknown",
                source_concept=known,
                target_concept=unknown,
                shared_structure=", ".join(shared),
                principle=f"Both share {', '.join(shared)} relations — "
                         f"what applies to {known} might apply to {unknown}.",
                confidence=0.3 + len(shared) * 0.1,
            )
        return None

    def _extract_principle(self, source_rel, target_rel) -> str:
        """Extract the abstract principle from two parallel relations."""
        return (
            f"Just as {source_rel.source} {source_rel.relation.value} {source_rel.target}, "
            f"{target_rel.source} {target_rel.relation.value} {target_rel.target}. "
            f"The principle: {source_rel.relation.value} is a pattern that repeats across domains."
        )

    def _get_domain(self, concept: str) -> str | None:
        """Determine which domain a concept belongs to."""
        from klomboagi.core.curriculum import CURRICULA
        concept_lower = concept.lower()
        for domain, facts in CURRICULA.items():
            for subj, _ in facts:
                if subj.lower() == concept_lower:
                    return domain
        # Check relations for domain tags
        for rel in self.relations.get_all_about(concept_lower)[:5]:
            if rel.source_domain and rel.source_domain not in ("conversation", "inference", "document"):
                return rel.source_domain
        return None

    def report(self) -> str:
        """Report discovered cross-domain transfers."""
        if not self.discovered_transfers:
            return "No cross-domain transfers discovered yet. Run scan_all() first."
        lines = [f"Cross-Domain Transfers ({len(self.discovered_transfers)}):"]
        for t in self.discovered_transfers[:15]:
            lines.append(f"  {t.source_domain}/{t.source_concept} ↔ "
                        f"{t.target_domain}/{t.target_concept}")
            lines.append(f"    Shared: {t.shared_structure}")
        return "\n".join(lines)
