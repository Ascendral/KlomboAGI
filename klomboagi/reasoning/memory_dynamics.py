"""
Memory Dynamics — stolen from ACT-R, NARS, and SOAR.

Three mechanisms that every serious cognitive architecture has:

1. TEMPORAL DECAY (ACT-R)
   B = ln(Σ t_i^(-0.5))
   Memories fade with power law. Recently and frequently accessed
   items stay strong. Old unused items decay toward zero.
   Most empirically validated equation in cognitive science.

2. CONFLICT PRIORITY BOOST (NARS)
   When two beliefs disagree, that disagreement becomes the
   HIGHEST priority signal. Not just a "surprise" — it should
   dominate attention until resolved. Contradictions are the
   most valuable information the system can encounter.

3. CHUNKING (SOAR)
   When the system reasons through a multi-step chain to reach
   a conclusion, compile that chain into a single direct rule.
   Next time the same pattern appears, skip the reasoning —
   fire the compiled rule instantly. This is how expertise develops.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


# ═══════════════════════════════════════
# 1. TEMPORAL DECAY (ACT-R base-level activation)
# ═══════════════════════════════════════

class ActivationDecay:
    """
    ACT-R base-level activation: B = ln(Σ t_i^(-d))

    Each access creates a trace. Activation is the log of
    the sum of temporal decays across all traces.

    d = 0.5 (ACT-R default decay parameter)
    """

    DECAY_PARAM = 0.5  # ACT-R's empirically validated decay rate

    def __init__(self) -> None:
        self._traces: dict[str, list[float]] = {}  # concept → list of access timestamps

    def access(self, concept: str) -> None:
        """Record an access to a concept."""
        now = time.time()
        if concept not in self._traces:
            self._traces[concept] = []
        self._traces[concept].append(now)
        # Keep last 50 traces per concept
        if len(self._traces[concept]) > 50:
            self._traces[concept] = self._traces[concept][-50:]

    def activation(self, concept: str) -> float:
        """
        Compute base-level activation for a concept.

        B = ln(Σ t_i^(-d))

        Higher = more accessible. Lower = harder to retrieve.
        """
        traces = self._traces.get(concept, [])
        if not traces:
            return -10.0  # Very low — never accessed

        now = time.time()
        total = 0.0
        for t in traces:
            age = max(0.1, now - t)  # seconds since access
            total += age ** (-self.DECAY_PARAM)

        if total <= 0:
            return -10.0
        return math.log(total)

    def retrievable(self, concept: str, threshold: float = -2.0) -> bool:
        """Can this concept be retrieved? (above threshold)"""
        return self.activation(concept) > threshold

    def top_active(self, n: int = 10) -> list[tuple[str, float]]:
        """Most active concepts right now."""
        activations = [(c, self.activation(c)) for c in self._traces]
        activations.sort(key=lambda x: x[1], reverse=True)
        return activations[:n]


# ═══════════════════════════════════════
# 2. CONFLICT PRIORITY BOOST (NARS)
# ═══════════════════════════════════════

@dataclass
class Conflict:
    """A detected conflict between beliefs."""
    belief_a: str
    belief_b: str
    severity: float      # 0-1, how bad is this conflict
    detected_at: float = 0.0
    resolved: bool = False
    resolution: str = ""

    @property
    def priority(self) -> float:
        """Conflicts get BOOSTED priority — they're the most important signal."""
        age = time.time() - self.detected_at if self.detected_at else 0
        # Priority decays slowly — conflicts stay important
        decay = max(0.1, 1.0 - age / 3600)  # 1 hour half-life
        return self.severity * decay * 2.0  # 2x boost over normal priority


class ConflictDetector:
    """
    Detects and prioritizes conflicts in the belief system.

    When beliefs disagree, that disagreement becomes the
    HIGHEST priority signal — above curiosity, above boredom,
    above everything. Contradictions must be resolved.
    """

    def __init__(self) -> None:
        self.active_conflicts: list[Conflict] = []
        self.resolved_conflicts: list[Conflict] = []

    def check(self, beliefs: dict) -> list[Conflict]:
        """Scan beliefs for conflicts. Returns new conflicts found."""
        new_conflicts = []

        belief_list = list(beliefs.values())
        for i, b1 in enumerate(belief_list):
            if not hasattr(b1, 'subject') or not b1.subject:
                continue
            for b2 in belief_list[i+1:]:
                if not hasattr(b2, 'subject') or not b2.subject:
                    continue
                if b1.subject == b2.subject and b1.predicate != b2.predicate:
                    # Same subject, different predicates — potential conflict
                    # Only flag if both have decent confidence
                    if b1.truth.confidence > 0.3 and b2.truth.confidence > 0.3:
                        # Check if already known
                        known = any(
                            c.belief_a == b1.statement and c.belief_b == b2.statement
                            for c in self.active_conflicts + self.resolved_conflicts
                        )
                        if not known:
                            conflict = Conflict(
                                belief_a=b1.statement,
                                belief_b=b2.statement,
                                severity=min(b1.truth.confidence, b2.truth.confidence),
                                detected_at=time.time(),
                            )
                            new_conflicts.append(conflict)
                            self.active_conflicts.append(conflict)

        return new_conflicts

    def highest_priority(self) -> Conflict | None:
        """The most urgent unresolved conflict."""
        unresolved = [c for c in self.active_conflicts if not c.resolved]
        if not unresolved:
            return None
        return max(unresolved, key=lambda c: c.priority)

    def resolve(self, belief_a: str, resolution: str) -> None:
        """Mark a conflict as resolved."""
        for c in self.active_conflicts:
            if c.belief_a == belief_a and not c.resolved:
                c.resolved = True
                c.resolution = resolution
                self.resolved_conflicts.append(c)
                break
        self.active_conflicts = [c for c in self.active_conflicts if not c.resolved]


# ═══════════════════════════════════════
# 3. CHUNKING (SOAR)
# ═══════════════════════════════════════

@dataclass
class Chunk:
    """
    A compiled reasoning rule.

    Instead of re-deriving A→B→C→D every time,
    store: IF A THEN D (with confidence from the chain).
    """
    condition: str        # what triggers this chunk
    conclusion: str       # what it produces
    chain: list[str]      # the original reasoning chain
    confidence: float     # product of chain confidences
    use_count: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "conclusion": self.conclusion,
            "chain_length": len(self.chain),
            "confidence": round(self.confidence, 3),
            "use_count": self.use_count,
        }


class ChunkCompiler:
    """
    Compiles multi-step reasoning chains into instant rules.

    When the system reasons: gravity → force → acceleration
    It creates a chunk: IF gravity THEN acceleration (confidence: X)

    Next time someone asks about gravity and acceleration,
    the chunk fires INSTANTLY instead of re-deriving.

    This is how expertise develops — deliberate becomes automatic.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}  # condition → chunk

    def compile(self, chain: list[str], conclusion: str,
                confidence: float) -> Chunk:
        """Compile a reasoning chain into a chunk."""
        if not chain:
            return None

        condition = chain[0]
        key = f"{condition}→{conclusion}"

        if key in self.chunks:
            # Strengthen existing chunk
            self.chunks[key].use_count += 1
            self.chunks[key].confidence = min(0.99,
                self.chunks[key].confidence + 0.05)
            return self.chunks[key]

        chunk = Chunk(
            condition=condition,
            conclusion=conclusion,
            chain=chain,
            confidence=confidence,
            created_at=time.time(),
        )
        self.chunks[key] = chunk
        return chunk

    def compile_from_path(self, relations_path: list) -> Chunk | None:
        """Compile a relation path into a chunk."""
        if len(relations_path) < 2:
            return None

        chain = [r.source for r in relations_path] + [relations_path[-1].target]
        confidence = 1.0
        for r in relations_path:
            confidence *= r.confidence

        return self.compile(chain, chain[-1], confidence)

    def lookup(self, condition: str) -> list[Chunk]:
        """Find chunks that fire for a given condition."""
        results = []
        condition_lower = condition.lower()
        for key, chunk in self.chunks.items():
            if chunk.condition.lower() in condition_lower or condition_lower in chunk.condition.lower():
                chunk.use_count += 1
                results.append(chunk)
        return results

    def stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "most_used": sorted(
                [(c.condition, c.conclusion, c.use_count) for c in self.chunks.values()],
                key=lambda x: x[2], reverse=True
            )[:5],
        }
