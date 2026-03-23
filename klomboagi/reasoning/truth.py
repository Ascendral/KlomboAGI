"""
NARS Truth Value System — adapted from Pei Wang's Non-Axiomatic Logic.

Every belief in KlomboAGI carries a truth value (frequency, confidence):
- frequency: what fraction of evidence supports this
- confidence: how much evidence we have (asymptotically approaches 1.0, never reaches it)

Learning happens through REVISION — combining evidence from multiple
observations. No training phase. No backpropagation. Pure evidence accumulation.

Based on:
- Pei Wang, "Non-Axiomatic Reasoning System" (1993-2025)
- NAL (Non-Axiomatic Logic) truth functions
- Adapted for KlomboAGI's structural reasoning

Key formulas:
  f = w+ / w                    (frequency = positive / total evidence)
  c = w / (w + k)               (confidence = evidence / (evidence + horizon))
  w = k * c / (1 - c)           (inverse: confidence to weight)

The HORIZON parameter (k) controls how much weight future evidence gets.
With k=1: 1 observation → c=0.50, 9 observations → c=0.90, 99 → c=0.99
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# The horizon parameter — how much future evidence could matter
# k=1 is Pei Wang's default. Higher k = more conservative (needs more evidence)
HORIZON = 1.0


@dataclass
class TruthValue:
    """
    A truth value representing evidential support for a belief.

    frequency: [0, 1] — what fraction of evidence supports this
    confidence: (0, 1) — how much evidence we have (never reaches 1.0)
    """
    frequency: float = 0.5      # Default: unknown (50/50)
    confidence: float = 0.0     # Default: no evidence

    def __post_init__(self):
        self.frequency = max(0.0, min(1.0, self.frequency))
        self.confidence = max(0.0, min(0.9999, self.confidence))  # Never exactly 1.0

    @staticmethod
    def from_evidence(positive: int, total: int) -> TruthValue:
        """Create a truth value from evidence counts."""
        if total == 0:
            return TruthValue(0.5, 0.0)
        f = positive / total
        c = w2c(total)
        return TruthValue(f, c)

    @staticmethod
    def from_single_observation(positive: bool = True) -> TruthValue:
        """One observation — weak but real evidence."""
        return TruthValue(1.0 if positive else 0.0, w2c(1))

    @property
    def weight(self) -> float:
        """Total evidence weight."""
        return c2w(self.confidence)

    @property
    def positive_weight(self) -> float:
        """Positive evidence weight."""
        return self.frequency * self.weight

    @property
    def negative_weight(self) -> float:
        """Negative evidence weight."""
        return (1 - self.frequency) * self.weight

    @property
    def expectation(self) -> float:
        """Expected truth — combines frequency and confidence.
        Used for decision-making: how much should I trust this?"""
        return self.confidence * (self.frequency - 0.5) + 0.5

    def is_positive(self) -> bool:
        """Is this more true than false?"""
        return self.frequency > 0.5

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Do we have enough evidence?"""
        return self.confidence >= threshold

    def __repr__(self) -> str:
        return f"<f={self.frequency:.2f}, c={self.confidence:.2f}>"

    def to_dict(self) -> dict:
        return {"frequency": round(self.frequency, 4),
                "confidence": round(self.confidence, 4)}

    @staticmethod
    def from_dict(d: dict) -> TruthValue:
        return TruthValue(d.get("frequency", 0.5), d.get("confidence", 0.0))


# ── Core conversion functions ──

def w2c(w: float) -> float:
    """Evidence weight to confidence. c = w / (w + k)"""
    return w / (w + HORIZON)

def c2w(c: float) -> float:
    """Confidence to evidence weight. w = k * c / (1 - c)"""
    if c >= 0.9999:
        return 10000.0  # Practical cap
    return HORIZON * c / (1 - c)


# ── Inference truth functions ──
# These compute the truth value of a conclusion from its premises.

def revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Combine two INDEPENDENT pieces of evidence about the same statement.
    This is THE learning mechanism — weak observations accumulate into strong beliefs.

    ⚠️ Premises must have independent evidence (check stamps to prevent double-counting)
    """
    w1 = c2w(tv1.confidence)
    w2 = c2w(tv2.confidence)
    w = w1 + w2
    if w == 0:
        return TruthValue(0.5, 0.0)
    f = (w1 * tv1.frequency + w2 * tv2.frequency) / w
    c = w2c(w)
    return TruthValue(f, c)


def deduction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Strong inference: A→B, B→C ∴ A→C
    Confidence bounded by product of inputs — chain only as strong as weakest link.
    """
    f = tv1.frequency * tv2.frequency
    c = tv1.frequency * tv2.frequency * tv1.confidence * tv2.confidence
    return TruthValue(f, c)


def induction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Weak inference: A→B, A→C ∴ B→C
    Low confidence — this is a GUESS based on shared cause.
    """
    f = tv2.frequency
    c = w2c(tv1.frequency * tv1.confidence * tv2.confidence)
    return TruthValue(f, c)


def abduction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Weak inference: A→C, B→C ∴ A→B
    Low confidence — this is a GUESS based on shared effect.
    """
    f = tv1.frequency
    c = w2c(tv2.frequency * tv1.confidence * tv2.confidence)
    return TruthValue(f, c)


def analogy(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    A→B, B↔C ∴ A→C
    Transfer through similarity — used for structural transfer.
    """
    f = tv1.frequency * tv2.frequency
    c = tv1.confidence * tv2.confidence * tv2.frequency
    return TruthValue(f, c)


def comparison(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    A→B, A→C ∴ B↔C
    How similar are B and C given they share cause A?
    """
    f0 = tv1.frequency * tv2.frequency
    f1 = tv1.frequency + tv2.frequency - f0
    f = f0 / f1 if f1 > 0 else 0.0
    c = w2c(f1 * tv1.confidence * tv2.confidence)
    return TruthValue(f, c)


def negation(tv: TruthValue) -> TruthValue:
    """Negate: if A has truth (f, c), NOT A has truth (1-f, c)."""
    return TruthValue(1.0 - tv.frequency, tv.confidence)


def conversion(tv: TruthValue) -> TruthValue:
    """Convert: if A→B has truth (f, c), then B→A has truth (1, f*c/(f*c+k))."""
    f = 1.0
    c = w2c(tv.frequency * tv.confidence)
    return TruthValue(f, c)


def temporal_projection(tv: TruthValue, time_distance: float,
                        decay_rate: float = 0.5) -> TruthValue:
    """
    Project a belief's confidence across time.
    Older beliefs have less confidence — the world changes.

    decay_rate: how fast confidence decays with time distance.
    0 = no decay, 1 = instant decay.
    """
    if time_distance <= 0:
        return tv
    decay = math.exp(-decay_rate * time_distance)
    return TruthValue(tv.frequency, tv.confidence * decay)


# ── Evidence stamp ──

@dataclass
class EvidenceStamp:
    """
    Tracks the evidential origin of a belief.
    Prevents double-counting — if two premises share evidence sources,
    they cannot be combined via revision.
    """
    sources: frozenset[int]     # Set of original evidence IDs
    creation_time: float = 0.0
    occurrence_time: float = 0.0

    @staticmethod
    def new(source_id: int, time: float = 0.0) -> EvidenceStamp:
        return EvidenceStamp(sources=frozenset([source_id]),
                             creation_time=time, occurrence_time=time)

    def overlaps(self, other: EvidenceStamp) -> bool:
        """Check if two stamps share evidence — if so, can't revise."""
        return bool(self.sources & other.sources)

    def merge(self, other: EvidenceStamp) -> EvidenceStamp:
        """Merge two stamps (union of evidence sources)."""
        return EvidenceStamp(
            sources=self.sources | other.sources,
            creation_time=max(self.creation_time, other.creation_time),
            occurrence_time=max(self.occurrence_time, other.occurrence_time),
        )

    def to_dict(self) -> dict:
        return {"sources": list(self.sources),
                "creation_time": self.creation_time,
                "occurrence_time": self.occurrence_time}

    @staticmethod
    def from_dict(d: dict) -> EvidenceStamp:
        return EvidenceStamp(
            sources=frozenset(d.get("sources", [])),
            creation_time=d.get("creation_time", 0.0),
            occurrence_time=d.get("occurrence_time", 0.0),
        )


# ── Belief ──

@dataclass
class Belief:
    """
    A statement with truth value and evidence tracking.
    This is what KlomboAGI stores in its knowledge graph.
    """
    statement: str              # What this belief says
    truth: TruthValue           # How true we think it is
    stamp: EvidenceStamp        # Where the evidence came from
    subject: str = ""           # What entity
    predicate: str = ""         # What property/relation
    source: str = "unknown"     # "human", "self", "search", etc.

    def revise_with(self, other: Belief) -> Belief | None:
        """
        Revise this belief with new evidence.
        Returns None if evidence overlaps (would double-count).
        """
        if self.stamp.overlaps(other.stamp):
            return None  # Can't combine — shared evidence

        new_truth = revision(self.truth, other.truth)
        new_stamp = self.stamp.merge(other.stamp)
        return Belief(
            statement=self.statement,
            truth=new_truth,
            stamp=new_stamp,
            subject=self.subject,
            predicate=self.predicate,
            source=f"revised({self.source}+{other.source})",
        )

    def __repr__(self) -> str:
        return f"Belief('{self.statement}' {self.truth} src={self.source})"

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "truth": self.truth.to_dict(),
            "stamp": self.stamp.to_dict(),
            "subject": self.subject,
            "predicate": self.predicate,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> Belief:
        return Belief(
            statement=d["statement"],
            truth=TruthValue.from_dict(d.get("truth", {})),
            stamp=EvidenceStamp.from_dict(d.get("stamp", {})),
            subject=d.get("subject", ""),
            predicate=d.get("predicate", ""),
            source=d.get("source", "unknown"),
        )
