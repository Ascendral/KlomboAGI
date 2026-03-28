"""
Learning Drive — the persistent engine that never stops.

Give it a goal like "learn everything about physics" and it will:
1. Assess what it currently knows
2. Find the biggest gaps
3. Plan a learning sequence
4. Execute: read articles, extract facts, build relations
5. Reassess — what new gaps emerged from what it learned?
6. Repeat forever

It's not a one-shot process. Every answer creates new questions.
Every article read reveals more unknowns. The system is DRIVEN
to keep filling gaps — that's the persistence trait made real.

Usage:
    drive = LearningDrive(genesis)
    drive.set_mission("learn everything about physics")
    drive.run(max_cycles=50)  # or run forever
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DriveCycle:
    """One cycle of the learning drive."""
    cycle_number: int
    topic_learned: str
    facts_before: int
    facts_after: int
    relations_before: int
    relations_after: int
    new_gaps: list[str]
    duration_seconds: float

    @property
    def facts_gained(self) -> int:
        return self.facts_after - self.facts_before

    @property
    def relations_gained(self) -> int:
        return self.relations_after - self.relations_before

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle_number,
            "topic": self.topic_learned,
            "facts_gained": self.facts_gained,
            "relations_gained": self.relations_gained,
            "new_gaps": self.new_gaps,
            "duration": round(self.duration_seconds, 1),
        }


@dataclass
class DriveReport:
    """Summary of a learning drive session."""
    mission: str
    cycles_completed: int
    total_facts_gained: int
    total_relations_gained: int
    topics_learned: list[str]
    remaining_gaps: list[str]
    total_duration: float

    def summary(self) -> str:
        lines = [
            f"Learning Drive Report: {self.mission}",
            f"  Cycles: {self.cycles_completed}",
            f"  Facts gained: +{self.total_facts_gained}",
            f"  Relations gained: +{self.total_relations_gained}",
            f"  Duration: {self.total_duration:.0f}s",
            f"  Topics: {', '.join(self.topics_learned[:10])}",
        ]
        if self.remaining_gaps:
            lines.append(f"  Remaining gaps ({len(self.remaining_gaps)}):")
            for g in self.remaining_gaps[:10]:
                lines.append(f"    ? {g}")
        return "\n".join(lines)


class LearningDrive:
    """
    The engine that never stops learning.

    Persistent. Autonomous. Always finding the next gap to fill.
    """

    def __init__(self, genesis) -> None:
        self.genesis = genesis
        self.mission: str = ""
        self.cycles: list[DriveCycle] = []
        self._explored_topics: set[str] = set()
        self._gap_queue: list[str] = []

        # Callbacks
        self.on_cycle: Callable[[DriveCycle], None] | None = None
        self.on_gap: Callable[[str], None] | None = None

    def set_mission(self, mission: str) -> None:
        """Set the high-level learning mission."""
        self.mission = mission
        # Extract seed topics from the mission
        stop = {"learn", "everything", "about", "all", "the", "and", "of", "in", "a", "an"}
        words = [w.lower() for w in mission.split() if w.lower() not in stop and len(w) > 2]
        self._gap_queue = words

    def run(self, max_cycles: int = 50) -> DriveReport:
        """
        Run the learning drive — keep learning until max_cycles or no gaps left.

        Each cycle:
        1. Pick the highest-priority gap
        2. Read and learn about it
        3. Extract new gaps from what we learned
        4. Add new gaps to the queue
        5. Report and continue
        """
        start_time = time.time()
        facts_start = len(self.genesis.base._beliefs)
        rels_start = self.genesis.relations.stats()["total_relations"]
        topics_learned = []

        for cycle_num in range(max_cycles):
            if not self._gap_queue:
                # Generate gaps from metacognition
                self._generate_gaps()

            if not self._gap_queue:
                # Self-model check: if existence value is high, try HARDER
                if hasattr(self.genesis, 'self_model'):
                    ev = self.genesis.self_model.existence_value()
                    if ev > 100:
                        # High existence value = lots to lose by stopping
                        # Generate broader gaps
                        self._generate_broad_gaps()
                if not self._gap_queue:
                    break  # Truly nothing left

            # Pick next topic
            topic = self._gap_queue.pop(0)

            # Skip if already explored
            if topic in self._explored_topics:
                continue
            self._explored_topics.add(topic)

            # Execute one learning cycle
            cycle = self._execute_cycle(cycle_num, topic)
            self.cycles.append(cycle)
            topics_learned.append(topic)

            if self.on_cycle:
                self.on_cycle(cycle)

            # Add new gaps to queue
            for gap in cycle.new_gaps:
                if gap not in self._explored_topics and gap not in self._gap_queue:
                    self._gap_queue.append(gap)
                    if self.on_gap:
                        self.on_gap(gap)

        # Final stats
        facts_end = len(self.genesis.base._beliefs)
        rels_end = self.genesis.relations.stats()["total_relations"]

        return DriveReport(
            mission=self.mission,
            cycles_completed=len(self.cycles),
            total_facts_gained=facts_end - facts_start,
            total_relations_gained=rels_end - rels_start,
            topics_learned=topics_learned,
            remaining_gaps=self._gap_queue[:20],
            total_duration=time.time() - start_time,
        )

    def _execute_cycle(self, cycle_num: int, topic: str) -> DriveCycle:
        """Execute one learning cycle for a topic."""
        start = time.time()
        facts_before = len(self.genesis.base._beliefs)
        rels_before = self.genesis.relations.stats()["total_relations"]

        # Read and learn
        self.genesis.read_and_learn(topic)

        facts_after = len(self.genesis.base._beliefs)
        rels_after = self.genesis.relations.stats()["total_relations"]

        # Find new gaps — concepts mentioned in new beliefs that we don't have
        new_gaps = self._find_new_gaps(facts_before)

        return DriveCycle(
            cycle_number=cycle_num,
            topic_learned=topic,
            facts_before=facts_before,
            facts_after=facts_after,
            relations_before=rels_before,
            relations_after=rels_after,
            new_gaps=new_gaps,
            duration_seconds=time.time() - start,
        )

    def _find_new_gaps(self, facts_before_count: int) -> list[str]:
        """Find concepts in newly learned facts that we should explore next."""
        gaps = []
        all_beliefs = list(self.genesis.base._beliefs.values())
        new_beliefs = all_beliefs[facts_before_count:]

        known_subjects = {b.subject for b in all_beliefs if hasattr(b, 'subject') and b.subject}

        for belief in new_beliefs:
            if not hasattr(belief, 'predicate') or not belief.predicate:
                continue
            # Extract potential topics from predicates
            pred_words = belief.predicate.lower().split()
            # Filter: must be a real concept (not common English, not a modifier)
            skip_words = {
                "oldest", "largest", "smallest", "greatest", "fastest",
                "often", "usually", "always", "never", "sometimes",
                "about", "around", "between", "through", "within",
                "called", "known", "based", "given", "described",
                "studied", "applied", "developed", "discovered",
                "important", "significant", "fundamental", "modern",
                "classical", "general", "special", "various", "several",
                "natural", "physical", "mathematical", "scientific",
                "different", "similar", "related", "other", "certain",
                "academic", "private", "public", "social", "human",
                "employed", "independent", "separate", "concerned",
                "sector", "discipline", "field", "branch", "area",
                "stipulated", "properties", "axioms",
            }
            for word in pred_words:
                if (len(word) > 5
                        and word not in self.genesis.base.COMMON_WORDS
                        and word not in skip_words
                        and word not in known_subjects
                        and word not in self._explored_topics
                        and not word.endswith(("ly", "ed", "ing", "tion", "ness"))):
                    gaps.append(word)

        # Deduplicate, cap at 5 new gaps per cycle
        seen = set()
        unique = []
        for g in gaps:
            if g not in seen:
                seen.add(g)
                unique.append(g)
        return unique[:5]

    def _generate_gaps(self) -> None:
        """Generate new learning targets from metacognition."""
        priorities = self.genesis.metacognition.identify_learning_priorities(
            self.genesis.base._beliefs, self.genesis.relations)

        # Extract DOMAIN NAMES from priorities, not random words
        meta_noise = {"study", "more", "learn", "only", "currently", "critical",
                      "gaps", "detected", "continue", "broadly", "relationships",
                      "improve", "answer", "rate", "reduce", "errors", "corrections",
                      "received", "confidence", "relations"}
        for p in priorities:
            words = [w for w in p.lower().split()
                    if len(w) > 4 and w not in meta_noise
                    and not w.endswith(("ing", "tion", "ness", "ment", "ally"))]
            for w in words:
                if w not in self._explored_topics and w not in self._gap_queue:
                    self._gap_queue.append(w)

    def _generate_broad_gaps(self) -> None:
        """
        When normal gap generation fails, look broader.
        Driven by self-model: high existence value = must keep learning.

        Finds concepts mentioned in relations that we don't have
        beliefs about — structural holes in the knowledge graph.
        """
        if not hasattr(self.genesis, 'relations'):
            return

        known_subjects = {b.subject for b in self.genesis.base._beliefs.values()
                         if hasattr(b, 'subject') and b.subject}

        for rel in self.genesis.relations._all:
            for concept in (rel.source, rel.target):
                if (concept not in known_subjects
                        and concept not in self._explored_topics
                        and concept not in self._gap_queue
                        and len(concept) > 3):
                    self._gap_queue.append(concept)
                    if len(self._gap_queue) >= 10:
                        return
