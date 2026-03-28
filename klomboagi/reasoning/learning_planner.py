"""
Learning Planner — goal-directed autonomous learning.

Instead of waiting for the human to say "learn about X", the system:
1. Identifies what it doesn't know (from metacognition + gaps)
2. Prioritizes what's most valuable to learn
3. Plans a learning sequence (prerequisites first)
4. Executes: search, read, extract, store, verify
5. Evaluates: did I actually learn it? What's next?

The loop: assess → plan → learn → verify → reassess

This is the difference between a reactive system and an autonomous learner.
"I want to understand quantum mechanics" →
  "First I need to understand waves" →
  "First I need to understand oscillation" →
  learns oscillation → learns waves → learns quantum mechanics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LearningStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class LearningGoal:
    """Something the system wants to learn."""
    topic: str
    reason: str                              # why learn this?
    priority: float = 0.5                    # 0-1
    prerequisites: list[str] = field(default_factory=list)
    status: LearningStatus = LearningStatus.PLANNED
    facts_before: int = 0
    facts_after: int = 0
    attempts: int = 0
    max_attempts: int = 3

    @property
    def facts_gained(self) -> int:
        return self.facts_after - self.facts_before

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "reason": self.reason,
            "priority": round(self.priority, 3),
            "prerequisites": self.prerequisites,
            "status": self.status.value,
            "facts_gained": self.facts_gained,
            "attempts": self.attempts,
        }


@dataclass
class LearningPlan:
    """An ordered sequence of learning goals."""
    goals: list[LearningGoal] = field(default_factory=list)
    completed: list[LearningGoal] = field(default_factory=list)

    def next_goal(self) -> LearningGoal | None:
        """Get the next goal whose prerequisites are met."""
        completed_topics = {g.topic for g in self.completed}
        for goal in self.goals:
            if goal.status == LearningStatus.PLANNED:
                # Check prerequisites
                prereqs_met = all(p in completed_topics for p in goal.prerequisites)
                if prereqs_met:
                    return goal
        return None

    def complete(self, topic: str) -> None:
        for i, g in enumerate(self.goals):
            if g.topic == topic:
                g.status = LearningStatus.COMPLETED
                self.completed.append(self.goals.pop(i))
                return

    def add(self, goal: LearningGoal) -> None:
        # Don't duplicate
        existing = {g.topic for g in self.goals} | {g.topic for g in self.completed}
        if goal.topic not in existing:
            self.goals.append(goal)

    def summary(self) -> str:
        lines = [f"Learning Plan ({len(self.goals)} pending, {len(self.completed)} done):"]
        for g in self.goals:
            prereq_str = f" [needs: {', '.join(g.prerequisites)}]" if g.prerequisites else ""
            lines.append(f"  {'→' if g.status == LearningStatus.IN_PROGRESS else '○'} "
                        f"{g.topic} ({g.priority:.0%}){prereq_str}")
        if self.completed:
            lines.append(f"\n  Completed:")
            for g in self.completed[-5:]:
                lines.append(f"  ✓ {g.topic} (+{g.facts_gained} facts)")
        return "\n".join(lines)


class LearningPlanner:
    """
    Plans and executes autonomous learning sequences.

    Given a target topic, figures out what to learn first,
    then learns it step by step.
    """

    # Common prerequisite chains
    PREREQUISITE_MAP: dict[str, list[str]] = {
        "quantum mechanics": ["physics", "wave", "energy", "probability"],
        "general relativity": ["physics", "gravity", "geometry", "calculus"],
        "calculus": ["algebra", "function", "limit"],
        "linear algebra": ["algebra", "matrix", "vector"],
        "statistics": ["probability", "mathematics"],
        "organic chemistry": ["chemistry", "atom", "chemical bond"],
        "genetics": ["biology", "dna", "cell"],
        "machine learning": ["statistics", "linear algebra", "algorithm"],
        "cryptography": ["mathematics", "number theory", "algorithm"],
        "game theory": ["mathematics", "economics", "probability"],
        "thermodynamics": ["physics", "energy", "temperature"],
        "electromagnetism": ["physics", "electric charge", "wave"],
        "ecology": ["biology", "evolution", "ecosystem"],
        "macroeconomics": ["economics", "gdp", "inflation"],
        "neuroscience": ["biology", "neuron", "chemistry"],
    }

    def __init__(self) -> None:
        self.plan = LearningPlan()
        self._knowledge_topics: set[str] = set()

    def update_knowledge(self, beliefs: dict) -> None:
        """Update what we already know based on current beliefs."""
        self._knowledge_topics = set()
        for statement, belief in beliefs.items():
            if hasattr(belief, 'subject') and belief.subject:
                self._knowledge_topics.add(belief.subject.lower())

    def plan_learning(self, target: str, reason: str = "",
                      priority: float = 0.7) -> LearningPlan:
        """
        Create a learning plan for a target topic.

        Identifies prerequisites, checks what we already know,
        and orders the learning sequence.
        """
        target_lower = target.lower()

        # Find prerequisites
        prereqs = self.PREREQUISITE_MAP.get(target_lower, [])

        # Filter out what we already know
        unknown_prereqs = [p for p in prereqs if p.lower() not in self._knowledge_topics]

        # Add prerequisites as goals (higher priority — must learn first)
        for prereq in unknown_prereqs:
            self.plan.add(LearningGoal(
                topic=prereq,
                reason=f"prerequisite for {target}",
                priority=priority + 0.1,  # slightly higher than target
            ))

        # Add the target itself
        self.plan.add(LearningGoal(
            topic=target,
            reason=reason or f"learning goal: {target}",
            priority=priority,
            prerequisites=unknown_prereqs,
        ))

        # Sort by priority (prerequisites naturally come first due to higher priority)
        self.plan.goals.sort(key=lambda g: g.priority, reverse=True)

        return self.plan

    def plan_from_gaps(self, beliefs: dict, relations,
                       metacognition=None) -> LearningPlan:
        """
        Auto-generate a learning plan from knowledge gaps.

        Uses metacognition to find weak domains, then plans
        learning to strengthen them.
        """
        self.update_knowledge(beliefs)

        if metacognition:
            priorities = metacognition.identify_learning_priorities(beliefs, relations)
            for p in priorities[:5]:
                # Extract the domain/topic from the priority message
                if "Study more" in p:
                    domain = p.split("Study more ")[1].split(" (")[0]
                    self.plan.add(LearningGoal(
                        topic=domain,
                        reason=p,
                        priority=0.8,
                    ))
                elif "relationships" in p.lower():
                    domain = p.split("in ")[1].split(" (")[0]
                    self.plan.add(LearningGoal(
                        topic=f"{domain} relationships",
                        reason=p,
                        priority=0.6,
                    ))

        return self.plan

    def execute_step(self, genesis) -> dict:
        """
        Execute one learning step — learn the next goal.

        Returns what happened.
        """
        self.update_knowledge(genesis.base._beliefs)

        goal = self.plan.next_goal()
        if not goal:
            return {"status": "no_goals", "message": "All learning goals completed or blocked."}

        goal.status = LearningStatus.IN_PROGRESS
        goal.facts_before = len(genesis.base._beliefs)
        goal.attempts += 1

        # Try to learn via read_and_learn
        try:
            result = genesis.read_and_learn(goal.topic)
            goal.facts_after = len(genesis.base._beliefs)

            if goal.facts_gained > 0:
                goal.status = LearningStatus.COMPLETED
                self.plan.complete(goal.topic)
                return {
                    "status": "learned",
                    "topic": goal.topic,
                    "facts_gained": goal.facts_gained,
                    "message": result,
                }
            elif goal.attempts >= goal.max_attempts:
                goal.status = LearningStatus.SKIPPED
                return {
                    "status": "skipped",
                    "topic": goal.topic,
                    "message": f"Could not learn about {goal.topic} after {goal.attempts} attempts.",
                }
            else:
                goal.status = LearningStatus.PLANNED  # Try again later
                return {
                    "status": "retry",
                    "topic": goal.topic,
                    "message": f"No new facts from {goal.topic}, will retry.",
                }
        except Exception as e:
            goal.status = LearningStatus.BLOCKED
            return {
                "status": "error",
                "topic": goal.topic,
                "message": str(e),
            }

    def execute_all(self, genesis, max_steps: int = 20) -> list[dict]:
        """Execute the entire learning plan autonomously."""
        results = []
        for _ in range(max_steps):
            if not self.plan.next_goal():
                break
            result = self.execute_step(genesis)
            results.append(result)
            if result["status"] in ("no_goals",):
                break
        return results
