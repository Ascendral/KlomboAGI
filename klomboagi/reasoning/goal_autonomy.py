"""
Goal Autonomy — the system sets its OWN goals and pursues them.

Not "human tells it what to learn." The system decides:
  "I'm weak in chemistry. I should learn more."
  "I keep failing questions about economics. That's my priority."
  "I noticed a gap between physics and biology. I should bridge it."

The goal formation cycle:
  1. ASSESS — what am I weak at? (from metacognition)
  2. FORMULATE — set a specific, measurable goal
  3. PLAN — break goal into steps
  4. EXECUTE — take the steps
  5. EVALUATE — did I achieve the goal?
  6. NEXT — what's the next goal?

Goals come from:
  - Knowledge gaps (metacognition identifies weak domains)
  - Failed questions (experiential learner tracks failures)
  - Structural holes (concepts with no relations)
  - Human interests (conversation memory tracks what human cares about)
  - Self-model (existence value drives growth)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AutonomousGoal:
    """A goal the system set for itself."""
    description: str
    reason: str              # why this goal was chosen
    source: str              # "weakness", "failure", "structural", "human_interest", "growth"
    priority: float = 0.5    # 0-1
    measurable: str = ""     # how to know if achieved
    steps: list[str] = field(default_factory=list)
    progress: float = 0.0
    status: str = "active"   # "active", "completed", "abandoned"
    created_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "reason": self.reason[:60],
            "source": self.source,
            "priority": round(self.priority, 2),
            "progress": round(self.progress, 2),
            "status": self.status,
        }


class GoalAutonomy:
    """
    Autonomous goal formation and pursuit.

    The system decides what to learn without being told.
    """

    MAX_ACTIVE_GOALS = 5

    def __init__(self, genesis) -> None:
        self.g = genesis
        self.goals: list[AutonomousGoal] = []

    def formulate_goals(self) -> list[AutonomousGoal]:
        """
        Assess current state and formulate goals autonomously.
        """
        new_goals = []

        # 1. From weaknesses (metacognition)
        if hasattr(self.g, 'metacognition'):
            priorities = self.g.metacognition.identify_learning_priorities(
                self.g.base._beliefs, self.g.relations)
            for p in priorities[:2]:
                new_goals.append(AutonomousGoal(
                    description=p,
                    reason="Metacognition identified this as a weakness",
                    source="weakness",
                    priority=0.8,
                    measurable="Confidence in this domain increases",
                    created_at=time.time(),
                ))

        # 2. From failed questions (experiential)
        if hasattr(self.g, 'experiential'):
            failed = [a for a in self.g.experiential.attempts if a.confidence < 0.3]
            topics = set()
            for a in failed[-10:]:
                words = [w for w in a.question.lower().split()
                        if len(w) > 4 and w not in {"what", "about", "does", "would"}]
                topics.update(words[:2])
            for topic in list(topics)[:2]:
                new_goals.append(AutonomousGoal(
                    description=f"Learn about {topic}",
                    reason=f"Failed to answer questions about {topic}",
                    source="failure",
                    priority=0.7,
                    measurable=f"Can answer questions about {topic}",
                    created_at=time.time(),
                ))

        # 3. From structural holes (concepts with beliefs but no relations)
        if hasattr(self.g, 'relations'):
            orphans = []
            for stmt, belief in list(self.g.base._beliefs.items())[:200]:
                if hasattr(belief, 'subject') and belief.subject:
                    rels = self.g.relations.get_all_about(belief.subject)
                    if not rels and len(belief.subject) > 3:
                        orphans.append(belief.subject)
            if orphans:
                new_goals.append(AutonomousGoal(
                    description=f"Connect {len(orphans)} isolated concepts to the knowledge graph",
                    reason=f"Found {len(orphans)} concepts with no relations",
                    source="structural",
                    priority=0.5,
                    steps=[f"Find relations for {c}" for c in orphans[:5]],
                    measurable="Orphan count decreases",
                    created_at=time.time(),
                ))

        # 4. From human interests (conversation memory)
        if hasattr(self.g, 'conversation_memory'):
            interests = self.g.conversation_memory.get_human_interests()
            unanswered = self.g.conversation_memory.get_unanswered()
            for q in unanswered[:2]:
                new_goals.append(AutonomousGoal(
                    description=f"Answer: {q}",
                    reason="Human asked this and I couldn't answer",
                    source="human_interest",
                    priority=0.9,  # Highest — human wanted this
                    measurable=f"Can answer: {q}",
                    created_at=time.time(),
                ))

        # Deduplicate and limit
        existing = {g.description for g in self.goals}
        for goal in new_goals:
            if goal.description not in existing:
                self.goals.append(goal)

        # Keep only top MAX_ACTIVE_GOALS
        active = [g for g in self.goals if g.status == "active"]
        active.sort(key=lambda g: g.priority, reverse=True)
        if len(active) > self.MAX_ACTIVE_GOALS:
            for g in active[self.MAX_ACTIVE_GOALS:]:
                g.status = "abandoned"

        return new_goals

    def pursue_next(self) -> dict:
        """Pursue the highest priority active goal."""
        active = [g for g in self.goals if g.status == "active"]
        if not active:
            self.formulate_goals()
            active = [g for g in self.goals if g.status == "active"]

        if not active:
            return {"status": "no_goals", "message": "No goals to pursue."}

        goal = max(active, key=lambda g: g.priority)

        # Execute based on source
        if goal.source in ("weakness", "failure"):
            topic = goal.description.replace("Learn about ", "").replace("Study more ", "")
            topic = topic.split("(")[0].strip()
            try:
                result = self.g.read_and_learn(topic)
                if "Could not read" not in result:
                    goal.progress = min(1.0, goal.progress + 0.5)
                    if goal.progress >= 1.0:
                        goal.status = "completed"
                        goal.completed_at = time.time()
                    return {"status": "progressed", "goal": goal.description, "result": result[:100]}
            except Exception:
                pass

        elif goal.source == "human_interest":
            question = goal.description.replace("Answer: ", "")
            result = self.g._active_learn(question)
            goal.progress = 0.5
            return {"status": "researched", "goal": goal.description, "result": result[:100]}

        elif goal.source == "structural":
            # Try to find relations for orphan concepts
            goal.progress = min(1.0, goal.progress + 0.2)
            if goal.progress >= 1.0:
                goal.status = "completed"
            return {"status": "connecting", "goal": goal.description}

        return {"status": "attempted", "goal": goal.description}

    def report(self) -> str:
        """Report on autonomous goals."""
        lines = ["Autonomous Goals:"]
        active = [g for g in self.goals if g.status == "active"]
        completed = [g for g in self.goals if g.status == "completed"]

        if active:
            lines.append(f"\n  Active ({len(active)}):")
            for g in sorted(active, key=lambda g: g.priority, reverse=True):
                bar = "█" * int(g.progress * 10) + "░" * (10 - int(g.progress * 10))
                lines.append(f"    [{bar}] {g.description[:50]} ({g.source}, {g.priority:.0%})")

        if completed:
            lines.append(f"\n  Completed ({len(completed)}):")
            for g in completed[-3:]:
                lines.append(f"    ✓ {g.description[:50]}")

        if not active and not completed:
            lines.append("  No goals yet. Run 'formulate' to generate goals.")

        return "\n".join(lines)
