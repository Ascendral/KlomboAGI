"""
Self-Improvement Loop — identify weaknesses, fix them autonomously.

The system analyzes its own performance and takes action:
  1. DIAGNOSE — what am I bad at?
  2. PRIORITIZE — what's most important to fix?
  3. ACT — go learn/practice/strengthen
  4. MEASURE — did it actually improve?
  5. REPEAT

Sources of weakness signals:
  - Low-confidence beliefs
  - Domains with few relations
  - Questions that triggered "I don't know"
  - Corrections from the human
  - Failed experiential attempts
  - Low scores in specific eval categories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Weakness:
    """An identified weakness in the system."""
    description: str
    severity: float        # 0-1
    category: str          # "knowledge", "reasoning", "language", "coverage"
    fix_action: str        # what to do about it
    fixed: bool = False


@dataclass
class ImprovementCycle:
    """One cycle of self-improvement."""
    weaknesses_found: list[Weakness]
    actions_taken: list[str]
    beliefs_before: int
    beliefs_after: int
    improvements: list[str]


class SelfImprover:
    """
    Identifies weaknesses and fixes them autonomously.
    """

    def __init__(self, genesis) -> None:
        self.g = genesis
        self.history: list[ImprovementCycle] = []

    def diagnose(self) -> list[Weakness]:
        """Identify current weaknesses."""
        weaknesses = []

        # 1. Low-confidence domains
        domain_confidence = self._assess_domain_confidence()
        for domain, avg_conf in domain_confidence.items():
            if avg_conf < 0.4:
                weaknesses.append(Weakness(
                    description=f"Low confidence in {domain} ({avg_conf:.0%})",
                    severity=1.0 - avg_conf,
                    category="knowledge",
                    fix_action=f"study {domain}",
                ))

        # 2. Domains with few relations
        if hasattr(self.g, 'relations'):
            from klomboagi.core.curriculum import CURRICULA
            for domain in CURRICULA:
                domain_rels = 0
                for subj, _ in CURRICULA[domain]:
                    domain_rels += len(self.g.relations.get_all_about(subj.lower()))
                if domain_rels < 5:
                    weaknesses.append(Weakness(
                        description=f"Few relations in {domain} ({domain_rels})",
                        severity=0.6,
                        category="coverage",
                        fix_action=f"learn relationships in {domain}",
                    ))

        # 3. Unanswered questions from past
        if hasattr(self.g, 'experiential'):
            failed = [a for a in self.g.experiential.attempts
                     if a.confidence < 0.3]
            if len(failed) > 3:
                topics = Counter(a.question.split()[-1] for a in failed)
                for topic, count in topics.most_common(3):
                    if len(topic) > 3:
                        weaknesses.append(Weakness(
                            description=f"Failed to answer about {topic} ({count} times)",
                            severity=0.7,
                            category="knowledge",
                            fix_action=f"learn about {topic}",
                        ))

        # 4. Corrections received
        if hasattr(self.g, 'failure_memory'):
            repeated = self.g.failure_memory.worst_mistakes(3)
            for f in repeated:
                if f.times_repeated > 1:
                    weaknesses.append(Weakness(
                        description=f"Repeated mistake: {f.description[:40]}",
                        severity=0.8,
                        category="reasoning",
                        fix_action=f.better_approach or "review approach",
                    ))

        weaknesses.sort(key=lambda w: w.severity, reverse=True)
        return weaknesses

    def improve(self, max_actions: int = 5) -> ImprovementCycle:
        """Run one improvement cycle: diagnose → prioritize → act → measure."""
        beliefs_before = len(self.g.base._beliefs)

        # DIAGNOSE
        weaknesses = self.diagnose()

        # PRIORITIZE — take top weaknesses
        actions_taken = []
        improvements = []

        for weakness in weaknesses[:max_actions]:
            action = weakness.fix_action

            if action.startswith("study ") or action.startswith("learn"):
                topic = action.split(" ", 1)[-1] if " " in action else action
                try:
                    result = self.g.read_and_learn(topic)
                    if "Could not read" not in result:
                        actions_taken.append(f"Studied {topic}")
                        weakness.fixed = True
                        improvements.append(f"Learned about {topic}")
                except Exception:
                    actions_taken.append(f"Failed to study {topic}")

        # Run inference
        if hasattr(self.g, 'relations'):
            inferred = self.g.relations.run_inference()
            if inferred:
                improvements.append(f"Inferred {len(inferred)} new relations")

        beliefs_after = len(self.g.base._beliefs)

        cycle = ImprovementCycle(
            weaknesses_found=weaknesses,
            actions_taken=actions_taken,
            beliefs_before=beliefs_before,
            beliefs_after=beliefs_after,
            improvements=improvements,
        )
        self.history.append(cycle)
        return cycle

    def report(self) -> str:
        """Report on self-improvement progress."""
        weaknesses = self.diagnose()
        lines = [f"Self-Improvement Report"]
        lines.append(f"  Weaknesses found: {len(weaknesses)}")
        for w in weaknesses[:5]:
            lines.append(f"    [{w.severity:.0%}] {w.description}")
            lines.append(f"         Fix: {w.fix_action}")
        if self.history:
            last = self.history[-1]
            lines.append(f"\n  Last cycle: +{last.beliefs_after - last.beliefs_before} beliefs")
            for imp in last.improvements:
                lines.append(f"    ✓ {imp}")
        return "\n".join(lines)

    def _assess_domain_confidence(self) -> dict[str, float]:
        """Average belief confidence per domain."""
        from klomboagi.core.curriculum import CURRICULA
        result = {}
        for domain, facts in CURRICULA.items():
            subjects = {s.lower() for s, _ in facts}
            confs = []
            for stmt, belief in self.g.base._beliefs.items():
                if hasattr(belief, 'subject') and belief.subject in subjects:
                    if hasattr(belief, 'truth'):
                        confs.append(belief.truth.confidence)
            if confs:
                result[domain] = sum(confs) / len(confs)
        return result
