"""
Skill Extraction — learn from success, mine from failure.

After a task completes:
- SUCCESS → extract the action sequence as a named, reusable skill
- FAILURE → extract the failure pattern as an anti-pattern to avoid

Skills are stored in datasets/skills/
Anti-patterns are stored in datasets/failure_cases/
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """A learned procedure that worked."""
    skill_id: str
    name: str
    domain: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    success_count: int = 1
    failure_count: int = 0
    confidence: float = 0.5
    learned_from: list[str] = field(default_factory=list)  # task IDs
    created_at: str = ""
    updated_at: str = ""

    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reliability"] = self.reliability()
        return d


@dataclass
class AntiPattern:
    """A failure pattern to avoid."""
    pattern_id: str
    description: str
    domain: str
    trigger: str                        # what condition causes this failure
    failed_action: str
    error_type: str
    frequency: int = 1
    avoidance: str = ""                 # what to do instead
    seen_in: list[str] = field(default_factory=list)  # task IDs

    def to_dict(self) -> dict:
        return asdict(self)


class SkillExtractor:
    """Extract skills from successful trajectories and anti-patterns from failures."""

    def __init__(self, skill_dir: str = "datasets/skills",
                 failure_dir: str = "datasets/failure_cases"):
        self.skill_dir = Path(skill_dir)
        self.failure_dir = Path(failure_dir)
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir.mkdir(parents=True, exist_ok=True)

    def extract_skill(self, trajectory: dict) -> Skill | None:
        """Extract a reusable skill from a successful trajectory."""
        if not trajectory.get("success"):
            return None

        steps = trajectory.get("steps", [])
        if not steps:
            return None

        # Build skill from the action sequence
        skill_steps = []
        for step in steps:
            if step.get("action") and step.get("outcome") != "failure":
                skill_steps.append({
                    "action": step["action"],
                    "phase": step.get("phase", ""),
                    "args": step.get("action_args", {}),
                })

        if not skill_steps:
            return None

        task_id = trajectory.get("task_id", "unknown")
        domain = trajectory.get("domain", "unknown")

        skill = Skill(
            skill_id=f"skill_{task_id}_{int(time.time())}",
            name=f"Procedure for: {trajectory.get('description', task_id)[:60]}",
            domain=domain,
            description=trajectory.get("description", ""),
            steps=skill_steps,
            learned_from=[task_id],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Save
        path = self.skill_dir / f"{skill.skill_id}.json"
        with open(path, "w") as f:
            json.dump(skill.to_dict(), f, indent=2)

        return skill

    def extract_anti_pattern(self, trajectory: dict) -> AntiPattern | None:
        """Extract a failure pattern from a failed trajectory."""
        if trajectory.get("success"):
            return None

        steps = trajectory.get("steps", [])
        failure_point = trajectory.get("failure_point")

        if failure_point is None or failure_point >= len(steps):
            return None

        failed_step = steps[failure_point]
        task_id = trajectory.get("task_id", "unknown")

        pattern = AntiPattern(
            pattern_id=f"anti_{task_id}_{int(time.time())}",
            description=f"Failed during: {failed_step.get('action', 'unknown')}",
            domain=trajectory.get("domain", "unknown"),
            trigger=failed_step.get("decision_reason", "unknown trigger"),
            failed_action=failed_step.get("action", "unknown"),
            error_type=failed_step.get("error", "unknown error"),
            seen_in=[task_id],
        )

        # Save
        path = self.failure_dir / f"{pattern.pattern_id}.json"
        with open(path, "w") as f:
            json.dump(pattern.to_dict(), f, indent=2)

        return pattern

    def find_skill(self, domain: str, description: str) -> Skill | None:
        """Find a relevant skill for a task."""
        best = None
        best_score = 0

        for path in self.skill_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("domain") == domain:
                    # Simple relevance: word overlap
                    skill_words = set(data.get("description", "").lower().split())
                    query_words = set(description.lower().split())
                    overlap = len(skill_words & query_words)
                    reliability = data.get("reliability", 0.5)
                    score = overlap * reliability

                    if score > best_score:
                        best_score = score
                        best = Skill(**{k: v for k, v in data.items() if k != "reliability"})
            except:
                pass

        return best

    def find_anti_patterns(self, domain: str, action: str) -> list[AntiPattern]:
        """Find known failure patterns for a domain/action."""
        patterns = []
        for path in self.failure_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("domain") == domain or data.get("failed_action") == action:
                    patterns.append(AntiPattern(**data))
            except:
                pass
        return patterns

    def get_stats(self) -> dict:
        skills = list(self.skill_dir.glob("*.json"))
        failures = list(self.failure_dir.glob("*.json"))
        return {
            "total_skills": len(skills),
            "total_anti_patterns": len(failures),
        }
