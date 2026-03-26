"""
Trajectory Storage — every action the system takes gets recorded.

A trajectory is a sequence of: state → action → observation → outcome.
These are the raw material for learning. Without trajectories, there's
nothing to learn FROM.

Storage: datasets/trajectories/<task_id>_<timestamp>.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Step:
    """One step in a trajectory."""
    step_id: int
    phase: str                      # perceive, plan, act, observe, reflect
    action: str                     # what the system did
    action_args: dict = field(default_factory=dict)
    observation: str = ""           # what happened
    decision_reason: str = ""       # WHY it chose this action
    memory_retrieved: list[str] = field(default_factory=list)
    memory_useful: bool = False     # did memory help?
    outcome: str = ""               # success, failure, partial
    error: str = ""
    duration_s: float = 0.0
    timestamp: str = ""


@dataclass
class Trajectory:
    """A complete record of one task attempt."""
    task_id: str
    domain: str
    description: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    interventions: int = 0
    total_steps: int = 0
    failure_point: int | None = None    # which step failed
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    skills_used: list[str] = field(default_factory=list)
    skills_learned: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_s: float = 0.0

    def add_step(self, phase: str, action: str, **kwargs) -> Step:
        step = Step(
            step_id=len(self.steps),
            phase=phase,
            action=action,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            **kwargs,
        )
        self.steps.append(step)
        self.total_steps = len(self.steps)
        return step

    def mark_success(self) -> None:
        self.success = True
        self.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def mark_failure(self, step_id: int, error: str = "") -> None:
        self.success = False
        self.failure_point = step_id
        if self.steps and step_id < len(self.steps):
            self.steps[step_id].error = error

    def to_dict(self) -> dict:
        return asdict(self)


class TrajectoryStore:
    """Persist and query trajectories."""

    def __init__(self, store_dir: str = "datasets/trajectories"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, traj: Trajectory) -> str:
        """Save a trajectory. Returns the file path."""
        filename = f"{traj.task_id}_{int(time.time())}.json"
        path = self.store_dir / filename
        with open(path, 'w') as f:
            json.dump(traj.to_dict(), f, indent=2, default=str)
        return str(path)

    def load(self, path: str) -> Trajectory:
        """Load a trajectory from file."""
        with open(path) as f:
            data = json.load(f)
        traj = Trajectory(
            task_id=data["task_id"],
            domain=data["domain"],
            description=data["description"],
        )
        traj.success = data.get("success", False)
        traj.interventions = data.get("interventions", 0)
        traj.total_steps = data.get("total_steps", 0)
        traj.skills_used = data.get("skills_used", [])
        traj.skills_learned = data.get("skills_learned", [])
        return traj

    def list_all(self, domain: str | None = None) -> list[str]:
        """List all trajectory files."""
        paths = sorted(self.store_dir.glob("*.json"))
        if domain:
            paths = [p for p in paths if domain in p.stem]
        return [str(p) for p in paths]

    def get_success_rate(self, domain: str | None = None) -> float:
        """Get success rate across all stored trajectories."""
        paths = self.list_all(domain)
        if not paths:
            return 0.0
        successes = 0
        for p in paths:
            try:
                with open(p) as f:
                    data = json.load(f)
                if data.get("success"):
                    successes += 1
            except:
                pass
        return successes / len(paths)

    def get_failure_patterns(self, limit: int = 10) -> list[dict]:
        """Extract common failure patterns from trajectories."""
        failures = []
        for p in self.list_all():
            try:
                with open(p) as f:
                    data = json.load(f)
                if not data.get("success") and data.get("steps"):
                    fp = data.get("failure_point", len(data["steps"]) - 1)
                    if fp is not None and fp < len(data["steps"]):
                        step = data["steps"][fp]
                        failures.append({
                            "task": data["task_id"],
                            "domain": data["domain"],
                            "failed_action": step.get("action", "unknown"),
                            "failed_phase": step.get("phase", "unknown"),
                            "error": step.get("error", "unknown"),
                        })
            except:
                pass
        return failures[:limit]
