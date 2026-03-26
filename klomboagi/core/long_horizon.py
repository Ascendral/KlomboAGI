"""
Long-Horizon Task Runner with checkpoint/resume.

Handles multi-step tasks that may take hours:
- Checkpoints before each step
- Resumes from last checkpoint on crash
- Retries failed steps with different approaches
- Tracks completion progress
"""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Checkpoint:
    task_id: str
    step: int
    total_steps: int
    state: dict = field(default_factory=dict)
    timestamp: str = ""
    completed_steps: list = field(default_factory=list)
    remaining_steps: list = field(default_factory=list)
    
    def save(self, path: str):
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @staticmethod
    def load(path: str) -> Checkpoint:
        with open(path) as f:
            return Checkpoint(**json.load(f))


class LongHorizonRunner:
    """Run multi-step tasks with checkpointing and recovery."""
    
    def __init__(self, checkpoint_dir: str = "datasets/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, task_id: str, steps: list,
            executor: Callable | None = None,
            max_retries: int = 2) -> dict:
        cp_path = self.checkpoint_dir / f"{task_id}.json"
        start = 0; completed = []
        
        if cp_path.exists():
            cp = Checkpoint.load(str(cp_path))
            start = cp.step; completed = cp.completed_steps
        
        results = list(completed)
        for i in range(start, len(steps)):
            Checkpoint(task_id=task_id, step=i, total_steps=len(steps),
                      completed_steps=results, remaining_steps=steps[i:]).save(str(cp_path))
            
            success = False
            for retry in range(max_retries + 1):
                try:
                    r = executor(steps[i]) if executor else {"step": i, "status": "ok"}
                    results.append(r); success = True; break
                except Exception as e:
                    if retry == max_retries:
                        return {"task_id": task_id, "completed": len(results),
                                "total": len(steps), "success": False,
                                "error": str(e), "failed_at": i, "can_resume": True}
        
        if cp_path.exists(): cp_path.unlink()
        return {"task_id": task_id, "completed": len(results),
                "total": len(steps), "success": True, "can_resume": False}
    
    def get_pending(self) -> list[str]:
        return [p.stem for p in self.checkpoint_dir.glob("*.json")]
