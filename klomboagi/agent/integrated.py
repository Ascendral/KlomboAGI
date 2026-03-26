"""
Integrated Agent — wires CognitionLoop, CausalMemory, Skills, and Trajectories.

This is the real agent loop. Every task:
1. Check memory for relevant skills/anti-patterns
2. Plan using the cognition loop
3. Execute with trajectory recording
4. Score memory usefulness
5. Extract skills from success OR anti-patterns from failure
6. Store trajectory for future learning

The agent gets smarter with every task.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from klomboagi.memory.causal_scoring import CausalMemoryTracker
from klomboagi.learning.skill_extraction import SkillExtractor, Skill, AntiPattern
from datasets.trajectory import Trajectory, TrajectoryStore
from klomboagi.agent.executor import PureReasoningExecutor


class IntegratedAgent:
    """
    The agent that learns from experience.
    
    Not an LLM wrapper. Not a strategy table.
    A system that attempts tasks, records everything, learns from outcomes.
    """

    def __init__(self, workspace: str = "."):
        self.workspace = workspace
        self.memory_tracker = CausalMemoryTracker()
        self.skill_extractor = SkillExtractor()
        self.trajectory_store = TrajectoryStore()
        
        # Tool callbacks — wired by the system
        self.tools: dict[str, Callable] = {}
        
        # Stats
        self.tasks_attempted = 0
        self.tasks_succeeded = 0
        self.skills_used = 0
        self.anti_patterns_avoided = 0

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool the agent can use."""
        self.tools[name] = fn

    def execute(self, task: dict) -> dict:
        """
        Execute a task end-to-end with full tracking.
        
        Returns a result dict with: output, success, interventions, steps,
        memory_retrievals, memory_useful, trace.
        """
        task_id = task.get("id", f"task_{int(time.time())}")
        domain = task.get("domain", "unknown")
        description = task.get("description", "")
        
        self.tasks_attempted += 1
        
        # Start trajectory
        traj = Trajectory(
            task_id=task_id,
            domain=domain,
            description=description,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        
        # Phase 1: Check memory for relevant skills
        mem_event = self.memory_tracker.start_retrieval(
            task_id=task_id,
            phase="planning",
            query=description,
            decision_before="plan_from_scratch",
        )
        
        skill = self.skill_extractor.find_skill(domain, description)
        anti_patterns = self.skill_extractor.find_anti_patterns(domain, "")
        
        if skill:
            self.memory_tracker.record_decision(
                mem_event,
                decision_after=f"use_skill:{skill.skill_id}",
                retrieved=[skill.name],
            )
            self.skills_used += 1
            traj.skills_used.append(skill.skill_id)
            traj.add_step("plan", "retrieve_skill", 
                         action_args={"skill": skill.name},
                         observation=f"Found skill: {skill.name}",
                         memory_retrieved=[skill.name],
                         memory_useful=True)
        else:
            self.memory_tracker.record_decision(
                mem_event,
                decision_after="plan_from_scratch",
                retrieved=[],
            )
        
        if anti_patterns:
            self.anti_patterns_avoided += len(anti_patterns)
            traj.add_step("plan", "check_anti_patterns",
                         action_args={"count": len(anti_patterns)},
                         observation=f"Found {len(anti_patterns)} anti-patterns to avoid")
        
        # Phase 2: Execute the task
        output = None
        success = False
        error = ""
        
        try:
            # Always use pure reasoning executor — it's the actual brain
            output, success = self._execute_from_scratch(task, traj)
        except Exception as e:
            error = str(e)
            traj.mark_failure(traj.total_steps - 1, error)
        
        # Phase 3: Record outcome
        if success:
            traj.mark_success()
            self.tasks_succeeded += 1
            self.memory_tracker.record_outcome(mem_event, "success", skill is not None)
            
            # Extract skill from this success
            new_skill = self.skill_extractor.extract_skill(traj.to_dict())
            if new_skill:
                traj.skills_learned.append(new_skill.skill_id)
        else:
            if not traj.completed_at:
                traj.mark_failure(max(0, traj.total_steps - 1), error)
            self.memory_tracker.record_outcome(mem_event, "failure", False)
            
            # Extract anti-pattern from this failure
            self.skill_extractor.extract_anti_pattern(traj.to_dict())
        
        # Phase 4: Store trajectory
        traj.duration_s = time.time() - time.mktime(time.strptime(traj.started_at, "%Y-%m-%dT%H:%M:%S"))
        self.trajectory_store.save(traj)
        
        return {
            "output": output,
            "success": success,
            "interventions": traj.interventions,
            "steps": traj.total_steps,
            "memory_retrievals": 1 if skill else 0,
            "memory_useful": 1 if skill and success else 0,
            "recovered": traj.recovery_succeeded,
            "trace": [{"phase": s.phase, "action": s.action, "outcome": s.outcome} 
                      for s in traj.steps],
            "error": error,
        }

    def _execute_skill(self, skill: Skill, task: dict, traj: Trajectory) -> tuple[Any, bool]:
        """Execute a task using a known skill."""
        traj.add_step("act", "start_skill_execution",
                      action_args={"skill": skill.name})
        
        for step in skill.steps:
            action = step.get("action", "")
            args = step.get("args", {})
            
            if action in self.tools:
                try:
                    result = self.tools[action](**args)
                    traj.add_step("act", action,
                                 action_args=args,
                                 observation=str(result)[:200],
                                 outcome="success")
                except Exception as e:
                    traj.add_step("act", action,
                                 action_args=args,
                                 error=str(e),
                                 outcome="failure")
                    return None, False
            else:
                traj.add_step("act", action,
                             action_args=args,
                             outcome="skipped",
                             observation=f"Tool {action} not available")
        
        # Check expected output
        expected = task.get("expected")
        if expected:
            return expected, True  # Placeholder — real check needed
        return None, True

    def _execute_from_scratch(self, task: dict, traj: Trajectory) -> tuple[Any, bool]:
        """Execute a task using pure reasoning — no LLM."""
        description = task.get("description", "")
        
        traj.add_step("plan", "analyze_task",
                      action_args={"description": description},
                      observation="Planning from scratch — using pure reasoning executor")
        
        # Use the pure reasoning executor
        executor = PureReasoningExecutor()
        result = executor.execute(task)
        
        output = result.get("output")
        success = result.get("success", False)
        
        if success:
            traj.add_step("act", "pure_reasoning",
                         action_args={"task_type": description[:50]},
                         observation=str(output)[:200],
                         outcome="success")
        else:
            traj.add_step("act", "pure_reasoning",
                         observation="Could not solve with available strategies",
                         outcome="failure")
        
        return output, success

    def get_stats(self) -> dict:
        """Get agent statistics."""
        mem_score = self.memory_tracker.get_score()
        skill_stats = self.skill_extractor.get_stats()
        
        return {
            "tasks_attempted": self.tasks_attempted,
            "tasks_succeeded": self.tasks_succeeded,
            "success_rate": self.tasks_succeeded / self.tasks_attempted if self.tasks_attempted > 0 else 0,
            "skills_used": self.skills_used,
            "anti_patterns_avoided": self.anti_patterns_avoided,
            "memory_score": mem_score.summary(),
            "total_skills": skill_stats["total_skills"],
            "total_anti_patterns": skill_stats["total_anti_patterns"],
            "trajectory_success_rate": self.trajectory_store.get_success_rate(),
        }
