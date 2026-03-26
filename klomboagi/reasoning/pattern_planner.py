"""
PatternPlanner — plans by finding structural patterns in past successes.

Instead of keyword matching or LLM generation, this planner:
1. Decomposes the current task into structural elements
2. Searches past trajectories for structurally similar tasks
3. Extracts the action sequence from the matching trajectory
4. Adapts it to the current task

This is the ARC solver's approach applied to task planning.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any


class PatternPlanner:
    """Plans by structural pattern matching against past trajectories."""

    def __init__(self, trajectory_dir: str = "datasets/trajectories",
                 skill_dir: str = "datasets/skills"):
        self.trajectory_dir = Path(trajectory_dir)
        self.skill_dir = Path(skill_dir)

    def plan(self, task: dict) -> list[dict] | None:
        """
        Generate a plan for a task by finding matching patterns.
        Returns a list of action steps, or None if no pattern found.
        """
        description = task.get("description", "")
        domain = task.get("domain", "")

        # Step 1: Decompose task into structural elements
        elements = self._decompose(description)

        # Step 2: Search for matching skills
        best_skill = self._find_matching_skill(elements, domain)
        if best_skill:
            return best_skill.get("steps", [])

        # Step 3: Search trajectories for similar structure
        best_trajectory = self._find_matching_trajectory(elements, domain)
        if best_trajectory:
            return self._extract_plan(best_trajectory)

        return None

    def _decompose(self, description: str) -> dict:
        """Decompose a task description into structural elements."""
        desc_lower = description.lower()
        words = set(desc_lower.split())

        # Identify structural patterns (like ARC solver categories)
        elements = {
            "action_type": self._identify_action(desc_lower),
            "input_type": self._identify_input(desc_lower),
            "output_type": self._identify_output(desc_lower),
            "operation": self._identify_operation(desc_lower),
            "keywords": words,
        }
        return elements

    def _identify_action(self, desc: str) -> str:
        if any(w in desc for w in ["fix", "repair", "debug", "correct"]): return "fix"
        if any(w in desc for w in ["write", "create", "generate", "build"]): return "create"
        if any(w in desc for w in ["find", "search", "extract", "identify"]): return "extract"
        if any(w in desc for w in ["count", "compute", "calculate"]): return "compute"
        if any(w in desc for w in ["sort", "rank", "order"]): return "sort"
        if any(w in desc for w in ["filter", "remove", "keep"]): return "filter"
        if any(w in desc for w in ["parse", "read", "analyze"]): return "parse"
        if any(w in desc for w in ["summarize", "describe"]): return "summarize"
        if any(w in desc for w in ["convert", "transform"]): return "transform"
        return "unknown"

    def _identify_input(self, desc: str) -> str:
        if any(w in desc for w in ["function", "code", "bug"]): return "code"
        if any(w in desc for w in ["text", "string", "word"]): return "text"
        if any(w in desc for w in ["number", "data", "list"]): return "data"
        if any(w in desc for w in ["log", "error", "output"]): return "log"
        if any(w in desc for w in ["file", "directory"]): return "file"
        return "unknown"

    def _identify_output(self, desc: str) -> str:
        if any(w in desc for w in ["count", "number", "total"]): return "number"
        if any(w in desc for w in ["list", "array"]): return "list"
        if any(w in desc for w in ["function", "code"]): return "code"
        if any(w in desc for w in ["summary", "sentence"]): return "text"
        if any(w in desc for w in ["dictionary", "map", "json"]): return "dict"
        return "unknown"

    def _identify_operation(self, desc: str) -> str:
        if "outlier" in desc: return "find_outliers"
        if "palindrome" in desc: return "check_palindrome"
        if "fibonacci" in desc: return "fibonacci"
        if "gcd" in desc or "greatest common" in desc: return "gcd"
        if "prime" in desc: return "prime_check"
        if "anagram" in desc: return "anagram_check"
        if "transpose" in desc: return "transpose"
        if "flatten" in desc: return "flatten"
        if "sort" in desc: return "sort"
        if "unique" in desc: return "deduplicate"
        if "count" in desc: return "count"
        if "parse" in desc: return "parse"
        if "extract" in desc: return "extract"
        if "filter" in desc: return "filter"
        return "general"

    def _find_matching_skill(self, elements: dict, domain: str) -> dict | None:
        """Find a skill that matches the structural elements."""
        if not self.skill_dir.exists():
            return None

        best_score = 0
        best_skill = None

        for path in self.skill_dir.glob("*.json"):
            try:
                with open(path) as f:
                    skill = json.load(f)

                # Score by structural similarity
                skill_desc = skill.get("description", "").lower()
                skill_elements = self._decompose(skill_desc)

                score = 0
                if skill_elements["action_type"] == elements["action_type"]: score += 3
                if skill_elements["input_type"] == elements["input_type"]: score += 2
                if skill_elements["output_type"] == elements["output_type"]: score += 2
                if skill_elements["operation"] == elements["operation"]: score += 5

                # Domain match bonus
                if skill.get("domain") == domain: score += 1

                # Keyword overlap
                overlap = len(skill_elements["keywords"] & elements["keywords"])
                score += overlap * 0.5

                if score > best_score and score >= 5:
                    best_score = score
                    best_skill = skill
            except:
                pass

        return best_skill

    def _find_matching_trajectory(self, elements: dict, domain: str) -> dict | None:
        """Find a trajectory with matching structure."""
        if not self.trajectory_dir.exists():
            return None

        best_score = 0
        best_traj = None

        for path in self.trajectory_dir.glob("*.json"):
            try:
                with open(path) as f:
                    traj = json.load(f)

                if not traj.get("success"):
                    continue

                traj_elements = self._decompose(traj.get("description", ""))

                score = 0
                if traj_elements["action_type"] == elements["action_type"]: score += 3
                if traj_elements["operation"] == elements["operation"]: score += 5

                if score > best_score and score >= 5:
                    best_score = score
                    best_traj = traj
            except:
                pass

        return best_traj

    def _extract_plan(self, trajectory: dict) -> list[dict]:
        """Extract action steps from a successful trajectory."""
        steps = trajectory.get("steps", [])
        return [{"action": s.get("action", ""), "args": s.get("action_args", {})}
                for s in steps if s.get("outcome") == "success"]
