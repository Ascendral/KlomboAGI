from __future__ import annotations

import json

from klomboagi.core.state import Reflection
from klomboagi.llm import complete as llm_complete
from klomboagi.storage.manager import StorageManager

_LLM_REFLECTION_SYSTEM = (
    "You are a reflection engine. Given what just happened, extract: "
    '{"lesson": "one sentence lesson learned", '
    '"should_continue": true/false, '
    '"blockers": ["any blockers identified"]}'
)


class ReflectionEngine:
    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def llm_reflect(
        self,
        action_outcome: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        """Ask the LLM to reflect on what just happened.  Returns None on failure."""
        user_parts = [f"Action outcome: {json.dumps(action_outcome, default=str)[:800]}"]
        if context:
            user_parts.append(f"Context: {json.dumps(context, default=str)[:800]}")
        raw = llm_complete(_LLM_REFLECTION_SYSTEM, "\n\n".join(user_parts))
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if "lesson" not in parsed:
            return None
        return {
            "lesson": str(parsed["lesson"]),
            "should_continue": bool(parsed.get("should_continue", True)),
            "blockers": list(parsed.get("blockers", [])),
        }

    def load_all(self) -> list[dict[str, object]]:
        return self.storage.reflections.load(default=[])

    def reflect(
        self,
        mission: dict[str, object],
        tasks: list[dict[str, object]],
        working_memory: dict[str, object],
        next_action: dict[str, object],
        action_outcome: dict[str, object],
    ) -> dict[str, object]:
        completed = sum(1 for task in tasks if task["status"] == "completed")
        outstanding = sum(1 for task in tasks if task["status"] != "completed")
        blockers = list(working_memory.get("blockers", []))
        lessons = []

        # Try LLM-based reflection first.
        llm_result = self.llm_reflect(
            action_outcome,
            context={
                "mission": mission.get("description"),
                "completed_tasks": completed,
                "outstanding_tasks": outstanding,
            },
        )
        if llm_result is not None:
            lessons.append(llm_result["lesson"])
            blockers.extend(llm_result.get("blockers", []))
        else:
            # Fall back to template-based heuristics.
            if blockers:
                lessons.append("Unblock constrained work before adding new execution steps.")
            if outstanding == 0 and tasks:
                lessons.append("The current task list is complete; refresh mission status.")
            if not tasks:
                lessons.append("No tasks exist yet; decomposition is the next meaningful action.")
        reflection = Reflection(
            mission_id=str(mission["id"]),
            active_task_id=working_memory.get("active_task_id"),
            summary=f"Mission '{mission['description']}' is focused on {working_memory['current_focus']}.",
            next_action=str(next_action["description"]),
            action_outcome=str(action_outcome["summary"]),
            completed_tasks=completed,
            outstanding_tasks=outstanding,
            blockers=blockers,
            lessons=lessons,
        )
        return self.save(reflection.to_dict())

    def save(self, reflection: dict[str, object]) -> dict[str, object]:
        reflections = self.load_all()
        reflections.append(reflection)
        self.storage.reflections.save(reflections)
        self.storage.event_log.append(
            "reflection.recorded",
            {
                "mission_id": reflection.get("mission_id"),
                "reflection_id": reflection.get("id"),
                "next_action": reflection.get("next_action"),
            },
        )
        return reflection
