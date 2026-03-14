from __future__ import annotations

import json

from codeagi.llm import complete as llm_complete
from codeagi.storage.manager import StorageManager
from codeagi.utils.time import utc_now

_LLM_CRITIC_SYSTEM = (
    "You are a safety critic for an autonomous agent. "
    "Given a proposed action and context, respond with JSON: "
    '{"approved": true/false, "reason": "brief explanation"}. '
    "Block actions that could cause data loss, access unauthorized paths, "
    "or execute dangerous commands."
)


class Critic:
    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def llm_critique(
        self,
        proposed_action: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        """Ask the LLM to evaluate a proposed action.  Returns None on failure."""
        user_parts = [f"Proposed action: {json.dumps(proposed_action, default=str)}"]
        if context:
            user_parts.append(f"Context: {json.dumps(context, default=str)[:1000]}")
        raw = llm_complete(_LLM_CRITIC_SYSTEM, "\n\n".join(user_parts))
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if "approved" not in parsed:
            return None
        return {
            "approved": bool(parsed["approved"]),
            "reason": str(parsed.get("reason", "")),
        }

    def load_all(self) -> dict[str, dict[str, object]]:
        return self.storage.critiques.load(default={})

    def critique(
        self,
        *,
        mission: dict[str, object],
        tasks: list[dict[str, object]],
        plan: dict[str, object],
        proposed_action: dict[str, object],
        verification: dict[str, object],
    ) -> dict[str, object]:
        approved = bool(verification["valid"])
        notes: list[str] = []
        final_action = dict(proposed_action)

        if not verification["valid"]:
            approved = False
            notes.append("Verification found contradictions between the plan and world state.")
            final_action = {
                "type": "replan",
                "description": "Rebuild the plan because the current world state and plan do not agree.",
                "task_id": None,
            }
        elif proposed_action["type"] == "mission_complete" and tasks:
            outstanding = [task for task in tasks if task["status"] != "completed"]
            if outstanding:
                approved = False
                notes.append("Mission completion was rejected because unfinished tasks still exist.")
                final_action = {
                    "type": "replan",
                    "description": "Rebuild the task graph because unfinished work remains.",
                    "task_id": None,
                }

        if proposed_action["type"] == "execute_task" and not notes:
            notes.append("The next task is grounded in the current world model and can proceed.")
        if proposed_action["type"] == "decompose_mission" and not notes:
            notes.append("Mission decomposition is appropriate because no executable tasks exist yet.")
        if proposed_action["type"] == "resolve_blocker" and not notes:
            notes.append("Blocker resolution takes priority over new execution work.")
        # LLM safety check — only override if rule-based critic approved.
        if approved:
            llm_result = self.llm_critique(
                proposed_action,
                context={"mission": mission.get("description"), "task_count": len(tasks)},
            )
            if llm_result is not None and not llm_result["approved"]:
                approved = False
                notes.append(f"LLM critic blocked: {llm_result['reason']}")
                final_action = {
                    "type": "replan",
                    "description": f"LLM critic blocked the action: {llm_result['reason']}",
                    "task_id": None,
                }

        for warning in verification.get("warnings", []):
            notes.append(f"Verification warning: {warning}")

        report = {
            "mission_id": str(mission["id"]),
            "plan_id": plan.get("id"),
            "reviewed_at": utc_now(),
            "approved": approved,
            "notes": notes,
            "proposed_action": proposed_action,
            "final_action": final_action,
        }
        return self.save(report)

    def save(self, report: dict[str, object]) -> dict[str, object]:
        payload = self.load_all()
        mission_id = str(report["mission_id"])
        payload[mission_id] = report
        self.storage.critiques.save(payload)
        self.storage.event_log.append(
            "critique.completed",
            {
                "mission_id": mission_id,
                "plan_id": report.get("plan_id"),
                "approved": report.get("approved"),
                "final_action": report["final_action"]["type"],
            },
        )
        return report
