"""
SkillForge — Promotes high-confidence CodeAGI procedures to shared CodeBot skills.

When CodeAGI discovers a reliable procedure through repeated successful execution,
SkillForge converts it into a CodeBot-compatible skill JSON file in the shared
skill store (~/.codebot/skills/). Both systems read from and write to this store.

Promotion criteria:
  - confidence >= 0.8
  - use_count >= 3
  - Not already promoted

This creates a feedback loop: CodeAGI discovers patterns through reflection,
CodeBot operationalizes them as reusable tool chains.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from codeagi.storage.manager import StorageManager
from codeagi.utils.time import utc_now


# Shared skill store location (same as CodeBot reads from)
def _shared_skills_dir() -> Path:
    """Resolve the shared skill store directory (~/.codebot/skills/)."""
    import os
    codebot_home = os.environ.get("CODEBOT_HOME", str(Path.home() / ".codebot"))
    return Path(codebot_home) / "skills"


# Action kind -> tool name mapping for CodeBot
_ACTION_TO_TOOL: dict[str, str] = {
    "search_files": "grep",
    "read_file": "read_file",
    "write_file": "write_file",
    "apply_patch": "edit_file",
    "run_command": "execute",
    "list_dir": "glob",
}


class SkillForge:
    """Promotes CodeAGI procedures to shared CodeBot skills."""

    # Thresholds for promotion
    MIN_CONFIDENCE = 0.8
    MIN_USE_COUNT = 3

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def scan_and_promote(self) -> list[dict[str, object]]:
        """Scan all procedures and promote eligible ones to shared skills.

        Returns list of newly promoted skill metadata dicts.
        """
        procedures = self.storage.procedures.load(default=[])
        promoted: list[dict[str, object]] = []

        for procedure in procedures:
            confidence = float(procedure.get("confidence", 0))
            use_count = int(procedure.get("use_count", 0))

            if confidence < self.MIN_CONFIDENCE:
                continue
            if use_count < self.MIN_USE_COUNT:
                continue

            skill_name = self._procedure_to_skill_name(procedure)
            if self._skill_exists(skill_name):
                # Already promoted — reinforce instead
                self._reinforce_skill(skill_name)
                continue

            skill = self._convert_to_skill(procedure, skill_name)
            if skill is not None:
                self._write_skill(skill)
                promoted.append(skill)
                self.storage.event_log.append(
                    "skillforge.promoted",
                    {
                        "procedure_id": procedure.get("id"),
                        "skill_name": skill_name,
                        "confidence": confidence,
                        "use_count": use_count,
                    },
                )

        return promoted

    def promote_one(self, procedure_id: str) -> dict[str, object] | None:
        """Force-promote a specific procedure regardless of thresholds."""
        procedures = self.storage.procedures.load(default=[])
        procedure = None
        for p in procedures:
            if p.get("id") == procedure_id:
                procedure = p
                break

        if procedure is None:
            return None

        skill_name = self._procedure_to_skill_name(procedure)
        if self._skill_exists(skill_name):
            self._reinforce_skill(skill_name)
            return None

        skill = self._convert_to_skill(procedure, skill_name)
        if skill is not None:
            self._write_skill(skill)
            self.storage.event_log.append(
                "skillforge.force_promoted",
                {
                    "procedure_id": procedure_id,
                    "skill_name": skill_name,
                },
            )
        return skill

    def list_shared_skills(self) -> list[dict[str, object]]:
        """List all skills in the shared store."""
        skills_dir = _shared_skills_dir()
        if not skills_dir.exists():
            return []

        skills: list[dict[str, object]] = []
        for f in sorted(skills_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("name") and data.get("steps"):
                    skills.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return skills

    def _convert_to_skill(
        self, procedure: dict[str, object], skill_name: str
    ) -> dict[str, object] | None:
        """Convert a CodeAGI procedure to a CodeBot skill definition."""
        steps_raw = list(procedure.get("steps", []))
        if not steps_raw:
            return None

        # Convert procedure steps to CodeBot skill steps
        skill_steps: list[dict[str, object]] = []
        for step_text in steps_raw:
            step_str = str(step_text)
            tool, args = self._parse_step(step_str)
            if tool:
                skill_steps.append({"tool": tool, "args": args})

        if not skill_steps:
            # Fallback: create a think step with the procedure's intent
            skill_steps.append({
                "tool": "think",
                "args": {"thought": f"Execute learned procedure: {procedure.get('title', 'unknown')}"},
            })

        now = datetime.now(timezone.utc).isoformat()
        return {
            "name": skill_name,
            "description": str(procedure.get("title", f"Promoted procedure: {skill_name}")),
            "trigger": str(procedure.get("trigger", "")),
            "steps": skill_steps,
            # Shared metadata
            "author": "codeagi",
            "confidence": float(procedure.get("confidence", 0.5)),
            "use_count": int(procedure.get("use_count", 0)),
            "origin": "promoted",
            "source_procedure_id": str(procedure.get("id", "")),
            "created_at": now,
            "updated_at": now,
        }

    def _parse_step(self, step_text: str) -> tuple[str | None, dict[str, object]]:
        """Parse a procedure step string into a tool name and args.

        Procedure steps look like:
          "Execute action: execute_task"
          "Review mission intent: Fix the login bug"
          "Record outcome: completed search"
        """
        step_lower = step_text.lower()

        # Try to extract action kind
        if "execute action:" in step_lower:
            action_part = step_text.split(":", 1)[-1].strip()
            # Check for known action kinds
            for action_kind, tool_name in _ACTION_TO_TOOL.items():
                if action_kind in action_part.lower():
                    return tool_name, {"description": action_part}
            # Default to think for unrecognized actions
            return "think", {"thought": f"Execute: {action_part}"}

        if "review mission" in step_lower or "review" in step_lower:
            intent = step_text.split(":", 1)[-1].strip() if ":" in step_text else step_text
            return "think", {"thought": f"Review intent: {intent}"}

        if "record outcome" in step_lower:
            return None, {}  # Skip recording steps — CodeBot handles this

        # Generic fallback
        return "think", {"thought": step_text}

    def _procedure_to_skill_name(self, procedure: dict[str, object]) -> str:
        """Derive a filesystem-safe skill name from a procedure."""
        title = str(procedure.get("title", "unknown"))
        # Convert to snake_case, keep only safe chars
        name = title.lower()
        name = name.replace(" ", "_").replace("-", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_")
        # Trim and deduplicate underscores
        while "__" in name:
            name = name.replace("__", "_")
        name = name.strip("_")
        # Prefix with codeagi_ to show origin
        if not name.startswith("codeagi_"):
            name = f"codeagi_{name}"
        return name[:64]  # Max 64 chars

    def _skill_exists(self, skill_name: str) -> bool:
        return (_shared_skills_dir() / f"{skill_name}.json").exists()

    def _reinforce_skill(self, skill_name: str) -> None:
        """Bump confidence and use_count on an existing shared skill."""
        skill_path = _shared_skills_dir() / f"{skill_name}.json"
        if not skill_path.exists():
            return
        try:
            skill = json.loads(skill_path.read_text())
            skill["use_count"] = int(skill.get("use_count", 0)) + 1
            skill["confidence"] = min(1.0, float(skill.get("confidence", 0.5)) + 0.05)
            skill["updated_at"] = datetime.now(timezone.utc).isoformat()
            skill_path.write_text(json.dumps(skill, indent=2) + "\n")
            self.storage.event_log.append(
                "skillforge.reinforced",
                {"skill_name": skill_name, "confidence": skill["confidence"]},
            )
        except (json.JSONDecodeError, OSError):
            pass

    def _write_skill(self, skill: dict[str, object]) -> None:
        """Write a skill to the shared store."""
        skills_dir = _shared_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skills_dir / f"{skill['name']}.json"
        skill_path.write_text(json.dumps(skill, indent=2) + "\n")
