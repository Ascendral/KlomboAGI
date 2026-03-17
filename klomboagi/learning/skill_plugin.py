"""
Skill Plugin — Self-modifying plugin system for KlomboAGI.

Agent writes Python plugin code, validates safety via blocklist,
stages in a sandbox directory, and registers as an executable skill.

Double safety gate:
  1. Hardcoded blocklist (no subprocess, no os.system, no eval, no network)
  2. Structural validation before promotion

Staging: ~/.codebot/plugins/staging/
Active:  ~/.codebot/plugins/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from klomboagi.learning.skill_forge import _shared_skills_dir
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


# APIs that plugins must not use
BLOCKED_APIS = [
    "subprocess",
    "os.system",
    "os.popen",
    "os.exec",
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "importlib",
    "shutil.rmtree",
    "socket",
    "http.client",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "open(",  # file I/O — plugins should use provided APIs only
]


def is_plugin_safe(source_code: str) -> str | None:
    """Check if plugin source code is safe. Returns None if safe, error message if blocked."""
    for blocked in BLOCKED_APIS:
        if blocked in source_code:
            return f'Plugin contains blocked API: "{blocked}". Plugins cannot use filesystem, network, process, or eval APIs.'
    return None


class SkillPlugin:
    """Creates, validates, and manages self-authored skill plugins."""

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def create(
        self,
        name: str,
        description: str,
        code: str,
        trigger: str = "",
    ) -> dict[str, object]:
        """Create a plugin in the staging directory.

        Returns result dict with success, message, and optional path.
        """
        if not name or not code:
            return {"success": False, "message": "name and code are required"}

        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        if not safe_name:
            return {"success": False, "message": "Invalid plugin name"}

        # Gate 1: Blocklist safety check
        safety_error = is_plugin_safe(code)
        if safety_error:
            self.storage.event_log.append("skill_plugin.blocked", {
                "name": safe_name, "reason": safety_error,
            })
            return {"success": False, "message": f"BLOCKED: {safety_error}"}

        # Write to staging
        staging_dir = self._staging_dir()
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Create skill JSON
        skill = {
            "name": f"plugin_{safe_name}",
            "description": description,
            "trigger": trigger,
            "steps": [{"tool": "think", "args": {"thought": f"Execute plugin: {safe_name}"}}],
            "author": "klomboagi-plugin-forge",
            "confidence": 0.5,
            "use_count": 0,
            "origin": "plugin",
            "plugin_code": code,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        # Generate hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        skill["plugin_hash"] = f"sha256:{code_hash}"

        skill_path = staging_dir / f"plugin_{safe_name}.json"
        skill_path.write_text(json.dumps(skill, indent=2))

        self.storage.event_log.append("skill_plugin.created", {
            "name": safe_name, "hash": skill["plugin_hash"],
        })

        return {
            "success": True,
            "message": f'Plugin "{safe_name}" created in staging',
            "path": str(skill_path),
            "hash": skill["plugin_hash"],
        }

    def validate(self, name: str) -> dict[str, object]:
        """Validate a staged plugin for safety and structure."""
        safe_name = f"plugin_{name}" if not name.startswith("plugin_") else name
        staging_path = self._staging_dir() / f"{safe_name}.json"

        if not staging_path.exists():
            return {"valid": False, "issues": [f'Plugin "{name}" not found in staging']}

        issues: list[str] = []
        try:
            data = json.loads(staging_path.read_text())
        except json.JSONDecodeError:
            return {"valid": False, "issues": ["Invalid JSON"]}

        # Check structure
        if not data.get("name"):
            issues.append("Missing name")
        if not data.get("steps"):
            issues.append("Missing steps")

        # Check plugin code safety
        code = str(data.get("plugin_code", ""))
        if code:
            safety_error = is_plugin_safe(code)
            if safety_error:
                issues.append(f"BLOCKED: {safety_error}")

            # Verify hash
            expected_hash = data.get("plugin_hash", "")
            actual_hash = f"sha256:{hashlib.sha256(code.encode()).hexdigest()}"
            if expected_hash and expected_hash != actual_hash:
                issues.append(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")

        return {"valid": len(issues) == 0, "issues": issues}

    def promote(self, name: str) -> dict[str, object]:
        """Move a validated plugin from staging to the active skill store."""
        safe_name = f"plugin_{name}" if not name.startswith("plugin_") else name
        staging_path = self._staging_dir() / f"{safe_name}.json"

        if not staging_path.exists():
            return {"success": False, "message": f'Plugin "{name}" not found in staging'}

        # Re-validate
        validation = self.validate(name)
        if not validation["valid"]:
            return {"success": False, "message": f"Validation failed: {validation['issues']}"}

        # Move to shared skills directory
        skills_dir = _shared_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)

        data = json.loads(staging_path.read_text())
        dest_path = skills_dir / f"{safe_name}.json"
        dest_path.write_text(json.dumps(data, indent=2))

        # Remove from staging
        staging_path.unlink()

        self.storage.event_log.append("skill_plugin.promoted", {
            "name": safe_name,
        })

        return {"success": True, "message": f'Plugin "{name}" promoted to active skills'}

    def remove(self, name: str) -> dict[str, object]:
        """Remove a plugin from staging and/or active directory."""
        safe_name = f"plugin_{name}" if not name.startswith("plugin_") else name
        removed = False

        # Remove from active
        active_path = _shared_skills_dir() / f"{safe_name}.json"
        if active_path.exists():
            active_path.unlink()
            removed = True

        # Remove from staging
        staging_path = self._staging_dir() / f"{safe_name}.json"
        if staging_path.exists():
            staging_path.unlink()
            removed = True

        return {
            "success": removed,
            "message": f'Plugin "{name}" removed' if removed else f'Plugin "{name}" not found',
        }

    def list_plugins(self) -> dict[str, list[str]]:
        """List active and staged plugins."""
        active = self._list_in_dir(_shared_skills_dir(), prefix="plugin_")
        staging = self._list_in_dir(self._staging_dir(), prefix="plugin_")
        return {"active": active, "staging": staging}

    # ── Internal ──

    def _staging_dir(self) -> Path:
        import os
        codebot_home = os.environ.get("CODEBOT_HOME", str(Path.home() / ".codebot"))
        return Path(codebot_home) / "plugins" / "staging"

    def _list_in_dir(self, directory: Path, prefix: str = "") -> list[str]:
        if not directory.exists():
            return []
        return [
            f.stem for f in sorted(directory.glob("*.json"))
            if f.stem.startswith(prefix)
        ]
