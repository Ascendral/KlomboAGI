from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def normalize_path(path: str) -> str:
    return Path(path).as_posix().strip("/")


@dataclass
class ActionRecord:
    tool: str
    success: bool
    summary: str = ""
    command: str | None = None
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    repo_id: str
    repo_path: str
    task_type: str
    request: str
    success: bool
    actions: list[ActionRecord]
    id: str = field(default_factory=lambda: new_id("episode"))
    mission_id: str | None = None
    plan_summary: str | None = None
    files_touched: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    user_feedback: str | None = None
    observed_preferences: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        payload["files_touched"] = [normalize_path(path) for path in self.files_touched]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Episode":
        actions = [
            action if isinstance(action, ActionRecord) else ActionRecord(**action)
            for action in payload.get("actions", [])
        ]
        return cls(
            id=str(payload.get("id") or new_id("episode")),
            mission_id=payload.get("mission_id"),
            repo_id=str(payload["repo_id"]),
            repo_path=str(payload["repo_path"]),
            task_type=str(payload["task_type"]),
            request=str(payload["request"]),
            plan_summary=payload.get("plan_summary"),
            actions=actions,
            files_touched=[normalize_path(path) for path in payload.get("files_touched", [])],
            commands=[str(command) for command in payload.get("commands", [])],
            success=bool(payload["success"]),
            stop_reason=payload.get("stop_reason"),
            user_feedback=payload.get("user_feedback"),
            observed_preferences=dict(payload.get("observed_preferences", {})),
            created_at=str(payload.get("created_at") or utc_now()),
        )


@dataclass
class RepoProfile:
    repo_id: str
    repo_path: str
    languages: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    repo_family: str | None = None
    command_families: dict[str, int] = field(default_factory=dict)
    preferred_test_commands: list[str] = field(default_factory=list)
    preferred_lint_commands: list[str] = field(default_factory=list)
    preferred_build_commands: list[str] = field(default_factory=list)
    common_paths: list[str] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    service_boundaries: list[str] = field(default_factory=list)
    ownership_zones: list[str] = field(default_factory=list)
    dependency_edges: list[str] = field(default_factory=list)
    architecture_summary: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    confidence: float = 0.5
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcedureMemory:
    name: str
    repo_id: str | None
    task_type: str
    trigger_terms: list[str]
    action_chain: list[str]
    id: str = field(default_factory=lambda: new_id("procedure"))
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    last_outcome: str = "unknown"
    source_episode_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AntiPatternMemory:
    name: str
    repo_id: str | None
    task_type: str
    trigger_terms: list[str]
    failing_chain: list[str]
    id: str = field(default_factory=lambda: new_id("anti"))
    failure_count: int = 0
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UserPreference:
    key: str
    value: Any
    repo_id: str | None = None
    id: str = field(default_factory=lambda: new_id("pref"))
    confidence: float = 0.5
    evidence_count: int = 1
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissionState:
    mission_id: str
    repo_id: str
    summary: str
    status: str
    last_plan: str | None = None
    attempted_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    next_best_step: str | None = None
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorReviewDecision:
    mission_id: str
    repo_id: str
    selected_option: str
    approved: bool = True
    selected_step: str | None = None
    notes: str | None = None
    id: str = field(default_factory=lambda: new_id("review"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
