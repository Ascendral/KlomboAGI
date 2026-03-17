"""
Autonomous Goal Refiner for KlomboAGI.

Takes a high-level mission and decomposes it into a dependency-ordered
task graph. Works with the Planner and ActionExecutor to break complex
missions into manageable subtasks that can be executed step by step.

Decomposition strategies:
  - LLM-based: Ask the LLM to propose subtasks
  - Heuristic: Pattern-match on mission description keywords
  - Recursive: Subtasks can be further decomposed (up to max_depth)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from klomboagi.core.mission import MissionManager
from klomboagi.llm import complete as llm_complete
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class SubtaskNode:
    id: str
    description: str
    parent_id: str | None
    depth: int
    status: str = "pending"  # pending | ready | in_progress | completed | failed | skipped
    dependencies: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    action_kind: str | None = None
    action_payload: dict[str, object] = field(default_factory=dict)
    output: str | None = None
    error: str | None = None
    created_at: str = ""
    completed_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "action_kind": self.action_kind,
            "action_payload": self.action_payload,
            "output": self.output,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


_LLM_DECOMPOSE_SYSTEM = (
    "You are a goal decomposition engine for an autonomous coding agent. "
    "Given a high-level mission, break it into 2-5 concrete subtasks with dependencies. "
    "Respond with JSON: {\"subtasks\": [{\"description\": \"...\", \"action_kind\": "
    "\"search_files|read_file|write_file|apply_patch|run_command|null\", "
    "\"depends_on\": [0]}]} where depends_on references sibling indices."
)


class GoalRefiner:
    """Decomposes missions into dependency-ordered subtask trees."""

    def __init__(self, storage: StorageManager, max_depth: int = 3) -> None:
        self.storage = storage
        self.missions = MissionManager(storage)
        self.max_depth = max_depth
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"subtask_{self._id_counter}"

    def decompose(self, mission: dict[str, object]) -> list[SubtaskNode]:
        """Decompose a mission into subtask nodes. Returns flat list of all nodes."""
        root_id = self._next_id()
        root = SubtaskNode(
            id=root_id,
            description=str(mission["description"]),
            parent_id=None,
            depth=0,
            created_at=utc_now(),
        )

        nodes: dict[str, SubtaskNode] = {root_id: root}
        self._expand(mission, root, nodes)
        self._update_ready(nodes)

        # Save decomposition strategy for learning
        strategies = self.storage.decomposition_strategies.load(default=[])
        strategies.append({
            "mission_id": str(mission["id"]),
            "description": str(mission["description"]),
            "node_count": len(nodes),
            "created_at": utc_now(),
        })
        self.storage.decomposition_strategies.save(strategies)

        self.storage.event_log.append(
            "goal.decomposed",
            {
                "mission_id": str(mission["id"]),
                "node_count": len(nodes),
                "max_depth": max(n.depth for n in nodes.values()),
            },
        )

        return list(nodes.values())

    def create_tasks_from_decomposition(
        self,
        mission: dict[str, object],
        nodes: list[SubtaskNode],
    ) -> list[dict[str, object]]:
        """Convert subtask nodes into real tasks in the MissionManager."""
        created: list[dict[str, object]] = []
        node_to_task: dict[str, str] = {}

        # Create tasks in dependency order (by depth then index)
        sorted_nodes = sorted(nodes, key=lambda n: (n.depth, n.id))

        for node in sorted_nodes:
            if node.depth == 0:
                continue  # Root node is the mission itself

            if node.subtasks:
                continue  # Non-leaf nodes don't need tasks

            # Map dependency node IDs to task IDs
            task_deps = []
            for dep_id in node.dependencies:
                if dep_id in node_to_task:
                    task_deps.append(node_to_task[dep_id])

            task = self.missions.create_task(
                str(mission["id"]),
                node.description,
                action_kind=node.action_kind,
                action_payload=node.action_payload or {},
                dependencies=task_deps,
            )
            node_to_task[node.id] = task.id
            created.append(task.to_dict() if hasattr(task, "to_dict") else {"id": task.id, "description": task.description})

        return created

    def get_ready_nodes(self, nodes: list[SubtaskNode]) -> list[SubtaskNode]:
        """Get leaf nodes that are ready to execute."""
        return [n for n in nodes if n.status == "ready" and not n.subtasks]

    def complete_node(self, nodes: list[SubtaskNode], node_id: str, output: str | None = None) -> None:
        """Mark a node as completed and update ready states."""
        node_map = {n.id: n for n in nodes}
        node = node_map.get(node_id)
        if not node:
            return

        node.status = "completed"
        node.output = output
        node.completed_at = utc_now()

        self._check_parent(node_map, node)
        self._update_ready(node_map)

    def fail_node(self, nodes: list[SubtaskNode], node_id: str, error: str) -> None:
        """Mark a node as failed, skip dependents."""
        node_map = {n.id: n for n in nodes}
        node = node_map.get(node_id)
        if not node:
            return

        node.status = "failed"
        node.error = error
        node.completed_at = utc_now()

        self._skip_dependents(node_map, node_id)
        self._check_parent(node_map, node)
        self._update_ready(node_map)

    def is_finished(self, nodes: list[SubtaskNode]) -> bool:
        """Check if the root node has reached a terminal state."""
        for node in nodes:
            if node.depth == 0:
                return node.status in ("completed", "failed")
        return False

    def summarize(self, nodes: list[SubtaskNode]) -> str:
        """Human-readable summary of decomposition state."""
        counts: dict[str, int] = {}
        for node in nodes:
            counts[node.status] = counts.get(node.status, 0) + 1

        root = next((n for n in nodes if n.depth == 0), None)
        lines = [
            f"Goal: {root.description if root else '?'}",
            f"Status: {root.status if root else '?'}",
            f"Nodes: {len(nodes)} ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})",
        ]

        # Build indented tree
        node_map = {n.id: n for n in nodes}
        if root:
            lines.append("")
            lines.extend(self._format_tree(node_map, root.id, 0))

        return "\n".join(lines)

    # ── Internal ──

    def _expand(self, mission: dict[str, object], node: SubtaskNode, nodes: dict[str, SubtaskNode]) -> None:
        """Recursively expand a node using LLM or heuristics."""
        if node.depth >= self.max_depth:
            return

        subtask_drafts = self._llm_decompose(mission, node)
        if not subtask_drafts:
            subtask_drafts = self._heuristic_decompose(node.description)
        if not subtask_drafts:
            return

        child_ids: list[str] = []
        for draft in subtask_drafts:
            child_id = self._next_id()
            child_ids.append(child_id)

            # Resolve relative dependencies
            deps: list[str] = []
            for ref in draft.get("depends_on", []):
                idx = int(ref) if isinstance(ref, (int, str)) else -1
                if 0 <= idx < len(child_ids):
                    deps.append(child_ids[idx])

            child = SubtaskNode(
                id=child_id,
                description=str(draft["description"]),
                parent_id=node.id,
                depth=node.depth + 1,
                action_kind=draft.get("action_kind"),
                action_payload=draft.get("action_payload", {}),
                dependencies=deps,
                created_at=utc_now(),
            )
            nodes[child_id] = child

            # Recurse if description is complex enough
            if node.depth + 1 < self.max_depth and len(child.description.split()) > 15:
                self._expand(mission, child, nodes)

        node.subtasks = child_ids

    def _llm_decompose(self, mission: dict[str, object], node: SubtaskNode) -> list[dict[str, object]] | None:
        """Use LLM to decompose a goal node."""
        prompt = (
            f"Mission: {mission['description']}\n"
            f"Current goal to decompose: {node.description}\n"
            f"Depth: {node.depth}/{self.max_depth}"
        )
        raw = llm_complete(_LLM_DECOMPOSE_SYSTEM, prompt)
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

        subtasks = parsed.get("subtasks", [])
        if not isinstance(subtasks, list) or len(subtasks) < 2:
            return None

        return subtasks

    def _heuristic_decompose(self, description: str) -> list[dict[str, object]] | None:
        """Pattern-match decomposition strategies."""
        lower = description.lower()

        if re.search(r"\b(fix|bug|errors?|crash|broken|debug)\b", lower):
            return [
                {"description": f"Search for error patterns: {_trunc(description)}", "action_kind": "search_files", "depends_on": []},
                {"description": "Read and analyze relevant files", "action_kind": "read_file", "depends_on": [0]},
                {"description": f"Apply fix: {_trunc(description)}", "action_kind": "apply_patch", "depends_on": [1]},
                {"description": "Run tests to verify fix", "action_kind": "run_command", "depends_on": [2]},
            ]

        if re.search(r"\b(add|create|implement|build|new|introduce)\b", lower):
            return [
                {"description": "Analyze codebase patterns", "action_kind": "search_files", "depends_on": []},
                {"description": f"Plan implementation: {_trunc(description)}", "action_kind": None, "depends_on": [0]},
                {"description": f"Write implementation: {_trunc(description)}", "action_kind": "write_file", "depends_on": [1]},
                {"description": "Write tests", "action_kind": "write_file", "depends_on": [2]},
                {"description": "Run tests", "action_kind": "run_command", "depends_on": [3]},
            ]

        if re.search(r"\b(refactor|restructure|reorganize|clean|simplify|optimize)\b", lower):
            return [
                {"description": f"Identify affected files: {_trunc(description)}", "action_kind": "search_files", "depends_on": []},
                {"description": "Run baseline tests", "action_kind": "run_command", "depends_on": []},
                {"description": "Apply refactoring", "action_kind": "apply_patch", "depends_on": [0, 1]},
                {"description": "Verify no regressions", "action_kind": "run_command", "depends_on": [2]},
            ]

        if re.search(r"\b(tests?|spec|coverage)\b", lower):
            return [
                {"description": f"Analyze code: {_trunc(description)}", "action_kind": "read_file", "depends_on": []},
                {"description": "Write test cases", "action_kind": "write_file", "depends_on": [0]},
                {"description": "Run tests", "action_kind": "run_command", "depends_on": [1]},
            ]

        if re.search(r"\b(research|investigate|explore|understand|audit)\b", lower):
            return [
                {"description": f"Search codebase: {_trunc(description)}", "action_kind": "search_files", "depends_on": []},
                {"description": "Read discovered files", "action_kind": "read_file", "depends_on": [0]},
                {"description": "Synthesize findings", "action_kind": None, "depends_on": [1]},
            ]

        return None

    def _update_ready(self, nodes: dict[str, SubtaskNode]) -> None:
        """Mark pending nodes as ready if all dependencies are completed."""
        completed = {nid for nid, n in nodes.items() if n.status == "completed"}
        for node in nodes.values():
            if node.status != "pending":
                continue
            if all(d in completed for d in node.dependencies):
                if node.subtasks:
                    node.status = "in_progress"
                else:
                    node.status = "ready"

    def _check_parent(self, nodes: dict[str, SubtaskNode], node: SubtaskNode) -> None:
        """Propagate completion/failure up to parent."""
        if not node.parent_id:
            return
        parent = nodes.get(node.parent_id)
        if not parent:
            return

        terminal = {"completed", "skipped", "failed"}
        all_terminal = all(
            nodes[cid].status in terminal
            for cid in parent.subtasks
            if cid in nodes
        )
        if not all_terminal:
            return

        any_failed = any(
            nodes[cid].status == "failed"
            for cid in parent.subtasks
            if cid in nodes
        )

        if any_failed:
            parent.status = "failed"
            parent.error = "One or more subtasks failed"
        else:
            parent.status = "completed"
            outputs = [
                nodes[cid].output
                for cid in parent.subtasks
                if cid in nodes and nodes[cid].output
            ]
            parent.output = "\n---\n".join(outputs) if outputs else None

        parent.completed_at = utc_now()

        if parent.parent_id:
            self._check_parent(nodes, parent)

    def _skip_dependents(self, nodes: dict[str, SubtaskNode], failed_id: str) -> None:
        """Skip nodes that depend on a failed node."""
        for node in nodes.values():
            if failed_id in node.dependencies and node.status == "pending":
                node.status = "skipped"
                self._skip_dependents(nodes, node.id)

    def _format_tree(self, nodes: dict[str, SubtaskNode], node_id: str, indent: int) -> list[str]:
        """Format a subtree for display."""
        node = nodes.get(node_id)
        if not node:
            return []

        prefix = "  " * indent
        icon = {"completed": "[x]", "ready": "[ ]", "in_progress": "[~]", "failed": "[!]", "skipped": "[-]"}.get(node.status, "[ ]")
        lines = [f"{prefix}{icon} [{node.status}] {node.description}"]

        for child_id in node.subtasks:
            lines.extend(self._format_tree(nodes, child_id, indent + 1))

        return lines


def _trunc(s: str, max_len: int = 80) -> str:
    return s[:max_len - 3] + "..." if len(s) > max_len else s
