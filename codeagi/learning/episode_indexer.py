"""
Episode Indexer — Cross-Session Learning for CodeAGI.

Records cognition cycle episodes, indexes tool chain patterns,
and feeds effective patterns back into future system context.

Episodes capture: mission, actions taken, tools used, outcomes,
and patterns discovered. The index aggregates patterns across
sessions for confidence adjustment and prompt injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from codeagi.storage.manager import StorageManager
from codeagi.utils.time import utc_now


@dataclass
class EpisodePattern:
    """A tool chain pattern observed during a session."""
    description: str
    tool_chain: list[str]
    effective: bool
    frequency: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "tool_chain": self.tool_chain,
            "effective": self.effective,
            "frequency": self.frequency,
        }


@dataclass
class Episode:
    """A recorded cognition cycle session."""
    session_id: str
    mission_id: str
    goal: str
    started_at: str
    ended_at: str
    tools_used: list[str]
    action_count: int
    success: bool
    outcomes: list[str]
    patterns: list[EpisodePattern] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "mission_id": self.mission_id,
            "goal": self.goal,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tools_used": self.tools_used,
            "action_count": self.action_count,
            "success": self.success,
            "outcomes": self.outcomes,
            "patterns": [p.to_dict() for p in self.patterns],
        }


@dataclass
class AggregatedPattern:
    """A pattern aggregated across multiple sessions."""
    description: str
    tool_chain: list[str]
    success_count: int = 0
    failure_count: int = 0
    total_occurrences: int = 0
    success_rate: float = 0.0
    last_seen: str = ""
    session_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "tool_chain": self.tool_chain,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_occurrences": self.total_occurrences,
            "success_rate": round(self.success_rate, 3),
            "last_seen": self.last_seen,
            "session_ids": self.session_ids,
        }


class EpisodeIndexer:
    """Records and indexes episodes for cross-session learning."""

    MAX_EPISODES = 100
    MAX_SESSION_IDS_PER_PATTERN = 20

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self._pattern_index: dict[str, AggregatedPattern] | None = None

    def record_episode(self, episode: Episode) -> None:
        """Record a completed episode and update the pattern index."""
        # Save episode to episodes store
        episodes = self.storage.episodes.load(default=[])
        episodes.append(episode.to_dict())
        if len(episodes) > self.MAX_EPISODES:
            episodes = episodes[-self.MAX_EPISODES:]
        self.storage.episodes.save(episodes)

        # Update pattern index
        self._update_index(episode)

        self.storage.event_log.append("episode_indexer.recorded", {
            "session_id": episode.session_id,
            "mission_id": episode.mission_id,
            "success": episode.success,
            "patterns": len(episode.patterns),
        })

    def build_episode(
        self,
        session_id: str,
        mission_id: str,
        goal: str,
        started_at: str,
        actions: list[dict[str, object]],
        success: bool,
        outcomes: list[str],
    ) -> Episode:
        """Build an episode from cycle data."""
        tools_used = list({
            str(a.get("tool_name") or a.get("action_type", "unknown"))
            for a in actions
        })
        tool_calls = [
            {"tool": str(a.get("tool_name") or a.get("action_type", "")),
             "success": bool(a.get("success", True))}
            for a in actions
        ]
        patterns = self.extract_patterns(tool_calls)

        return Episode(
            session_id=session_id,
            mission_id=mission_id,
            goal=goal,
            started_at=started_at,
            ended_at=utc_now(),
            tools_used=tools_used,
            action_count=len(actions),
            success=success,
            outcomes=outcomes,
            patterns=patterns,
        )

    def extract_patterns(
        self, tool_calls: list[dict[str, object]]
    ) -> list[EpisodePattern]:
        """Extract tool chain patterns from a sequence of tool calls."""
        if len(tool_calls) < 2:
            return []

        pattern_map: dict[str, EpisodePattern] = {}

        for window_size in range(2, min(5, len(tool_calls) + 1)):
            for i in range(len(tool_calls) - window_size + 1):
                window = tool_calls[i : i + window_size]
                chain = [str(t.get("tool", "")) for t in window]
                key = " → ".join(chain)

                if key not in pattern_map:
                    effective = all(t.get("success", True) for t in window)
                    pattern_map[key] = EpisodePattern(
                        description=key,
                        tool_chain=chain,
                        effective=effective,
                        frequency=1,
                    )
                else:
                    pattern_map[key].frequency += 1

        # Keep patterns seen more than once or that are effective
        return [p for p in pattern_map.values() if p.frequency > 1 or p.effective]

    def get_top_patterns(self, n: int = 3) -> list[AggregatedPattern]:
        """Get top N patterns by success rate."""
        index = self._load_index()
        patterns = list(index.values())

        return sorted(
            [p for p in patterns if p.total_occurrences >= 2],
            key=lambda p: (p.success_rate, p.total_occurrences),
            reverse=True,
        )[:n]

    def get_anti_patterns(self, n: int = 3) -> list[AggregatedPattern]:
        """Get patterns with low success rate to avoid."""
        index = self._load_index()
        patterns = list(index.values())

        return sorted(
            [p for p in patterns if p.total_occurrences >= 3 and p.success_rate < 0.3],
            key=lambda p: p.success_rate,
        )[:n]

    def build_context_block(self) -> str:
        """Build a context block for injection into cognition prompts."""
        top = self.get_top_patterns(3)
        anti = self.get_anti_patterns(2)

        if not top and not anti:
            return ""

        lines = ["Cross-session patterns:"]

        if top:
            lines.append("Effective approaches:")
            for p in top:
                rate = round(p.success_rate * 100)
                lines.append(f"  - {' → '.join(p.tool_chain)} ({rate}% success, {p.total_occurrences}x)")

        if anti:
            lines.append("Approaches to avoid:")
            for p in anti:
                rate = round(p.success_rate * 100)
                lines.append(f"  - {' → '.join(p.tool_chain)} ({rate}% success)")

        return "\n".join(lines)

    def summarize(self) -> str:
        """Get a summary of cross-session learning state."""
        episodes = self.storage.episodes.load(default=[])
        index = self._load_index()

        if not episodes:
            return "No cross-session data recorded."

        top = self.get_top_patterns(3)
        lines = [f"Cross-Session Learning: {len(episodes)} episodes, {len(index)} patterns"]

        if top:
            lines.append("Top patterns:")
            for p in top:
                rate = round(p.success_rate * 100)
                lines.append(f"  {' → '.join(p.tool_chain)} — {rate}% success ({p.total_occurrences}x)")

        return "\n".join(lines)

    def prune(self, keep_count: int = 50) -> int:
        """Remove old episodes, keeping the most recent N."""
        episodes = self.storage.episodes.load(default=[])
        if len(episodes) <= keep_count:
            return 0

        pruned = len(episodes) - keep_count
        self.storage.episodes.save(episodes[-keep_count:])
        return pruned

    # ── Internal ──

    def _load_index(self) -> dict[str, AggregatedPattern]:
        """Load the pattern index from storage."""
        if self._pattern_index is not None:
            return self._pattern_index

        raw = self.storage.episode_index.load(default={})
        index: dict[str, AggregatedPattern] = {}

        if isinstance(raw, dict):
            for key, val in raw.items():
                if isinstance(val, dict):
                    index[key] = AggregatedPattern(
                        description=str(val.get("description", key)),
                        tool_chain=list(val.get("tool_chain", [])),
                        success_count=int(val.get("success_count", 0)),
                        failure_count=int(val.get("failure_count", 0)),
                        total_occurrences=int(val.get("total_occurrences", 0)),
                        success_rate=float(val.get("success_rate", 0.0)),
                        last_seen=str(val.get("last_seen", "")),
                        session_ids=list(val.get("session_ids", [])),
                    )

        self._pattern_index = index
        return index

    def _update_index(self, episode: Episode) -> None:
        """Update the pattern index with patterns from an episode."""
        index = self._load_index()

        for pattern in episode.patterns:
            key = ":".join(pattern.tool_chain)

            if key not in index:
                index[key] = AggregatedPattern(
                    description=pattern.description,
                    tool_chain=pattern.tool_chain,
                )

            agg = index[key]
            agg.total_occurrences += pattern.frequency
            if pattern.effective:
                agg.success_count += pattern.frequency
            else:
                agg.failure_count += pattern.frequency
            agg.success_rate = (
                agg.success_count / agg.total_occurrences
                if agg.total_occurrences > 0
                else 0.0
            )
            agg.last_seen = episode.ended_at

            if episode.session_id not in agg.session_ids:
                agg.session_ids.append(episode.session_id)
                if len(agg.session_ids) > self.MAX_SESSION_IDS_PER_PATTERN:
                    agg.session_ids = agg.session_ids[-self.MAX_SESSION_IDS_PER_PATTERN:]

        self._pattern_index = index

        # Persist
        serialized = {k: v.to_dict() for k, v in index.items()}
        self.storage.episode_index.save(serialized)
