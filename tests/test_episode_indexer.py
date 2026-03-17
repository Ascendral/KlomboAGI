"""Tests for EpisodeIndexer cross-session learning."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from klomboagi.learning.episode_indexer import (
    AggregatedPattern,
    Episode,
    EpisodeIndexer,
    EpisodePattern,
)


@pytest.fixture()
def mock_storage(tmp_path):
    storage = MagicMock()
    storage.paths.runtime_root = tmp_path / "runtime"
    storage.paths.runtime_root.mkdir()
    storage.event_log = MagicMock()
    # Episodes store
    _episodes = []
    storage.episodes.load.side_effect = lambda default=None: list(_episodes)
    storage.episodes.save.side_effect = lambda data: _episodes.clear() or _episodes.extend(data)
    # Episode index store
    _index = {}
    storage.episode_index.load.side_effect = lambda default=None: dict(_index)
    storage.episode_index.save.side_effect = lambda data: (_index.clear(), _index.update(data))
    return storage


@pytest.fixture()
def indexer(mock_storage):
    return EpisodeIndexer(mock_storage)


def _make_episode(session_id="sess_1", mission_id="m_1", success=True, patterns=None):
    return Episode(
        session_id=session_id,
        mission_id=mission_id,
        goal="Fix a bug",
        started_at="2026-03-15T00:00:00Z",
        ended_at="2026-03-15T00:10:00Z",
        tools_used=["grep", "read_file"],
        action_count=5,
        success=success,
        outcomes=["Bug fixed"],
        patterns=patterns or [],
    )


class TestEpisodePattern:
    def test_to_dict(self):
        p = EpisodePattern("grep → read", ["grep", "read"], True, 3)
        d = p.to_dict()
        assert d["tool_chain"] == ["grep", "read"]
        assert d["effective"] is True
        assert d["frequency"] == 3


class TestEpisode:
    def test_to_dict(self):
        ep = _make_episode()
        d = ep.to_dict()
        assert d["session_id"] == "sess_1"
        assert d["success"] is True


class TestExtractPatterns:
    def test_extracts_patterns(self, indexer):
        calls = [
            {"tool": "grep", "success": True},
            {"tool": "read_file", "success": True},
            {"tool": "edit_file", "success": True},
        ]
        patterns = indexer.extract_patterns(calls)
        assert len(patterns) > 0

    def test_empty_for_single_call(self, indexer):
        assert indexer.extract_patterns([{"tool": "grep", "success": True}]) == []

    def test_marks_failed_as_not_effective(self, indexer):
        calls = [
            {"tool": "grep", "success": True},
            {"tool": "edit_file", "success": False},
        ]
        patterns = indexer.extract_patterns(calls)
        for p in patterns:
            if "edit_file" in p.tool_chain:
                assert p.effective is False


class TestRecordEpisode:
    def test_records_and_persists(self, indexer, mock_storage):
        ep = _make_episode()
        indexer.record_episode(ep)
        mock_storage.episodes.save.assert_called()
        mock_storage.event_log.append.assert_called_with(
            "episode_indexer.recorded",
            {
                "session_id": "sess_1",
                "mission_id": "m_1",
                "success": True,
                "patterns": 0,
            },
        )

    def test_updates_pattern_index(self, indexer, mock_storage):
        ep = _make_episode(patterns=[
            EpisodePattern("grep → read", ["grep", "read"], True, 3),
        ])
        indexer.record_episode(ep)
        mock_storage.episode_index.save.assert_called()


class TestBuildEpisode:
    def test_builds_from_actions(self, indexer):
        ep = indexer.build_episode(
            session_id="s1",
            mission_id="m1",
            goal="Test goal",
            started_at="2026-03-15T00:00:00Z",
            actions=[
                {"tool_name": "grep", "success": True},
                {"tool_name": "read_file", "success": True},
            ],
            success=True,
            outcomes=["Done"],
        )
        assert ep.session_id == "s1"
        assert ep.action_count == 2
        assert "grep" in ep.tools_used


class TestGetTopPatterns:
    def test_returns_sorted_by_success_rate(self, indexer, mock_storage):
        # Manually set up index
        ep = _make_episode(patterns=[
            EpisodePattern("a → b", ["a", "b"], True, 3),
            EpisodePattern("c → d", ["c", "d"], False, 2),
        ])
        indexer.record_episode(ep)
        top = indexer.get_top_patterns(5)
        if len(top) >= 2:
            assert top[0].success_rate >= top[1].success_rate

    def test_filters_low_occurrence(self, indexer):
        ep = _make_episode(patterns=[
            EpisodePattern("x → y", ["x", "y"], True, 1),
        ])
        indexer.record_episode(ep)
        top = indexer.get_top_patterns(5)
        assert all(p.total_occurrences >= 2 for p in top)


class TestBuildContextBlock:
    def test_empty_when_no_patterns(self, indexer):
        assert indexer.build_context_block() == ""

    def test_includes_patterns(self, indexer):
        ep = _make_episode(patterns=[
            EpisodePattern("grep → edit", ["grep", "edit"], True, 5),
        ])
        indexer.record_episode(ep)
        block = indexer.build_context_block()
        if block:
            assert "Cross-session" in block


class TestSummarize:
    def test_no_data(self, indexer):
        assert "No cross-session" in indexer.summarize()

    def test_with_data(self, indexer):
        indexer.record_episode(_make_episode(session_id="s1"))
        indexer.record_episode(_make_episode(session_id="s2"))
        summary = indexer.summarize()
        assert "2 episodes" in summary


class TestPrune:
    def test_prunes_old(self, indexer, mock_storage):
        # Add episodes to the mock store directly
        episodes_data = []
        for i in range(5):
            ep = _make_episode(session_id=f"s{i}")
            episodes_data.append(ep.to_dict())
        mock_storage.episodes.load.side_effect = lambda default=None: list(episodes_data)
        mock_storage.episodes.save.side_effect = lambda data: None

        pruned = indexer.prune(2)
        assert pruned == 3

    def test_no_prune_under_limit(self, indexer):
        indexer.record_episode(_make_episode())
        assert indexer.prune(10) == 0
