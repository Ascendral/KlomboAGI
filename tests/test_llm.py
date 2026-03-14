"""Tests for LLM integration — provider, planner, critic, and reflection fallbacks.

All tests run without an actual LLM server.  HTTP calls are mocked using a
simple monkey-patch on urllib.request.urlopen.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codeagi.llm.provider import complete
from codeagi.reasoning.planner import Planner
from codeagi.reasoning.critic import Critic
from codeagi.learning.reflection import ReflectionEngine
from codeagi.storage.manager import StorageManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal file-like object returned by a mocked urlopen."""

    def __init__(self, body: dict) -> None:
        self._data = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _openai_response(content: str) -> dict:
    """Build a minimal OpenAI chat-completion response dict."""
    return {
        "choices": [{"message": {"content": content}}],
    }


def _mock_urlopen_factory(content: str):
    """Return a callable that replaces urlopen and returns *content*."""
    def _mock_urlopen(req, *, timeout=None):
        return _FakeResponse(_openai_response(content))
    return _mock_urlopen


def _mock_urlopen_error(req, *, timeout=None):
    """Simulate a connection-refused error."""
    raise OSError("Connection refused")


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------

class TestLLMProvider(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["CODEAGI_LLM_ENABLED"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_LLM_ENABLED", None)

    def test_returns_empty_string_when_disabled(self) -> None:
        os.environ["CODEAGI_LLM_ENABLED"] = "0"
        result = complete("system", "user")
        self.assertEqual(result, "")

    def test_returns_empty_string_when_api_unreachable(self) -> None:
        with patch("codeagi.llm.provider.urllib.request.urlopen", side_effect=OSError("Connection refused")):
            result = complete("system", "user")
        self.assertEqual(result, "")

    def test_returns_content_on_success(self) -> None:
        expected = "Hello from the LLM"
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory(expected)):
            result = complete("system", "user")
        self.assertEqual(result, expected)

    def test_returns_empty_on_empty_choices(self) -> None:
        def _mock(req, *, timeout=None):
            return _FakeResponse({"choices": []})
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock):
            result = complete("system", "user")
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# Planner fallback tests
# ---------------------------------------------------------------------------

class TestPlannerLLMFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_LLM_ENABLED"] = "1"
        self.storage = StorageManager.bootstrap()
        self.planner = Planner(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_LLM_ENABLED", None)
        self.temp_dir.cleanup()

    def test_draft_task_falls_back_when_llm_unavailable(self) -> None:
        mission = {"id": "m1", "description": "search repo for deploy_app"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", side_effect=OSError):
            task = self.planner.draft_task(mission)
        self.assertEqual(task["action_kind"], "search_files")

    def test_draft_task_falls_back_when_llm_disabled(self) -> None:
        os.environ["CODEAGI_LLM_ENABLED"] = "0"
        mission = {"id": "m1", "description": "search repo for deploy_app"}
        task = self.planner.draft_task(mission)
        self.assertEqual(task["action_kind"], "search_files")

    def test_draft_task_uses_llm_when_available(self) -> None:
        llm_response = json.dumps({
            "action": "read_file",
            "description": "Read the deploy script",
            "args": {"path": "deploy.sh"},
        })
        mission = {"id": "m1", "description": "inspect deployment"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory(llm_response)):
            task = self.planner.draft_task(mission)
        self.assertEqual(task["action_kind"], "read_file")
        self.assertEqual(task["action_payload"]["path"], "deploy.sh")

    def test_llm_plan_returns_none_on_bad_json(self) -> None:
        mission = {"id": "m1", "description": "do something"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory("not json")):
            result = self.planner.llm_plan(mission)
        self.assertIsNone(result)

    def test_llm_plan_returns_none_on_invalid_action(self) -> None:
        llm_response = json.dumps({"action": "destroy_everything", "description": "bad"})
        mission = {"id": "m1", "description": "do something"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory(llm_response)):
            result = self.planner.llm_plan(mission)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Critic fallback tests
# ---------------------------------------------------------------------------

class TestCriticLLMFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_LLM_ENABLED"] = "1"
        self.storage = StorageManager.bootstrap()
        self.critic = Critic(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_LLM_ENABLED", None)
        self.temp_dir.cleanup()

    def test_critique_falls_back_to_rules_when_llm_unavailable(self) -> None:
        mission = {"id": "m1", "description": "test"}
        plan = {"id": "p1", "steps": []}
        action = {"type": "execute_task", "description": "do it", "task_id": "t1"}
        verification = {"valid": True, "warnings": []}
        with patch("codeagi.llm.provider.urllib.request.urlopen", side_effect=OSError):
            result = self.critic.critique(
                mission=mission, tasks=[], plan=plan,
                proposed_action=action, verification=verification,
            )
        self.assertTrue(result["approved"])

    def test_llm_critic_can_block_action(self) -> None:
        llm_response = json.dumps({"approved": False, "reason": "dangerous"})
        mission = {"id": "m1", "description": "test"}
        plan = {"id": "p1", "steps": []}
        action = {"type": "execute_task", "description": "rm -rf /", "task_id": "t1"}
        verification = {"valid": True, "warnings": []}
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory(llm_response)):
            result = self.critic.critique(
                mission=mission, tasks=[], plan=plan,
                proposed_action=action, verification=verification,
            )
        self.assertFalse(result["approved"])
        self.assertEqual(result["final_action"]["type"], "replan")

    def test_llm_critique_returns_none_on_bad_json(self) -> None:
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory("nope")):
            result = self.critic.llm_critique({"type": "test"})
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Reflection fallback tests
# ---------------------------------------------------------------------------

class TestReflectionLLMFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
        os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
        os.environ["CODEAGI_LLM_ENABLED"] = "1"
        self.storage = StorageManager.bootstrap()
        self.reflection = ReflectionEngine(self.storage)

    def tearDown(self) -> None:
        os.environ.pop("CODEAGI_RUNTIME_ROOT", None)
        os.environ.pop("CODEAGI_LONG_TERM_ROOT", None)
        os.environ.pop("CODEAGI_LLM_ENABLED", None)
        self.temp_dir.cleanup()

    def test_reflect_falls_back_to_templates_when_llm_unavailable(self) -> None:
        mission = {"id": "m1", "description": "test mission"}
        tasks = [{"id": "t1", "status": "completed"}]
        memory = {"blockers": [], "current_focus": "test"}
        action = {"type": "execute_task", "description": "do it", "task_id": "t1"}
        outcome = {"summary": "task done", "status": "completed"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", side_effect=OSError):
            result = self.reflection.reflect(mission, tasks, memory, action, outcome)
        self.assertIn("lessons", result)
        self.assertTrue(any("complete" in lesson.lower() for lesson in result["lessons"]))

    def test_reflect_uses_llm_lesson_when_available(self) -> None:
        llm_response = json.dumps({
            "lesson": "Always validate inputs before processing.",
            "should_continue": True,
            "blockers": [],
        })
        mission = {"id": "m1", "description": "test mission"}
        tasks = [{"id": "t1", "status": "queued"}]
        memory = {"blockers": [], "current_focus": "test"}
        action = {"type": "execute_task", "description": "do it", "task_id": "t1"}
        outcome = {"summary": "task done", "status": "completed"}
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory(llm_response)):
            result = self.reflection.reflect(mission, tasks, memory, action, outcome)
        self.assertTrue(any("validate" in lesson.lower() for lesson in result["lessons"]))

    def test_llm_reflect_returns_none_on_bad_json(self) -> None:
        with patch("codeagi.llm.provider.urllib.request.urlopen", _mock_urlopen_factory("not json")):
            result = self.reflection.llm_reflect({"summary": "test"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
