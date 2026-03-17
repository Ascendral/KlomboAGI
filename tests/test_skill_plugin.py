"""Tests for SkillPlugin self-modifying plugin system."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from klomboagi.learning.skill_plugin import SkillPlugin, is_plugin_safe


@pytest.fixture()
def mock_storage(tmp_path):
    storage = MagicMock()
    storage.paths.runtime_root = tmp_path / "runtime"
    storage.paths.runtime_root.mkdir()
    storage.event_log = MagicMock()
    return storage


@pytest.fixture()
def skills_dir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr("klomboagi.learning.skill_plugin._shared_skills_dir", lambda: d)
    return d


@pytest.fixture()
def staging_dir(tmp_path, monkeypatch):
    d = tmp_path / "plugins" / "staging"
    monkeypatch.setenv("CODEBOT_HOME", str(tmp_path))
    return d


@pytest.fixture()
def plugin(mock_storage):
    return SkillPlugin(mock_storage)


class TestIsPluginSafe:
    def test_allows_safe_code(self):
        assert is_plugin_safe("x = 1 + 2") is None

    def test_blocks_subprocess(self):
        assert "subprocess" in is_plugin_safe("import subprocess")

    def test_blocks_eval(self):
        assert "eval(" in is_plugin_safe("result = eval('1+1')")

    def test_blocks_os_system(self):
        assert "os.system" in is_plugin_safe("os.system('rm -rf /')")

    def test_blocks_network(self):
        assert "socket" in is_plugin_safe("import socket")

    def test_blocks_requests(self):
        assert "requests" in is_plugin_safe("import requests")

    def test_blocks_open(self):
        assert "open(" in is_plugin_safe("f = open('/etc/passwd')")


class TestSkillPluginCreate:
    def test_creates_plugin_in_staging(self, plugin, staging_dir):
        result = plugin.create("hello", "Says hello", "x = 'hello'")
        assert result["success"] is True
        assert "staging" in result["message"]
        assert (staging_dir / "plugin_hello.json").exists()

    def test_blocks_dangerous_code(self, plugin, staging_dir):
        result = plugin.create("evil", "Bad plugin", "import subprocess; subprocess.run(['rm', '-rf', '/'])")
        assert result["success"] is False
        assert "BLOCKED" in result["message"]

    def test_requires_name_and_code(self, plugin):
        result = plugin.create("", "No name", "code")
        assert result["success"] is False

    def test_generates_hash(self, plugin, staging_dir):
        result = plugin.create("hashed", "Has hash", "x = 42")
        assert result["success"] is True
        data = json.loads((staging_dir / "plugin_hashed.json").read_text())
        assert data["plugin_hash"].startswith("sha256:")


class TestSkillPluginValidate:
    def test_validates_good_plugin(self, plugin, staging_dir):
        plugin.create("good", "Good plugin", "x = 1")
        result = plugin.validate("good")
        assert result["valid"] is True

    def test_rejects_missing_plugin(self, plugin):
        result = plugin.validate("nonexistent")
        assert result["valid"] is False

    def test_detects_hash_mismatch(self, plugin, staging_dir):
        plugin.create("tampered", "Will be tampered", "x = 1")
        # Tamper with the code
        path = staging_dir / "plugin_tampered.json"
        data = json.loads(path.read_text())
        data["plugin_code"] = "MODIFIED CODE"
        path.write_text(json.dumps(data))
        result = plugin.validate("tampered")
        assert any("Hash mismatch" in i for i in result["issues"])


class TestSkillPluginPromote:
    def test_promotes_to_active(self, plugin, staging_dir, skills_dir):
        plugin.create("promote_me", "Promote test", "x = 1")
        result = plugin.promote("promote_me")
        assert result["success"] is True
        assert (skills_dir / "plugin_promote_me.json").exists()
        assert not (staging_dir / "plugin_promote_me.json").exists()

    def test_rejects_missing(self, plugin):
        result = plugin.promote("ghost")
        assert result["success"] is False


class TestSkillPluginRemove:
    def test_removes_staged_plugin(self, plugin, staging_dir):
        plugin.create("removable", "Remove me", "x = 1")
        result = plugin.remove("removable")
        assert result["success"] is True

    def test_reports_not_found(self, plugin):
        result = plugin.remove("nonexistent")
        assert result["success"] is False


class TestSkillPluginList:
    def test_lists_plugins(self, plugin, staging_dir):
        plugin.create("listed", "List test", "x = 1")
        result = plugin.list_plugins()
        assert "plugin_listed" in result["staging"]
