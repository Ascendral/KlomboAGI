"""Tests for SkillEvolution engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from codeagi.learning.skill_evolution import EvolutionReport, SkillEvolution, SkillTestResult


@pytest.fixture()
def mock_storage(tmp_path):
    storage = MagicMock()
    storage.paths.runtime_root = tmp_path / "runtime"
    storage.paths.runtime_root.mkdir()
    storage.event_log = MagicMock()
    return storage


@pytest.fixture()
def skills_dir(tmp_path, monkeypatch):
    """Create a temporary shared skills directory."""
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr(
        "codeagi.learning.skill_evolution._shared_skills_dir", lambda: d
    )
    return d


@pytest.fixture()
def evolution(mock_storage):
    return SkillEvolution(mock_storage)


def _write_skill(skills_dir, name, confidence=0.5, use_count=0, steps=None):
    skill = {
        "name": name,
        "description": f"Skill: {name}",
        "steps": steps or [{"tool": "think", "args": {"thought": "test"}}],
        "author": "test",
        "confidence": confidence,
        "use_count": use_count,
        "origin": "test",
        "created_at": "2026-03-15T00:00:00Z",
        "updated_at": "2026-03-15T00:00:00Z",
    }
    (skills_dir / f"{name}.json").write_text(json.dumps(skill, indent=2))
    return skill


class TestSkillTestResult:
    def test_to_dict(self):
        r = SkillTestResult(
            skill_name="test_skill",
            passed=True,
            message="OK",
            duration_ms=1.5,
            tested_at="2026-03-15T00:00:00Z",
        )
        d = r.to_dict()
        assert d["skill_name"] == "test_skill"
        assert d["passed"] is True


class TestTestSkill:
    def test_passes_valid_skill(self, evolution):
        skill = {"name": "valid", "steps": [{"tool": "think", "args": {}}]}
        result = evolution.test_skill(skill)
        assert result.passed is True
        assert "Structural" in result.message

    def test_fails_empty_steps(self, evolution):
        skill = {"name": "empty", "steps": []}
        result = evolution.test_skill(skill)
        assert result.passed is False
        assert "no steps" in result.message

    def test_fails_invalid_step(self, evolution):
        skill = {"name": "bad", "steps": [{"not_tool": "x"}]}
        result = evolution.test_skill(skill)
        assert result.passed is False
        assert "invalid tool" in result.message

    def test_custom_runner(self, evolution):
        skill = {"name": "custom", "steps": [{"tool": "think", "args": {}}]}
        runner = lambda s: {"passed": True, "message": "Custom OK"}
        result = evolution.test_skill(skill, test_runner=runner)
        assert result.passed is True
        assert "Custom OK" in result.message

    def test_custom_runner_exception(self, evolution):
        skill = {"name": "crash", "steps": [{"tool": "think", "args": {}}]}
        runner = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
        result = evolution.test_skill(skill, test_runner=runner)
        assert result.passed is False
        assert "boom" in result.message


class TestRetireFailures:
    def test_retires_low_confidence(self, evolution, skills_dir):
        _write_skill(skills_dir, "weak_skill", confidence=0.05)
        skills = [{"name": "weak_skill", "confidence": 0.05, "steps": [{"tool": "t", "args": {}}]}]
        retired = evolution.retire_failures(skills)
        assert "weak_skill" in retired
        assert not (skills_dir / "weak_skill.json").exists()
        assert (skills_dir / "retired" / "weak_skill.json").exists()

    def test_keeps_healthy_skills(self, evolution, skills_dir):
        _write_skill(skills_dir, "strong_skill", confidence=0.9)
        skills = [{"name": "strong_skill", "confidence": 0.9, "steps": [{"tool": "t", "args": {}}]}]
        retired = evolution.retire_failures(skills)
        assert retired == []
        assert (skills_dir / "strong_skill.json").exists()


class TestGenerateVariants:
    def test_creates_variant_for_top_performer(self, evolution, skills_dir):
        _write_skill(skills_dir, "top_skill", confidence=0.9, use_count=5)
        skills = [{"name": "top_skill", "confidence": 0.9, "use_count": 5, "steps": [{"tool": "grep", "args": {}}]}]
        evolved = evolution.generate_variants(skills)
        assert len(evolved) == 1
        assert evolved[0].startswith("top_skill_v")
        # Variant file should exist
        variant_path = skills_dir / f"{evolved[0]}.json"
        assert variant_path.exists()
        data = json.loads(variant_path.read_text())
        assert data["origin"] == "evolved"
        assert data["confidence"] == pytest.approx(0.72, abs=0.01)

    def test_skips_low_confidence(self, evolution, skills_dir):
        skills = [{"name": "mediocre", "confidence": 0.5, "use_count": 10, "steps": [{"tool": "t", "args": {}}]}]
        evolved = evolution.generate_variants(skills)
        assert evolved == []

    def test_skips_low_use_count(self, evolution, skills_dir):
        skills = [{"name": "new_skill", "confidence": 0.9, "use_count": 1, "steps": [{"tool": "t", "args": {}}]}]
        evolved = evolution.generate_variants(skills)
        assert evolved == []


class TestComposeSkills:
    def test_composes_complementary_skills(self, evolution, skills_dir):
        skills = [
            {
                "name": "reader",
                "confidence": 0.8,
                "steps": [{"tool": "read_file", "args": {}}],
            },
            {
                "name": "writer",
                "confidence": 0.8,
                "steps": [{"tool": "write_file", "args": {}}],
            },
        ]
        composed = evolution.compose_skills(skills)
        assert len(composed) == 1
        assert composed[0].startswith("composed_reader_writer")
        data = json.loads((skills_dir / f"{composed[0]}.json").read_text())
        assert data["origin"] == "composed"
        assert len(data["steps"]) == 2

    def test_skips_overlapping_skills(self, evolution, skills_dir):
        skills = [
            {"name": "a", "confidence": 0.8, "steps": [{"tool": "think", "args": {}}]},
            {"name": "b", "confidence": 0.8, "steps": [{"tool": "think", "args": {}}]},
        ]
        composed = evolution.compose_skills(skills)
        assert composed == []

    def test_limits_compositions_per_cycle(self, evolution, skills_dir):
        # Create 5 skills with different tools to trigger many compositions
        skills = [
            {"name": f"s{i}", "confidence": 0.9, "steps": [{"tool": f"tool_{i}", "args": {}}]}
            for i in range(5)
        ]
        composed = evolution.compose_skills(skills)
        assert len(composed) <= 3


class TestFullEvolveCycle:
    def test_full_cycle(self, evolution, skills_dir, mock_storage):
        _write_skill(skills_dir, "healthy", confidence=0.9, use_count=5)
        _write_skill(skills_dir, "dying", confidence=0.05, use_count=1)
        report = evolution.evolve()
        assert len(report.tested) == 2
        assert len(report.retired) >= 1
        assert "dying" in report.retired
        mock_storage.event_log.append.assert_called()

    def test_format_report(self):
        report = EvolutionReport(
            tested=[
                SkillTestResult("a", True, "OK", 1.0),
                SkillTestResult("b", False, "fail", 1.0),
            ],
            retired=["b"],
            evolved=["a_v1"],
            composed=[],
            timestamp="2026-03-15T00:00:00Z",
        )
        text = SkillEvolution.format_report(report)
        assert "2 (1 passed)" in text
        assert "Retired: 1" in text
        assert "Evolved: 1" in text
