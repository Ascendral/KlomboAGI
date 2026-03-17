"""
Skill Evolution Engine for KlomboAGI.

Skills test themselves, measure success rates, retire low-confidence,
generate variants for top performers, and compose complementary skills
into higher-order skills.

Evolution cycle: test → retire → evolve → compose
Runs on daemon trigger or scheduled interval.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from klomboagi.learning.skill_forge import _shared_skills_dir
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class SkillTestResult:
    skill_name: str
    passed: bool
    message: str
    duration_ms: float
    tested_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_name": self.skill_name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "tested_at": self.tested_at,
        }


@dataclass
class EvolutionReport:
    tested: list[SkillTestResult] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    evolved: list[str] = field(default_factory=list)
    composed: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "tested": [t.to_dict() for t in self.tested],
            "retired": self.retired,
            "evolved": self.evolved,
            "composed": self.composed,
            "timestamp": self.timestamp,
        }


class SkillEvolution:
    """Evolves shared skills: test → retire → evolve → compose."""

    RETIRE_THRESHOLD = 0.1
    EVOLVE_THRESHOLD = 0.7
    MIN_USES_FOR_EVOLUTION = 3

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def evolve(
        self,
        test_runner: object | None = None,
    ) -> EvolutionReport:
        """Run a full evolution cycle. Returns report."""
        skills = self._load_skills()
        report = EvolutionReport(timestamp=utc_now())

        # Phase 1: Test all skills
        for skill in skills:
            result = self.test_skill(skill, test_runner)
            report.tested.append(result)
            self._reinforce(skill, result.passed)

        # Phase 2: Retire failures
        report.retired = self.retire_failures(skills)

        # Phase 3: Generate variants for top performers
        report.evolved = self.generate_variants(skills)

        # Phase 4: Compose complementary skills
        report.composed = self.compose_skills(skills)

        # Persist report
        self._persist_report(report)

        self.storage.event_log.append("skill_evolution.completed", {
            "tested": len(report.tested),
            "passed": sum(1 for t in report.tested if t.passed),
            "retired": len(report.retired),
            "evolved": len(report.evolved),
            "composed": len(report.composed),
        })

        return report

    def test_skill(
        self,
        skill: dict[str, object],
        test_runner: object | None = None,
    ) -> SkillTestResult:
        """Test a single skill structurally, optionally with a custom runner."""
        import time
        start = time.monotonic()

        name = str(skill.get("name", "unknown"))
        steps = list(skill.get("steps", []))

        # Structural validation
        if not steps:
            return SkillTestResult(
                skill_name=name,
                passed=False,
                message="Skill has no steps",
                duration_ms=round((time.monotonic() - start) * 1000, 1),
                tested_at=utc_now(),
            )

        for step in steps:
            if not isinstance(step, dict) or not step.get("tool"):
                return SkillTestResult(
                    skill_name=name,
                    passed=False,
                    message=f"Step has invalid tool: {step}",
                    duration_ms=round((time.monotonic() - start) * 1000, 1),
                    tested_at=utc_now(),
                )

        # Custom test runner
        if test_runner and callable(test_runner):
            try:
                result = test_runner(skill)
                return SkillTestResult(
                    skill_name=name,
                    passed=bool(result.get("passed", False)) if isinstance(result, dict) else bool(result),
                    message=str(result.get("message", "Custom test")) if isinstance(result, dict) else str(result),
                    duration_ms=round((time.monotonic() - start) * 1000, 1),
                    tested_at=utc_now(),
                )
            except Exception as exc:
                return SkillTestResult(
                    skill_name=name,
                    passed=False,
                    message=f"Test error: {exc}",
                    duration_ms=round((time.monotonic() - start) * 1000, 1),
                    tested_at=utc_now(),
                )

        return SkillTestResult(
            skill_name=name,
            passed=True,
            message="Structural validation passed",
            duration_ms=round((time.monotonic() - start) * 1000, 1),
            tested_at=utc_now(),
        )

    def retire_failures(self, skills: list[dict[str, object]]) -> list[str]:
        """Move low-confidence skills to retired/ directory."""
        retired: list[str] = []
        skills_dir = _shared_skills_dir()
        retired_dir = skills_dir / "retired"

        for skill in skills:
            confidence = float(skill.get("confidence", 0.5))
            if confidence >= self.RETIRE_THRESHOLD:
                continue

            name = str(skill.get("name", ""))
            src = skills_dir / f"{name}.json"
            if not src.exists():
                continue

            try:
                retired_dir.mkdir(parents=True, exist_ok=True)
                data = json.loads(src.read_text())
                data["retired"] = True
                data["retired_at"] = utc_now()
                (retired_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
                src.unlink()
                retired.append(name)
            except (json.JSONDecodeError, OSError):
                continue

        return retired

    def generate_variants(self, skills: list[dict[str, object]]) -> list[str]:
        """Generate evolved variants of top-performing skills."""
        created: list[str] = []
        skills_dir = _shared_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)

        for skill in skills:
            confidence = float(skill.get("confidence", 0.5))
            use_count = int(skill.get("use_count", 0))

            if confidence < self.EVOLVE_THRESHOLD or use_count < self.MIN_USES_FOR_EVOLUTION:
                continue

            name = str(skill.get("name", ""))
            # Generate short suffix from timestamp
            import time
            suffix = hex(int(time.time() * 1000))[-4:]
            variant_name = f"{name}_v{suffix}"
            variant_path = skills_dir / f"{variant_name}.json"

            if variant_path.exists():
                continue

            # Variant: add a planning step before execution
            steps = list(skill.get("steps", []))
            variant = {
                "name": variant_name,
                "description": f"[Evolved] {skill.get('description', '')}",
                "steps": [
                    {"tool": "think", "args": {"thought": f"Planning execution of evolved skill: {name}"}},
                    *steps,
                ],
                "author": skill.get("author", "klomboagi"),
                "confidence": confidence * 0.8,
                "use_count": 0,
                "origin": "evolved",
                "parent_skill": name,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }

            if skill.get("trigger"):
                variant["trigger"] = skill["trigger"]

            try:
                variant_path.write_text(json.dumps(variant, indent=2))
                created.append(variant_name)
            except OSError:
                continue

        return created

    def compose_skills(self, skills: list[dict[str, object]]) -> list[str]:
        """Compose complementary skills with non-overlapping tool sets."""
        composed: list[str] = []
        skills_dir = _shared_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            s for s in skills
            if float(s.get("confidence", 0.5)) >= self.EVOLVE_THRESHOLD
        ]

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]

                tools_a = {step.get("tool") for step in a.get("steps", []) if isinstance(step, dict)}
                tools_b = {step.get("tool") for step in b.get("steps", []) if isinstance(step, dict)}
                overlap = tools_a & tools_b

                # Only compose if < 50% overlap
                max_size = max(len(tools_a), len(tools_b), 1)
                if len(overlap) / max_size >= 0.5:
                    continue

                name_a = str(a.get("name", "a"))
                name_b = str(b.get("name", "b"))
                composed_name = f"composed_{name_a}_{name_b}"[:64]
                composed_path = skills_dir / f"{composed_name}.json"

                if composed_path.exists():
                    continue

                combined = {
                    "name": composed_name,
                    "description": f"[Composed] {a.get('description', '')} + {b.get('description', '')}",
                    "steps": list(a.get("steps", [])) + list(b.get("steps", [])),
                    "author": "klomboagi",
                    "confidence": min(
                        float(a.get("confidence", 0.5)),
                        float(b.get("confidence", 0.5)),
                    ) * 0.7,
                    "use_count": 0,
                    "origin": "composed",
                    "source_skills": [name_a, name_b],
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }

                try:
                    composed_path.write_text(json.dumps(combined, indent=2))
                    composed.append(composed_name)
                except OSError:
                    continue

                # Limit per cycle
                if len(composed) >= 3:
                    return composed

        return composed

    @staticmethod
    def format_report(report: EvolutionReport) -> str:
        """Format an evolution report for display."""
        passed = sum(1 for t in report.tested if t.passed)
        lines = [
            "Skill Evolution Report",
            f"  Tested: {len(report.tested)} ({passed} passed)",
            f"  Retired: {len(report.retired)}"
            + (f" ({', '.join(report.retired)})" if report.retired else ""),
            f"  Evolved: {len(report.evolved)}"
            + (f" ({', '.join(report.evolved)})" if report.evolved else ""),
            f"  Composed: {len(report.composed)}"
            + (f" ({', '.join(report.composed)})" if report.composed else ""),
        ]
        return "\n".join(lines)

    # ── Internal ──

    def _load_skills(self) -> list[dict[str, object]]:
        """Load all skills from the shared store."""
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

    def _reinforce(self, skill: dict[str, object], success: bool) -> None:
        """Update a skill's confidence based on test result."""
        name = str(skill.get("name", ""))
        skill_path = _shared_skills_dir() / f"{name}.json"
        if not skill_path.exists():
            return

        try:
            data = json.loads(skill_path.read_text())
            current = float(data.get("confidence", 0.5))
            data["confidence"] = (
                min(current + 0.05, 1.0) if success
                else max(current - 0.1, 0.0)
            )
            data["updated_at"] = utc_now()
            skill_path.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    def _persist_report(self, report: EvolutionReport) -> None:
        """Save the evolution report to storage."""
        try:
            health_dir = self.storage.paths.runtime_root / "health"
            health_dir.mkdir(parents=True, exist_ok=True)
            (health_dir / "last-evolution-report.json").write_text(
                json.dumps(report.to_dict(), indent=2)
            )
        except OSError:
            pass
