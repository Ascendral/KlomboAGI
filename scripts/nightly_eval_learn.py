from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.harness import EvalHarness
from klomboagi.learning.skill_extraction import SkillExtractor
from klomboagi.memory.causal_scoring import CausalMemoryTracker


@dataclass
class NightlyEvalLearnReport:
    run_id: str
    timestamp: str
    processed_trajectories: int = 0
    skipped_trajectories: int = 0
    extracted_skills: int = 0
    extracted_anti_patterns: int = 0
    total_skills: int = 0
    total_anti_patterns: int = 0
    memory_summary: dict[str, Any] = field(default_factory=dict)
    eval_summary: dict[str, Any] | None = None
    previous_eval_comparison: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NightlyStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"processed_trajectories": {}, "runs": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"processed_trajectories": {}, "runs": []}

    def save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _memory_summary(memory_dir: Path) -> dict[str, Any]:
    tracker = CausalMemoryTracker(store_path=str(memory_dir))
    score = tracker.lifetime_score()
    return {
        "total_retrievals": score.total_retrievals,
        "decisions_changed": score.decisions_changed,
        "outcomes_improved": score.outcomes_improved,
        "outcomes_worsened": score.outcomes_worsened,
        "no_effect": score.no_effect,
        "usefulness_rate": score.usefulness_rate(),
        "change_rate": score.change_rate(),
        "summary": score.summary(),
    }


def _latest_report_id(report_dir: Path) -> str | None:
    reports = sorted(report_dir.glob("eval_*.json"), key=lambda item: item.stat().st_mtime)
    if not reports:
        return None
    return reports[-1].stem


def _load_agent(module_name: str, factory_name: str) -> Any:
    module = importlib.import_module(module_name)
    if hasattr(module, factory_name):
        return getattr(module, factory_name)()
    if hasattr(module, "AGENT"):
        return getattr(module, "AGENT")
    raise ValueError(
        f"Module '{module_name}' did not expose factory '{factory_name}' or AGENT."
    )


def run_nightly(
    *,
    trajectory_dir: str | Path = "datasets/trajectories",
    skill_dir: str | Path = "datasets/skills",
    failure_dir: str | Path = "datasets/failure_cases",
    memory_dir: str | Path = "datasets/memory_events",
    eval_dir: str | Path = "evals/hidden",
    eval_report_dir: str | Path = "evals/reports",
    nightly_dir: str | Path = "datasets/nightly",
    domain: str | None = None,
    agent: Any | None = None,
    agent_module: str | None = None,
    agent_factory: str = "build_agent",
    skip_evals: bool = False,
) -> NightlyEvalLearnReport:
    trajectory_dir = Path(trajectory_dir)
    skill_dir = Path(skill_dir)
    failure_dir = Path(failure_dir)
    memory_dir = Path(memory_dir)
    eval_dir = Path(eval_dir)
    eval_report_dir = Path(eval_report_dir)
    nightly_dir = Path(nightly_dir)

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    eval_report_dir.mkdir(parents=True, exist_ok=True)
    report_dir = nightly_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"nightly_{time.strftime('%Y%m%d_%H%M%S')}"
    report = NightlyEvalLearnReport(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    state_store = NightlyStateStore(nightly_dir / "state.json")
    state = state_store.load()
    processed = state.setdefault("processed_trajectories", {})
    runs = state.setdefault("runs", [])

    extractor = SkillExtractor(skill_dir=str(skill_dir), failure_dir=str(failure_dir))

    for trajectory_path in sorted(trajectory_dir.glob("*.json")):
        fingerprint = _sha256_file(trajectory_path)
        if fingerprint in processed:
            report.skipped_trajectories += 1
            continue

        data = _load_json(trajectory_path)
        extracted_kind = "none"
        skill = extractor.extract_skill(data)
        anti_pattern = extractor.extract_anti_pattern(data)
        if skill is not None:
            report.extracted_skills += 1
            extracted_kind = "skill"
        if anti_pattern is not None:
            report.extracted_anti_patterns += 1
            extracted_kind = "anti_pattern"

        processed[fingerprint] = {
            "path": str(trajectory_path),
            "processed_at": report.timestamp,
            "extracted": extracted_kind,
        }
        report.processed_trajectories += 1

    stats = extractor.get_stats()
    report.total_skills = int(stats["total_skills"])
    report.total_anti_patterns = int(stats["total_anti_patterns"])
    report.memory_summary = _memory_summary(memory_dir)

    if agent is None and agent_module:
        agent = _load_agent(agent_module, agent_factory)

    if skip_evals:
        report.notes.append("Eval suite skipped by request.")
    elif agent is None:
        report.notes.append("No eval agent provided; skipped hidden eval suite.")
    else:
        previous_report_id = _latest_report_id(eval_report_dir)
        harness = EvalHarness(eval_dir=str(eval_dir), report_dir=str(eval_report_dir))
        eval_report = harness.run_all(domain=domain, agent=agent)
        report.eval_summary = eval_report.to_dict()
        if previous_report_id and previous_report_id != eval_report.run_id:
            report.previous_eval_comparison = harness.compare(previous_report_id, eval_report.run_id)

    runs.append(
        {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "processed_trajectories": report.processed_trajectories,
            "extracted_skills": report.extracted_skills,
            "extracted_anti_patterns": report.extracted_anti_patterns,
            "eval_run_id": report.eval_summary["run_id"] if report.eval_summary else None,
        }
    )
    state_store.save(state)

    report_path = report_dir / f"{report.run_id}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nightly_eval_learn",
        description="Extract skills from trajectories, mine failures, score memory, and optionally run evals.",
    )
    parser.add_argument("--trajectory-dir", default="datasets/trajectories")
    parser.add_argument("--skill-dir", default="datasets/skills")
    parser.add_argument("--failure-dir", default="datasets/failure_cases")
    parser.add_argument("--memory-dir", default="datasets/memory_events")
    parser.add_argument("--eval-dir", default="evals/hidden")
    parser.add_argument("--eval-report-dir", default="evals/reports")
    parser.add_argument("--nightly-dir", default="datasets/nightly")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--agent-module", default=None)
    parser.add_argument("--agent-factory", default="build_agent")
    parser.add_argument("--skip-evals", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_nightly(
        trajectory_dir=args.trajectory_dir,
        skill_dir=args.skill_dir,
        failure_dir=args.failure_dir,
        memory_dir=args.memory_dir,
        eval_dir=args.eval_dir,
        eval_report_dir=args.eval_report_dir,
        nightly_dir=args.nightly_dir,
        domain=args.domain,
        agent_module=args.agent_module,
        agent_factory=args.agent_factory,
        skip_evals=args.skip_evals,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
