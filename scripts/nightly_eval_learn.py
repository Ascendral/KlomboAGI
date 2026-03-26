#!/usr/bin/env python3
"""
Nightly Eval & Learn — makes the system improve overnight.

This script is the closed loop that turns raw experience into intelligence.
Run it nightly (or after any development session).

It:
1. Loads latest trajectories and episodes
2. Extracts successful skills
3. Mines failed trajectories into anti-patterns
4. Runs the hidden eval suite
5. Compares against the previous baseline
6. Saves report and REFUSES to promote regressions

Usage:
    python scripts/nightly_eval_learn.py              # Full cycle
    python scripts/nightly_eval_learn.py --eval-only  # Just run evals
    python scripts/nightly_eval_learn.py --learn-only # Just extract skills
    python scripts/nightly_eval_learn.py --report     # Show latest report
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.trajectory import TrajectoryStore
from klomboagi.learning.skill_extraction import SkillExtractor
from klomboagi.memory.causal_scoring import CausalMemoryTracker
from klomboagi.agent.integrated import IntegratedAgent
from evals.harness import EvalHarness


def learn_from_trajectories():
    """Extract skills from successes, anti-patterns from failures."""
    print("\n📚 LEARNING FROM TRAJECTORIES")
    print("=" * 50)

    store = TrajectoryStore()
    extractor = SkillExtractor()

    trajectories = store.list_all()
    print(f"  Trajectories found: {len(trajectories)}")

    new_skills = 0
    new_anti_patterns = 0

    for path in trajectories:
        try:
            with open(path) as f:
                traj = json.load(f)

            if traj.get("success"):
                skill = extractor.extract_skill(traj)
                if skill:
                    new_skills += 1
            else:
                anti = extractor.extract_anti_pattern(traj)
                if anti:
                    new_anti_patterns += 1
        except Exception as e:
            print(f"  Warning: could not process {path}: {e}")

    stats = extractor.get_stats()
    print(f"  New skills extracted: {new_skills}")
    print(f"  New anti-patterns: {new_anti_patterns}")
    print(f"  Total skills: {stats['total_skills']}")
    print(f"  Total anti-patterns: {stats['total_anti_patterns']}")

    return stats


def run_evals():
    """Run the hidden eval suite."""
    print("\n🧪 RUNNING HIDDEN EVAL SUITE")
    print("=" * 50)

    harness = EvalHarness()
    agent = IntegratedAgent()

    report = harness.run_all(agent=agent)
    print(report.summary())

    return report


def compare_with_baseline(report):
    """Compare current run against the best previous run."""
    print("\n📊 COMPARING WITH BASELINE")
    print("=" * 50)

    report_dir = Path("evals/reports")
    previous_reports = sorted(report_dir.glob("eval_*.json"))

    if len(previous_reports) < 2:
        print("  No previous baseline to compare against.")
        return True  # No regression possible

    # Load the second-to-last report (last one is the current run)
    baseline_path = previous_reports[-2]
    with open(baseline_path) as f:
        baseline = json.load(f)

    current = report.to_dict()

    print(f"  Baseline: {baseline_path.name}")
    print(f"  Current:  eval_{int(time.time())}")
    print()

    metrics = [
        ("Success Rate", baseline.get("success_rate", 0), current.get("success_rate", 0)),
        ("Intervention Rate", baseline.get("intervention_rate", 0), current.get("intervention_rate", 0)),
        ("Autonomy Horizon", baseline.get("avg_autonomy_horizon", 0), current.get("avg_autonomy_horizon", 0)),
        ("Recovery Rate", baseline.get("recovery_rate", 0), current.get("recovery_rate", 0)),
        ("Memory Usefulness", baseline.get("memory_usefulness", 0), current.get("memory_usefulness", 0)),
    ]

    regressions = []
    improvements = []

    for name, old, new in metrics:
        delta = new - old
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")

        # For intervention rate, lower is better
        if name == "Intervention Rate":
            is_regression = delta > 0.01
            is_improvement = delta < -0.01
        else:
            is_regression = delta < -0.01
            is_improvement = delta > 0.01

        status = "🔴 REGRESSION" if is_regression else ("🟢 IMPROVED" if is_improvement else "⚪ STABLE")
        print(f"  {name}: {old:.3f} → {new:.3f} {direction} {status}")

        if is_regression:
            regressions.append(name)
        elif is_improvement:
            improvements.append(name)

    print()
    if regressions:
        print(f"  ⛔ REGRESSIONS DETECTED: {', '.join(regressions)}")
        print(f"  ⛔ DO NOT PROMOTE THIS BUILD")
        return False
    elif improvements:
        print(f"  ✅ Improvements: {', '.join(improvements)}")
        print(f"  ✅ Safe to promote")
        return True
    else:
        print(f"  ⚪ No significant changes")
        return True


def show_report():
    """Show the latest eval report."""
    report_dir = Path("evals/reports")
    reports = sorted(report_dir.glob("eval_*.json"))
    if not reports:
        print("No eval reports found.")
        return

    with open(reports[-1]) as f:
        data = json.load(f)

    print(f"\nLatest report: {reports[-1].name}")
    print(f"  Success: {data.get('tasks_succeeded', 0)}/{data.get('tasks_attempted', 0)} ({100*data.get('success_rate', 0):.1f}%)")
    print(f"  Interventions: {data.get('total_interventions', 0)}")
    print(f"  Autonomy: {data.get('avg_autonomy_horizon', 0):.1f} steps")
    print(f"  Memory useful: {100*data.get('memory_usefulness', 0):.0f}%")


def main():
    args = sys.argv[1:]

    if "--report" in args:
        show_report()
        return

    if "--learn-only" in args:
        learn_from_trajectories()
        return

    if "--eval-only" in args:
        report = run_evals()
        compare_with_baseline(report)
        return

    # Full cycle
    print("🌙 NIGHTLY EVAL & LEARN CYCLE")
    print("=" * 50)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Learn from trajectories
    learn_stats = learn_from_trajectories()

    # Step 2: Run evals
    report = run_evals()

    # Step 3: Compare with baseline
    safe = compare_with_baseline(report)

    # Step 4: Summary
    print(f"\n{'='*50}")
    print(f"NIGHTLY CYCLE COMPLETE")
    print(f"{'='*50}")
    print(f"  Skills: {learn_stats['total_skills']}")
    print(f"  Anti-patterns: {learn_stats['total_anti_patterns']}")
    print(f"  Eval: {report.tasks_succeeded}/{report.tasks_attempted}")
    print(f"  Promote: {'✅ YES' if safe else '⛔ NO — REGRESSIONS'}")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
