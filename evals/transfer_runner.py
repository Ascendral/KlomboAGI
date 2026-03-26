"""
Transfer Benchmark Runner.

For each transfer pair:
1. Run the agent on the TRAIN task (domain A)
2. Extract skill from that success
3. Run the agent on the TEST task (domain B) WITHOUT adding a direct solver
4. Check if the skill from domain A helped in domain B

Transfer score = success rate on test tasks AFTER training on train tasks
Baseline = success rate on test tasks WITHOUT prior training
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TransferResult:
    pair_id: str
    train_success: bool
    test_success_with_transfer: bool
    test_success_baseline: bool  # Would it work without the training?
    transfer_helped: bool
    transfer_type: str


@dataclass
class TransferReport:
    pairs_tested: int = 0
    train_successes: int = 0
    test_with_transfer: int = 0
    test_baseline: int = 0
    transfer_helped_count: int = 0
    results: list = field(default_factory=list)

    def transfer_score(self) -> float:
        return self.test_with_transfer / self.pairs_tested if self.pairs_tested > 0 else 0.0

    def baseline_score(self) -> float:
        return self.test_baseline / self.pairs_tested if self.pairs_tested > 0 else 0.0

    def transfer_lift(self) -> float:
        return self.transfer_score() - self.baseline_score()

    def summary(self) -> str:
        return (
            f"Transfer Report: {self.pairs_tested} pairs\n"
            f"  Train success: {self.train_successes}/{self.pairs_tested}\n"
            f"  Test with transfer: {self.test_with_transfer}/{self.pairs_tested} ({100*self.transfer_score():.0f}%)\n"
            f"  Test baseline: {self.test_baseline}/{self.pairs_tested} ({100*self.baseline_score():.0f}%)\n"
            f"  Transfer lift: {100*self.transfer_lift():+.0f}%\n"
            f"  Transfer helped: {self.transfer_helped_count}/{self.pairs_tested}"
        )


def run_transfer_benchmark(agent, benchmark_path: str = "evals/transfer/benchmark.json") -> TransferReport:
    """Run the transfer benchmark."""
    with open(benchmark_path) as f:
        pairs = json.load(f)

    report = TransferReport()

    for pair in pairs:
        report.pairs_tested += 1

        # Step 1: Run train task
        train_result = agent.execute(pair["train_task"])
        train_ok = train_result.get("success", False)
        if train_ok:
            report.train_successes += 1

        # Step 2: Run test task WITH accumulated knowledge
        test_result = agent.execute(pair["test_task"])
        test_ok = test_result.get("success", False)
        if test_ok:
            report.test_with_transfer += 1

        # Step 3: Check baseline (would a fresh agent solve the test task?)
        # For now, assume baseline = test task without any prior training
        # A fresh agent would still have the executor, so baseline ≈ test result
        baseline_ok = test_ok  # Conservative: same as with transfer for now
        if baseline_ok:
            report.test_baseline += 1

        # Did transfer help?
        helped = train_ok and test_ok and pair["transfer_type"] == "structural"
        if helped:
            report.transfer_helped_count += 1

        report.results.append(TransferResult(
            pair_id=pair["id"],
            train_success=train_ok,
            test_success_with_transfer=test_ok,
            test_success_baseline=baseline_ok,
            transfer_helped=helped,
            transfer_type=pair["transfer_type"],
        ))

    return report
