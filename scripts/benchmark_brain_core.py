from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from klomboagi import brain_core as bridge


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    elapsed_s: float

    @property
    def per_call_ms(self) -> float:
        if self.iterations == 0:
            return 0.0
        return (self.elapsed_s / self.iterations) * 1000.0

    def to_dict(self) -> dict[str, float | int | str]:
        payload = asdict(self)
        payload["per_call_ms"] = self.per_call_ms
        return payload


def _time_it(name: str, iterations: int, fn: Callable[[], object]) -> BenchmarkResult:
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - started
    return BenchmarkResult(name=name, iterations=iterations, elapsed_s=elapsed)


def benchmark(iterations: int = 2000) -> dict[str, object]:
    memories = [
        "database indexing strategy for write-heavy workloads",
        "deployment rollback checklist",
        "frontend release note drafting workflow",
        "search repository for TODO markers before editing",
    ]
    candidates = [
        {
            "id": "search",
            "description": "Search workspace for pattern: deploy_app",
            "action_kind": "search_files",
            "action_payload": {"path": ".", "pattern": "deploy_app"},
            "estimated_cost": 0.2,
        },
        {
            "id": "read",
            "description": "Read file deploy.md",
            "action_kind": "read_file",
            "action_payload": {"path": "deploy.md"},
            "estimated_cost": 0.3,
        },
        {
            "id": "patch",
            "description": "Patch file deploy.md",
            "action_kind": "apply_patch",
            "action_payload": {"path": "deploy.md"},
            "estimated_cost": 0.6,
        },
    ]
    anti_patterns = ["Patch file deploy.md permission denied"]

    retrieval_result = _time_it(
        "retrieve_memory",
        iterations,
        lambda: bridge.retrieve_memory("deployment strategy deploy_app", memories, limit=3),
    )
    scoring_result = _time_it(
        "score_plan_candidates",
        iterations,
        lambda: bridge.score_plan_candidates("inspect deployment code", candidates, anti_patterns),
    )
    return {
        "native_available": bridge.native_available(),
        "results": [retrieval_result.to_dict(), scoring_result.to_dict()],
    }


def main() -> int:
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
