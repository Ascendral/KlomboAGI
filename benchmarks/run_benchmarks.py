#!/usr/bin/env python3
"""Executable benchmarks for CodeAGI core operations.

Times mission creation, task creation, and run_cycle execution.
Measures memory usage via tracemalloc.
Reports results as JSON to stdout.

Usage:
    python3 benchmarks/run_benchmarks.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codeagi.core.loop import RuntimeLoop
from codeagi.core.mission import MissionManager
from codeagi.storage.manager import StorageManager


def _setup_env(base: Path) -> None:
    os.environ["CODEAGI_RUNTIME_ROOT"] = str(base / "runtime")
    os.environ["CODEAGI_LONG_TERM_ROOT"] = str(base / "long_term")
    os.environ["CODEAGI_MAX_CYCLE_STEPS"] = "3"


def _teardown_env() -> None:
    for key in ("CODEAGI_RUNTIME_ROOT", "CODEAGI_LONG_TERM_ROOT", "CODEAGI_MAX_CYCLE_STEPS"):
        os.environ.pop(key, None)


def bench_mission_creation(manager: MissionManager, n: int = 20) -> dict:
    """Benchmark creating N missions."""
    tracemalloc.start()
    start = time.perf_counter()
    for i in range(n):
        manager.create_mission(f"Benchmark mission {i}", priority=50)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "operation": "mission_creation",
        "iterations": n,
        "total_seconds": round(elapsed, 6),
        "avg_seconds": round(elapsed / n, 6),
        "peak_memory_bytes": peak,
    }


def bench_task_creation(manager: MissionManager, mission_id: str, n: int = 30) -> dict:
    """Benchmark creating N tasks for a single mission."""
    tracemalloc.start()
    start = time.perf_counter()
    for i in range(n):
        manager.create_task(mission_id, f"Benchmark task {i}")
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "operation": "task_creation",
        "iterations": n,
        "total_seconds": round(elapsed, 6),
        "avg_seconds": round(elapsed / n, 6),
        "peak_memory_bytes": peak,
    }


def bench_run_cycle(storage: StorageManager) -> dict:
    """Benchmark a single run_cycle invocation."""
    runtime = RuntimeLoop(storage)
    tracemalloc.start()
    start = time.perf_counter()
    payload = runtime.run_cycle()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "operation": "run_cycle",
        "iterations": 1,
        "total_seconds": round(elapsed, 6),
        "avg_seconds": round(elapsed, 6),
        "peak_memory_bytes": peak,
        "cycle_status": payload.get("status"),
        "stop_reason": (payload.get("cycle_trace") or {}).get("stop_reason"),
    }


def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="codeagi_bench_")
    base = Path(tmpdir)
    _setup_env(base)
    try:
        storage = StorageManager.bootstrap()
        manager = MissionManager(storage)

        results: list[dict] = []

        # 1. Mission creation benchmark
        results.append(bench_mission_creation(manager))

        # 2. Task creation benchmark (create a fresh mission for tasks)
        mission = manager.create_mission("Benchmark task host")
        results.append(bench_task_creation(manager, mission.id))

        # 3. run_cycle benchmark — needs a fresh storage so the cycle has
        #    exactly one active mission with tasks to work on
        storage2 = StorageManager.bootstrap()
        mgr2 = MissionManager(storage2)
        m = mgr2.create_mission("Cycle benchmark mission")
        mgr2.create_task(m.id, "First benchmark task")
        mgr2.create_task(m.id, "Second benchmark task")
        results.append(bench_run_cycle(storage2))

        report = {
            "benchmark_suite": "codeagi_v0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        }
        print(json.dumps(report, indent=2))
    finally:
        _teardown_env()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
