"""
Persistent Daemon Mode for KlomboAGI.

`klomboagi daemon start` runs an always-on background cognition agent that:
  - Watches for new missions in the queue
  - Runs cognition cycles on active missions
  - Self-monitors health
  - Handles graceful shutdown via signals
  - Exponential backoff during idle periods
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class DaemonJob:
    id: str
    job_type: str  # mission | health_check | maintenance
    description: str
    payload: dict[str, object] = field(default_factory=dict)
    priority: int = 5
    status: str = "pending"  # pending | running | completed | failed
    created_at: str = ""
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "description": self.description,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "result": self.result,
            "error": self.error,
        }


class KlomboAGIDaemon:
    """Always-on cognition daemon for KlomboAGI."""

    def __init__(
        self,
        storage: StorageManager,
        *,
        tick_interval: float = 30.0,
        health_interval: float = 300.0,
        idle_threshold: float = 120.0,
        max_backoff: float = 300.0,
    ) -> None:
        self.storage = storage
        self.tick_interval = tick_interval
        self.health_interval = health_interval
        self.idle_threshold = idle_threshold
        self.max_backoff = max_backoff

        self.state: str = "stopped"  # stopped | starting | running | idle | processing | stopping
        self.job_queue: list[DaemonJob] = []
        self.active_job: DaemonJob | None = None
        self._id_counter = 0
        self._running = False
        self._last_activity = time.time()
        self._current_backoff = tick_interval
        self._last_health_check = 0.0

        self.on_execute_job: callable | None = None  # type: ignore[type-arg]

    def start(self) -> None:
        """Start the daemon loop. Blocks until stop() is called."""
        if self.state != "stopped":
            raise RuntimeError(f"Daemon is already {self.state}")

        self.state = "starting"
        self._running = True
        self._write_pid()
        self._load_queue()

        self.storage.event_log.append("daemon.started", {"pid": os.getpid()})
        self.state = "running"

        self._setup_signals()

        try:
            while self._running:
                self._tick()
                time.sleep(min(self._current_backoff, self.tick_interval))
        finally:
            self._cleanup()

    def start_nonblocking(self) -> None:
        """Start without blocking (for testing). Must call tick() manually."""
        if self.state != "stopped":
            raise RuntimeError(f"Daemon is already {self.state}")
        self.state = "starting"
        self._running = True
        self._write_pid()
        self._load_queue()
        self.state = "running"

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False
        self.state = "stopping"
        self.storage.event_log.append("daemon.stopping", {})

    def force_stop(self) -> None:
        """Immediately clean up without waiting for loop."""
        self._running = False
        self._cleanup()

    def enqueue(self, job_type: str, description: str, payload: dict[str, object] | None = None, priority: int = 5) -> DaemonJob:
        """Add a job to the queue."""
        self._id_counter += 1
        job = DaemonJob(
            id=f"djob_{self._id_counter}",
            job_type=job_type,
            description=description,
            payload=payload or {},
            priority=priority,
            created_at=utc_now(),
        )
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda j: j.priority)
        self._save_queue()
        return job

    def get_queue(self) -> list[DaemonJob]:
        return list(self.job_queue)

    def get_state(self) -> str:
        return self.state

    def status(self) -> dict[str, object]:
        return {
            "state": self.state,
            "pending_jobs": len([j for j in self.job_queue if j.status == "pending"]),
            "active_job": self.active_job.to_dict() if self.active_job else None,
            "backoff_seconds": round(self._current_backoff, 1),
        }

    def tick(self) -> None:
        """Run one tick (for testing)."""
        self._tick()

    # ── Internal ──

    def _tick(self) -> None:
        if self.state not in ("running", "idle"):
            return

        # Health check
        now = time.time()
        if now - self._last_health_check >= self.health_interval:
            self._health_tick()
            self._last_health_check = now

        # Process next job
        pending = [j for j in self.job_queue if j.status == "pending"]
        if not pending:
            self.state = "idle"
            self._apply_backoff()
            return

        self._current_backoff = self.tick_interval
        self._last_activity = time.time()
        self.state = "processing"

        job = pending[0]
        self._execute_job(job)
        self.state = "running"

    def _execute_job(self, job: DaemonJob) -> None:
        job.status = "running"
        self.active_job = job

        try:
            if self.on_execute_job:
                result = self.on_execute_job(job)
                job.result = str(result) if result else "completed"
            else:
                job.result = f"Job {job.id} processed: {job.description}"
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        finally:
            self.active_job = None
            self.job_queue = [j for j in self.job_queue if j.id != job.id]
            self._save_queue()

    def _health_tick(self) -> None:
        self.storage.event_log.append("daemon.health_check", {"state": self.state})

    def _apply_backoff(self) -> None:
        elapsed = time.time() - self._last_activity
        if elapsed > self.idle_threshold:
            self._current_backoff = min(self._current_backoff * 1.5, self.max_backoff)

    def _setup_signals(self) -> None:
        def handler(signum: int, frame: object) -> None:
            self.stop()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _write_pid(self) -> None:
        pid_dir = self.storage.paths.runtime_root / "daemon"
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / "pid").write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        pid_file = self.storage.paths.runtime_root / "daemon" / "pid"
        if pid_file.exists():
            pid_file.unlink()

    def _load_queue(self) -> None:
        queue_file = self.storage.paths.runtime_root / "daemon" / "queue.json"
        if queue_file.exists():
            try:
                data = json.loads(queue_file.read_text())
                self.job_queue = [DaemonJob(**j) for j in data]
                for j in self.job_queue:
                    if j.status == "running":
                        j.status = "pending"
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_queue(self) -> None:
        queue_dir = self.storage.paths.runtime_root / "daemon"
        queue_dir.mkdir(parents=True, exist_ok=True)
        (queue_dir / "queue.json").write_text(
            json.dumps([j.to_dict() for j in self.job_queue], indent=2)
        )

    def _cleanup(self) -> None:
        self._save_queue()
        self._remove_pid()
        self.state = "stopped"
        self.storage.event_log.append("daemon.stopped", {})
