"""Tests for CodeAGIDaemon persistent daemon mode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from codeagi.core.daemon import CodeAGIDaemon, DaemonJob


@pytest.fixture()
def mock_storage(tmp_path):
    storage = MagicMock()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    storage.paths.runtime_root = runtime_root
    storage.event_log = MagicMock()
    return storage


@pytest.fixture()
def daemon(mock_storage):
    return CodeAGIDaemon(mock_storage, tick_interval=1.0, health_interval=10.0)


class TestDaemonJob:
    def test_to_dict(self):
        job = DaemonJob(
            id="djob_1",
            job_type="mission",
            description="Test job",
            payload={"key": "value"},
            priority=3,
            status="pending",
            created_at="2026-03-15T00:00:00Z",
        )
        d = job.to_dict()
        assert d["id"] == "djob_1"
        assert d["job_type"] == "mission"
        assert d["priority"] == 3
        assert d["payload"] == {"key": "value"}

    def test_defaults(self):
        job = DaemonJob(id="djob_2", job_type="health_check", description="Check")
        assert job.priority == 5
        assert job.status == "pending"
        assert job.result is None
        assert job.error is None


class TestDaemonLifecycle:
    def test_initial_state(self, daemon):
        assert daemon.get_state() == "stopped"

    def test_start_nonblocking(self, daemon):
        daemon.start_nonblocking()
        assert daemon.get_state() == "running"

    def test_start_nonblocking_rejects_double_start(self, daemon):
        daemon.start_nonblocking()
        with pytest.raises(RuntimeError, match="already"):
            daemon.start_nonblocking()

    def test_stop(self, daemon):
        daemon.start_nonblocking()
        daemon.stop()
        assert daemon.get_state() == "stopping"

    def test_force_stop(self, daemon):
        daemon.start_nonblocking()
        daemon.force_stop()
        assert daemon.get_state() == "stopped"

    def test_pid_file_written(self, daemon, mock_storage, tmp_path):
        daemon.start_nonblocking()
        pid_file = tmp_path / "runtime" / "daemon" / "pid"
        assert pid_file.exists()

    def test_pid_file_removed_on_cleanup(self, daemon, mock_storage, tmp_path):
        daemon.start_nonblocking()
        daemon.force_stop()
        pid_file = tmp_path / "runtime" / "daemon" / "pid"
        assert not pid_file.exists()


class TestJobQueue:
    def test_enqueue_and_get_queue(self, daemon):
        daemon.start_nonblocking()
        job = daemon.enqueue("mission", "Do something")
        assert job.id == "djob_1"
        assert len(daemon.get_queue()) == 1

    def test_priority_sorting(self, daemon):
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Low priority", priority=10)
        daemon.enqueue("mission", "High priority", priority=1)
        queue = daemon.get_queue()
        assert queue[0].priority == 1
        assert queue[1].priority == 10

    def test_enqueue_persists_to_file(self, daemon, tmp_path):
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Persisted job")
        queue_file = tmp_path / "runtime" / "daemon" / "queue.json"
        assert queue_file.exists()
        data = json.loads(queue_file.read_text())
        assert len(data) == 1
        assert data[0]["description"] == "Persisted job"


class TestTickProcessing:
    def test_tick_processes_pending_job(self, daemon):
        daemon.start_nonblocking()
        job = daemon.enqueue("mission", "Process me")
        daemon.tick()
        # Job should be removed from queue after processing
        assert len(daemon.get_queue()) == 0

    def test_tick_calls_execute_callback(self, daemon):
        callback = MagicMock(return_value="done")
        daemon.on_execute_job = callback
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Callback test")
        daemon.tick()
        callback.assert_called_once()

    def test_tick_goes_idle_when_no_jobs(self, daemon):
        daemon.start_nonblocking()
        daemon.tick()
        assert daemon.get_state() == "idle"

    def test_tick_handles_job_failure(self, daemon):
        def failing_callback(job):
            raise ValueError("Something broke")

        daemon.on_execute_job = failing_callback
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Will fail")
        daemon.tick()
        # Queue should be empty (failed job removed)
        assert len(daemon.get_queue()) == 0

    def test_tick_returns_to_running_after_processing(self, daemon):
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Process me")
        daemon.tick()
        assert daemon.get_state() == "running"


class TestStatus:
    def test_status_dict(self, daemon):
        daemon.start_nonblocking()
        daemon.enqueue("mission", "Job 1")
        status = daemon.status()
        assert status["state"] == "running"
        assert status["pending_jobs"] == 1
        assert status["active_job"] is None

    def test_status_after_idle(self, daemon):
        daemon.start_nonblocking()
        daemon.tick()  # No jobs → idle
        status = daemon.status()
        assert status["state"] == "idle"


class TestQueuePersistence:
    def test_load_queue_on_start(self, daemon, tmp_path):
        # Pre-populate queue file
        queue_dir = tmp_path / "runtime" / "daemon"
        queue_dir.mkdir(parents=True, exist_ok=True)
        jobs = [
            {
                "id": "djob_99",
                "job_type": "mission",
                "description": "Persisted job",
                "payload": {},
                "priority": 5,
                "status": "pending",
                "created_at": "2026-03-15T00:00:00Z",
                "result": None,
                "error": None,
            }
        ]
        (queue_dir / "queue.json").write_text(json.dumps(jobs))

        daemon.start_nonblocking()
        assert len(daemon.get_queue()) == 1
        assert daemon.get_queue()[0].id == "djob_99"

    def test_running_jobs_reset_to_pending_on_load(self, daemon, tmp_path):
        queue_dir = tmp_path / "runtime" / "daemon"
        queue_dir.mkdir(parents=True, exist_ok=True)
        jobs = [
            {
                "id": "djob_50",
                "job_type": "mission",
                "description": "Was running",
                "payload": {},
                "priority": 5,
                "status": "running",
                "created_at": "2026-03-15T00:00:00Z",
                "result": None,
                "error": None,
            }
        ]
        (queue_dir / "queue.json").write_text(json.dumps(jobs))

        daemon.start_nonblocking()
        assert daemon.get_queue()[0].status == "pending"
