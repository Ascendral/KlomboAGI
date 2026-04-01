"""
System Observer — KlomboAGI's eyes on its own machine.

Continuously monitors system metrics, detects anomalies, and feeds
observations into the brain as beliefs. This is what makes Klombo
aware of its own physical state — not just a snapshot, but a living
stream of "I notice my RAM is climbing."

Runs as a background thread inside the server process.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import psutil


@dataclass
class SystemSnapshot:
    """Point-in-time system state."""
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    disk_percent: float
    disk_free_gb: float
    net_bytes_sent: int
    net_bytes_recv: int
    process_count: int
    top_process: str = ""
    top_process_cpu: float = 0.0
    top_process_mem: float = 0.0


@dataclass
class Anomaly:
    """Something unusual the observer noticed."""
    timestamp: float
    category: str  # cpu, ram, disk, network, process
    severity: str  # info, warning, critical
    message: str
    value: float
    threshold: float


class SystemObserver:
    """Watches the machine and notices things.

    Collects snapshots at a regular interval, maintains a rolling history,
    detects anomalies by comparing current values to recent baselines.
    """

    def __init__(
        self,
        interval: float = 30.0,
        history_size: int = 120,  # 120 snapshots * 30s = 1 hour of history
        on_anomaly: Callable[[Anomaly], None] | None = None,
        on_snapshot: Callable[[SystemSnapshot], None] | None = None,
    ):
        self.interval = interval
        self.history: deque[SystemSnapshot] = deque(maxlen=history_size)
        self.anomalies: deque[Anomaly] = deque(maxlen=200)
        self.on_anomaly = on_anomaly
        self.on_snapshot = on_snapshot
        self._running = False
        self._thread: threading.Thread | None = None
        self._prev_net = None

        # Thresholds
        self.cpu_warn = 80.0
        self.cpu_crit = 95.0
        self.ram_warn = 85.0
        self.ram_crit = 95.0
        self.disk_warn = 90.0
        self.disk_crit = 95.0

    def start(self) -> None:
        """Start observing in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop observing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def take_snapshot(self) -> SystemSnapshot:
        """Take a single snapshot of the system."""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        net = psutil.net_io_counters()
        if self._prev_net is None:
            net_sent, net_recv = 0, 0
        else:
            net_sent = net.bytes_sent - self._prev_net.bytes_sent
            net_recv = net.bytes_recv - self._prev_net.bytes_recv
        self._prev_net = net

        # Find top CPU process
        top_name = ""
        top_cpu = 0.0
        top_mem = 0.0
        try:
            procs = []
            for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"]):
                try:
                    info = p.info
                    if info["cpu_percent"] and info["cpu_percent"] > top_cpu:
                        top_cpu = info["cpu_percent"]
                        top_name = info["name"] or ""
                        top_mem = info["memory_percent"] or 0.0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu,
            ram_percent=mem.percent,
            ram_available_gb=mem.available / (1024 ** 3),
            disk_percent=disk.percent,
            disk_free_gb=disk.free / (1024 ** 3),
            net_bytes_sent=net_sent,
            net_bytes_recv=net_recv,
            process_count=len(psutil.pids()),
            top_process=top_name,
            top_process_cpu=top_cpu,
            top_process_mem=top_mem,
        )

    def check_anomalies(self, snap: SystemSnapshot) -> list[Anomaly]:
        """Check a snapshot for anomalies."""
        anomalies = []

        # CPU
        if snap.cpu_percent >= self.cpu_crit:
            anomalies.append(Anomaly(
                snap.timestamp, "cpu", "critical",
                f"CPU at {snap.cpu_percent:.0f}% (top: {snap.top_process} at {snap.top_process_cpu:.0f}%)",
                snap.cpu_percent, self.cpu_crit))
        elif snap.cpu_percent >= self.cpu_warn:
            anomalies.append(Anomaly(
                snap.timestamp, "cpu", "warning",
                f"CPU at {snap.cpu_percent:.0f}%",
                snap.cpu_percent, self.cpu_warn))

        # RAM
        if snap.ram_percent >= self.ram_crit:
            anomalies.append(Anomaly(
                snap.timestamp, "ram", "critical",
                f"RAM at {snap.ram_percent:.0f}% — only {snap.ram_available_gb:.1f}GB free",
                snap.ram_percent, self.ram_crit))
        elif snap.ram_percent >= self.ram_warn:
            anomalies.append(Anomaly(
                snap.timestamp, "ram", "warning",
                f"RAM at {snap.ram_percent:.0f}%",
                snap.ram_percent, self.ram_warn))

        # Disk
        if snap.disk_percent >= self.disk_crit:
            anomalies.append(Anomaly(
                snap.timestamp, "disk", "critical",
                f"Disk at {snap.disk_percent:.0f}% — only {snap.disk_free_gb:.0f}GB free",
                snap.disk_percent, self.disk_crit))
        elif snap.disk_percent >= self.disk_warn:
            anomalies.append(Anomaly(
                snap.timestamp, "disk", "warning",
                f"Disk at {snap.disk_percent:.0f}%",
                snap.disk_percent, self.disk_warn))

        # Trend detection: RAM climbing steadily
        if len(self.history) >= 10:
            recent_ram = [s.ram_percent for s in list(self.history)[-10:]]
            if all(recent_ram[i] < recent_ram[i + 1] for i in range(len(recent_ram) - 1)):
                delta = recent_ram[-1] - recent_ram[0]
                if delta > 5:
                    anomalies.append(Anomaly(
                        snap.timestamp, "ram", "info",
                        f"RAM climbing steadily: +{delta:.1f}% over last {len(recent_ram)} readings",
                        delta, 5.0))

        # Trend detection: CPU sustained high
        if len(self.history) >= 5:
            recent_cpu = [s.cpu_percent for s in list(self.history)[-5:]]
            if all(c > 60 for c in recent_cpu):
                avg = sum(recent_cpu) / len(recent_cpu)
                anomalies.append(Anomaly(
                    snap.timestamp, "cpu", "info",
                    f"CPU sustained at {avg:.0f}% for {len(recent_cpu)} readings",
                    avg, 60.0))

        return anomalies

    def get_summary(self) -> dict:
        """Current system summary."""
        if not self.history:
            snap = self.take_snapshot()
            self.history.append(snap)

        latest = self.history[-1]
        recent_anomalies = [
            {"category": a.category, "severity": a.severity,
             "message": a.message, "time": a.timestamp}
            for a in list(self.anomalies)[-10:]
        ]

        # Compute averages
        if len(self.history) > 1:
            avg_cpu = sum(s.cpu_percent for s in self.history) / len(self.history)
            avg_ram = sum(s.ram_percent for s in self.history) / len(self.history)
        else:
            avg_cpu = latest.cpu_percent
            avg_ram = latest.ram_percent

        return {
            "current": {
                "cpu_percent": round(latest.cpu_percent, 1),
                "ram_percent": round(latest.ram_percent, 1),
                "ram_available_gb": round(latest.ram_available_gb, 1),
                "disk_percent": round(latest.disk_percent, 1),
                "disk_free_gb": round(latest.disk_free_gb, 0),
                "process_count": latest.process_count,
                "top_process": latest.top_process,
            },
            "averages": {
                "cpu_percent": round(avg_cpu, 1),
                "ram_percent": round(avg_ram, 1),
            },
            "observations": len(self.history),
            "anomalies": recent_anomalies,
            "uptime_hours": round((time.time() - self.history[0].timestamp) / 3600, 1) if self.history else 0,
        }

    def _loop(self) -> None:
        """Main observation loop."""
        while self._running:
            try:
                snap = self.take_snapshot()
                self.history.append(snap)

                if self.on_snapshot:
                    self.on_snapshot(snap)

                anomalies = self.check_anomalies(snap)
                for a in anomalies:
                    self.anomalies.append(a)
                    if self.on_anomaly:
                        self.on_anomaly(a)

            except Exception:
                pass  # Observer must never crash the host

            time.sleep(self.interval)
