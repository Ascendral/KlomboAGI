"""
Execution Auditor for KlomboAGI.

Monitors action execution patterns, detects anomalies (repeated failures,
loops, cascading errors), and records telemetry for self-healing.

Integrates with the cognition loop to track every action outcome and
create fix missions when patterns indicate systemic problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class ActionTelemetry:
    action_type: str
    tool_name: str | None
    success: bool
    duration_ms: float
    error: str | None = None
    mission_id: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "mission_id": self.mission_id,
            "timestamp": self.timestamp,
        }


@dataclass
class Anomaly:
    anomaly_type: str  # repeated_failure | loop_detected | error_cascade
    severity: str  # warning | critical
    description: str
    tool_name: str
    evidence: list[str] = field(default_factory=list)
    fix_description: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "tool_name": self.tool_name,
            "evidence": self.evidence,
            "fix_description": self.fix_description,
        }


class ExecutionAuditor:
    """Tracks action telemetry and detects anomalies."""

    def __init__(self, storage: StorageManager, max_history: int = 200) -> None:
        self.storage = storage
        self.max_history = max_history
        self._history: list[ActionTelemetry] = []
        self._failure_threshold = 3
        self._loop_threshold = 5

    def record(self, telemetry: ActionTelemetry) -> list[Anomaly]:
        """Record an action execution and return any detected anomalies."""
        if not telemetry.timestamp:
            telemetry.timestamp = utc_now()

        self._history.append(telemetry)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # Persist telemetry
        store_data = self.storage.execution_telemetry.load(default=[])
        store_data.append(telemetry.to_dict())
        # Keep bounded
        if len(store_data) > self.max_history:
            store_data = store_data[-self.max_history:]
        self.storage.execution_telemetry.save(store_data)

        return self.detect()

    def detect(self) -> list[Anomaly]:
        """Run all anomaly detectors."""
        anomalies: list[Anomaly] = []

        repeated = self._detect_repeated_failure()
        if repeated:
            anomalies.append(repeated)

        loop = self._detect_loop()
        if loop:
            anomalies.append(loop)

        cascade = self._detect_error_cascade()
        if cascade:
            anomalies.append(cascade)

        # Log anomalies
        for anomaly in anomalies:
            self.storage.event_log.append(
                "auditor.anomaly",
                anomaly.to_dict(),
            )

        return anomalies

    def get_history(self) -> list[ActionTelemetry]:
        return list(self._history)

    def get_tool_stats(self, tool_name: str) -> dict[str, object]:
        matching = [t for t in self._history if t.tool_name == tool_name]
        failures = sum(1 for t in matching if not t.success)
        avg_ms = sum(t.duration_ms for t in matching) / len(matching) if matching else 0
        return {"total": len(matching), "failures": failures, "avg_duration_ms": round(avg_ms)}

    def summarize(self) -> str:
        if not self._history:
            return "No action telemetry recorded."

        tool_map: dict[str, dict[str, int | float]] = {}
        for t in self._history:
            name = t.tool_name or t.action_type
            if name not in tool_map:
                tool_map[name] = {"total": 0, "failures": 0, "total_ms": 0.0}
            tool_map[name]["total"] += 1
            if not t.success:
                tool_map[name]["failures"] += 1
            tool_map[name]["total_ms"] += t.duration_ms

        lines = [f"Execution Audit ({len(self._history)} actions):"]
        for name, stats in tool_map.items():
            avg = round(stats["total_ms"] / stats["total"]) if stats["total"] else 0
            fail_str = f" ({stats['failures']} failed)" if stats["failures"] else ""
            lines.append(f"  {name}: {stats['total']} calls, avg {avg}ms{fail_str}")
        return "\n".join(lines)

    def reset(self) -> None:
        self._history = []

    # ── Detectors ──

    def _detect_repeated_failure(self) -> Anomaly | None:
        recent = self._history[-self._failure_threshold:]
        if len(recent) < self._failure_threshold:
            return None

        last_tool = recent[-1].tool_name or recent[-1].action_type
        same = [t for t in recent if (t.tool_name or t.action_type) == last_tool]
        if len(same) < self._failure_threshold:
            return None
        if not all(not t.success for t in same):
            return None

        errors = [t.error or "unknown" for t in same[-3:]]
        return Anomaly(
            anomaly_type="repeated_failure",
            severity="critical",
            description=f'"{last_tool}" failed {len(same)} consecutive times',
            tool_name=last_tool,
            evidence=errors,
            fix_description=f"Investigate repeated {last_tool} failures: {errors[0]}",
        )

    def _detect_loop(self) -> Anomaly | None:
        if len(self._history) < self._loop_threshold:
            return None

        recent = self._history[-self._loop_threshold:]
        sigs = [f"{t.tool_name or t.action_type}:{t.success}" for t in recent]
        if len(set(sigs)) == 1:
            name = recent[0].tool_name or recent[0].action_type
            return Anomaly(
                anomaly_type="loop_detected",
                severity="critical",
                description=f'Loop: "{name}" called {self._loop_threshold} times identically',
                tool_name=name,
                evidence=sigs,
                fix_description=f"Break loop — stop calling {name}",
            )
        return None

    def _detect_error_cascade(self) -> Anomaly | None:
        window = self._history[-10:]
        if len(window) < 5:
            return None

        failures = [t for t in window if not t.success]
        if len(failures) < 4:
            return None

        tool_names = list({t.tool_name or t.action_type for t in failures})
        if len(tool_names) < 2:
            return None

        return Anomaly(
            anomaly_type="error_cascade",
            severity="critical",
            description=f"Error cascade: {len(failures)}/10 recent actions failed across {len(tool_names)} tools",
            tool_name=", ".join(tool_names),
            evidence=[f"{t.tool_name}: {t.error or 'failed'}" for t in failures],
            fix_description="Multiple tools failing — check system health",
        )
