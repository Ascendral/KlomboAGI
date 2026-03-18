from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
from tempfile import TemporaryDirectory
from typing import Any

from klombo.engine import KlomboEngine
from klombo.models import utc_now


@dataclass
class BenchmarkScenario:
    name: str
    repo_id: str
    task_type: str
    request: str
    setup_episodes: list[dict[str, Any]] = field(default_factory=list)
    setup_missions: list[dict[str, Any]] = field(default_factory=list)
    expected_procedure_tool: str | None = None
    expected_anti_pattern_tool: str | None = None
    expected_preference_key: str | None = None
    expected_semantic_substring: str | None = None
    expected_resume_step: str | None = None


class BenchmarkHarness:
    """Evaluation harness for memory-informed retrieval quality."""

    def __init__(
        self,
        engine: KlomboEngine,
        *,
        signing_key: str | bytes | None = None,
        signing_key_provider: Callable[[], str | bytes | None] | None = None,
        signing_key_env: str = "KLOMBO_BENCHMARK_SIGNING_KEY",
        persist_generated_key: bool = True,
    ) -> None:
        self.engine = engine
        self.signing_key = signing_key
        self.signing_key_provider = signing_key_provider
        self.signing_key_env = signing_key_env
        self.persist_generated_key = persist_generated_key
        self._cached_signing_key: bytes | None = None
        self._cached_signing_meta: dict[str, str] | None = None

    def run(self, scenarios: list[BenchmarkScenario]) -> dict[str, Any]:
        results = []
        for scenario in scenarios:
            for mission in scenario.setup_missions:
                self.engine.record_mission_state(mission)
            for episode in scenario.setup_episodes:
                self.engine.record_episode(episode)

            context = self.engine.get_planning_context(
                repo_id=scenario.repo_id,
                request=scenario.request,
                task_type=scenario.task_type,
            )
            resume = None
            if scenario.setup_missions:
                resume = self.engine.resume_context(scenario.setup_missions[0]["mission_id"])

            procedure_hit = self._tool_hit(context.get("procedures", []), scenario.expected_procedure_tool, "action_chain")
            anti_pattern_hit = self._tool_hit(
                context.get("anti_patterns", []),
                scenario.expected_anti_pattern_tool,
                "failing_chain",
            )
            preference_hit = self._preference_hit(context.get("preferences", []), scenario.expected_preference_key)
            semantic_hit = self._semantic_hit(context.get("semantic_facts", []), scenario.expected_semantic_substring)
            resume_hit = self._resume_hit(resume, scenario.expected_resume_step)
            results.append(
                {
                    "name": scenario.name,
                    "procedure_hit": procedure_hit,
                    "anti_pattern_hit": anti_pattern_hit,
                    "preference_hit": preference_hit,
                    "semantic_hit": semantic_hit,
                    "resume_hit": resume_hit,
                    "context": context,
                    "resume_context": resume,
                }
            )

        summary = {
            "kind": "single_mode",
            "generated_at": utc_now(),
            "scenario_count": len(results),
            "procedure_hit_rate": self._rate(results, "procedure_hit"),
            "anti_pattern_hit_rate": self._rate(results, "anti_pattern_hit"),
            "preference_hit_rate": self._rate(results, "preference_hit"),
            "semantic_hit_rate": self._rate(results, "semantic_hit"),
            "resume_hit_rate": self._rate(results, "resume_hit"),
            "results": results,
        }
        self._record_benchmark_summary(summary)
        return summary

    def compare_memory_modes(self, scenarios: list[BenchmarkScenario]) -> dict[str, Any]:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            cold_engine = KlomboEngine(base / "cold")
            warm_engine = KlomboEngine(base / "warm")
            warm_summary = BenchmarkHarness(warm_engine).run(scenarios)
            cold_summary = BenchmarkHarness(cold_engine).run(
                [
                    BenchmarkScenario(
                        name=scenario.name,
                        repo_id=scenario.repo_id,
                        task_type=scenario.task_type,
                        request=scenario.request,
                        setup_missions=scenario.setup_missions,
                        expected_procedure_tool=scenario.expected_procedure_tool,
                        expected_anti_pattern_tool=scenario.expected_anti_pattern_tool,
                        expected_preference_key=scenario.expected_preference_key,
                        expected_semantic_substring=scenario.expected_semantic_substring,
                        expected_resume_step=scenario.expected_resume_step,
                    )
                    for scenario in scenarios
                ]
            )
            comparison = {
                "kind": "memory_comparison",
                "generated_at": utc_now(),
                "scenario_count": len(scenarios),
                "memory_on": warm_summary,
                "memory_off": cold_summary,
                "procedure_lift": round(
                    warm_summary["procedure_hit_rate"] - cold_summary["procedure_hit_rate"],
                    3,
                ),
                "anti_pattern_lift": round(
                    warm_summary["anti_pattern_hit_rate"] - cold_summary["anti_pattern_hit_rate"],
                    3,
                ),
                "preference_lift": round(
                    warm_summary["preference_hit_rate"] - cold_summary["preference_hit_rate"],
                    3,
                ),
                "semantic_lift": round(
                    warm_summary["semantic_hit_rate"] - cold_summary["semantic_hit_rate"],
                    3,
                ),
                "resume_lift": round(
                    warm_summary["resume_hit_rate"] - cold_summary["resume_hit_rate"],
                    3,
                ),
            }
            self._record_benchmark_summary(comparison)
            return comparison

    def verify_history(self) -> dict[str, Any]:
        payload = self.engine.benchmark_summary()
        history = list(payload.get("history", []))
        issues = []
        previous_signature = ""
        for index, envelope in enumerate(history):
            summary = envelope.get("summary")
            signature = envelope.get("signature", "")
            if not isinstance(summary, dict):
                issues.append(f"history[{index}] missing summary")
                continue
            expected = self._sign_summary(summary, previous_signature)
            if not hmac.compare_digest(signature, expected):
                issues.append(f"history[{index}] signature mismatch")
            if envelope.get("previous_signature", "") != previous_signature:
                issues.append(f"history[{index}] previous signature mismatch")
            previous_signature = signature
        return {"valid": not issues, "issues": issues, "count": len(history)}

    def _tool_hit(self, items: list[dict[str, Any]], expected_tool: str | None, field: str) -> bool:
        if not expected_tool:
            return True
        for item in items:
            if expected_tool in item.get(field, []):
                return True
        return False

    def _preference_hit(self, items: list[dict[str, Any]], expected_key: str | None) -> bool:
        if not expected_key:
            return True
        return any(item.get("key") == expected_key for item in items)

    def _semantic_hit(self, items: list[str], expected_substring: str | None) -> bool:
        if not expected_substring:
            return True
        lowered = expected_substring.lower()
        return any(lowered in item.lower() for item in items)

    def _resume_hit(self, item: dict[str, Any] | None, expected_step: str | None) -> bool:
        if not expected_step:
            return True
        if not item:
            return False
        return item.get("next_best_step") == expected_step

    def _rate(self, results: list[dict[str, Any]], key: str) -> float:
        if not results:
            return 0.0
        hits = sum(1 for item in results if item.get(key))
        return round(hits / len(results), 3)

    def _record_benchmark_summary(self, summary: dict[str, Any]) -> None:
        payload = self.engine.benchmark_summary()
        history = list(payload.get("history", []))
        regressions = list(payload.get("regressions", []))
        previous = payload.get("latest")
        previous_summary = previous.get("summary") if isinstance(previous, dict) and "summary" in previous else previous
        if previous_summary:
            regressions.extend(self._detect_regressions(previous_summary, summary))
        previous_signature = history[-1]["signature"] if history else ""
        envelope = {
            "summary": summary,
            "previous_signature": previous_signature,
            "signature": self._sign_summary(summary, previous_signature),
            "signature_meta": self._signing_metadata(),
            "signed_at": utc_now(),
        }
        history.append(envelope)
        stored = {
            "history": history[-20:],
            "latest": envelope,
            "regressions": regressions[-20:],
        }
        self.engine.storage.save_json(self.engine.storage.benchmark_runs_file, stored)

    def _detect_regressions(self, previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
        metrics = [
            "procedure_lift",
            "anti_pattern_lift",
            "preference_lift",
            "semantic_lift",
            "resume_lift",
            "procedure_hit_rate",
            "anti_pattern_hit_rate",
            "preference_hit_rate",
            "semantic_hit_rate",
            "resume_hit_rate",
        ]
        regressions = []
        for metric in metrics:
            if metric not in previous or metric not in current:
                continue
            if float(current[metric]) < float(previous[metric]):
                regressions.append(
                    {
                        "metric": metric,
                        "previous": previous[metric],
                        "current": current[metric],
                        "detected_at": utc_now(),
                    }
                )
        return regressions

    def _sign_summary(self, summary: dict[str, Any], previous_signature: str) -> str:
        key, _meta = self._resolve_signing_key()
        payload = json.dumps(
            {"summary": summary, "previous_signature": previous_signature},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    def _signing_metadata(self) -> dict[str, str]:
        _key, meta = self._resolve_signing_key()
        return dict(meta)

    def _resolve_signing_key(self) -> tuple[bytes, dict[str, str]]:
        if self._cached_signing_key is not None and self._cached_signing_meta is not None:
            return self._cached_signing_key, dict(self._cached_signing_meta)

        resolved_key: bytes
        source: str

        if self.signing_key is not None:
            resolved_key = self._coerce_signing_key(self.signing_key)
            source = "explicit"
        elif self.signing_key_provider is not None:
            provided = self.signing_key_provider()
            if provided:
                resolved_key = self._coerce_signing_key(provided)
                source = "provider"
            else:
                resolved_key, source = self._load_env_or_file_key()
        else:
            resolved_key, source = self._load_env_or_file_key()

        meta = {
            "source": source,
            "fingerprint": hashlib.sha256(resolved_key).hexdigest()[:16],
        }
        self._cached_signing_key = resolved_key
        self._cached_signing_meta = dict(meta)
        return resolved_key, meta

    def _load_env_or_file_key(self) -> tuple[bytes, str]:
        env_value = os.environ.get(self.signing_key_env)
        if env_value:
            return env_value.encode("utf-8"), f"env:{self.signing_key_env}"

        key_file = self.engine.storage.benchmark_signing_key_file
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip().encode("utf-8"), "state_file"

        key = secrets.token_hex(32)
        if self.persist_generated_key:
            key_file.write_text(key, encoding="utf-8")
            return key.encode("utf-8"), "generated+persisted"
        return key.encode("utf-8"), "generated+ephemeral"

    def _coerce_signing_key(self, value: str | bytes) -> bytes:
        return value if isinstance(value, bytes) else value.encode("utf-8")
