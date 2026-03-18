from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from klombo.models import (
    AntiPatternMemory,
    Episode,
    MissionState,
    OperatorReviewDecision,
    ProcedureMemory,
    RepoProfile,
    TransferReview,
    UserPreference,
    utc_now,
)
from klombo.storage import KlomboStorage


_EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".php": "php",
    ".liquid": "liquid",
    ".css": "css",
    ".html": "html",
    ".md": "markdown",
}

_FRAMEWORK_HINTS = {
    "pytest": "pytest",
    "eslint": "eslint",
    "ruff": "ruff",
    "vitest": "vitest",
    "jest": "jest",
    "react": "react",
    "next": "nextjs",
    "shopify": "shopify-theme",
    "liquid": "shopify-theme",
}

_PYTHON_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([\.a-zA-Z0-9_]+)\s+import\s+")
_PYTHON_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)")
_JS_IMPORT_RE = re.compile(r"""from\s+["']([^"']+)["']""")
_JS_REQUIRE_RE = re.compile(r"""require\(\s*["']([^"']+)["']\s*\)""")
_JS_DYNAMIC_IMPORT_RE = re.compile(r"""import\(\s*["']([^"']+)["']\s*\)""")


class KlomboEngine:
    """Standalone learning core with no direct CodeBot attachment."""

    def __init__(self, root: str | Path) -> None:
        self.storage = KlomboStorage(root)

    def record_episode(self, episode: Episode | dict[str, Any]) -> dict[str, Any]:
        materialized = episode if isinstance(episode, Episode) else Episode.from_dict(episode)
        self.storage.append_jsonl(self.storage.episodes_file, materialized.to_dict())

        repo_profiles = self.storage.load_json(self.storage.repo_profiles_file, {})
        procedures = self.storage.load_json(self.storage.procedures_file, [])
        anti_patterns = self.storage.load_json(self.storage.anti_patterns_file, [])
        preferences = self.storage.load_json(self.storage.preferences_file, [])

        repo_profiles[materialized.repo_id] = self._update_repo_profile(
            repo_profiles.get(materialized.repo_id),
            materialized,
        )
        procedures = self._update_procedures(procedures, materialized)
        anti_patterns = self._update_anti_patterns(anti_patterns, materialized)
        preferences = self._update_preferences(preferences, materialized)

        procedures, anti_patterns, preferences = self._maintain_memories(
            procedures,
            anti_patterns,
            preferences,
        )

        self.storage.save_json(self.storage.repo_profiles_file, repo_profiles)
        self.storage.save_json(self.storage.procedures_file, procedures)
        self.storage.save_json(self.storage.anti_patterns_file, anti_patterns)
        self.storage.save_json(self.storage.preferences_file, preferences)

        return {
            "episode_id": materialized.id,
            "repo_id": materialized.repo_id,
            "success": materialized.success,
            "action_chain": self._action_chain(materialized),
            "updated_repo_profile": repo_profiles[materialized.repo_id],
        }

    def record_mission_state(self, state: MissionState | dict[str, Any]) -> dict[str, Any]:
        materialized = state if isinstance(state, MissionState) else MissionState(**state)
        missions = self.storage.load_json(self.storage.missions_file, {})
        missions[materialized.mission_id] = materialized.to_dict()
        self.storage.save_json(self.storage.missions_file, missions)
        return materialized.to_dict()

    def record_operator_review(
        self,
        decision: OperatorReviewDecision | dict[str, Any],
    ) -> dict[str, Any]:
        materialized = (
            decision
            if isinstance(decision, OperatorReviewDecision)
            else OperatorReviewDecision(**decision)
        )
        if not materialized.context_signature:
            missions = self.storage.load_json(self.storage.missions_file, {})
            mission = missions.get(materialized.mission_id)
            if mission:
                materialized.context_signature = self._mission_context_signature(mission)
        reviews = self.storage.load_json(self.storage.operator_reviews_file, {})
        reviews[materialized.mission_id] = materialized.to_dict()
        self.storage.save_json(self.storage.operator_reviews_file, reviews)
        return materialized.to_dict()

    def record_transfer_review(
        self,
        review: TransferReview | dict[str, Any],
    ) -> dict[str, Any]:
        materialized = review if isinstance(review, TransferReview) else TransferReview(**review)
        reviews = self.storage.load_json(self.storage.transfer_reviews_file, [])
        reviews.append(materialized.to_dict())
        self.storage.save_json(self.storage.transfer_reviews_file, reviews[-200:])
        return materialized.to_dict()

    def scan_repo(self, repo_id: str, repo_path: str | Path, *, max_files: int = 250) -> dict[str, Any]:
        root = Path(repo_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Repo path does not exist: {root}")

        repo_profiles = self.storage.load_json(self.storage.repo_profiles_file, {})
        current = repo_profiles.get(repo_id)
        profile = RepoProfile(**current) if current else RepoProfile(repo_id=repo_id, repo_path=str(root))
        languages = Counter(profile.languages)
        common_paths = Counter(profile.common_paths)
        frameworks = set(profile.frameworks)
        package_managers = set(profile.package_managers)
        external_dependencies = Counter(profile.external_dependencies)
        command_families = dict(profile.command_families)
        entrypoints = set(profile.entrypoints)
        test_dirs = set(profile.test_dirs)
        service_boundaries = set(profile.service_boundaries)
        ownership_zones = set(profile.ownership_zones)
        dependency_edges = set(profile.dependency_edges)
        dependency_hotspots = set(profile.dependency_hotspots)
        dependency_layers = list(profile.dependency_layers)
        architecture_summary = list(profile.architecture_summary)
        semantic_facts = list(profile.semantic_facts)

        scanned_files = 0
        for path in root.rglob("*"):
            if scanned_files >= max_files:
                break
            if any(part in {".git", "node_modules", "dist", "build", "__pycache__"} for part in path.parts):
                continue
            if path.is_dir():
                continue
            scanned_files += 1
            suffix = path.suffix.lower()
            relative = path.relative_to(root).as_posix()
            if suffix in _EXTENSION_LANGUAGE_MAP:
                languages[_EXTENSION_LANGUAGE_MAP[suffix]] += 1
            parent = path.parent.relative_to(root).as_posix().strip(".")
            ownership_zone = self._derive_ownership_zone(relative)
            if ownership_zone:
                ownership_zones.add(ownership_zone)
            if parent:
                common_paths[parent.strip("/")] += 1
                if "test" in parent.lower():
                    test_dirs.add(parent.strip("/"))
                boundary = self._derive_service_boundary(relative)
                if boundary:
                    service_boundaries.add(boundary)
            if self._is_entrypoint(path.name, relative):
                entrypoints.add(relative)
            file_dependency_edges, file_external_dependencies = self._extract_dependency_signals(path, relative)
            for edge in file_dependency_edges:
                dependency_edges.add(edge)
            for dependency in file_external_dependencies:
                external_dependencies[dependency] += 1
            lowered_name = path.name.lower()
            if lowered_name == "package.json":
                self._scan_package_json(
                    path,
                    frameworks,
                    package_managers,
                    external_dependencies,
                    command_families,
                    semantic_facts,
                    profile,
                )
            elif lowered_name == "pyproject.toml":
                self._scan_pyproject(
                    path,
                    frameworks,
                    package_managers,
                    external_dependencies,
                    command_families,
                    semantic_facts,
                    profile,
                )
            elif lowered_name in {"requirements.txt", "gemfile", "cargo.toml", "go.mod"}:
                self._scan_manifest_name(path, lowered_name, frameworks, package_managers, external_dependencies, semantic_facts)
            elif lowered_name in {"pnpm-lock.yaml", "package-lock.json", "yarn.lock"}:
                self._scan_lockfile_name(lowered_name, package_managers, semantic_facts)
            if "auth" in relative.lower():
                fact = f"Auth-related code detected in {repo_id}: {relative}"
                if fact not in semantic_facts:
                    semantic_facts.append(fact)
        dependency_hotspots = self._derive_dependency_hotspots(dependency_edges)
        for zone in sorted(ownership_zones)[:4]:
            fact = f"Ownership zone detected in {repo_id}: {zone}"
            if fact not in semantic_facts:
                semantic_facts.append(fact)
        for edge in sorted(dependency_edges)[:4]:
            fact = f"Dependency edge detected in {repo_id}: {edge}"
            if fact not in semantic_facts:
                semantic_facts.append(fact)
        for dependency in [name for name, _count in external_dependencies.most_common(4)]:
            fact = f"External dependency detected in {repo_id}: {dependency}"
            if fact not in semantic_facts:
                semantic_facts.append(fact)
        dependency_layers = self._derive_dependency_layers(ownership_zones, dependency_edges, dependency_hotspots)
        for layer in dependency_layers[:4]:
            fact = f"Dependency layer detected in {repo_id}: {layer}"
            if fact not in semantic_facts:
                semantic_facts.append(fact)
        architecture_summary = self._build_architecture_summary(
            repo_id=repo_id,
            languages=languages,
            frameworks=frameworks,
            entrypoints=entrypoints,
            test_dirs=test_dirs,
            service_boundaries=service_boundaries,
            ownership_zones=ownership_zones,
            dependency_edges=dependency_edges,
            dependency_hotspots=dependency_hotspots,
            dependency_layers=dependency_layers,
            external_dependencies=external_dependencies,
        )

        profile.repo_path = str(root)
        profile.languages = [name for name, _count in languages.most_common(8)]
        profile.common_paths = [name for name, _count in common_paths.most_common(10)]
        profile.frameworks = sorted(frameworks)
        profile.external_dependencies = [name for name, _count in external_dependencies.most_common(12)]
        profile.repo_family = self._derive_repo_family(profile.languages, sorted(frameworks), sorted(package_managers))
        profile.package_managers = sorted(package_managers)
        profile.command_families = command_families
        profile.entrypoints = sorted(entrypoints)[:8]
        profile.test_dirs = sorted(test_dirs)[:8]
        profile.service_boundaries = sorted(service_boundaries)[:10]
        profile.ownership_zones = sorted(ownership_zones)[:10]
        profile.dependency_edges = sorted(dependency_edges)[:12]
        profile.dependency_hotspots = sorted(dependency_hotspots)[:8]
        profile.dependency_layers = dependency_layers[:10]
        profile.architecture_summary = architecture_summary[:10]
        profile.semantic_facts = semantic_facts[-20:]
        profile.confidence = min(1.0, max(profile.confidence, 0.7))
        profile.updated_at = utc_now()

        repo_profiles[repo_id] = profile.to_dict()
        self.storage.save_json(self.storage.repo_profiles_file, repo_profiles)
        return profile.to_dict()

    def resume_context(self, mission_id: str) -> dict[str, Any] | None:
        missions = self.storage.load_json(self.storage.missions_file, {})
        mission = missions.get(mission_id)
        if mission is None:
            return None

        procedures = self.storage.load_json(self.storage.procedures_file, [])
        anti_patterns = self.storage.load_json(self.storage.anti_patterns_file, [])
        repo_profiles = self.storage.load_json(self.storage.repo_profiles_file, {})
        operator_reviews = self.storage.load_json(self.storage.operator_reviews_file, {})
        repo_profile = repo_profiles.get(mission["repo_id"])
        review_decision = operator_reviews.get(mission_id)
        applied_review_decision, decision_status, decision_reason = self._resolve_operator_review_state(
            mission,
            review_decision,
        )
        query_tokens = self._tokens(
            f"{mission.get('summary', '')} {' '.join(mission.get('blocked_actions', []))} {' '.join(mission.get('attempted_actions', []))}"
        )
        scored_procedures = sorted(
            (self._score_procedure(item, mission["repo_id"], "resume", query_tokens) for item in procedures),
            key=lambda pair: pair["score"],
            reverse=True,
        )
        scored_anti_patterns = sorted(
            (self._score_anti_pattern(item, mission["repo_id"], "resume", query_tokens) for item in anti_patterns),
            key=lambda pair: pair["score"],
            reverse=True,
        )

        hints: list[str] = []
        blocked_actions = list(mission.get("blocked_actions", []))
        if blocked_actions:
            hints.append(f"Avoid retrying blocked action unchanged: {blocked_actions[0]}")
        if scored_anti_patterns and scored_anti_patterns[0]["score"] > 0:
            chain = " -> ".join(scored_anti_patterns[0]["item"].get("failing_chain", []))
            hints.append(f"Known anti-pattern in this repo: {chain}")
        if scored_procedures and scored_procedures[0]["score"] > 0:
            chain = " -> ".join(scored_procedures[0]["item"].get("action_chain", []))
            hints.append(f"Recovered work usually succeeds via: {chain}")
        if repo_profile and repo_profile.get("preferred_test_commands"):
            hints.append(f"Prefer targeted verification command: {repo_profile['preferred_test_commands'][0]}")
        layer_hints = self._build_layer_hints(
            repo_profile,
            query_tokens,
            fallback_to_hotspots=True,
        )
        hints.extend(layer_hints[:2])

        suggested_next_step = mission.get("next_best_step")
        if not suggested_next_step and scored_procedures and scored_procedures[0]["score"] > 0:
            suggested_next_step = f"Try recovered chain: {' -> '.join(scored_procedures[0]['item'].get('action_chain', []))}"
        recovery_plan, conflicts, chosen_strategy = self._resolve_recovery_plan(
            mission=mission,
            repo_profile=repo_profile,
            scored_procedures=scored_procedures,
            scored_anti_patterns=scored_anti_patterns,
        )
        suggested_next_step, chosen_strategy = self._apply_operator_review_decision(
            mission=mission,
            recovery_plan=recovery_plan,
            suggested_next_step=suggested_next_step,
            chosen_strategy=chosen_strategy,
            review_decision=applied_review_decision,
        )
        operator_review = self._build_operator_review(
            mission=mission,
            recovery_plan=recovery_plan,
            conflicts=conflicts,
            chosen_strategy=chosen_strategy,
            review_decision=review_decision,
            decision_status=decision_status,
            decision_reason=decision_reason,
        )

        return {
            **mission,
            "suggested_next_step": suggested_next_step,
            "recovery_hints": hints[:4],
            "layer_hints": layer_hints[:3],
            "recovery_plan": recovery_plan,
            "conflicts": conflicts,
            "chosen_strategy": chosen_strategy,
            "review_required": operator_review["required"],
            "operator_review": operator_review,
            "avoid_action_chains": [
                item["item"].get("failing_chain", [])
                for item in scored_anti_patterns[:3]
                if item["score"] > 0
            ],
            "supporting_procedures": [
                item["item"]
                for item in scored_procedures[:3]
                if item["score"] > 0
            ],
            "repo_profile": repo_profile,
        }

    def maintain_memories(self) -> dict[str, int]:
        procedures = self.storage.load_json(self.storage.procedures_file, [])
        anti_patterns = self.storage.load_json(self.storage.anti_patterns_file, [])
        preferences = self.storage.load_json(self.storage.preferences_file, [])
        kept_procedures, kept_anti_patterns, kept_preferences = self._maintain_memories(
            procedures,
            anti_patterns,
            preferences,
        )
        self.storage.save_json(self.storage.procedures_file, kept_procedures)
        self.storage.save_json(self.storage.anti_patterns_file, kept_anti_patterns)
        self.storage.save_json(self.storage.preferences_file, kept_preferences)
        return {
            "procedures": len(kept_procedures),
            "anti_patterns": len(kept_anti_patterns),
            "preferences": len(kept_preferences),
        }

    def get_planning_context(
        self,
        *,
        repo_id: str,
        request: str,
        task_type: str,
        max_items: int = 3,
    ) -> dict[str, Any]:
        self.maintain_memories()
        repo_profiles = self.storage.load_json(self.storage.repo_profiles_file, {})
        procedures = self.storage.load_json(self.storage.procedures_file, [])
        anti_patterns = self.storage.load_json(self.storage.anti_patterns_file, [])
        preferences = self.storage.load_json(self.storage.preferences_file, [])

        query_tokens = self._tokens(f"{task_type} {request}")
        scored_procedures = sorted(
            (self._score_procedure(item, repo_id, task_type, query_tokens) for item in procedures),
            key=lambda pair: pair["score"],
            reverse=True,
        )
        scored_anti_patterns = sorted(
            (self._score_anti_pattern(item, repo_id, task_type, query_tokens) for item in anti_patterns),
            key=lambda pair: pair["score"],
            reverse=True,
        )

        repo_profile = repo_profiles.get(repo_id)
        active_preferences = self._rank_preferences(preferences, repo_id, max_items)
        semantic_facts = self._select_semantic_facts(repo_profile, query_tokens, max_items)
        layer_hints = self._build_layer_hints(repo_profile, query_tokens)
        transfer_candidates, transfer_controls = self._find_transfer_candidates(
            repo_profiles,
            procedures,
            repo_id,
            task_type,
            query_tokens,
        )

        return {
            "repo_profile": repo_profile,
            "procedures": [item["item"] for item in scored_procedures if item["score"] > 0][:max_items],
            "anti_patterns": [item["item"] for item in scored_anti_patterns if item["score"] > 0][:max_items],
            "preferences": active_preferences,
            "semantic_facts": [item["fact"] for item in semantic_facts],
            "layer_hints": layer_hints[:max_items],
            "transfer_candidates": transfer_candidates[:max_items],
            "transfer_controls": transfer_controls,
            "explanations": {
                "procedures": [item for item in scored_procedures if item["score"] > 0][:max_items],
                "anti_patterns": [item for item in scored_anti_patterns if item["score"] > 0][:max_items],
                "preferences": self._explain_preferences(active_preferences),
                "semantic_facts": semantic_facts,
                "layer_hints": layer_hints[:max_items],
                "transfer_candidates": transfer_candidates[:max_items],
                "transfer_controls": transfer_controls,
            },
            "generated_at": utc_now(),
        }

    def benchmark_summary(self) -> dict[str, Any]:
        return self.storage.load_json(
            self.storage.benchmark_runs_file,
            {"history": [], "latest": None, "regressions": []},
        )

    def _update_repo_profile(self, current: dict[str, Any] | None, episode: Episode) -> dict[str, Any]:
        profile = RepoProfile(**current) if current else RepoProfile(repo_id=episode.repo_id, repo_path=episode.repo_path)
        languages = set(profile.languages)
        package_managers = set(profile.package_managers)
        frameworks = set(profile.frameworks)
        external_dependencies = set(profile.external_dependencies)
        command_families = dict(profile.command_families)
        common_paths = set(profile.common_paths)
        entrypoints = set(profile.entrypoints)
        test_dirs = set(profile.test_dirs)
        service_boundaries = set(profile.service_boundaries)
        ownership_zones = set(profile.ownership_zones)
        dependency_hotspots = set(profile.dependency_hotspots)
        dependency_layers = list(profile.dependency_layers)
        semantic_facts = list(profile.semantic_facts)

        for path in episode.files_touched:
            suffix = Path(path).suffix.lower()
            language = _EXTENSION_LANGUAGE_MAP.get(suffix)
            if language:
                languages.add(language)
            if suffix in {".tsx", ".jsx"}:
                frameworks.add("react")
            if suffix == ".liquid":
                frameworks.add("shopify-theme")
            parent = Path(path).parent.as_posix().strip("/")
            if parent and parent != ".":
                common_paths.add(parent)
                if "test" in parent.lower():
                    test_dirs.add(parent)
                boundary = self._derive_service_boundary(path)
                if boundary:
                    service_boundaries.add(boundary)
            ownership_zone = self._derive_ownership_zone(path)
            if ownership_zone:
                ownership_zones.add(ownership_zone)
            if self._is_entrypoint(Path(path).name, path):
                entrypoints.add(path)
            for token, framework in _FRAMEWORK_HINTS.items():
                if token in path.lower():
                    frameworks.add(framework)

        for command in episode.commands:
            lower = command.lower()
            family = self._classify_command(lower)
            command_families[family] = int(command_families.get(family, 0)) + 1
            if "pnpm" in lower:
                package_managers.add("pnpm")
            if "npm" in lower:
                package_managers.add("npm")
            if "yarn" in lower:
                package_managers.add("yarn")
            for token, framework in _FRAMEWORK_HINTS.items():
                if token in lower:
                    frameworks.add(framework)
            if family == "test":
                profile.preferred_test_commands = self._bump_command(profile.preferred_test_commands, command)
            if family == "lint":
                profile.preferred_lint_commands = self._bump_command(profile.preferred_lint_commands, command)
            if family == "build":
                profile.preferred_build_commands = self._bump_command(profile.preferred_build_commands, command)

        if episode.success:
            for command in episode.commands[:2]:
                fact = f"Reliable command in {episode.repo_id}: {command}"
                if fact not in semantic_facts:
                    semantic_facts.append(fact)
            for path in sorted(common_paths)[:2]:
                fact = f"Frequently touched path in {episode.repo_id}: {path}"
                if fact not in semantic_facts:
                    semantic_facts.append(fact)
            for framework in sorted(frameworks)[:2]:
                fact = f"Detected framework in {episode.repo_id}: {framework}"
                if fact not in semantic_facts:
                    semantic_facts.append(fact)

        profile.languages = sorted(languages)
        profile.package_managers = sorted(package_managers)
        profile.frameworks = sorted(frameworks)
        profile.external_dependencies = sorted(external_dependencies)[:12]
        profile.repo_family = self._derive_repo_family(profile.languages, profile.frameworks, profile.package_managers)
        profile.command_families = command_families
        profile.common_paths = sorted(common_paths)[:10]
        profile.entrypoints = sorted(entrypoints)[:8]
        profile.test_dirs = sorted(test_dirs)[:8]
        profile.service_boundaries = sorted(service_boundaries)[:10]
        profile.ownership_zones = sorted(ownership_zones)[:10]
        profile.dependency_hotspots = sorted(dependency_hotspots)[:8]
        profile.dependency_layers = dependency_layers[:10]
        profile.architecture_summary = self._build_architecture_summary(
            repo_id=episode.repo_id,
            languages=Counter(profile.languages),
            frameworks=frameworks,
            entrypoints=entrypoints,
            test_dirs=test_dirs,
            service_boundaries=service_boundaries,
            ownership_zones=ownership_zones,
            dependency_edges=set(profile.dependency_edges),
            dependency_hotspots=dependency_hotspots,
            dependency_layers=dependency_layers,
            external_dependencies=Counter(profile.external_dependencies),
        )[:10]
        profile.semantic_facts = semantic_facts[-14:]
        profile.confidence = min(1.0, profile.confidence + (0.05 if episode.success else 0.01))
        profile.updated_at = utc_now()
        return profile.to_dict()

    def _update_procedures(self, current: list[dict[str, Any]], episode: Episode) -> list[dict[str, Any]]:
        chain = self._action_chain(episode)
        if not chain:
            return current
        match_index = self._find_chain_match(current, chain, episode.repo_id, episode.task_type, key="action_chain")
        if match_index is None:
            if not episode.success:
                return current
            item = ProcedureMemory(
                name=f"{episode.task_type} via {' -> '.join(chain)}",
                repo_id=episode.repo_id,
                task_type=episode.task_type,
                trigger_terms=sorted(self._tokens(episode.request))[:12],
                action_chain=chain,
                success_count=1,
                confidence=0.65,
                last_outcome="success",
                source_episode_ids=[episode.id],
            )
            current.append(item.to_dict())
            return current

        item = dict(current[match_index])
        if episode.success:
            item["success_count"] = int(item.get("success_count", 0)) + 1
            item["confidence"] = min(1.0, float(item.get("confidence", 0.5)) + 0.1)
            item["last_outcome"] = "success"
        else:
            item["failure_count"] = int(item.get("failure_count", 0)) + 1
            item["confidence"] = max(0.0, float(item.get("confidence", 0.5)) - 0.12)
            item["last_outcome"] = "failure"
        item["updated_at"] = utc_now()
        item["source_episode_ids"] = list(dict.fromkeys([*item.get("source_episode_ids", []), episode.id]))[-10:]
        current[match_index] = item
        return current

    def _update_anti_patterns(self, current: list[dict[str, Any]], episode: Episode) -> list[dict[str, Any]]:
        if episode.success:
            return current
        chain = self._action_chain(episode)
        if not chain:
            return current
        match_index = self._find_chain_match(current, chain, episode.repo_id, episode.task_type, key="failing_chain")
        evidence = [action.summary for action in episode.actions if action.summary]
        if match_index is None:
            item = AntiPatternMemory(
                name=f"Avoid {' -> '.join(chain)} for {episode.task_type}",
                repo_id=episode.repo_id,
                task_type=episode.task_type,
                trigger_terms=sorted(self._tokens(episode.request))[:12],
                failing_chain=chain,
                failure_count=1,
                confidence=0.65,
                evidence=evidence[:3],
                source_episode_ids=[episode.id],
            )
            current.append(item.to_dict())
            return current

        item = dict(current[match_index])
        item["failure_count"] = int(item.get("failure_count", 0)) + 1
        item["confidence"] = min(1.0, float(item.get("confidence", 0.5)) + 0.1)
        item["updated_at"] = utc_now()
        item["source_episode_ids"] = list(dict.fromkeys([*item.get("source_episode_ids", []), episode.id]))[-10:]
        item["evidence"] = list(dict.fromkeys([*item.get("evidence", []), *evidence]))[-5:]
        current[match_index] = item
        return current

    def _update_preferences(self, current: list[dict[str, Any]], episode: Episode) -> list[dict[str, Any]]:
        if not episode.observed_preferences:
            return current
        indexed = {
            (item.get("repo_id"), item["key"], str(item["value"])): idx
            for idx, item in enumerate(current)
        }
        for key, value in episode.observed_preferences.items():
            scoped_lookup = (episode.repo_id, str(key), str(value))
            global_lookup = (None, str(key), str(value))
            if scoped_lookup not in indexed:
                pref = UserPreference(key=str(key), value=value, repo_id=episode.repo_id, confidence=0.6)
                current.append(pref.to_dict())
                indexed[scoped_lookup] = len(current) - 1
            else:
                idx = indexed[scoped_lookup]
                item = dict(current[idx])
                item["evidence_count"] = int(item.get("evidence_count", 1)) + 1
                item["confidence"] = min(1.0, float(item.get("confidence", 0.5)) + 0.08)
                item["updated_at"] = utc_now()
                current[idx] = item

            if self._seen_across_multiple_repos(current, str(key), value):
                if global_lookup not in indexed:
                    pref = UserPreference(key=str(key), value=value, repo_id=None, confidence=0.58)
                    current.append(pref.to_dict())
                    indexed[global_lookup] = len(current) - 1
                else:
                    idx = indexed[global_lookup]
                    item = dict(current[idx])
                    item["evidence_count"] = int(item.get("evidence_count", 1)) + 1
                    item["confidence"] = min(1.0, float(item.get("confidence", 0.5)) + 0.05)
                    item["updated_at"] = utc_now()
                    current[idx] = item
        return current

    def _maintain_memories(
        self,
        procedures: list[dict[str, Any]],
        anti_patterns: list[dict[str, Any]],
        preferences: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            self._decay_and_prune_list(procedures, floor=0.18, max_age_days=120, decay_per_day=0.0025),
            self._decay_and_prune_list(anti_patterns, floor=0.2, max_age_days=180, decay_per_day=0.0015),
            self._decay_and_prune_list(preferences, floor=0.22, max_age_days=240, decay_per_day=0.001),
        )

    def _decay_and_prune_list(
        self,
        items: list[dict[str, Any]],
        *,
        floor: float,
        max_age_days: int,
        decay_per_day: float,
    ) -> list[dict[str, Any]]:
        kept = []
        for item in items:
            aged = dict(item)
            age_days = self._age_days(str(item.get("updated_at", utc_now())))
            decayed_confidence = max(0.0, float(item.get("confidence", 0.5)) - (age_days * decay_per_day))
            aged["confidence"] = round(decayed_confidence, 4)
            if age_days > max_age_days and decayed_confidence < floor:
                continue
            kept.append(aged)
        return kept

    def _select_semantic_facts(
        self,
        repo_profile: dict[str, Any] | None,
        query_tokens: set[str],
        max_items: int,
    ) -> list[dict[str, Any]]:
        if not repo_profile:
            return []
        facts = []
        for fact in repo_profile.get("semantic_facts", []):
            fact_tokens = self._tokens(fact)
            score = len(query_tokens.intersection(fact_tokens)) if query_tokens else 0
            reasons = []
            if score:
                reasons.append(f"matched {score} query tokens")
            facts.append({"fact": fact, "score": score, "reasons": reasons or ["repo fact fallback"]})
        facts.sort(key=lambda item: item["score"], reverse=True)
        preferred = [item for item in facts if item["score"] > 0][:max_items]
        return preferred or facts[:max_items]

    def _score_procedure(
        self,
        item: dict[str, Any],
        repo_id: str,
        task_type: str,
        query_tokens: set[str],
    ) -> dict[str, Any]:
        score = float(item.get("confidence", 0.0))
        reasons = [f"base confidence {round(float(item.get('confidence', 0.0)), 3)}"]
        if item.get("repo_id") == repo_id:
            score += 1.0
            reasons.append("repo match +1.0")
        if item.get("task_type") == task_type:
            score += 0.8
            reasons.append("task-type match +0.8")
        overlap = len(query_tokens.intersection(set(item.get("trigger_terms", []))))
        if overlap:
            delta = overlap * 0.2
            score += delta
            reasons.append(f"trigger overlap +{round(delta, 3)}")
        success_bonus = min(int(item.get("success_count", 0)), 5) * 0.1
        if success_bonus:
            score += success_bonus
            reasons.append(f"success history +{round(success_bonus, 3)}")
        failure_penalty = int(item.get("failure_count", 0)) * 0.08
        if failure_penalty:
            score -= failure_penalty
            reasons.append(f"failure penalty -{round(failure_penalty, 3)}")
        return {"score": round(score, 4), "item": item, "reasons": reasons}

    def _score_anti_pattern(
        self,
        item: dict[str, Any],
        repo_id: str,
        task_type: str,
        query_tokens: set[str],
    ) -> dict[str, Any]:
        score = float(item.get("confidence", 0.0))
        reasons = [f"base confidence {round(float(item.get('confidence', 0.0)), 3)}"]
        if item.get("repo_id") == repo_id:
            score += 1.0
            reasons.append("repo match +1.0")
        if item.get("task_type") == task_type:
            score += 0.8
            reasons.append("task-type match +0.8")
        overlap = len(query_tokens.intersection(set(item.get("trigger_terms", []))))
        if overlap:
            delta = overlap * 0.2
            score += delta
            reasons.append(f"trigger overlap +{round(delta, 3)}")
        failure_bonus = min(int(item.get("failure_count", 0)), 5) * 0.15
        if failure_bonus:
            score += failure_bonus
            reasons.append(f"failure history +{round(failure_bonus, 3)}")
        return {"score": round(score, 4), "item": item, "reasons": reasons}

    def _explain_preferences(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        explained = []
        for item in items:
            scope = f"repo-scoped to {item['repo_id']}" if item.get("repo_id") else "global preference"
            explained.append(
                {
                    "item": item,
                    "score": round(float(item.get("confidence", 0.0)), 4),
                    "reasons": [
                        scope,
                        f"confidence {round(float(item.get('confidence', 0.0)), 3)}",
                        f"evidence count {int(item.get('evidence_count', 0))}",
                    ],
                }
            )
        return explained

    def _rank_preferences(self, preferences: list[dict[str, Any]], repo_id: str, max_items: int) -> list[dict[str, Any]]:
        ranked = sorted(
            preferences,
            key=lambda item: (
                item.get("repo_id") != repo_id,
                item.get("repo_id") is None,
                -float(item.get("confidence", 0.0)),
                -int(item.get("evidence_count", 0)),
            ),
        )
        repo_specific = [item for item in ranked if item.get("repo_id") == repo_id]
        global_items = [item for item in ranked if item.get("repo_id") is None]
        return [*repo_specific, *global_items][:max_items]

    def _seen_across_multiple_repos(self, preferences: list[dict[str, Any]], key: str, value: Any) -> bool:
        repo_ids = {
            item.get("repo_id")
            for item in preferences
            if item.get("repo_id") and item.get("key") == key and item.get("value") == value
        }
        return len(repo_ids) >= 2

    def _find_transfer_candidates(
        self,
        repo_profiles: dict[str, dict[str, Any]],
        procedures: list[dict[str, Any]],
        repo_id: str,
        task_type: str,
        query_tokens: set[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current = repo_profiles.get(repo_id)
        if not current or not current.get("repo_family"):
            return [], self._empty_transfer_controls()
        current_family = current.get("repo_family")
        layer_signal = self._transfer_layer_signal(current, query_tokens)
        transfer_reviews = self.storage.load_json(self.storage.transfer_reviews_file, [])
        candidates = []
        review_count = 0
        guided_count = 0
        blocked_count = 0
        matched_review_count = 0
        for other_id, profile in repo_profiles.items():
            if other_id == repo_id:
                continue
            if profile.get("repo_family") != current_family:
                continue
            matching_procedures = [
                item
                for item in procedures
                if item.get("repo_id") == other_id and item.get("task_type") == task_type
            ]
            if not matching_procedures:
                continue
            representative_procedure = max(
                matching_procedures,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    int(item.get("success_count", 0)),
                    len(item.get("action_chain", [])),
                ),
            )
            procedure_signature = self._procedure_signature(representative_procedure)
            strongest = float(representative_procedure.get("confidence", 0.0))
            strongest_success_count = int(representative_procedure.get("success_count", 0))
            base_transfer_score = round(
                (strongest * 0.7)
                + (min(strongest_success_count, 3) * 0.05)
                + (min(len(matching_procedures), 3) * 0.08)
                + (float(profile.get("confidence", 0.5)) * 0.25),
                4,
            )
            history = self._summarize_transfer_history(
                transfer_reviews,
                repo_family=current_family,
                task_type=task_type,
                procedure_signature=procedure_signature,
            )
            matched_review_count += history["matched_reviews"]
            transfer_score = round(
                base_transfer_score + history["history_adjustment"] + layer_signal["adjustment"],
                4,
            )
            policy = self._score_transfer_candidate(transfer_score)
            candidate = {
                "repo_id": other_id,
                "repo_family": current_family,
                "matching_procedure_count": len(matching_procedures),
                "procedure_signature": procedure_signature,
                "strongest_confidence": round(strongest, 4),
                "strongest_success_count": strongest_success_count,
                "base_transfer_score": base_transfer_score,
                "history_adjustment": history["history_adjustment"],
                "layer_adjustment": layer_signal["adjustment"],
                "matched_layers": layer_signal["matched_layers"],
                "accepted_review_count": history["accepted_count"],
                "rejected_review_count": history["rejected_count"],
                "transfer_score": transfer_score,
                "transfer_tier": policy["tier"],
                "review_required": policy["review_required"],
                "apply_mode": policy["apply_mode"],
                "reasons": [
                    f"same repo family {current_family}",
                    f"{len(matching_procedures)} matching procedures for task type {task_type}",
                    f"base transfer score {base_transfer_score}",
                    f"review history adjustment {history['history_adjustment']}",
                    f"layer adjustment {layer_signal['adjustment']}",
                    *history["reasons"],
                    *layer_signal["reasons"],
                    f"transfer score {transfer_score}",
                    *policy["reasons"],
                ],
            }
            if policy["tier"] == "blocked":
                blocked_count += 1
                continue
            if policy["review_required"]:
                review_count += 1
            else:
                guided_count += 1
            candidates.append(candidate)
        return (
            sorted(
                candidates,
                key=lambda item: (
                    item["transfer_score"],
                    item["strongest_confidence"],
                    item["matching_procedure_count"],
                ),
                reverse=True,
            ),
            {
                "policy": "decision_aware_transfer_v0.10",
                "guided_threshold": 0.9,
                "review_threshold": 0.72,
                "eligible_without_review": guided_count,
                "eligible_with_review": review_count,
                "blocked_candidates": blocked_count,
                "matched_review_count": matched_review_count,
                "review_required": review_count > 0,
            },
        )

    def _empty_transfer_controls(self) -> dict[str, Any]:
        return {
            "policy": "decision_aware_transfer_v0.10",
            "guided_threshold": 0.9,
            "review_threshold": 0.72,
            "eligible_without_review": 0,
            "eligible_with_review": 0,
            "blocked_candidates": 0,
            "matched_review_count": 0,
            "review_required": False,
        }

    def _transfer_layer_signal(self, repo_profile: dict[str, Any], query_tokens: set[str]) -> dict[str, Any]:
        matched_layers = self._match_dependency_layers(repo_profile, query_tokens)
        if not matched_layers:
            return {"adjustment": 0.0, "matched_layers": [], "reasons": ["request did not match a risky dependency layer"]}

        highest_risk = max(
            (self._layer_risk_weight(role) for _zone, role in matched_layers),
            default=0.0,
        )
        adjustment = round(-highest_risk, 4)
        reasons = [
            f"request touches {role} layer {zone}"
            for zone, role in matched_layers[:2]
        ]
        return {
            "adjustment": adjustment,
            "matched_layers": [f"{zone}:{role}" for zone, role in matched_layers[:3]],
            "reasons": reasons,
        }

    def _layer_risk_weight(self, role: str) -> float:
        if role == "foundation":
            return 0.08
        if role == "orchestration":
            return 0.04
        if role == "shared":
            return 0.03
        return 0.0

    def _score_transfer_candidate(self, transfer_score: float) -> dict[str, Any]:
        if transfer_score >= 0.9:
            return {
                "tier": "guided",
                "review_required": False,
                "apply_mode": "reference",
                "reasons": ["score cleared guided transfer threshold"],
            }
        if transfer_score >= 0.72:
            return {
                "tier": "review",
                "review_required": True,
                "apply_mode": "operator_review",
                "reasons": ["score requires operator review before transfer is applied"],
            }
        return {
            "tier": "blocked",
            "review_required": True,
            "apply_mode": "withhold",
            "reasons": ["score stayed below transfer threshold"],
        }

    def _summarize_transfer_history(
        self,
        transfer_reviews: list[dict[str, Any]],
        *,
        repo_family: str,
        task_type: str,
        procedure_signature: str,
    ) -> dict[str, Any]:
        accepted_count = 0
        rejected_count = 0
        for item in transfer_reviews:
            if item.get("repo_family") != repo_family:
                continue
            if item.get("task_type") != task_type:
                continue
            if item.get("procedure_signature") != procedure_signature:
                continue
            if item.get("accepted", False):
                accepted_count += 1
            else:
                rejected_count += 1
        history_adjustment = round((min(accepted_count, 3) * 0.05) - (min(rejected_count, 3) * 0.12), 4)
        reasons = []
        if accepted_count:
            reasons.append(f"{accepted_count} accepted transfer reviews matched this procedure")
        if rejected_count:
            reasons.append(f"{rejected_count} rejected transfer reviews matched this procedure")
        if not reasons:
            reasons.append("no matching transfer review history yet")
        return {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "matched_reviews": accepted_count + rejected_count,
            "history_adjustment": history_adjustment,
            "reasons": reasons,
        }

    def _procedure_signature(self, procedure: dict[str, Any]) -> str:
        return " -> ".join(str(tool) for tool in procedure.get("action_chain", []))

    def _mission_context_signature(self, mission: dict[str, Any]) -> str:
        payload = {
            "repo_id": mission.get("repo_id"),
            "summary": mission.get("summary"),
            "status": mission.get("status"),
            "last_plan": mission.get("last_plan"),
            "attempted_actions": mission.get("attempted_actions", []),
            "blocked_actions": mission.get("blocked_actions", []),
            "next_best_step": mission.get("next_best_step"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _resolve_operator_review_state(
        self,
        mission: dict[str, Any],
        review_decision: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        if not review_decision:
            return None, "none", None
        approved = bool(review_decision.get("approved", True))
        if not approved:
            return None, "rejected", "Previous operator review did not approve auto-reuse."
        stored_signature = review_decision.get("context_signature")
        if stored_signature and stored_signature != self._mission_context_signature(mission):
            return None, "invalidated", "Prior operator approval was invalidated because the mission context changed."
        return review_decision, "approved", None

    def _resolve_recovery_plan(
        self,
        *,
        mission: dict[str, Any],
        repo_profile: dict[str, Any] | None,
        scored_procedures: list[dict[str, Any]],
        scored_anti_patterns: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str], str]:
        blocked_actions = list(mission.get("blocked_actions", []))
        conflicts: list[str] = []
        plan: list[dict[str, Any]] = []

        if blocked_actions:
            plan.append(
                {
                    "priority": 1,
                    "step": f"Do not repeat blocked action unchanged: {blocked_actions[0]}",
                    "source": "blocked_action",
                }
            )

        top_anti = scored_anti_patterns[0] if scored_anti_patterns and scored_anti_patterns[0]["score"] > 0 else None
        top_proc = scored_procedures[0] if scored_procedures and scored_procedures[0]["score"] > 0 else None

        if top_anti:
            failing_chain = " -> ".join(top_anti["item"].get("failing_chain", []))
            plan.append(
                {
                    "priority": 2,
                    "step": f"Avoid known failing chain: {failing_chain}",
                    "source": "anti_pattern",
                }
            )
        if top_proc:
            action_chain = " -> ".join(top_proc["item"].get("action_chain", []))
            plan.append(
                {
                    "priority": 3,
                    "step": f"Use strongest recovered procedure: {action_chain}",
                    "source": "procedure",
                }
            )

        if repo_profile and repo_profile.get("preferred_test_commands"):
            plan.append(
                {
                    "priority": 4,
                    "step": f"Verify with targeted command: {repo_profile['preferred_test_commands'][0]}",
                    "source": "repo_profile",
                }
            )

        chosen_strategy = "follow_saved_next_step"
        if top_proc and blocked_actions:
            blocked_text = " ".join(blocked_actions).lower()
            chain_tokens = " ".join(top_proc["item"].get("action_chain", [])).lower()
            if any(token in blocked_text for token in chain_tokens.split()):
                conflicts.append("Top recovered procedure overlaps with a blocked action pattern.")
                chosen_strategy = "prefer_safe_variation"
        elif top_proc:
            chosen_strategy = "use_recovered_procedure"
        elif top_anti:
            chosen_strategy = "avoid_known_anti_pattern"

        return sorted(plan, key=lambda item: item["priority"]), conflicts, chosen_strategy

    def _build_operator_review(
        self,
        *,
        mission: dict[str, Any],
        recovery_plan: list[dict[str, Any]],
        conflicts: list[str],
        chosen_strategy: str,
        review_decision: dict[str, Any] | None = None,
        decision_status: str = "none",
        decision_reason: str | None = None,
    ) -> dict[str, Any]:
        required = bool(conflicts) and decision_status != "approved"
        options = []
        next_step = mission.get("next_best_step")
        if next_step:
            options.append(
                {
                    "id": "follow_saved_next_step",
                    "label": "Follow saved next step",
                    "step": next_step,
                }
            )
        if recovery_plan:
            options.append(
                {
                    "id": "apply_recovery_plan",
                    "label": "Apply the highest-priority safe recovery step",
                    "step": recovery_plan[0]["step"],
                }
            )
        if mission.get("blocked_actions"):
            options.append(
                {
                    "id": "pause_and_replan",
                    "label": "Pause and request a fresh plan",
                    "step": f"Replan around blocked action: {mission['blocked_actions'][0]}",
                }
            )
        unique_options = []
        seen = set()
        for option in options:
            marker = (option["id"], option["step"])
            if marker in seen:
                continue
            seen.add(marker)
            unique_options.append(option)
        if decision_status == "approved":
            summary = f"Operator approved strategy: {review_decision.get('selected_option', chosen_strategy)}"
        elif decision_status in {"rejected", "invalidated"} and decision_reason:
            summary = decision_reason
        elif conflicts:
            summary = conflicts[0]
        else:
            summary = "No operator review required."
        return {
            "required": required,
            "summary": summary,
            "recommended_option": chosen_strategy,
            "options": unique_options[:3],
            "conflicts": conflicts,
            "decision_status": decision_status,
            "decision": review_decision,
        }

    def _apply_operator_review_decision(
        self,
        *,
        mission: dict[str, Any],
        recovery_plan: list[dict[str, Any]],
        suggested_next_step: str | None,
        chosen_strategy: str,
        review_decision: dict[str, Any] | None,
    ) -> tuple[str | None, str]:
        if not review_decision or not review_decision.get("approved", True):
            return suggested_next_step, chosen_strategy

        selected_option = str(review_decision.get("selected_option") or chosen_strategy)
        selected_step = review_decision.get("selected_step")
        if selected_option == "apply_recovery_plan" and recovery_plan:
            return str(selected_step or recovery_plan[0]["step"]), selected_option
        if selected_option == "pause_and_replan":
            return str(selected_step or "Pause and request a fresh plan"), selected_option
        if selected_option == "follow_saved_next_step":
            return str(selected_step or mission.get("next_best_step") or suggested_next_step), selected_option
        return suggested_next_step, selected_option

    def _find_chain_match(
        self,
        items: list[dict[str, Any]],
        chain: list[str],
        repo_id: str,
        task_type: str,
        *,
        key: str,
    ) -> int | None:
        for index, item in enumerate(items):
            if item.get(key) == chain and item.get("repo_id") == repo_id and item.get("task_type") == task_type:
                return index
        return None

    def _derive_repo_family(
        self,
        languages: list[str],
        frameworks: list[str],
        package_managers: list[str],
    ) -> str:
        language = languages[0] if languages else "unknown"
        framework = frameworks[0] if frameworks else "generic"
        package_manager = package_managers[0] if package_managers else "none"
        return f"{language}:{framework}:{package_manager}"

    def _derive_service_boundary(self, relative_path: str) -> str | None:
        normalized = relative_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")
        if len(parts) < 2:
            return None
        interesting = {"src", "app", "services", "service", "api", "server", "components", "lib"}
        for index, part in enumerate(parts[:-1]):
            if part in interesting and index + 1 < len(parts):
                return "/".join(parts[: index + 2])
        if any(token in normalized.lower() for token in ("auth", "checkout", "deploy", "migration")):
            return "/".join(parts[:2])
        return None

    def _derive_ownership_zone(self, relative_path: str) -> str | None:
        normalized = relative_path.replace("\\", "/").strip("/")
        if not normalized:
            return None
        boundary = self._derive_service_boundary(normalized)
        if boundary:
            return boundary
        parts = [part for part in normalized.split("/") if part and part != "."]
        if not parts:
            return None
        if parts[0] in {"tests", "test", "docs", "config", "scripts"}:
            return parts[0]
        if len(parts) >= 2 and parts[0] in {"src", "app", "klombo"}:
            return "/".join(parts[:2])
        return parts[0]

    def _extract_dependency_signals(self, path: Path, relative_path: str) -> tuple[list[str], list[str]]:
        if path.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            return [], []
        source_zone = self._derive_ownership_zone(relative_path)
        if not source_zone:
            return [], []
        try:
            if path.stat().st_size > 200_000:
                return [], []
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        edges = set()
        external_dependencies = set()
        for line in text.splitlines()[:200]:
            for reference in self._extract_import_references(line, path.suffix.lower()):
                target_zone = self._resolve_dependency_zone(reference, relative_path)
                if target_zone:
                    edges.add(f"{source_zone} -> {target_zone}")
                external_dependency = self._normalize_external_dependency(reference)
                if external_dependency:
                    external_dependencies.add(external_dependency)
        return sorted(edges), sorted(external_dependencies)

    def _extract_import_references(self, line: str, suffix: str) -> list[str]:
        refs: list[str] = []
        if suffix == ".py":
            from_match = _PYTHON_FROM_IMPORT_RE.match(line)
            if from_match:
                refs.append(from_match.group(1))
            import_match = _PYTHON_IMPORT_RE.match(line)
            if import_match:
                for chunk in import_match.group(1).split(","):
                    refs.append(chunk.strip().split(" as ")[0].strip())
            return [ref for ref in refs if ref]
        refs.extend(_JS_IMPORT_RE.findall(line))
        refs.extend(_JS_REQUIRE_RE.findall(line))
        refs.extend(_JS_DYNAMIC_IMPORT_RE.findall(line))
        return [ref for ref in refs if ref]

    def _resolve_dependency_zone(self, reference: str, relative_path: str) -> str | None:
        cleaned = reference.strip()
        if not cleaned:
            return None
        if cleaned.startswith("@/"):
            return self._derive_ownership_zone(cleaned[2:])
        if cleaned.startswith("."):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative_path), cleaned))
            return self._derive_ownership_zone(resolved)
        if "/" in cleaned:
            return self._derive_ownership_zone(cleaned)
        if "." in cleaned:
            normalized = cleaned.lstrip(".").replace(".", "/")
            return self._derive_ownership_zone(normalized)
        return None

    def _normalize_external_dependency(self, reference: str) -> str | None:
        cleaned = reference.strip()
        if not cleaned or cleaned.startswith(".") or cleaned.startswith("@/"):
            return None
        root = cleaned.replace("\\", "/")
        local_roots = {
            "src",
            "app",
            "lib",
            "server",
            "services",
            "service",
            "components",
            "tests",
            "test",
            "docs",
            "config",
            "scripts",
            "styles",
            "assets",
            "public",
            "klombo",
        }
        if "/" in root:
            first = root.split("/", 1)[0]
            if first in local_roots:
                return None
            if root.startswith("@"):
                parts = root.split("/")
                return "/".join(parts[:2]) if len(parts) >= 2 else root
            return first
        if "." in root:
            first = root.split(".", 1)[0]
            if first in local_roots:
                return None
            return first
        return root

    def _derive_dependency_hotspots(self, dependency_edges: set[str]) -> set[str]:
        inbound = Counter()
        for edge in dependency_edges:
            _source, _arrow, target = edge.partition(" -> ")
            if target:
                inbound[target] += 1
        return {zone for zone, count in inbound.items() if count >= 1}

    def _derive_dependency_layers(
        self,
        ownership_zones: set[str],
        dependency_edges: set[str],
        dependency_hotspots: set[str],
    ) -> list[str]:
        inbound = Counter()
        outbound = Counter()
        for edge in dependency_edges:
            source, _arrow, target = edge.partition(" -> ")
            if source and target:
                outbound[source] += 1
                inbound[target] += 1
        layers = []
        for zone in sorted(ownership_zones):
            if zone in dependency_hotspots and inbound[zone] >= max(1, outbound[zone]):
                role = "foundation"
            elif outbound[zone] > inbound[zone]:
                role = "orchestration"
            elif inbound[zone] > 0:
                role = "shared"
            else:
                role = "leaf"
            layers.append(f"{zone}:{role}")
        return layers

    def _parse_dependency_layers(self, repo_profile: dict[str, Any] | None) -> list[tuple[str, str]]:
        if not repo_profile:
            return []
        parsed = []
        for entry in repo_profile.get("dependency_layers", []):
            zone, separator, role = str(entry).partition(":")
            if zone and separator and role:
                parsed.append((zone, role))
        return parsed

    def _match_dependency_layers(
        self,
        repo_profile: dict[str, Any] | None,
        query_tokens: set[str],
    ) -> list[tuple[str, str]]:
        if not repo_profile or not query_tokens:
            return []
        matched = []
        hotspots = set(repo_profile.get("dependency_hotspots", []))
        for zone, role in self._parse_dependency_layers(repo_profile):
            zone_tokens = self._tokens(zone.replace("/", " "))
            if query_tokens.intersection(zone_tokens):
                boosted_role = "foundation" if zone in hotspots and role in {"shared", "leaf"} else role
                matched.append((zone, boosted_role))
        return matched

    def _build_layer_hints(
        self,
        repo_profile: dict[str, Any] | None,
        query_tokens: set[str],
        *,
        fallback_to_hotspots: bool = False,
    ) -> list[str]:
        if not repo_profile:
            return []
        matched_layers = self._match_dependency_layers(repo_profile, query_tokens)
        if not matched_layers and fallback_to_hotspots:
            matched_layers = [
                (zone, "foundation")
                for zone in repo_profile.get("dependency_hotspots", [])[:1]
            ]
        hints = []
        for zone, role in matched_layers[:3]:
            if role == "foundation":
                hints.append(f"Treat {zone} as a shared foundation layer; prefer narrow edits and verify dependents.")
            elif role == "orchestration":
                hints.append(f"Treat {zone} as an orchestration layer; validate downstream calls before broad refactors.")
            elif role == "shared":
                hints.append(f"{zone} is shared across multiple zones; verify adjacent surfaces after changes.")
            else:
                hints.append(f"Keep work localized in {zone} before escalating to shared layers.")
        return hints

    def _is_entrypoint(self, filename: str, relative_path: str) -> bool:
        lowered = filename.lower()
        normalized = relative_path.lower()
        return lowered in {
            "main.py",
            "__main__.py",
            "app.py",
            "manage.py",
            "index.ts",
            "index.tsx",
            "main.ts",
            "main.tsx",
            "server.ts",
            "server.js",
            "theme.liquid",
        } or normalized.endswith("layout/theme.liquid")

    def _build_architecture_summary(
        self,
        *,
        repo_id: str,
        languages: Counter,
        frameworks: set[str],
        entrypoints: set[str],
        test_dirs: set[str],
        service_boundaries: set[str],
        ownership_zones: set[str],
        dependency_edges: set[str],
        dependency_hotspots: set[str],
        dependency_layers: list[str],
        external_dependencies: Counter,
    ) -> list[str]:
        summary = []
        if languages:
            summary.append(f"Primary language in {repo_id}: {languages.most_common(1)[0][0]}")
        if frameworks:
            summary.append(f"Detected frameworks in {repo_id}: {', '.join(sorted(frameworks)[:3])}")
        if entrypoints:
            summary.append(f"Entrypoints in {repo_id}: {', '.join(sorted(entrypoints)[:3])}")
        if test_dirs:
            summary.append(f"Test directories in {repo_id}: {', '.join(sorted(test_dirs)[:3])}")
        if service_boundaries:
            summary.append(f"Service boundaries in {repo_id}: {', '.join(sorted(service_boundaries)[:4])}")
        if ownership_zones:
            summary.append(f"Ownership zones in {repo_id}: {', '.join(sorted(ownership_zones)[:4])}")
        if dependency_edges:
            summary.append(f"Dependency edges in {repo_id}: {', '.join(sorted(dependency_edges)[:3])}")
        if dependency_hotspots:
            summary.append(f"Dependency hubs in {repo_id}: {', '.join(sorted(dependency_hotspots)[:4])}")
        if dependency_layers:
            summary.append(
                f"Dependency layers in {repo_id}: {', '.join(item.replace(':', ' (', 1) + ')' for item in dependency_layers[:4])}"
            )
        if external_dependencies:
            summary.append(
                f"External dependencies in {repo_id}: {', '.join(name for name, _count in external_dependencies.most_common(4))}"
            )
        return summary

    def _action_chain(self, episode: Episode) -> list[str]:
        chain = []
        for action in episode.actions:
            tool = action.tool.strip().lower()
            if tool:
                chain.append(tool)
        return chain

    def _tokens(self, text: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {token for token in cleaned.split() if len(token) > 2}

    def _bump_command(self, current: list[str], command: str) -> list[str]:
        unique = [item for item in current if item != command]
        return [command, *unique][:5]

    def _classify_command(self, command: str) -> str:
        if any(token in command for token in ("pytest", "unittest", "vitest", "jest", " test")):
            return "test"
        if any(token in command for token in ("lint", "ruff", "eslint")):
            return "lint"
        if "build" in command:
            return "build"
        if any(token in command for token in ("migrate", "migration")):
            return "migrate"
        if any(token in command for token in ("dev", "serve", "start")):
            return "serve"
        return "other"

    def _scan_package_json(
        self,
        path: Path,
        frameworks: set[str],
        package_managers: set[str],
        external_dependencies: Counter,
        command_families: dict[str, int],
        semantic_facts: list[str],
        profile: RepoProfile,
    ) -> None:
        package_managers.add("npm")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        scripts = dict(data.get("scripts", {}))
        deps = {
            **dict(data.get("dependencies", {})),
            **dict(data.get("devDependencies", {})),
        }
        for name in deps:
            normalized = self._normalize_manifest_dependency(name)
            if normalized:
                external_dependencies[normalized] += 1
            for token, framework in _FRAMEWORK_HINTS.items():
                if token in name.lower():
                    frameworks.add(framework)
        for script_name, command in scripts.items():
            family = self._classify_command(str(command).lower())
            command_families[family] = int(command_families.get(family, 0)) + 1
            if family == "test":
                profile.preferred_test_commands = self._bump_command(profile.preferred_test_commands, str(command))
            if family == "lint":
                profile.preferred_lint_commands = self._bump_command(profile.preferred_lint_commands, str(command))
            if family == "build":
                profile.preferred_build_commands = self._bump_command(profile.preferred_build_commands, str(command))
            fact = f"package.json script in {profile.repo_id}: {script_name} -> {command}"
            if fact not in semantic_facts:
                semantic_facts.append(fact)

    def _scan_pyproject(
        self,
        path: Path,
        frameworks: set[str],
        package_managers: set[str],
        external_dependencies: Counter,
        command_families: dict[str, int],
        semantic_facts: list[str],
        profile: RepoProfile,
    ) -> None:
        text = path.read_text(encoding="utf-8")
        package_managers.add("python")
        lowered = text.lower()
        if "[tool.pytest" in lowered or "pytest" in lowered:
            frameworks.add("pytest")
            command_families["test"] = int(command_families.get("test", 0)) + 1
            profile.preferred_test_commands = self._bump_command(profile.preferred_test_commands, "pytest")
        if "[tool.ruff" in lowered or "ruff" in lowered:
            frameworks.add("ruff")
            command_families["lint"] = int(command_families.get("lint", 0)) + 1
            profile.preferred_lint_commands = self._bump_command(profile.preferred_lint_commands, "ruff check .")
        for dependency in self._extract_pyproject_dependency_names(text):
            normalized = self._normalize_manifest_dependency(dependency)
            if normalized:
                external_dependencies[normalized] += 1
        if "[project]" in lowered:
            fact = f"Python project metadata detected in {profile.repo_id}: pyproject.toml"
            if fact not in semantic_facts:
                semantic_facts.append(fact)

    def _scan_manifest_name(
        self,
        path: Path,
        name: str,
        frameworks: set[str],
        package_managers: set[str],
        external_dependencies: Counter,
        semantic_facts: list[str],
    ) -> None:
        if name == "requirements.txt":
            package_managers.add("python")
            frameworks.add("python")
            semantic_facts.append("Python dependency manifest detected: requirements.txt")
            for dependency in self._extract_requirements_dependency_names(path):
                normalized = self._normalize_manifest_dependency(dependency)
                if normalized:
                    external_dependencies[normalized] += 1
        elif name == "gemfile":
            package_managers.add("bundler")
            frameworks.add("ruby")
            semantic_facts.append("Ruby dependency manifest detected: Gemfile")
        elif name == "cargo.toml":
            package_managers.add("cargo")
            frameworks.add("rust")
            semantic_facts.append("Rust dependency manifest detected: Cargo.toml")
        elif name == "go.mod":
            package_managers.add("go")
            frameworks.add("go")
            semantic_facts.append("Go module manifest detected: go.mod")

    def _scan_lockfile_name(
        self,
        name: str,
        package_managers: set[str],
        semantic_facts: list[str],
    ) -> None:
        if name == "pnpm-lock.yaml":
            package_managers.add("pnpm")
        elif name == "package-lock.json":
            package_managers.add("npm")
        elif name == "yarn.lock":
            package_managers.add("yarn")
        semantic_facts.append(f"Lockfile detected: {name}")

    def _extract_requirements_dependency_names(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        names = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            candidate = re.split(r"[<>=!~\[]", stripped, maxsplit=1)[0].strip()
            if candidate:
                names.append(candidate)
        return names

    def _extract_pyproject_dependency_names(self, text: str) -> list[str]:
        names = []
        for match in re.findall(r'["\']([A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,\-]+\])?(?:\s*[<>=!~].*?)?)["\']', text):
            candidate = re.split(r"[<>=!~\[]", match, maxsplit=1)[0].strip()
            if candidate and candidate.lower() not in {"python", "sample"}:
                names.append(candidate)
        return names

    def _normalize_manifest_dependency(self, name: str) -> str | None:
        cleaned = name.strip().strip("\"'").replace("\\", "/")
        if not cleaned:
            return None
        if cleaned.startswith("@"):
            parts = cleaned.split("/")
            return "/".join(parts[:2]) if len(parts) >= 2 else cleaned
        return cleaned.split("/", 1)[0]

    def _age_days(self, timestamp: str) -> float:
        try:
            then = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
        now = datetime.now(timezone.utc)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (now - then).total_seconds() / 86400.0)
