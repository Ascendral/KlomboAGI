from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from klombo.models import (
    AntiPatternMemory,
    Episode,
    MissionState,
    ProcedureMemory,
    RepoProfile,
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
        command_families = dict(profile.command_families)
        entrypoints = set(profile.entrypoints)
        test_dirs = set(profile.test_dirs)
        service_boundaries = set(profile.service_boundaries)
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
            if parent:
                common_paths[parent.strip("/")] += 1
                if "test" in parent.lower():
                    test_dirs.add(parent.strip("/"))
                boundary = self._derive_service_boundary(relative)
                if boundary:
                    service_boundaries.add(boundary)
            if self._is_entrypoint(path.name, relative):
                entrypoints.add(relative)
            lowered_name = path.name.lower()
            if lowered_name == "package.json":
                self._scan_package_json(path, frameworks, package_managers, command_families, semantic_facts, profile)
            elif lowered_name == "pyproject.toml":
                self._scan_pyproject(path, frameworks, package_managers, command_families, semantic_facts, profile)
            elif lowered_name in {"requirements.txt", "gemfile", "cargo.toml", "go.mod"}:
                self._scan_manifest_name(lowered_name, frameworks, package_managers, semantic_facts)
            elif lowered_name in {"pnpm-lock.yaml", "package-lock.json", "yarn.lock"}:
                self._scan_lockfile_name(lowered_name, package_managers, semantic_facts)
            if "auth" in relative.lower():
                fact = f"Auth-related code detected in {repo_id}: {relative}"
                if fact not in semantic_facts:
                    semantic_facts.append(fact)
        architecture_summary = self._build_architecture_summary(
            repo_id=repo_id,
            languages=languages,
            frameworks=frameworks,
            entrypoints=entrypoints,
            test_dirs=test_dirs,
            service_boundaries=service_boundaries,
        )

        profile.repo_path = str(root)
        profile.languages = [name for name, _count in languages.most_common(8)]
        profile.common_paths = [name for name, _count in common_paths.most_common(10)]
        profile.frameworks = sorted(frameworks)
        profile.repo_family = self._derive_repo_family(profile.languages, sorted(frameworks), sorted(package_managers))
        profile.package_managers = sorted(package_managers)
        profile.command_families = command_families
        profile.entrypoints = sorted(entrypoints)[:8]
        profile.test_dirs = sorted(test_dirs)[:8]
        profile.service_boundaries = sorted(service_boundaries)[:10]
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
        repo_profile = repo_profiles.get(mission["repo_id"])
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

        suggested_next_step = mission.get("next_best_step")
        if not suggested_next_step and scored_procedures and scored_procedures[0]["score"] > 0:
            suggested_next_step = f"Try recovered chain: {' -> '.join(scored_procedures[0]['item'].get('action_chain', []))}"
        recovery_plan, conflicts, chosen_strategy = self._resolve_recovery_plan(
            mission=mission,
            repo_profile=repo_profile,
            scored_procedures=scored_procedures,
            scored_anti_patterns=scored_anti_patterns,
        )

        return {
            **mission,
            "suggested_next_step": suggested_next_step,
            "recovery_hints": hints[:4],
            "recovery_plan": recovery_plan,
            "conflicts": conflicts,
            "chosen_strategy": chosen_strategy,
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
        transfer_candidates = self._find_transfer_candidates(repo_profiles, procedures, repo_id, task_type)

        return {
            "repo_profile": repo_profile,
            "procedures": [item["item"] for item in scored_procedures if item["score"] > 0][:max_items],
            "anti_patterns": [item["item"] for item in scored_anti_patterns if item["score"] > 0][:max_items],
            "preferences": active_preferences,
            "semantic_facts": [item["fact"] for item in semantic_facts],
            "transfer_candidates": transfer_candidates[:max_items],
            "explanations": {
                "procedures": [item for item in scored_procedures if item["score"] > 0][:max_items],
                "anti_patterns": [item for item in scored_anti_patterns if item["score"] > 0][:max_items],
                "preferences": self._explain_preferences(active_preferences),
                "semantic_facts": semantic_facts,
                "transfer_candidates": transfer_candidates[:max_items],
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
        command_families = dict(profile.command_families)
        common_paths = set(profile.common_paths)
        entrypoints = set(profile.entrypoints)
        test_dirs = set(profile.test_dirs)
        service_boundaries = set(profile.service_boundaries)
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
        profile.repo_family = self._derive_repo_family(profile.languages, profile.frameworks, profile.package_managers)
        profile.command_families = command_families
        profile.common_paths = sorted(common_paths)[:10]
        profile.entrypoints = sorted(entrypoints)[:8]
        profile.test_dirs = sorted(test_dirs)[:8]
        profile.service_boundaries = sorted(service_boundaries)[:10]
        profile.architecture_summary = self._build_architecture_summary(
            repo_id=episode.repo_id,
            languages=Counter(profile.languages),
            frameworks=frameworks,
            entrypoints=entrypoints,
            test_dirs=test_dirs,
            service_boundaries=service_boundaries,
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
    ) -> list[dict[str, Any]]:
        current = repo_profiles.get(repo_id)
        if not current or not current.get("repo_family"):
            return []
        current_family = current.get("repo_family")
        candidates = []
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
            strongest = max(float(item.get("confidence", 0.0)) for item in matching_procedures)
            candidates.append(
                {
                    "repo_id": other_id,
                    "repo_family": current_family,
                    "matching_procedure_count": len(matching_procedures),
                    "strongest_confidence": round(strongest, 4),
                    "reasons": [
                        f"same repo family {current_family}",
                        f"{len(matching_procedures)} matching procedures for task type {task_type}",
                    ],
                }
            )
        return sorted(
            candidates,
            key=lambda item: (item["strongest_confidence"], item["matching_procedure_count"]),
            reverse=True,
        )

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
        if "[project]" in lowered:
            fact = f"Python project metadata detected in {profile.repo_id}: pyproject.toml"
            if fact not in semantic_facts:
                semantic_facts.append(fact)

    def _scan_manifest_name(
        self,
        name: str,
        frameworks: set[str],
        package_managers: set[str],
        semantic_facts: list[str],
    ) -> None:
        if name == "requirements.txt":
            package_managers.add("python")
            frameworks.add("python")
            semantic_facts.append("Python dependency manifest detected: requirements.txt")
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

    def _age_days(self, timestamp: str) -> float:
        try:
            then = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
        now = datetime.now(timezone.utc)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (now - then).total_seconds() / 86400.0)
