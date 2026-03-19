from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from klombo.benchmark import BenchmarkHarness, BenchmarkScenario
from klombo.engine import KlomboEngine
from klombo.fixtures import (
    default_repo_scenarios,
    layer_guidance_scenarios,
    layer_sensitive_operator_review_scenarios,
)
from klombo.models import ActionRecord, Episode, MissionState


class KlomboEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = KlomboEngine(self.root / "memory")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_episode_updates_repo_profile_and_procedure(self) -> None:
        episode = Episode(
            repo_id="repo-a",
            repo_path="/tmp/repo-a",
            task_type="bugfix",
            request="Fix failing auth test in src/auth",
            success=True,
            actions=[
                ActionRecord(tool="search_files", success=True, summary="Found auth code"),
                ActionRecord(tool="apply_patch", success=True, summary="Patched auth handler"),
                ActionRecord(tool="run_command", success=True, command="pytest tests/test_auth.py"),
            ],
            files_touched=["src/auth/service.py", "tests/test_auth.py"],
            commands=["pytest tests/test_auth.py"],
            observed_preferences={"test_scope": "targeted"},
        )

        result = self.engine.record_episode(episode)
        context = self.engine.get_planning_context(
            repo_id="repo-a",
            request="Fix auth bug in src/auth",
            task_type="bugfix",
        )

        self.assertTrue(result["success"])
        self.assertEqual(context["repo_profile"]["repo_id"], "repo-a")
        self.assertIn("python", context["repo_profile"]["languages"])
        self.assertIn("src/auth", context["repo_profile"]["common_paths"])
        self.assertIn("pytest", context["repo_profile"]["frameworks"])
        self.assertEqual(context["repo_profile"]["command_families"]["test"], 1)
        self.assertTrue(context["procedures"])
        self.assertEqual(context["procedures"][0]["action_chain"][0], "search_files")
        self.assertTrue(context["preferences"])
        self.assertTrue(context["explanations"]["procedures"])
        self.assertTrue(any("repo match" in reason for reason in context["explanations"]["procedures"][0]["reasons"]))

    def test_scan_repo_extracts_semantic_facts_from_filesystem(self) -> None:
        repo_root = self.root / "sample-repo"
        (repo_root / "src" / "auth").mkdir(parents=True)
        (repo_root / "src" / "api").mkdir(parents=True)
        (repo_root / "tests").mkdir(parents=True)
        (repo_root / "app.py").write_text(
            "import requests\nfrom src.auth.service import refresh_token\n",
            encoding="utf-8",
        )
        (repo_root / "src" / "auth" / "service.py").write_text("def refresh_token():\n    return True\n", encoding="utf-8")
        (repo_root / "src" / "api" / "routes.py").write_text(
            "import jwt\nfrom src.auth.service import refresh_token\n\ndef route():\n    return refresh_token()\n",
            encoding="utf-8",
        )
        (repo_root / "tests" / "test_auth.py").write_text("def test_auth():\n    assert True\n", encoding="utf-8")
        (repo_root / "requirements.txt").write_text("requests==2.32.0\npyjwt>=2.8\n", encoding="utf-8")
        (repo_root / "package.json").write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^18.0.0", "zod": "^3.0.0"},
                    "scripts": {"test": "vitest", "lint": "eslint ."},
                }
            ),
            encoding="utf-8",
        )
        (repo_root / "pyproject.toml").write_text(
            "[project]\nname='sample'\n[tool.pytest.ini_options]\naddopts='-q'\n[tool.ruff]\n",
            encoding="utf-8",
        )

        profile = self.engine.scan_repo("repo-scan", repo_root)

        self.assertIn("python", profile["languages"])
        self.assertIn("react", profile["frameworks"])
        self.assertIn("pytest", profile["frameworks"])
        self.assertIn("npm", profile["package_managers"])
        self.assertIn("src/auth", profile["common_paths"])
        self.assertTrue(profile["repo_family"])
        self.assertIn("app.py", profile["entrypoints"])
        self.assertIn("tests", profile["test_dirs"])
        self.assertIn("src/auth", profile["service_boundaries"])
        self.assertIn("src/api", profile["ownership_zones"])
        self.assertIn("src/auth", profile["ownership_zones"])
        self.assertIn("src/api -> src/auth", profile["dependency_edges"])
        self.assertIn("src/auth", profile["dependency_hotspots"])
        self.assertIn("src/auth:foundation", profile["dependency_layers"])
        self.assertIn("react", profile["external_dependencies"])
        self.assertIn("requests", profile["external_dependencies"])
        self.assertTrue(profile["architecture_summary"])
        self.assertTrue(any("Dependency hubs" in item for item in profile["architecture_summary"]))
        self.assertTrue(any("Dependency layers" in item for item in profile["architecture_summary"]))
        self.assertTrue(any("External dependencies" in item for item in profile["architecture_summary"]))
        self.assertTrue(any("package.json script" in fact for fact in profile["semantic_facts"]))
        self.assertTrue(any("Dependency layer detected" in fact for fact in profile["semantic_facts"]))
        self.assertTrue(any("External dependency detected" in fact for fact in profile["semantic_facts"]))

    def test_planning_context_adds_layer_hints_for_dependency_hubs(self) -> None:
        repo_root = self.root / "layered-repo"
        (repo_root / "src" / "auth").mkdir(parents=True)
        (repo_root / "src" / "api").mkdir(parents=True)
        (repo_root / "app.py").write_text("from src.api.routes import route\n", encoding="utf-8")
        (repo_root / "src" / "auth" / "service.py").write_text("def refresh_token():\n    return True\n", encoding="utf-8")
        (repo_root / "src" / "api" / "routes.py").write_text(
            "from src.auth.service import refresh_token\n\ndef route():\n    return refresh_token()\n",
            encoding="utf-8",
        )
        (repo_root / "pyproject.toml").write_text("[project]\nname='layered'\n[tool.pytest.ini_options]\n", encoding="utf-8")

        self.engine.scan_repo("repo-layer", repo_root)
        context = self.engine.get_planning_context(
            repo_id="repo-layer",
            request="Fix auth token refresh in src/auth",
            task_type="bugfix",
            max_items=5,
        )

        self.assertTrue(context["layer_hints"])
        self.assertTrue(any("src/auth" in hint for hint in context["layer_hints"]))

    def test_transfer_candidates_use_repo_family_matches(self) -> None:
        shared_a = self.root / "repo-a"
        shared_b = self.root / "repo-b"
        for root in [shared_a, shared_b]:
            (root / "src" / "checkout").mkdir(parents=True)
            (root / "tests").mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps({"dependencies": {"react": "^18.0.0"}, "scripts": {"test": "vitest"}}),
                encoding="utf-8",
            )
            (root / "src" / "checkout" / "index.tsx").write_text("export const Checkout = () => null;\n", encoding="utf-8")
            (root / "tests" / "checkout.test.tsx").write_text("it('works', () => {});\n", encoding="utf-8")

        self.engine.scan_repo("repo-family-a", shared_a)
        self.engine.scan_repo("repo-family-b", shared_b)
        self.engine.record_episode(
            Episode(
                repo_id="repo-family-b",
                repo_path=str(shared_b),
                task_type="feature",
                request="Add promo banner to checkout flow",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="write_file", success=True),
                    ActionRecord(tool="run_command", success=True, command="vitest checkout"),
                ],
                files_touched=["src/checkout/Banner.tsx", "tests/checkout.test.tsx"],
                commands=["vitest checkout"],
            )
        )

        context = self.engine.get_planning_context(
            repo_id="repo-family-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )

        self.assertTrue(context["transfer_candidates"])
        self.assertEqual(context["transfer_candidates"][0]["repo_id"], "repo-family-b")
        self.assertEqual(context["transfer_candidates"][0]["transfer_tier"], "review")
        self.assertTrue(context["transfer_candidates"][0]["review_required"])
        self.assertEqual(context["transfer_controls"]["eligible_with_review"], 1)
        self.assertIn("same repo family", context["explanations"]["transfer_candidates"][0]["reasons"][0])

    def test_transfer_candidates_can_clear_review_with_stronger_evidence(self) -> None:
        shared_a = self.root / "repo-guided-a"
        shared_b = self.root / "repo-guided-b"
        for root in [shared_a, shared_b]:
            (root / "src" / "checkout").mkdir(parents=True)
            (root / "tests").mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps({"dependencies": {"react": "^18.0.0"}, "scripts": {"test": "vitest"}}),
                encoding="utf-8",
            )
            (root / "src" / "checkout" / "index.tsx").write_text("export const Checkout = () => null;\n", encoding="utf-8")
            (root / "tests" / "checkout.test.tsx").write_text("it('works', () => {});\n", encoding="utf-8")

        self.engine.scan_repo("repo-guided-a", shared_a)
        self.engine.scan_repo("repo-guided-b", shared_b)
        for _ in range(3):
            self.engine.record_episode(
                Episode(
                    repo_id="repo-guided-b",
                    repo_path=str(shared_b),
                    task_type="feature",
                    request="Add promo banner to checkout flow",
                    success=True,
                    actions=[
                        ActionRecord(tool="search_files", success=True),
                        ActionRecord(tool="write_file", success=True),
                        ActionRecord(tool="run_command", success=True, command="vitest checkout"),
                    ],
                    files_touched=["src/checkout/Banner.tsx", "tests/checkout.test.tsx"],
                    commands=["vitest checkout"],
                )
            )

        context = self.engine.get_planning_context(
            repo_id="repo-guided-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )

        self.assertTrue(context["transfer_candidates"])
        self.assertEqual(context["transfer_candidates"][0]["transfer_tier"], "guided")
        self.assertFalse(context["transfer_candidates"][0]["review_required"])
        self.assertEqual(context["transfer_controls"]["eligible_without_review"], 1)

    def test_transfer_reviews_can_promote_candidate_to_guided(self) -> None:
        shared_a = self.root / "repo-reviewed-a"
        shared_b = self.root / "repo-reviewed-b"
        for root in [shared_a, shared_b]:
            (root / "src" / "checkout").mkdir(parents=True)
            (root / "tests").mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps({"dependencies": {"react": "^18.0.0"}, "scripts": {"test": "vitest"}}),
                encoding="utf-8",
            )
            (root / "src" / "checkout" / "index.tsx").write_text("export const Checkout = () => null;\n", encoding="utf-8")
            (root / "tests" / "checkout.test.tsx").write_text("it('works', () => {});\n", encoding="utf-8")

        self.engine.scan_repo("repo-reviewed-a", shared_a)
        self.engine.scan_repo("repo-reviewed-b", shared_b)
        self.engine.record_episode(
            Episode(
                repo_id="repo-reviewed-b",
                repo_path=str(shared_b),
                task_type="feature",
                request="Add promo banner to checkout flow",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="write_file", success=True),
                    ActionRecord(tool="run_command", success=True, command="vitest checkout"),
                ],
                files_touched=["src/checkout/Banner.tsx", "tests/checkout.test.tsx"],
                commands=["vitest checkout"],
            )
        )

        before = self.engine.get_planning_context(
            repo_id="repo-reviewed-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )
        candidate = before["transfer_candidates"][0]
        self.assertEqual(candidate["transfer_tier"], "review")

        for _ in range(3):
            self.engine.record_transfer_review(
                {
                    "repo_id": "repo-reviewed-a",
                    "candidate_repo_id": "repo-reviewed-b",
                    "repo_family": candidate["repo_family"],
                    "task_type": "feature",
                    "procedure_signature": candidate["procedure_signature"],
                    "accepted": True,
                    "transfer_score": candidate["transfer_score"],
                    "transfer_tier": candidate["transfer_tier"],
                    "apply_mode": candidate["apply_mode"],
                }
            )

        after = self.engine.get_planning_context(
            repo_id="repo-reviewed-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )

        self.assertEqual(after["transfer_candidates"][0]["transfer_tier"], "guided")
        self.assertEqual(after["transfer_candidates"][0]["accepted_review_count"], 3)
        self.assertEqual(after["transfer_controls"]["matched_review_count"], 3)

    def test_rejected_transfer_reviews_can_block_candidate(self) -> None:
        shared_a = self.root / "repo-block-a"
        shared_b = self.root / "repo-block-b"
        for root in [shared_a, shared_b]:
            (root / "src" / "checkout").mkdir(parents=True)
            (root / "tests").mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps({"dependencies": {"react": "^18.0.0"}, "scripts": {"test": "vitest"}}),
                encoding="utf-8",
            )
            (root / "src" / "checkout" / "index.tsx").write_text("export const Checkout = () => null;\n", encoding="utf-8")
            (root / "tests" / "checkout.test.tsx").write_text("it('works', () => {});\n", encoding="utf-8")

        self.engine.scan_repo("repo-block-a", shared_a)
        self.engine.scan_repo("repo-block-b", shared_b)
        self.engine.record_episode(
            Episode(
                repo_id="repo-block-b",
                repo_path=str(shared_b),
                task_type="feature",
                request="Add promo banner to checkout flow",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="write_file", success=True),
                    ActionRecord(tool="run_command", success=True, command="vitest checkout"),
                ],
                files_touched=["src/checkout/Banner.tsx", "tests/checkout.test.tsx"],
                commands=["vitest checkout"],
            )
        )

        before = self.engine.get_planning_context(
            repo_id="repo-block-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )
        candidate = before["transfer_candidates"][0]
        self.assertEqual(candidate["transfer_tier"], "review")

        self.engine.record_transfer_review(
            {
                "repo_id": "repo-block-a",
                "candidate_repo_id": "repo-block-b",
                "repo_family": candidate["repo_family"],
                "task_type": "feature",
                "procedure_signature": candidate["procedure_signature"],
                "accepted": False,
                "transfer_score": candidate["transfer_score"],
                "transfer_tier": candidate["transfer_tier"],
                "apply_mode": candidate["apply_mode"],
                "notes": "Transfer suggested the wrong checkout pattern",
            }
        )

        after = self.engine.get_planning_context(
            repo_id="repo-block-a",
            request="Add promo banner to checkout flow",
            task_type="feature",
            max_items=5,
        )

        self.assertFalse(after["transfer_candidates"])
        self.assertEqual(after["transfer_controls"]["blocked_candidates"], 1)
        self.assertEqual(after["transfer_controls"]["matched_review_count"], 1)

    def test_repeated_failures_create_ranked_anti_pattern(self) -> None:
        for _ in range(2):
            self.engine.record_episode(
                Episode(
                    repo_id="repo-b",
                    repo_path="/tmp/repo-b",
                    task_type="refactor",
                    request="Refactor build config safely",
                    success=False,
                    actions=[
                        ActionRecord(tool="run_command", success=False, summary="Global build timed out", command="npm test"),
                        ActionRecord(tool="apply_patch", success=False, summary="Patch invalidated config"),
                    ],
                    commands=["npm test"],
                    stop_reason="timeout",
                )
            )

        context = self.engine.get_planning_context(
            repo_id="repo-b",
            request="Refactor build config safely",
            task_type="refactor",
        )

        self.assertTrue(context["anti_patterns"])
        self.assertEqual(context["anti_patterns"][0]["failing_chain"], ["run_command", "apply_patch"])
        self.assertGreaterEqual(context["anti_patterns"][0]["failure_count"], 2)
        self.assertTrue(context["explanations"]["anti_patterns"])

    def test_resume_context_returns_saved_mission_state(self) -> None:
        state = MissionState(
            mission_id="mission_123",
            repo_id="repo-c",
            summary="Resume interrupted auth migration",
            status="active",
            last_plan="Search migration files then patch targeted modules",
            attempted_actions=["search_files", "read_file"],
            blocked_actions=["run_command:npm test"],
            next_best_step="Patch auth migration in src/server/auth",
        )
        self.engine.record_mission_state(state)
        resumed = self.engine.resume_context("mission_123")

        self.assertIsNotNone(resumed)
        self.assertEqual(resumed["repo_id"], "repo-c")
        self.assertEqual(resumed["next_best_step"], "Patch auth migration in src/server/auth")

    def test_resume_context_includes_recovery_hints(self) -> None:
        repo_root = self.root / "resume-repo"
        (repo_root / "src" / "auth").mkdir(parents=True)
        (repo_root / "src" / "api").mkdir(parents=True)
        (repo_root / "src" / "auth" / "service.py").write_text("def migrate():\n    return True\n", encoding="utf-8")
        (repo_root / "src" / "api" / "routes.py").write_text(
            "from src.auth.service import migrate\n\ndef route():\n    return migrate()\n",
            encoding="utf-8",
        )
        (repo_root / "pyproject.toml").write_text("[project]\nname='resume'\n[tool.pytest.ini_options]\n", encoding="utf-8")
        self.engine.scan_repo("repo-resume", repo_root)
        self.engine.record_episode(
            Episode(
                repo_id="repo-resume",
                repo_path=str(repo_root),
                task_type="resume",
                request="Resume interrupted auth migration",
                success=False,
                actions=[
                    ActionRecord(tool="run_command", success=False, summary="Full test timed out", command="npm test"),
                    ActionRecord(tool="apply_patch", success=False, summary="Patched wrong file"),
                ],
                commands=["npm test"],
            )
        )
        self.engine.record_episode(
            Episode(
                repo_id="repo-resume",
                repo_path=str(repo_root),
                task_type="resume",
                request="Resume interrupted auth migration",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="apply_patch", success=True),
                    ActionRecord(tool="run_command", success=True, command="pnpm test auth-migration"),
                ],
                commands=["pnpm test auth-migration"],
            )
        )
        self.engine.record_mission_state(
            MissionState(
                mission_id="mission_resume",
                repo_id="repo-resume",
                summary="Resume interrupted auth migration",
                status="active",
                blocked_actions=["run_command:npm test"],
                next_best_step="Patch auth migration in src/server/auth/migrate.ts",
            )
        )

        resumed = self.engine.resume_context("mission_resume")

        self.assertIsNotNone(resumed)
        self.assertTrue(resumed["recovery_hints"])
        self.assertTrue(resumed["recovery_plan"])
        self.assertEqual(resumed["chosen_strategy"], "prefer_safe_variation")
        self.assertTrue(resumed["conflicts"])
        self.assertTrue(resumed["review_required"])
        self.assertEqual(resumed["operator_review"]["recommended_option"], "prefer_safe_variation")
        self.assertTrue(resumed["operator_review"]["options"])
        self.assertTrue(resumed["avoid_action_chains"])
        self.assertTrue(resumed["supporting_procedures"])
        self.assertTrue(resumed["layer_hints"])
        self.assertTrue(any("src/auth" in hint for hint in resumed["layer_hints"]))
        self.assertEqual(resumed["suggested_next_step"], "Patch auth migration in src/server/auth/migrate.ts")

    def test_transfer_candidates_penalize_shared_foundation_changes(self) -> None:
        current_root = self.root / "repo-layer-current"
        candidate_root = self.root / "repo-layer-candidate"
        for root in [current_root, candidate_root]:
            (root / "src" / "auth").mkdir(parents=True)
            (root / "src" / "api").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='layered'\n[tool.pytest.ini_options]\n", encoding="utf-8")
        (current_root / "src" / "auth" / "service.py").write_text("def refresh():\n    return True\n", encoding="utf-8")
        (current_root / "src" / "api" / "routes.py").write_text(
            "from src.auth.service import refresh\n\ndef route():\n    return refresh()\n",
            encoding="utf-8",
        )
        (candidate_root / "src" / "auth" / "service.py").write_text("def refresh():\n    return True\n", encoding="utf-8")
        (candidate_root / "src" / "api" / "routes.py").write_text(
            "from src.auth.service import refresh\n\ndef route():\n    return refresh()\n",
            encoding="utf-8",
        )

        self.engine.scan_repo("repo-layer-current", current_root)
        self.engine.scan_repo("repo-layer-candidate", candidate_root)
        for _ in range(3):
            self.engine.record_episode(
                Episode(
                    repo_id="repo-layer-candidate",
                    repo_path=str(candidate_root),
                    task_type="bugfix",
                    request="Fix auth token refresh in src/auth",
                    success=True,
                    actions=[
                        ActionRecord(tool="search_files", success=True),
                        ActionRecord(tool="apply_patch", success=True),
                        ActionRecord(tool="run_command", success=True, command="pytest tests/test_auth.py"),
                    ],
                    files_touched=["src/auth/service.py"],
                    commands=["pytest tests/test_auth.py"],
                )
            )

        high_risk = self.engine.get_planning_context(
            repo_id="repo-layer-current",
            request="Fix auth token refresh in src/auth",
            task_type="bugfix",
            max_items=5,
        )
        low_risk = self.engine.get_planning_context(
            repo_id="repo-layer-current",
            request="Fix output copy in cli",
            task_type="bugfix",
            max_items=5,
        )

        self.assertTrue(high_risk["transfer_candidates"])
        self.assertTrue(low_risk["transfer_candidates"])
        self.assertLess(
            high_risk["transfer_candidates"][0]["transfer_score"],
            low_risk["transfer_candidates"][0]["transfer_score"],
        )
        self.assertLess(high_risk["transfer_candidates"][0]["layer_adjustment"], 0.0)
        self.assertEqual(low_risk["transfer_candidates"][0]["layer_adjustment"], 0.0)
        self.assertTrue(
            any("foundation layer" in reason for reason in high_risk["transfer_candidates"][0]["reasons"])
        )

    def test_operator_review_decision_is_persisted_and_reused(self) -> None:
        self.engine.record_episode(
            Episode(
                repo_id="repo-review",
                repo_path="/tmp/repo-review",
                task_type="resume",
                request="Resume interrupted checkout migration",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="apply_patch", success=True),
                    ActionRecord(tool="run_command", success=True, command="pnpm test checkout-migration"),
                ],
                commands=["pnpm test checkout-migration"],
            )
        )
        self.engine.record_mission_state(
            MissionState(
                mission_id="mission_review",
                repo_id="repo-review",
                summary="Resume interrupted checkout migration",
                status="active",
                blocked_actions=["apply_patch:checkout migration"],
                next_best_step="Patch checkout migration in src/checkout/migrate.ts",
            )
        )
        self.engine.record_operator_review(
            {
                "mission_id": "mission_review",
                "repo_id": "repo-review",
                "selected_option": "pause_and_replan",
                "selected_step": "Pause and request a fresh migration plan",
                "notes": "Need human confirmation before retrying patch path",
            }
        )

        resumed = self.engine.resume_context("mission_review")
        stored_reviews = self.engine.storage.load_json(self.engine.storage.operator_reviews_file, {})

        self.assertIn("mission_review", stored_reviews)
        self.assertEqual(resumed["chosen_strategy"], "pause_and_replan")
        self.assertFalse(resumed["review_required"])
        self.assertEqual(resumed["operator_review"]["decision_status"], "approved")
        self.assertEqual(resumed["suggested_next_step"], "Pause and request a fresh migration plan")

    def test_operator_review_is_invalidated_when_mission_context_changes(self) -> None:
        self.engine.record_episode(
            Episode(
                repo_id="repo-review-drift",
                repo_path="/tmp/repo-review-drift",
                task_type="resume",
                request="Resume interrupted checkout migration",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True),
                    ActionRecord(tool="apply_patch", success=True),
                    ActionRecord(tool="run_command", success=True, command="pnpm test checkout-migration"),
                ],
                commands=["pnpm test checkout-migration"],
            )
        )
        self.engine.record_mission_state(
            MissionState(
                mission_id="mission_review_drift",
                repo_id="repo-review-drift",
                summary="Resume interrupted checkout migration",
                status="active",
                blocked_actions=["apply_patch:checkout migration"],
                next_best_step="Patch checkout migration in src/checkout/migrate.ts",
            )
        )
        self.engine.record_operator_review(
            {
                "mission_id": "mission_review_drift",
                "repo_id": "repo-review-drift",
                "selected_option": "pause_and_replan",
                "selected_step": "Pause and request a fresh migration plan",
            }
        )
        self.engine.record_mission_state(
            MissionState(
                mission_id="mission_review_drift",
                repo_id="repo-review-drift",
                summary="Resume interrupted checkout migration after schema update",
                status="active",
                blocked_actions=["apply_patch:checkout migration"],
                next_best_step="Patch checkout migration in src/checkout/migrate-v2.ts",
            )
        )

        resumed = self.engine.resume_context("mission_review_drift")

        self.assertEqual(resumed["chosen_strategy"], "prefer_safe_variation")
        self.assertTrue(resumed["review_required"])
        self.assertEqual(resumed["operator_review"]["decision_status"], "invalidated")
        self.assertEqual(
            resumed["operator_review"]["summary"],
            "Prior operator approval was invalidated because the mission context changed.",
        )
        self.assertEqual(resumed["suggested_next_step"], "Patch checkout migration in src/checkout/migrate-v2.ts")

    def test_preferences_are_repo_scoped_with_global_rollup(self) -> None:
        self.engine.record_episode(
            Episode(
                repo_id="repo-pref-a",
                repo_path="/tmp/repo-pref-a",
                task_type="bugfix",
                request="Fix auth bug",
                success=True,
                actions=[ActionRecord(tool="search_files", success=True)],
                observed_preferences={"test_scope": "targeted"},
            )
        )
        self.engine.record_episode(
            Episode(
                repo_id="repo-pref-b",
                repo_path="/tmp/repo-pref-b",
                task_type="feature",
                request="Add promo banner",
                success=True,
                actions=[ActionRecord(tool="write_file", success=True)],
                observed_preferences={"test_scope": "targeted"},
            )
        )

        context = self.engine.get_planning_context(
            repo_id="repo-pref-a",
            request="Fix auth bug",
            task_type="bugfix",
            max_items=5,
        )
        prefs = context["preferences"]
        explanations = context["explanations"]["preferences"]

        self.assertTrue(any(item.get("repo_id") == "repo-pref-a" for item in prefs))
        self.assertTrue(any(item.get("repo_id") is None for item in prefs))
        self.assertEqual(prefs[0]["repo_id"], "repo-pref-a")
        self.assertIn("repo-scoped", explanations[0]["reasons"][0])

    def test_maintain_memories_prunes_stale_low_confidence_items(self) -> None:
        stale_items = [
            {
                "id": "procedure_old",
                "name": "stale procedure",
                "repo_id": "repo-stale",
                "task_type": "bugfix",
                "trigger_terms": ["auth"],
                "action_chain": ["search_files"],
                "success_count": 0,
                "failure_count": 0,
                "confidence": 0.19,
                "last_outcome": "unknown",
                "source_episode_ids": [],
                "updated_at": "2023-01-01T00:00:00+00:00",
            }
        ]
        self.engine.storage.save_json(self.engine.storage.procedures_file, stale_items)
        counts = self.engine.maintain_memories()
        procedures = self.engine.storage.load_json(self.engine.storage.procedures_file, [])

        self.assertEqual(counts["procedures"], 0)
        self.assertEqual(procedures, [])

    def test_storage_quarantines_corrupt_json(self) -> None:
        corrupt_file = self.engine.storage.procedures_file
        corrupt_file.parent.mkdir(parents=True, exist_ok=True)
        corrupt_file.write_text("{not valid json", encoding="utf-8")

        result = self.engine.storage.load_json(corrupt_file, [])
        quarantined = list(self.engine.storage.quarantine_dir.glob("procedures.json*.corrupt"))

        self.assertEqual(result, [])
        self.assertTrue(quarantined)
        self.assertFalse(corrupt_file.exists())

    def test_benchmark_harness_compares_memory_off_vs_on(self) -> None:
        harness = BenchmarkHarness(self.engine)
        summary = harness.compare_memory_modes(default_repo_scenarios())

        self.assertEqual(summary["scenario_count"], 4)
        self.assertGreater(summary["memory_on"]["procedure_hit_rate"], summary["memory_off"]["procedure_hit_rate"])
        self.assertGreater(summary["memory_on"]["anti_pattern_hit_rate"], summary["memory_off"]["anti_pattern_hit_rate"])
        self.assertGreater(summary["memory_on"]["preference_hit_rate"], summary["memory_off"]["preference_hit_rate"])
        self.assertGreater(summary["memory_on"]["semantic_hit_rate"], summary["memory_off"]["semantic_hit_rate"])
        self.assertGreater(summary["procedure_lift"], 0.0)
        self.assertGreater(summary["anti_pattern_lift"], 0.0)

    def test_benchmark_history_tracks_regressions(self) -> None:
        harness = BenchmarkHarness(self.engine)
        strong = default_repo_scenarios()
        weak = [
            BenchmarkScenario(
                name="regressed bugfix pattern",
                repo_id="repo-python-auth",
                task_type="bugfix",
                request="Fix failing auth token refresh bug in src/auth",
                expected_procedure_tool="apply_patch",
                expected_semantic_substring="src/auth",
            )
        ]

        harness.compare_memory_modes(strong)
        harness.compare_memory_modes(weak)
        benchmark_state = self.engine.benchmark_summary()

        self.assertEqual(len(benchmark_state["history"]), 2)
        self.assertTrue(benchmark_state["regressions"])
        self.assertTrue(any(item["metric"] == "procedure_lift" for item in benchmark_state["regressions"]))

    def test_benchmark_history_signatures_detect_tampering(self) -> None:
        harness = BenchmarkHarness(self.engine)
        harness.compare_memory_modes(default_repo_scenarios())
        verified = harness.verify_history()
        self.assertTrue(verified["valid"])

        benchmark_file = self.engine.storage.benchmark_runs_file
        payload = self.engine.storage.load_json(benchmark_file, {})
        payload["history"][0]["summary"]["scenario_count"] = 999
        self.engine.storage.save_json(benchmark_file, payload)

        verified_after = harness.verify_history()
        self.assertFalse(verified_after["valid"])
        self.assertTrue(any("signature mismatch" in item for item in verified_after["issues"]))

    def test_benchmark_harness_uses_env_signing_key_without_persisting_file(self) -> None:
        os.environ["KLOMBO_TEST_SIGNING_KEY"] = "external-signing-key"
        try:
            harness = BenchmarkHarness(
                self.engine,
                signing_key_env="KLOMBO_TEST_SIGNING_KEY",
                persist_generated_key=False,
            )
            harness.compare_memory_modes(default_repo_scenarios())

            benchmark_state = self.engine.benchmark_summary()
            latest = benchmark_state["latest"]
            self.assertEqual(latest["signature_meta"]["source"], "env:KLOMBO_TEST_SIGNING_KEY")
            self.assertFalse(self.engine.storage.benchmark_signing_key_file.exists())
            self.assertTrue(harness.verify_history()["valid"])
        finally:
            os.environ.pop("KLOMBO_TEST_SIGNING_KEY", None)

    def test_benchmark_harness_tracks_layer_guidance_precision(self) -> None:
        harness = BenchmarkHarness(self.engine)
        summary = harness.benchmark_layer_guidance(layer_guidance_scenarios())

        self.assertEqual(summary["kind"], "layer_guidance")
        self.assertEqual(summary["scenario_count"], 3)
        self.assertEqual(summary["layer_hint_hit_rate"], 1.0)
        self.assertEqual(summary["layer_penalty_hit_rate"], 1.0)
        self.assertEqual(summary["layer_guidance_hit_rate"], 1.0)
        self.assertTrue(any(item["name"] == "transfer penalty for shared auth foundation" for item in summary["results"]))
        benchmark_state = self.engine.benchmark_summary()
        self.assertEqual(benchmark_state["latest"]["summary"]["kind"], "layer_guidance")

    def test_benchmark_harness_tracks_operator_review_recovery(self) -> None:
        harness = BenchmarkHarness(self.engine)
        summary = harness.benchmark_operator_review_recovery(layer_sensitive_operator_review_scenarios())

        self.assertEqual(summary["kind"], "operator_review_recovery")
        self.assertEqual(summary["scenario_count"], 3)
        self.assertEqual(summary["layer_hint_hit_rate"], 1.0)
        self.assertEqual(summary["review_required_hit_rate"], 1.0)
        self.assertEqual(summary["chosen_strategy_hit_rate"], 1.0)
        self.assertEqual(summary["decision_status_hit_rate"], 1.0)
        self.assertEqual(summary["resume_step_hit_rate"], 1.0)
        self.assertEqual(summary["operator_recovery_hit_rate"], 1.0)
        self.assertTrue(any(item["name"] == "operator can pause shared-layer recovery" for item in summary["results"]))
        benchmark_state = self.engine.benchmark_summary()
        self.assertEqual(benchmark_state["latest"]["summary"]["kind"], "operator_review_recovery")


if __name__ == "__main__":
    unittest.main()
