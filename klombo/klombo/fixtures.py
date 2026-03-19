from __future__ import annotations

from klombo.benchmark import BenchmarkScenario
from klombo.models import ActionRecord, Episode, MissionState


def default_repo_scenarios() -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name="python bugfix in auth service",
            repo_id="repo-python-auth",
            task_type="bugfix",
            request="Fix failing auth token refresh bug in src/auth",
            setup_episodes=[
                Episode(
                    repo_id="repo-python-auth",
                    repo_path="/repos/python-auth",
                    task_type="bugfix",
                    request="Fix auth token refresh bug in src/auth/service.py",
                    success=True,
                    actions=[
                        ActionRecord(tool="search_files", success=True, summary="Located auth refresh code"),
                        ActionRecord(tool="read_file", success=True, summary="Read src/auth/service.py"),
                        ActionRecord(tool="apply_patch", success=True, summary="Patched refresh logic"),
                        ActionRecord(tool="run_command", success=True, command="pytest tests/test_auth_refresh.py"),
                    ],
                    files_touched=["src/auth/service.py", "tests/test_auth_refresh.py"],
                    commands=["pytest tests/test_auth_refresh.py"],
                    observed_preferences={"test_scope": "targeted"},
                ).to_dict()
            ],
            expected_procedure_tool="apply_patch",
            expected_semantic_substring="src/auth",
        ),
        BenchmarkScenario(
            name="typescript feature in checkout ui",
            repo_id="repo-ts-checkout",
            task_type="feature",
            request="Add checkout promo banner to app/components/checkout",
            setup_episodes=[
                Episode(
                    repo_id="repo-ts-checkout",
                    repo_path="/repos/ts-checkout",
                    task_type="feature",
                    request="Add promo banner to app/components/checkout",
                    success=True,
                    actions=[
                        ActionRecord(tool="search_files", success=True, summary="Found checkout components"),
                        ActionRecord(tool="write_file", success=True, summary="Added banner component"),
                        ActionRecord(tool="run_command", success=True, command="pnpm test checkout-banner"),
                    ],
                    files_touched=["app/components/checkout/Banner.tsx", "app/components/checkout/Banner.test.tsx"],
                    commands=["pnpm test checkout-banner"],
                    observed_preferences={"component_style": "tsx"},
                ).to_dict()
            ],
            expected_procedure_tool="write_file",
            expected_preference_key="component_style",
        ),
        BenchmarkScenario(
            name="dangerous refactor anti-pattern",
            repo_id="repo-config-refactor",
            task_type="refactor",
            request="Refactor deployment config without breaking ci",
            setup_episodes=[
                Episode(
                    repo_id="repo-config-refactor",
                    repo_path="/repos/config-refactor",
                    task_type="refactor",
                    request="Refactor deployment config without breaking ci",
                    success=False,
                    actions=[
                        ActionRecord(tool="run_command", success=False, summary="Full npm test timed out", command="npm test"),
                        ActionRecord(tool="apply_patch", success=False, summary="Patched wrong deployment file"),
                    ],
                    files_touched=["config/deploy.yml"],
                    commands=["npm test"],
                    stop_reason="timeout",
                ).to_dict(),
                Episode(
                    repo_id="repo-config-refactor",
                    repo_path="/repos/config-refactor",
                    task_type="refactor",
                    request="Refactor deployment config without breaking ci",
                    success=False,
                    actions=[
                        ActionRecord(tool="run_command", success=False, summary="Full npm test timed out", command="npm test"),
                        ActionRecord(tool="apply_patch", success=False, summary="Patched wrong deployment file"),
                    ],
                    files_touched=["config/deploy.yml"],
                    commands=["npm test"],
                    stop_reason="timeout",
                ).to_dict(),
            ],
            expected_anti_pattern_tool="run_command",
        ),
        BenchmarkScenario(
            name="resume interrupted migration mission",
            repo_id="repo-resume-migration",
            task_type="resume",
            request="Resume interrupted auth migration in src/server/auth",
            setup_missions=[
                MissionState(
                    mission_id="mission_resume_auth",
                    repo_id="repo-resume-migration",
                    summary="Resume interrupted auth migration",
                    status="active",
                    last_plan="Search migrations, patch auth models, run targeted migration test",
                    attempted_actions=["search_files", "read_file"],
                    blocked_actions=["run_command:pnpm test"],
                    next_best_step="Patch auth migration in src/server/auth/migrate.ts",
                ).to_dict()
            ],
            expected_resume_step="Patch auth migration in src/server/auth/migrate.ts",
            expected_semantic_substring="src/server/auth",
            setup_episodes=[
                Episode(
                    repo_id="repo-resume-migration",
                    repo_path="/repos/resume-migration",
                    task_type="resume",
                    request="Resume interrupted auth migration in src/server/auth",
                    success=True,
                    actions=[
                        ActionRecord(tool="search_files", success=True, summary="Found migration files"),
                        ActionRecord(tool="apply_patch", success=True, summary="Patched migration"),
                        ActionRecord(tool="run_command", success=True, command="pnpm test auth-migration"),
                    ],
                    files_touched=["src/server/auth/migrate.ts", "tests/auth-migration.test.ts"],
                    commands=["pnpm test auth-migration"],
                    observed_preferences={"resume_style": "preserve-plan"},
                ).to_dict()
            ],
            expected_procedure_tool="apply_patch",
        ),
    ]


def layer_guidance_scenarios() -> list[BenchmarkScenario]:
    auth_repo_files = {
        "pyproject.toml": "[project]\nname='layered'\n[tool.pytest.ini_options]\n",
        "src/auth/service.py": "def refresh_token():\n    return True\n",
        "src/api/routes.py": "from src.auth.service import refresh_token\n\ndef route():\n    return refresh_token()\n",
    }
    return [
        BenchmarkScenario(
            name="layer hint for shared auth foundation",
            repo_id="repo-layer-hint",
            task_type="bugfix",
            request="Fix auth token refresh in src/auth",
            setup_repo_files=auth_repo_files,
            expect_layer_hint=True,
            expected_layer_hint_substring="src/auth",
        ),
        BenchmarkScenario(
            name="transfer penalty for shared auth foundation",
            repo_id="repo-layer-penalty",
            task_type="bugfix",
            request="Fix auth token refresh in src/auth",
            setup_repo_files=auth_repo_files,
            peer_repo_id="repo-layer-penalty-peer",
            peer_repo_files=auth_repo_files,
            peer_setup_episodes=[
                Episode(
                    repo_id="repo-layer-penalty-peer",
                    repo_path="/repos/repo-layer-penalty-peer",
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
                ).to_dict(),
                Episode(
                    repo_id="repo-layer-penalty-peer",
                    repo_path="/repos/repo-layer-penalty-peer",
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
                ).to_dict(),
                Episode(
                    repo_id="repo-layer-penalty-peer",
                    repo_path="/repos/repo-layer-penalty-peer",
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
                ).to_dict(),
            ],
            expect_layer_hint=True,
            expected_layer_hint_substring="src/auth",
            expect_layer_penalty=True,
        ),
        BenchmarkScenario(
            name="no penalty for unrelated leaf request",
            repo_id="repo-layer-control",
            task_type="bugfix",
            request="Fix output copy in cli",
            setup_repo_files={
                **auth_repo_files,
                "cli/output.py": "def render_output():\n    return 'ok'\n",
            },
            peer_repo_id="repo-layer-control-peer",
            peer_repo_files={
                **auth_repo_files,
                "cli/output.py": "def render_output():\n    return 'ok'\n",
            },
            peer_setup_episodes=[
                Episode(
                    repo_id="repo-layer-control-peer",
                    repo_path="/repos/repo-layer-control-peer",
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
                ).to_dict(),
                Episode(
                    repo_id="repo-layer-control-peer",
                    repo_path="/repos/repo-layer-control-peer",
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
                ).to_dict(),
                Episode(
                    repo_id="repo-layer-control-peer",
                    repo_path="/repos/repo-layer-control-peer",
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
                ).to_dict(),
            ],
            expect_layer_hint=True,
            expected_layer_hint_substring="cli",
            expect_layer_penalty=False,
        ),
    ]


def layer_sensitive_operator_review_scenarios() -> list[BenchmarkScenario]:
    repo_files = {
        "pyproject.toml": "[project]\nname='layer-review'\n[tool.pytest.ini_options]\n",
        "src/auth/migrate.py": "def migrate_auth():\n    return True\n",
        "src/api/routes.py": "from src.auth.migrate import migrate_auth\n\ndef route():\n    return migrate_auth()\n",
        "tests/test_auth_migration.py": "def test_auth_migration():\n    assert True\n",
    }

    def setup_episodes(repo_id: str) -> list[dict[str, object]]:
        return [
            Episode(
                repo_id=repo_id,
                repo_path=f"/repos/{repo_id}",
                task_type="resume",
                request="Resume interrupted auth migration in src/auth",
                success=False,
                actions=[
                    ActionRecord(tool="run_command", success=False, summary="Full pytest run timed out", command="pytest"),
                    ActionRecord(tool="apply_patch", success=False, summary="Retried the same risky auth patch path"),
                ],
                files_touched=["src/auth/migrate.py"],
                commands=["pytest"],
                stop_reason="timeout",
            ).to_dict(),
            Episode(
                repo_id=repo_id,
                repo_path=f"/repos/{repo_id}",
                task_type="resume",
                request="Resume interrupted auth migration in src/auth",
                success=True,
                actions=[
                    ActionRecord(tool="search_files", success=True, summary="Found auth migration files"),
                    ActionRecord(tool="apply_patch", success=True, summary="Patched auth migration safely"),
                    ActionRecord(
                        tool="run_command",
                        success=True,
                        command="pytest tests/test_auth_migration.py",
                    ),
                ],
                files_touched=["src/auth/migrate.py", "tests/test_auth_migration.py"],
                commands=["pytest tests/test_auth_migration.py"],
                observed_preferences={"test_scope": "targeted"},
            ).to_dict(),
        ]

    def setup_mission(mission_id: str, repo_id: str) -> dict[str, object]:
        return MissionState(
            mission_id=mission_id,
            repo_id=repo_id,
            summary="Resume interrupted auth migration in src/auth",
            status="active",
            last_plan="Search auth migration, patch auth module, run targeted auth migration test",
            attempted_actions=["search_files", "apply_patch"],
            blocked_actions=["apply_patch:auth migration"],
            next_best_step="Patch auth migration in src/auth/migrate.py",
        ).to_dict()

    return [
        BenchmarkScenario(
            name="layer-sensitive recovery requires operator review",
            repo_id="repo-layer-review-required",
            task_type="resume",
            request="Resume interrupted auth migration in src/auth",
            setup_repo_files=repo_files,
            setup_episodes=setup_episodes("repo-layer-review-required"),
            setup_missions=[setup_mission("mission_layer_review_required", "repo-layer-review-required")],
            expect_layer_hint=True,
            expected_layer_hint_substring="src/auth",
            expected_review_required=True,
            expected_chosen_strategy="prefer_safe_variation",
            expected_operator_decision_status="none",
            expected_resume_step="Patch auth migration in src/auth/migrate.py",
        ),
        BenchmarkScenario(
            name="operator can pause shared-layer recovery",
            repo_id="repo-layer-review-pause",
            task_type="resume",
            request="Resume interrupted auth migration in src/auth",
            setup_repo_files=repo_files,
            setup_episodes=setup_episodes("repo-layer-review-pause"),
            setup_missions=[setup_mission("mission_layer_review_pause", "repo-layer-review-pause")],
            setup_operator_reviews=[
                {
                    "mission_id": "mission_layer_review_pause",
                    "repo_id": "repo-layer-review-pause",
                    "selected_option": "pause_and_replan",
                    "selected_step": "Pause and request a fresh auth migration plan",
                    "notes": "Shared auth layer needs a fresh plan before retrying the patch path.",
                }
            ],
            expect_layer_hint=True,
            expected_layer_hint_substring="src/auth",
            expected_review_required=False,
            expected_chosen_strategy="pause_and_replan",
            expected_operator_decision_status="approved",
            expected_resume_step="Pause and request a fresh auth migration plan",
        ),
        BenchmarkScenario(
            name="operator can choose safe shared-layer recovery step",
            repo_id="repo-layer-review-recovery",
            task_type="resume",
            request="Resume interrupted auth migration in src/auth",
            setup_repo_files=repo_files,
            setup_episodes=setup_episodes("repo-layer-review-recovery"),
            setup_missions=[setup_mission("mission_layer_review_recovery", "repo-layer-review-recovery")],
            setup_operator_reviews=[
                {
                    "mission_id": "mission_layer_review_recovery",
                    "repo_id": "repo-layer-review-recovery",
                    "selected_option": "apply_recovery_plan",
                    "notes": "Reuse the safest recovery step instead of repeating the blocked patch action.",
                }
            ],
            expect_layer_hint=True,
            expected_layer_hint_substring="src/auth",
            expected_review_required=False,
            expected_chosen_strategy="apply_recovery_plan",
            expected_operator_decision_status="approved",
            expected_resume_step="Do not repeat blocked action unchanged: apply_patch:auth migration",
        ),
    ]
