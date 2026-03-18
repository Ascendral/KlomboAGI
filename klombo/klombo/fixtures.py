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
