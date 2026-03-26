from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_CORE_MANIFEST = REPO_ROOT / "brain_core" / "Cargo.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_brain_core",
        description="Build the Rust brain_core extension with maturin when available.",
    )
    parser.add_argument("--release", action="store_true", help="Build the extension in release mode.")
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="Fail if the native module cannot be built or imported.",
    )
    return parser


def native_module_available() -> bool:
    try:
        module = importlib.import_module("brain_core")
    except Exception:
        return False
    return hasattr(module, "retrieve_memory") and hasattr(module, "score_plan")


def maturin_available() -> bool:
    if shutil.which("maturin") is not None:
        return True
    try:
        importlib.import_module("maturin")
        return True
    except Exception:
        return False


def maturin_command() -> list[str]:
    if shutil.which("maturin") is not None:
        return ["maturin"]
    return [sys.executable, "-m", "maturin"]


def build_native(*, release: bool = False) -> dict[str, Any]:
    if not BRAIN_CORE_MANIFEST.exists():
        return {"ok": False, "reason": f"Missing manifest: {BRAIN_CORE_MANIFEST}"}

    if not maturin_available():
        return {
            "ok": False,
            "reason": "maturin is not installed; install dev dependencies or `pip install maturin`.",
        }

    command = [*maturin_command(), "develop", "--manifest-path", str(BRAIN_CORE_MANIFEST)]
    if release:
        command.append("--release")

    env = dict(os.environ)
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    ok = completed.returncode == 0 and native_module_available()
    return {
        "ok": ok,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "native_available": native_module_available(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_native(release=args.release)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_native and not result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
