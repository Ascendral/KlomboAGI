"""
General Action Layer — filesystem, shell, code execution.

These are the system's HANDS. Without them it can only compute in memory.
With them it can actually DO things in the real world.

All actions are guarded by the safety policy.
"""

from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
from typing import Any


class GeneralTools:
    """Safe, bounded tools for real-world digital work."""

    def __init__(self, workspace: str = ".", allowed_commands: set | None = None):
        self.workspace = Path(workspace).resolve()
        self.allowed_commands = allowed_commands or {
            "ls", "cat", "head", "tail", "wc", "grep", "find",
            "python3", "node", "npm", "pip", "git",
            "curl", "wget",
        }

    # === Filesystem ===

    def read_file(self, path: str) -> str:
        """Read a file's contents."""
        full = (self.workspace / path).resolve()
        if not str(full).startswith(str(self.workspace)):
            return f"Error: path escapes workspace"
        if not full.exists():
            return f"Error: file not found: {path}"
        return full.read_text(errors="replace")[:50000]

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        full = (self.workspace / path).resolve()
        if not str(full).startswith(str(self.workspace)):
            return f"Error: path escapes workspace"
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def list_dir(self, path: str = ".") -> list[str]:
        """List directory contents."""
        full = (self.workspace / path).resolve()
        if not str(full).startswith(str(self.workspace)):
            return ["Error: path escapes workspace"]
        if not full.exists():
            return [f"Error: directory not found: {path}"]
        return sorted(str(p.relative_to(full)) for p in full.iterdir())

    def search_files(self, pattern: str, path: str = ".") -> list[str]:
        """Search for files matching a glob pattern."""
        full = (self.workspace / path).resolve()
        if not str(full).startswith(str(self.workspace)):
            return ["Error: path escapes workspace"]
        return sorted(str(p.relative_to(self.workspace)) for p in full.rglob(pattern))[:100]

    def grep(self, pattern: str, path: str = ".") -> list[str]:
        """Search file contents for a pattern."""
        full = (self.workspace / path).resolve()
        results = []
        for fpath in full.rglob("*"):
            if fpath.is_file() and fpath.suffix in (".py", ".js", ".ts", ".json", ".md", ".txt", ".yaml", ".yml"):
                try:
                    content = fpath.read_text(errors="replace")
                    for i, line in enumerate(content.split("\n"), 1):
                        if pattern.lower() in line.lower():
                            results.append(f"{fpath.relative_to(self.workspace)}:{i}: {line.strip()}")
                except:
                    pass
        return results[:100]

    # === Shell ===

    def execute(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command safely."""
        # Check command is allowed
        cmd_name = command.split()[0] if command else ""
        if cmd_name not in self.allowed_commands:
            return f"Error: command '{cmd_name}' not in allowed list: {self.allowed_commands}"

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(self.workspace),
            )
            output = result.stdout + result.stderr
            return output[:10000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    # === Code Execution ===

    def run_python(self, code: str, timeout: int = 10) -> str:
        """Execute Python code in a subprocess."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(self.workspace),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nError: {result.stderr}"
            return output[:10000]
        except subprocess.TimeoutExpired:
            return "Error: code execution timed out"
        except Exception as e:
            return f"Error: {e}"

    def run_tests(self, test_path: str = "tests/", timeout: int = 60) -> str:
        """Run pytest on a directory."""
        return self.execute(f"python3 -m pytest {test_path} -q --tb=short", timeout=timeout)

    # === Structured Data ===

    def parse_json(self, text: str) -> Any:
        """Parse JSON string."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON: {e}"

    def to_json(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, indent=2, default=str)

    # === Tool Registry ===

    def get_tools(self) -> dict:
        """Return all tools as a name → function mapping."""
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_dir": self.list_dir,
            "search_files": self.search_files,
            "grep": self.grep,
            "execute": self.execute,
            "run_python": self.run_python,
            "run_tests": self.run_tests,
            "parse_json": self.parse_json,
            "to_json": self.to_json,
        }
