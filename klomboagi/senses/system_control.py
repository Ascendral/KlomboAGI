"""
System Control — KlomboAGI can act on its machine.

Safe command execution with allowlists and guardrails.
Klombo can manage processes, check network, open apps,
but can NOT delete files, modify system configs, or sudo.
"""

from __future__ import annotations

import os
import re
import subprocess
import signal
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    allowed: bool
    blocked_reason: str = ""


# Commands that are always safe
SAFE_COMMANDS = {
    "ls", "pwd", "whoami", "hostname", "date", "uptime", "df", "du",
    "ps", "top", "htop", "free", "uname", "sw_vers", "system_profiler",
    "networksetup", "ifconfig", "ping", "curl", "dig", "nslookup",
    "open", "pmset", "caffeinate", "say",
    "launchctl", "diskutil",
    "python3", "pip3", "brew",
    "git",
    "cat", "head", "tail", "wc", "sort", "uniq", "grep", "find",
    "echo",
}

# Patterns that are NEVER allowed
BLOCKED_PATTERNS = [
    r"sudo\s",
    r"\brm\s+-rf\b",
    r"\brm\s+-r\b",
    r"\brm\s+/",
    r"\bmkfs\b",
    r"\bdd\s+",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r">\s*/etc/",
    r">\s*/System/",
    r">\s*/usr/",
    r"\bchmod\s+777\b",
    r"\bchown\s+root\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bcurl\b.*\|\s*bash",
    r"\beval\b",
    r"\bexec\b",
]


class SystemControl:
    """Execute safe system commands."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.history: list[CommandResult] = []

    def execute(self, command: str) -> CommandResult:
        """Execute a command if it passes safety checks."""
        # Check blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                result = CommandResult(
                    command=command, stdout="", stderr="",
                    returncode=-1, allowed=False,
                    blocked_reason=f"Blocked: matches dangerous pattern '{pattern}'")
                self.history.append(result)
                return result

        # Check if base command is in allowlist
        base_cmd = command.strip().split()[0] if command.strip() else ""
        # Handle paths like /usr/bin/python3
        base_cmd = os.path.basename(base_cmd)

        if base_cmd not in SAFE_COMMANDS:
            result = CommandResult(
                command=command, stdout="", stderr="",
                returncode=-1, allowed=False,
                blocked_reason=f"Command '{base_cmd}' not in allowlist")
            self.history.append(result)
            return result

        # Execute
        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self.timeout)
            result = CommandResult(
                command=command,
                stdout=proc.stdout[:10000],  # Cap output
                stderr=proc.stderr[:5000],
                returncode=proc.returncode,
                allowed=True)
        except subprocess.TimeoutExpired:
            result = CommandResult(
                command=command, stdout="", stderr=f"Timed out after {self.timeout}s",
                returncode=-1, allowed=True)
        except Exception as e:
            result = CommandResult(
                command=command, stdout="", stderr=str(e),
                returncode=-1, allowed=True)

        self.history.append(result)
        return result

    def kill_process(self, pid: int) -> CommandResult:
        """Kill a process by PID (SIGTERM, not SIGKILL)."""
        try:
            os.kill(pid, signal.SIGTERM)
            return CommandResult(
                command=f"kill {pid}", stdout=f"Sent SIGTERM to {pid}",
                stderr="", returncode=0, allowed=True)
        except ProcessLookupError:
            return CommandResult(
                command=f"kill {pid}", stdout="",
                stderr=f"No process with PID {pid}", returncode=1, allowed=True)
        except PermissionError:
            return CommandResult(
                command=f"kill {pid}", stdout="",
                stderr=f"Permission denied for PID {pid}", returncode=1, allowed=True)

    def list_processes(self, sort_by: str = "cpu") -> list[dict]:
        """List top processes by CPU or memory."""
        import psutil
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": round(info["cpu_percent"] or 0, 1),
                    "mem": round(info["memory_percent"] or 0, 1),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        key = "cpu" if sort_by == "cpu" else "mem"
        procs.sort(key=lambda p: p[key], reverse=True)
        return procs[:20]

    def network_status(self) -> dict:
        """Quick network status check."""
        result = self.execute("ping -c 1 -t 3 8.8.8.8")
        connected = result.returncode == 0

        interfaces = {}
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            if name.startswith("lo"):
                continue
            for addr in addrs:
                if addr.family.name == "AF_INET":
                    interfaces[name] = addr.address

        return {
            "connected": connected,
            "interfaces": interfaces,
        }

    def open_app(self, app_name: str) -> CommandResult:
        """Open a macOS application."""
        # Sanitize app name
        safe_name = re.sub(r'[^a-zA-Z0-9 .\-]', '', app_name)
        return self.execute(f'open -a "{safe_name}"')
