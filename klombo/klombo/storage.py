from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from klombo.models import utc_now


class KlomboStorage:
    """Simple file-backed storage for the standalone Klombo core."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.root / "state"
        self.logs_dir = self.root / "logs"
        self.quarantine_dir = self.root / "quarantine"
        self.tmp_dir = self.root / "tmp"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def episodes_file(self) -> Path:
        return self.logs_dir / "episodes.jsonl"

    @property
    def repo_profiles_file(self) -> Path:
        return self.state_dir / "repo_profiles.json"

    @property
    def procedures_file(self) -> Path:
        return self.state_dir / "procedures.json"

    @property
    def anti_patterns_file(self) -> Path:
        return self.state_dir / "anti_patterns.json"

    @property
    def preferences_file(self) -> Path:
        return self.state_dir / "preferences.json"

    @property
    def missions_file(self) -> Path:
        return self.state_dir / "missions.json"

    @property
    def operator_reviews_file(self) -> Path:
        return self.state_dir / "operator_reviews.json"

    @property
    def benchmark_runs_file(self) -> Path:
        return self.state_dir / "benchmark_runs.json"

    @property
    def benchmark_signing_key_file(self) -> Path:
        return self.state_dir / "benchmark_signing_key.txt"

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._quarantine(path)
            return default

    def save_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.tmp_dir / f"{path.name}.{utc_now().replace(':', '-')}.tmp"
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temp_path.open("r+", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        had_error = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                had_error = True
        if had_error and not rows:
            self._quarantine(path)
        return rows

    def _quarantine(self, path: Path) -> None:
        if not path.exists():
            return
        target = self.quarantine_dir / f"{path.name}.{utc_now().replace(':', '-')}.corrupt"
        try:
            os.replace(path, target)
        except OSError:
            pass
