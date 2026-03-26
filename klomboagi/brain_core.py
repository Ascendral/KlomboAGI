from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

_native_brain_core = None
try:
    candidate = importlib.import_module("brain_core")
    if hasattr(candidate, "retrieve_memory") and hasattr(candidate, "score_plan"):
        _native_brain_core = candidate
except Exception:
    _native_brain_core = None


def native_available() -> bool:
    return _native_brain_core is not None


def retrieve_memory(query: str, memories: list[str], limit: int = 3) -> list[tuple[str, float]]:
    if _native_brain_core is not None:
        return list(_native_brain_core.retrieve_memory(query, memories, limit))

    query_tokens = _tokenize(query)
    hits: list[tuple[str, float]] = []
    for memory in memories:
        memory_tokens = _tokenize(memory)
        overlap = len(query_tokens.intersection(memory_tokens))
        if not overlap:
            continue
        denominator = max(len(query_tokens), len(memory_tokens)) or 1
        hits.append((memory, overlap / denominator))
    hits.sort(key=lambda item: item[1], reverse=True)
    return hits[:limit]


def score_plan_candidates(
    goal: str,
    candidates: list[dict[str, object]],
    anti_patterns: list[str] | None = None,
) -> list[dict[str, object]]:
    anti_patterns = anti_patterns or []
    if _native_brain_core is not None:
        payload = [
            (
                str(candidate["id"]),
                str(candidate["description"]),
                str(candidate.get("action_kind") or ""),
                float(candidate.get("estimated_cost", 0.0)),
            )
            for candidate in candidates
        ]
        results = _native_brain_core.score_plan(goal, payload, anti_patterns)
        normalized = []
        for candidate_id, score, reasons in results:
            normalized.append(
                {
                    "candidate_id": candidate_id,
                    "score": float(score),
                    "reasons": list(reasons),
                }
            )
        return sorted(normalized, key=lambda item: item["score"], reverse=True)

    goal_tokens = _tokenize(goal)
    scored: list[dict[str, object]] = []
    for candidate in candidates:
        description = str(candidate["description"])
        description_tokens = _tokenize(description)
        overlap = len(goal_tokens.intersection(description_tokens))
        penalty = 0.0
        for pattern in anti_patterns:
            pattern_tokens = _tokenize(pattern)
            token_overlap = len(description_tokens.intersection(pattern_tokens))
            if token_overlap:
                penalty += token_overlap * 0.25
            if _normalized_phrase(description) and _normalized_phrase(description) in _normalized_phrase(pattern):
                penalty += 2.0
        estimated_cost = float(candidate.get("estimated_cost", 0.0))
        score = overlap - estimated_cost - penalty
        scored.append(
            {
                "candidate_id": str(candidate["id"]),
                "score": score,
                "reasons": [
                    f"goal_overlap={overlap:.2f}",
                    f"estimated_cost={estimated_cost:.2f}",
                    f"anti_pattern_penalty={penalty:.2f}",
                ],
            }
        )
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def transfer_score(source_tags: list[str], target_tags: list[str]) -> tuple[float, list[str], list[str]]:
    if _native_brain_core is not None:
        overlap, shared, missing = _native_brain_core.transfer_score(source_tags, target_tags)
        return float(overlap), list(shared), list(missing)

    source = set(_tokenize(" ".join(source_tags)))
    target = set(_tokenize(" ".join(target_tags)))
    shared = sorted(source.intersection(target))
    missing = sorted(target.difference(source))
    denominator = max(len(source), len(target)) or 1
    return len(shared) / denominator, shared, missing


def reliability_score(successes: int, failures: int) -> float:
    if _native_brain_core is not None:
        return float(_native_brain_core.reliability_score(successes, failures))
    total = successes + failures
    return successes / total if total else 0.0


def load_failure_patterns() -> list[str]:
    root = Path(os.getenv("KLOMBOAGI_FAILURE_CASE_DIR", "datasets/failure_cases"))
    if not root.exists():
        return []

    patterns: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        pieces = [
            str(data.get("description", "")),
            str(data.get("trigger", "")),
            str(data.get("failed_action", "")),
            str(data.get("error_type", "")),
            str(data.get("avoidance", "")),
        ]
        joined = " ".join(piece for piece in pieces if piece).strip()
        if joined:
            patterns.append(joined)
    return patterns


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token
    }


def _normalized_phrase(text: str) -> str:
    return " ".join(
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token
    )
