"""
Abstraction Engine — the missing piece.

This module extracts structural patterns from episodes, not content.
When the system sees 3 dogs, it doesn't learn "dog" — it learns "three",
"group", "sameness", "category". The structure underneath, not the surface.

The algorithm:
1. Take two or more episodes (experiences)
2. Align them structurally (what roles do the parts play?)
3. Find what's INVARIANT across them (the structure)
4. Find what VARIES (the surface details)
5. The invariant IS the abstraction
6. Store it as a reusable schema that can be applied to new situations

This is analogous to how the human brain forms concepts:
- See red ball, blue ball, green ball → abstract "color varies, ball-ness is invariant"
- See fix bug in Python, fix bug in Go → abstract "the debugging process, not the language"
- See 3 dogs, 3 cats, 3 trees → abstract "three-ness, not dog-ness"

No LLM calls. Pure algorithmic structure matching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class StructuralElement:
    """A single element in a structural representation."""
    role: str           # The functional role: "agent", "action", "target", "result", "count", "container"
    value: Any          # The concrete value in this instance
    type_tag: str       # Abstract type: "entity", "operation", "quantity", "state", "relation"
    children: list[StructuralElement] = field(default_factory=list)

    def signature(self) -> str:
        """Role + type without concrete value = structural signature."""
        child_sigs = tuple(c.signature() for c in self.children)
        return f"{self.role}:{self.type_tag}({child_sigs})"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "value": self.value,
            "type_tag": self.type_tag,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Abstraction:
    """A learned structural pattern — the invariant across episodes."""
    id: str
    name: str
    schema: list[dict]          # The structural skeleton (roles + types, no values)
    invariants: list[str]       # What stays the same across instances
    variables: list[str]        # What changes across instances (slots to fill)
    source_episodes: list[str]  # Episode IDs this was extracted from
    instance_count: int = 0     # How many times this pattern has been seen
    confidence: float = 0.0     # How reliable this abstraction is
    transfer_count: int = 0     # How many times applied to a NEW domain
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schema": self.schema,
            "invariants": self.invariants,
            "variables": self.variables,
            "source_episodes": self.source_episodes,
            "instance_count": self.instance_count,
            "confidence": self.confidence,
            "transfer_count": self.transfer_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AbstractionEngine:
    """
    Extracts structural patterns from episodes.

    The core algorithm:
    1. Decompose each episode into structural elements
    2. Align elements across episodes by role
    3. Find invariant structure (same roles, same types, same relations)
    4. Find variable slots (same role, different values)
    5. Create an abstraction = invariant structure + variable slots
    6. Test abstraction against new episodes (does it fit?)
    7. Strengthen or weaken based on fit
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def load_all(self) -> list[dict]:
        """Load all known abstractions."""
        return self.storage.load_json("abstractions", default=[])

    def save_all(self, abstractions: list[dict]) -> None:
        """Persist abstractions."""
        self.storage.save_json("abstractions", abstractions)

    # ── Step 1: Decompose an episode into structural elements ──

    def decompose(self, episode: dict) -> list[StructuralElement]:
        """
        Break an episode into its structural elements.

        An episode has: mission, tasks, actions, outcomes.
        We extract the ROLES each piece plays, not the content.
        """
        elements = []

        # The goal/mission = what was being attempted
        if "description" in episode or "mission" in episode:
            goal_text = episode.get("description", episode.get("mission", {}).get("description", ""))
            elements.append(StructuralElement(
                role="goal",
                value=goal_text,
                type_tag="intention",
            ))

        # The actions taken = the process
        actions = episode.get("actions", episode.get("steps", []))
        for i, action in enumerate(actions):
            if isinstance(action, dict):
                action_type = action.get("type", action.get("action", "unknown"))
                target = action.get("target", action.get("path", action.get("args", {})))
                result = action.get("result", action.get("outcome", action.get("status", "unknown")))
            else:
                action_type = str(action)
                target = None
                result = None

            elements.append(StructuralElement(
                role="step",
                value=action_type,
                type_tag="operation",
                children=[
                    StructuralElement(role="position", value=i, type_tag="quantity"),
                    StructuralElement(role="target", value=target, type_tag="entity"),
                    StructuralElement(role="result", value=result, type_tag="state"),
                ],
            ))

        # The outcome = what happened
        outcome = episode.get("outcome", episode.get("status", "unknown"))
        success = episode.get("success", outcome in ("completed", "success", "passed"))
        elements.append(StructuralElement(
            role="outcome",
            value=outcome,
            type_tag="state",
            children=[
                StructuralElement(role="success", value=success, type_tag="boolean"),
                StructuralElement(role="step_count", value=len(actions), type_tag="quantity"),
            ],
        ))

        # Quantities — this is where "three-ness" lives
        # Count distinct types of things
        type_counts: dict[str, int] = {}
        for el in elements:
            type_counts[el.type_tag] = type_counts.get(el.type_tag, 0) + 1
        for type_tag, count in type_counts.items():
            if count > 1:
                elements.append(StructuralElement(
                    role="quantity",
                    value=count,
                    type_tag="quantity",
                    children=[
                        StructuralElement(role="of_type", value=type_tag, type_tag="category"),
                    ],
                ))

        return elements

    # ── Step 2: Align two episodes structurally ──

    def align(self, elements_a: list[StructuralElement], elements_b: list[StructuralElement]) -> list[tuple[StructuralElement | None, StructuralElement | None]]:
        """
        Align elements from two episodes by their structural role.
        Returns pairs: (element_from_a, element_from_b) or (None, element) for unmatched.
        """
        aligned = []
        used_b = set()

        for ea in elements_a:
            best_match = None
            best_score = 0.0
            for i, eb in enumerate(elements_b):
                if i in used_b:
                    continue
                score = self._alignment_score(ea, eb)
                if score > best_score:
                    best_score = score
                    best_match = (i, eb)

            if best_match and best_score >= 0.5:
                used_b.add(best_match[0])
                aligned.append((ea, best_match[1]))
            else:
                aligned.append((ea, None))

        # Add unmatched from B
        for i, eb in enumerate(elements_b):
            if i not in used_b:
                aligned.append((None, eb))

        return aligned

    def _alignment_score(self, a: StructuralElement, b: StructuralElement) -> float:
        """Score how well two elements align structurally (ignoring concrete values)."""
        score = 0.0

        # Same role = strong signal
        if a.role == b.role:
            score += 0.5

        # Same type = structural match
        if a.type_tag == b.type_tag:
            score += 0.3

        # Same number of children = similar complexity
        if len(a.children) == len(b.children):
            score += 0.1

        # Same value = exact match (less interesting for abstraction)
        if a.value == b.value:
            score += 0.1

        return score

    # ── Step 3: Extract invariants and variables ──

    def extract_pattern(self, aligned: list[tuple[StructuralElement | None, StructuralElement | None]]) -> tuple[list[str], list[str]]:
        """
        From aligned elements, determine what's invariant and what varies.

        Invariant: same role, same type, same value across episodes
        Variable: same role, same type, DIFFERENT value across episodes
        """
        invariants = []
        variables = []

        for ea, eb in aligned:
            if ea is None or eb is None:
                # Only in one episode — not part of the pattern
                continue

            sig_a = ea.signature()
            sig_b = eb.signature()

            if sig_a == sig_b and ea.value == eb.value:
                # Exact structural AND value match = invariant
                invariants.append(f"{ea.role}:{ea.type_tag}={ea.value}")
            elif sig_a == sig_b:
                # Structural match but different values = variable slot
                variables.append(f"{ea.role}:{ea.type_tag}=[{ea.value} | {eb.value}]")
            else:
                # Different structure = not part of this pattern
                pass

        return invariants, variables

    # ── Step 4: Create or update an abstraction ──

    def abstract(self, episodes: list[dict]) -> Abstraction | None:
        """
        Given 2+ episodes, extract the structural abstraction.

        This is the core operation:
        - Decompose each episode
        - Align pairwise
        - Find what's invariant across ALL episodes
        - What varies = the slots in the schema
        """
        if len(episodes) < 2:
            return None

        # Decompose all episodes
        decomposed = [self.decompose(ep) for ep in episodes]

        # Align first pair to get initial pattern
        aligned = self.align(decomposed[0], decomposed[1])
        invariants, variables = self.extract_pattern(aligned)

        # Refine against remaining episodes
        for i in range(2, len(decomposed)):
            aligned_next = self.align(decomposed[0], decomposed[i])
            inv_next, var_next = self.extract_pattern(aligned_next)

            # Invariants must hold across ALL episodes
            invariants = [inv for inv in invariants if inv in inv_next]

            # Variables accumulate
            existing_var_roles = {v.split("=")[0].split("[")[0] for v in variables}
            for v in var_next:
                role = v.split("=")[0].split("[")[0]
                if role not in existing_var_roles:
                    variables.append(v)

        if not invariants:
            return None  # No common structure found

        # Build the schema (structural skeleton)
        schema = []
        for inv in invariants:
            parts = inv.split("=", 1)
            role_type = parts[0]
            role, type_tag = role_type.split(":", 1)
            schema.append({"role": role, "type": type_tag, "fixed_value": parts[1] if len(parts) > 1 else None})
        for var in variables:
            role_type = var.split("=")[0].split("[")[0]
            role, type_tag = role_type.split(":", 1)
            schema.append({"role": role, "type": type_tag, "fixed_value": None, "is_variable": True})

        # Generate ID from structural signature
        sig = json.dumps(sorted(invariants), sort_keys=True)
        abs_id = "abs_" + hashlib.sha256(sig.encode()).hexdigest()[:12]

        # Name it by its invariant structure
        name = self._generate_name(invariants, variables)

        episode_ids = [ep.get("id", str(i)) for i, ep in enumerate(episodes)]
        now = utc_now()

        abstraction = Abstraction(
            id=abs_id,
            name=name,
            schema=schema,
            invariants=invariants,
            variables=variables,
            source_episodes=episode_ids,
            instance_count=len(episodes),
            confidence=min(0.9, 0.3 + 0.15 * len(episodes)),  # More instances = higher confidence
            created_at=now,
            updated_at=now,
        )

        # Store it
        all_abstractions = self.load_all()
        # Check if we already have this pattern
        existing = next((a for a in all_abstractions if a["id"] == abs_id), None)
        if existing:
            existing["instance_count"] += len(episodes)
            existing["confidence"] = min(0.95, existing["confidence"] + 0.05)
            existing["source_episodes"].extend(episode_ids)
            existing["updated_at"] = now
        else:
            all_abstractions.append(abstraction.to_dict())

        self.save_all(all_abstractions)

        self.storage.event_log.append(
            "abstraction.formed",
            {"id": abs_id, "name": name, "invariant_count": len(invariants), "variable_count": len(variables)},
        )

        return abstraction

    # ── Step 5: Match a new episode against known abstractions ──

    def match(self, episode: dict) -> list[tuple[dict, float]]:
        """
        Given a new episode, find which abstractions it matches.
        Returns (abstraction, fit_score) pairs sorted by score.
        """
        elements = self.decompose(episode)
        all_abstractions = self.load_all()
        matches = []

        for abstraction in all_abstractions:
            score = self._fit_score(elements, abstraction)
            if score > 0.3:
                matches.append((abstraction, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _fit_score(self, elements: list[StructuralElement], abstraction: dict) -> float:
        """How well do these elements fit this abstraction's schema?"""
        schema = abstraction.get("schema", [])
        if not schema:
            return 0.0

        matched = 0
        for slot in schema:
            role = slot["role"]
            type_tag = slot["type"]
            # Find an element that fills this slot
            for el in elements:
                if el.role == role and el.type_tag == type_tag:
                    matched += 1
                    break

        return matched / len(schema) if schema else 0.0

    # ── Step 6: Apply abstraction to predict ──

    def predict(self, episode_so_far: dict, abstraction: dict) -> dict | None:
        """
        Given a partial episode and a matching abstraction,
        predict what should happen next based on the pattern.

        This is where abstraction becomes useful:
        "I've seen this pattern before. Last time, the next step was X."
        """
        elements = self.decompose(episode_so_far)
        schema = abstraction.get("schema", [])

        # Find which schema slots are already filled
        filled_roles = {el.role for el in elements}

        # Find unfilled slots = predictions
        predictions = {}
        for slot in schema:
            if slot["role"] not in filled_roles:
                if slot.get("fixed_value"):
                    predictions[slot["role"]] = {
                        "predicted_value": slot["fixed_value"],
                        "type": slot["type"],
                        "confidence": abstraction.get("confidence", 0.5),
                        "source": abstraction["id"],
                    }
                else:
                    predictions[slot["role"]] = {
                        "predicted_value": None,  # Variable — can't predict exact value
                        "type": slot["type"],
                        "confidence": abstraction.get("confidence", 0.5) * 0.5,  # Less confident for variables
                        "source": abstraction["id"],
                        "note": "variable slot — structure known, value unknown",
                    }

        return predictions if predictions else None

    def _generate_name(self, invariants: list[str], variables: list[str]) -> str:
        """Generate a human-readable name for an abstraction."""
        # Use the invariant structure to name it
        roles = []
        for inv in invariants[:3]:  # First 3 invariants
            parts = inv.split(":", 1)
            roles.append(parts[0])

        var_roles = []
        for var in variables[:2]:
            parts = var.split(":", 1)
            var_roles.append(parts[0])

        name = "pattern:" + "+".join(roles)
        if var_roles:
            name += " varying:" + "+".join(var_roles)

        return name
