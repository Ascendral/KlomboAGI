"""
Causal Model — learning "X causes Y" from experience.

Not correlation. Not co-occurrence. Causation.

The difference:
- Correlation: "Every time I see rain, I see umbrellas"
- Causation: "Rain causes people to use umbrellas"
- The test: If I INTERVENE and remove umbrellas, does rain stop? No.
            If I INTERVENE and stop rain, do umbrellas disappear? Yes.

This is Pearl's do-calculus in code:
- Level 1: Association — P(Y|X) — what do I observe?
- Level 2: Intervention — P(Y|do(X)) — what happens if I ACT?
- Level 3: Counterfactual — P(Y_x|X',Y') — what WOULD have happened?

The algorithm:
1. Observe action-outcome pairs from episodes
2. Build a directed graph of causal relationships
3. Strengthen edges that survive intervention tests
4. Weaken edges that fail intervention tests
5. Use the graph to predict outcomes of novel actions
6. Use counterfactuals to explain failures

No LLM. Pure graph operations + statistical tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


@dataclass
class CausalEdge:
    """A directed causal relationship: cause → effect."""
    cause: str
    effect: str
    strength: float = 0.0       # -1.0 to 1.0 (negative = inhibits)
    observations: int = 0       # How many times we've seen this pair
    interventions: int = 0      # How many times we've tested this deliberately
    successes: int = 0          # Times the effect followed the cause
    failures: int = 0           # Times the cause happened without the effect
    confounders: list[str] = field(default_factory=list)  # Known third variables
    context: str = ""           # Under what conditions this holds
    last_observed: str = ""

    @property
    def confidence(self) -> float:
        """How confident are we this is truly causal?"""
        if self.observations == 0:
            return 0.0
        # Intervention data is worth more than observation
        obs_weight = min(self.observations / 10, 1.0) * 0.4
        int_weight = min(self.interventions / 3, 1.0) * 0.6
        success_rate = self.successes / max(self.successes + self.failures, 1)
        return (obs_weight + int_weight) * success_rate

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "observations": self.observations,
            "interventions": self.interventions,
            "successes": self.successes,
            "failures": self.failures,
            "confounders": self.confounders,
            "context": self.context,
            "confidence": self.confidence,
            "last_observed": self.last_observed,
        }

    @staticmethod
    def from_dict(d: dict) -> CausalEdge:
        return CausalEdge(
            cause=d["cause"],
            effect=d["effect"],
            strength=d.get("strength", 0.0),
            observations=d.get("observations", 0),
            interventions=d.get("interventions", 0),
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            confounders=d.get("confounders", []),
            context=d.get("context", ""),
            last_observed=d.get("last_observed", ""),
        )


class CausalGraph:
    """
    A directed graph of causal relationships learned from experience.

    This is not a static knowledge graph. It updates continuously:
    - New observations strengthen or weaken edges
    - Interventions (deliberate tests) provide stronger evidence
    - Confounders are tracked and edges are adjusted
    - The graph can predict, explain, and suggest experiments
    """

    def __init__(self) -> None:
        self.edges: dict[str, CausalEdge] = {}  # key = "cause->effect"
        self.nodes: set[str] = set()

    def _key(self, cause: str, effect: str) -> str:
        return f"{cause}->{effect}"

    def observe(self, cause: str, effect: str, success: bool = True, context: str = "") -> CausalEdge:
        """
        Record an observation: cause happened, then effect happened (or didn't).

        This is Level 1 (association). Weaker than intervention.
        """
        key = self._key(cause, effect)
        self.nodes.add(cause)
        self.nodes.add(effect)

        if key not in self.edges:
            self.edges[key] = CausalEdge(cause=cause, effect=effect, context=context)

        edge = self.edges[key]
        edge.observations += 1
        edge.last_observed = utc_now()
        if success:
            edge.successes += 1
        else:
            edge.failures += 1

        # Update strength based on success rate
        total = edge.successes + edge.failures
        if total > 0:
            edge.strength = (edge.successes / total) * 2 - 1  # Map to [-1, 1]

        return edge

    def intervene(self, cause: str, effect: str, result: bool) -> CausalEdge:
        """
        Record a deliberate intervention: we MADE cause happen and observed effect.

        This is Level 2 (do-calculus). Stronger evidence than passive observation.
        Intervention breaks confounders — if we force X and Y still happens,
        X likely causes Y (not just correlated through Z).
        """
        key = self._key(cause, effect)
        self.nodes.add(cause)
        self.nodes.add(effect)

        if key not in self.edges:
            self.edges[key] = CausalEdge(cause=cause, effect=effect)

        edge = self.edges[key]
        edge.interventions += 1
        edge.observations += 1
        edge.last_observed = utc_now()
        if result:
            edge.successes += 1
            # Intervention success strengthens more than observation
            edge.strength = min(1.0, edge.strength + 0.2)
        else:
            edge.failures += 1
            edge.strength = max(-1.0, edge.strength - 0.3)  # Failure weakens more

        return edge

    def get_causes(self, effect: str) -> list[CausalEdge]:
        """What causes this effect? Returns edges sorted by strength."""
        causes = [e for e in self.edges.values() if e.effect == effect]
        causes.sort(key=lambda e: e.strength, reverse=True)
        return causes

    def get_effects(self, cause: str) -> list[CausalEdge]:
        """What does this cause? Returns edges sorted by strength."""
        effects = [e for e in self.edges.values() if e.cause == cause]
        effects.sort(key=lambda e: e.strength, reverse=True)
        return effects

    def predict(self, action: str) -> list[tuple[str, float, float]]:
        """
        Given an action, predict what will happen.
        Returns: [(predicted_effect, probability, confidence), ...]
        """
        effects = self.get_effects(action)
        predictions = []
        for edge in effects:
            if edge.strength > 0:
                prob = (edge.strength + 1) / 2  # Map [-1,1] to [0,1]
                predictions.append((edge.effect, prob, edge.confidence))
        return predictions

    def explain(self, effect: str) -> list[tuple[str, float, float]]:
        """
        Given an observed effect, explain what likely caused it.
        Returns: [(likely_cause, strength, confidence), ...]
        """
        causes = self.get_causes(effect)
        explanations = []
        for edge in causes:
            if edge.strength > 0:
                explanations.append((edge.cause, edge.strength, edge.confidence))
        return explanations

    def counterfactual(self, cause: str, effect: str) -> dict:
        """
        Level 3: What WOULD have happened if cause hadn't occurred?

        Looks at the graph structure:
        - Are there other causes for this effect?
        - How strong are they?
        - Would the effect have happened anyway?
        """
        edge = self.edges.get(self._key(cause, effect))
        if not edge:
            return {"answer": "unknown", "reason": "no causal link observed"}

        # Find alternative causes for the same effect
        alt_causes = [e for e in self.get_causes(effect) if e.cause != cause]

        if not alt_causes:
            return {
                "answer": "effect_would_not_occur",
                "reason": f"'{cause}' is the only known cause of '{effect}'",
                "confidence": edge.confidence,
            }

        # Check if alternative causes are strong enough
        strong_alts = [e for e in alt_causes if e.strength > 0.3 and e.confidence > 0.1]

        if strong_alts:
            best_alt = strong_alts[0]
            return {
                "answer": "effect_might_still_occur",
                "reason": f"'{best_alt.cause}' also causes '{effect}' (strength={best_alt.strength:.2f})",
                "confidence": best_alt.confidence,
                "alternative_cause": best_alt.cause,
            }

        return {
            "answer": "effect_probably_would_not_occur",
            "reason": f"No strong alternative causes found for '{effect}'",
            "confidence": edge.confidence * 0.8,
        }

    def suggest_experiment(self) -> dict | None:
        """
        Find the weakest causal link and suggest an intervention to test it.

        This is how the system actively seeks knowledge:
        "I'm not sure if X causes Y. Let me test it."
        """
        # Find edges with high observations but low interventions
        uncertain = []
        for edge in self.edges.values():
            if edge.observations >= 3 and edge.interventions < 2:
                uncertain.append(edge)

        if not uncertain:
            # Find edges with low confidence overall
            uncertain = [e for e in self.edges.values() if e.confidence < 0.5 and e.observations >= 2]

        if not uncertain:
            return None

        # Pick the one we're least sure about
        uncertain.sort(key=lambda e: e.confidence)
        target = uncertain[0]

        return {
            "type": "intervention",
            "cause": target.cause,
            "effect": target.effect,
            "current_strength": target.strength,
            "current_confidence": target.confidence,
            "reason": f"Observed {target.observations} times but only tested {target.interventions} times. Need intervention data.",
            "instruction": f"Deliberately perform '{target.cause}' and observe whether '{target.effect}' occurs.",
        }

    def find_confounders(self, cause: str, effect: str) -> list[str]:
        """
        Find potential confounders — variables that might cause BOTH
        the cause and the effect, creating a spurious correlation.

        If Z → X and Z → Y, then X and Y correlate but X doesn't cause Y.
        """
        confounders = []
        for node in self.nodes:
            if node == cause or node == effect:
                continue
            # Does this node cause both the cause and the effect?
            causes_cause = self._key(node, cause) in self.edges and self.edges[self._key(node, cause)].strength > 0.3
            causes_effect = self._key(node, effect) in self.edges and self.edges[self._key(node, effect)].strength > 0.3
            if causes_cause and causes_effect:
                confounders.append(node)
        return confounders

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> CausalGraph:
        graph = CausalGraph()
        graph.nodes = set(d.get("nodes", []))
        for k, v in d.get("edges", {}).items():
            graph.edges[k] = CausalEdge.from_dict(v)
        return graph


class CausalModel:
    """
    Persistent causal model that learns from the system's experiences.

    Wraps CausalGraph with storage and episode processing.
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self.graph = self._load()

    def _load(self) -> CausalGraph:
        data = self.storage.load_json("causal_graph", default={"nodes": [], "edges": {}})
        return CausalGraph.from_dict(data)

    def save(self) -> None:
        self.storage.save_json("causal_graph", self.graph.to_dict())

    def learn_from_episode(self, episode: dict) -> list[CausalEdge]:
        """
        Extract causal relationships from an episode.

        Each action → outcome pair is a potential causal link.
        Sequential actions create chains: A → B → C.
        """
        learned = []
        actions = episode.get("actions", episode.get("steps", []))
        outcome = episode.get("outcome", episode.get("status", "unknown"))
        success = episode.get("success", outcome in ("completed", "success", "passed"))

        # Each action potentially causes the next action's context
        for i in range(len(actions) - 1):
            curr = actions[i]
            next_action = actions[i + 1]

            curr_name = curr.get("type", curr.get("action", str(curr))) if isinstance(curr, dict) else str(curr)
            next_name = next_action.get("type", next_action.get("action", str(next_action))) if isinstance(next_action, dict) else str(next_action)

            edge = self.graph.observe(curr_name, next_name, success=True)
            learned.append(edge)

        # Last action → overall outcome
        if actions:
            last = actions[-1]
            last_name = last.get("type", last.get("action", str(last))) if isinstance(last, dict) else str(last)
            edge = self.graph.observe(last_name, outcome, success=success)
            learned.append(edge)

        # First action → overall outcome (full chain)
        if actions:
            first = actions[0]
            first_name = first.get("type", first.get("action", str(first))) if isinstance(first, dict) else str(first)
            edge = self.graph.observe(first_name, outcome, success=success)
            learned.append(edge)

        self.save()
        self.storage.event_log.append(
            "causal.learned",
            {"episode_id": episode.get("id", "unknown"), "edges_updated": len(learned)},
        )
        return learned

    def predict_outcome(self, action: str) -> list[tuple[str, float, float]]:
        """What will happen if I do this?"""
        return self.graph.predict(action)

    def explain_outcome(self, effect: str) -> list[tuple[str, float, float]]:
        """Why did this happen?"""
        return self.graph.explain(effect)

    def what_if_not(self, cause: str, effect: str) -> dict:
        """What would have happened without this cause?"""
        return self.graph.counterfactual(cause, effect)

    def what_should_i_test(self) -> dict | None:
        """What causal link should I test next?"""
        return self.graph.suggest_experiment()
