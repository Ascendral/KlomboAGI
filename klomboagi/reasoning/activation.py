"""
Spreading Activation — neurons fire in all directions simultaneously.

When a concept is activated, activation spreads through ALL its connections:
is_a, causes, part_of, uses, opposite_of — everything. Each hop decays
the signal. Concepts that receive activation from MULTIPLE sources become
highly activated — they're where the thinking concentrates.

This is how real neural networks work. Not deep learning — actual biology.
A neuron fires, its neighbors fire, their neighbors fire. What emerges
from the overlapping activation patterns IS the thought.

Usage:
    network = ActivationNetwork(relation_store, beliefs)
    result = network.activate(["gravity", "energy"])
    # → returns concepts sorted by activation strength
    # gravity activates: force, acceleration, geometry
    # energy activates: work, heat, conservation
    # OVERLAP: force appears in both → highly activated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from klomboagi.core.relations import RelationStore, RelationType


@dataclass
class ActivationNode:
    """A concept with an activation level."""
    name: str
    activation: float = 0.0
    sources: list[str] = field(default_factory=list)  # what activated this
    hops: int = 0  # how many steps from the original stimulus

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "activation": round(self.activation, 4),
            "sources": self.sources,
            "hops": self.hops,
        }


@dataclass
class ActivationResult:
    """The result of a spreading activation pass."""
    seeds: list[str]
    activated: list[ActivationNode]   # sorted by activation, highest first
    total_nodes: int
    convergence_points: list[str]     # nodes activated by multiple seeds

    def top(self, n: int = 10) -> list[ActivationNode]:
        return self.activated[:n]

    def to_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "top_activated": [n.to_dict() for n in self.activated[:20]],
            "total_nodes": self.total_nodes,
            "convergence_points": self.convergence_points,
        }


class ActivationNetwork:
    """
    Spreading activation across the knowledge graph.

    Simulates neural firing: activation spreads from seed concepts
    through all relation types simultaneously, decaying with distance.
    Convergence points (nodes receiving signal from multiple seeds)
    get boosted — they're where the thinking concentrates.
    """

    DECAY = 0.6           # signal loses 40% per hop
    CONVERGENCE_BOOST = 2.0  # bonus for multi-source activation
    MAX_HOPS = 4          # don't spread further than this
    MIN_ACTIVATION = 0.01 # stop spreading below this threshold

    def __init__(self, relations: RelationStore,
                 beliefs: dict | None = None) -> None:
        self.relations = relations
        self.beliefs = beliefs or {}

    def activate(self, seeds: list[str], initial_strength: float = 1.0) -> ActivationResult:
        """
        Activate seed concepts and let activation spread.

        All seeds fire simultaneously. Each spreads through all
        relation types. Convergence points (concepts reached by
        multiple seeds) get boosted.
        """
        # Track activation per concept
        activations: dict[str, float] = defaultdict(float)
        sources: dict[str, set] = defaultdict(set)
        hops: dict[str, int] = {}

        # Fire each seed simultaneously
        for seed in seeds:
            self._spread(seed, seed, initial_strength, 0,
                        activations, sources, hops, set())

        # Also check beliefs for connections
        if self.beliefs:
            for seed in seeds:
                seed_lower = seed.lower()
                for statement, belief in self.beliefs.items():
                    if hasattr(belief, 'subject') and belief.subject == seed_lower:
                        pred = belief.predicate
                        strength = initial_strength * self.DECAY * belief.truth.confidence
                        if strength > self.MIN_ACTIVATION:
                            activations[pred] += strength
                            sources[pred].add(seed)
                            if pred not in hops:
                                hops[pred] = 1
                    elif hasattr(belief, 'predicate') and belief.predicate == seed_lower:
                        subj = belief.subject
                        strength = initial_strength * self.DECAY * 0.5 * belief.truth.confidence
                        if strength > self.MIN_ACTIVATION:
                            activations[subj] += strength
                            sources[subj].add(seed)
                            if subj not in hops:
                                hops[subj] = 1

        # Apply convergence boost — concepts reached by multiple seeds
        convergence_points = []
        for concept, src_set in sources.items():
            if len(src_set) > 1:
                activations[concept] *= self.CONVERGENCE_BOOST
                convergence_points.append(concept)

        # Remove seeds from results (we already know about them)
        for seed in seeds:
            activations.pop(seed, None)

        # Build sorted result
        nodes = []
        for name, activation in activations.items():
            if activation >= self.MIN_ACTIVATION:
                nodes.append(ActivationNode(
                    name=name,
                    activation=activation,
                    sources=sorted(sources[name]),
                    hops=hops.get(name, 0),
                ))

        nodes.sort(key=lambda n: n.activation, reverse=True)

        return ActivationResult(
            seeds=seeds,
            activated=nodes,
            total_nodes=len(nodes),
            convergence_points=convergence_points,
        )

    def _spread(self, seed: str, current: str, strength: float, depth: int,
                activations: dict[str, float],
                sources: dict[str, set],
                hops: dict[str, int],
                visited: set) -> None:
        """Recursively spread activation from current node."""
        if depth > self.MAX_HOPS or strength < self.MIN_ACTIVATION:
            return
        if current in visited:
            return
        visited.add(current)

        # Activate current node
        if current != seed:  # Don't count the seed itself
            activations[current] += strength
            sources[current].add(seed)
            if current not in hops or depth < hops[current]:
                hops[current] = depth

        # Spread through forward relations
        for rel in self.relations.get_forward(current):
            next_strength = strength * self.DECAY * rel.confidence
            self._spread(seed, rel.target, next_strength, depth + 1,
                         activations, sources, hops, visited.copy())

        # Spread through backward relations
        for rel in self.relations.get_backward(current):
            next_strength = strength * self.DECAY * rel.confidence * 0.7  # backward slightly weaker
            self._spread(seed, rel.source, next_strength, depth + 1,
                         activations, sources, hops, visited.copy())

    def think_about(self, concepts: list[str]) -> str:
        """
        What comes to mind when thinking about these concepts?

        Returns the top activated concepts with explanations.
        """
        result = self.activate(concepts)

        if not result.activated:
            return f"Nothing comes to mind for: {', '.join(concepts)}"

        lines = [f"Thinking about {', '.join(concepts)}..."]

        if result.convergence_points:
            lines.append(f"\nConvergence points (where ideas meet):")
            for cp in result.convergence_points[:5]:
                node = next((n for n in result.activated if n.name == cp), None)
                if node:
                    lines.append(f"  * {cp} (activation: {node.activation:.2f}, "
                                f"from: {', '.join(node.sources)})")

        lines.append(f"\nMost activated concepts:")
        for node in result.top(10):
            marker = " *" if node.name in result.convergence_points else ""
            lines.append(f"  {node.name:30s} {node.activation:.3f} "
                        f"(hop {node.hops}, from: {', '.join(node.sources)}){marker}")

        return "\n".join(lines)
