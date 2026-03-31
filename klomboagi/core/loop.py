"""
RuntimeLoop compatibility shim.

Delegates to Genesis's unified cognition pipeline (autonomous mode).
All mission execution now happens through Genesis.run_cycle(), which
has direct access to the full reasoning stack instead of bridging.

This shim preserves the RuntimeLoop(storage, genesis) constructor
signature for backwards compatibility with CLI, tests, benchmarks,
and any external callers.
"""

from __future__ import annotations

import warnings

from klomboagi.storage.manager import StorageManager


class RuntimeLoop:
    """
    Compatibility wrapper — delegates to Genesis autonomous mode.

    One brain, two modes:
    - Interactive: genesis.hear(message)
    - Autonomous: genesis.run_cycle() (via this shim or directly)
    """

    def __init__(self, storage: StorageManager, genesis=None) -> None:
        if genesis is None:
            from klomboagi.core.genesis import Genesis
            genesis = Genesis()
        self._genesis = genesis
        if not hasattr(genesis, '_autonomous') or not genesis._autonomous:
            genesis._init_autonomous(storage)
        self.storage = storage

        # Expose subsystems that tests and callers access directly
        self.missions = genesis._autonomous.missions
        self.working_memory = genesis._autonomous.mission_wm
        self.planner = genesis._autonomous.planner
        self.verifier = genesis._autonomous.verifier
        self.critic = genesis._autonomous.critic
        self.reflection = genesis._autonomous.reflection
        self.consolidator = genesis._autonomous.consolidator
        self.semantic_memory = genesis._autonomous.semantic_memory
        self.autonomy = genesis._autonomous.autonomy
        self.world_model = genesis._autonomous.world_model
        self.executor = genesis._autonomous.executor
        self.skill_forge = genesis._autonomous.skill_forge
        self.episode_indexer = genesis._autonomous.episode_indexer
        self.execution_auditor = genesis._autonomous.execution_auditor
        self.executive = genesis._autonomous.executive
        self.scheduler = genesis._autonomous.scheduler
        self.genesis = genesis  # Preserve the old bridge attribute
        self.max_cycle_steps = genesis._autonomous.max_cycle_steps

    def initialize(self) -> dict[str, object]:
        return self._genesis.initialize_autonomous()

    def status(self) -> dict[str, object]:
        return self._genesis.autonomous_status()

    def run_cycle(self) -> dict[str, object]:
        return self._genesis.run_cycle()
