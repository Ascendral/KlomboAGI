"""
Autonomous Mode — mission execution subsystems.

Groups the subsystems needed for autonomous mission execution:
planning, verification, critique, action execution, reflection,
memory consolidation, skill forging, and world modeling.

These are complementary to Genesis's reasoning stack (which handles
questions and learning). Together they form one brain with two modes:
interactive (hear) and autonomous (run_cycle).
"""

from __future__ import annotations

from klomboagi.action.executor import ActionExecutor
from klomboagi.core.executive import Executive
from klomboagi.core.mission import MissionManager
from klomboagi.core.scheduler import Scheduler
from klomboagi.evals.autonomy import AutonomyEvaluator
from klomboagi.evals.execution_auditor import ExecutionAuditor
from klomboagi.learning.consolidation import MemoryConsolidator
from klomboagi.learning.episode_indexer import EpisodeIndexer
from klomboagi.learning.reflection import ReflectionEngine
from klomboagi.learning.semantic import SemanticMemory
from klomboagi.learning.skill_forge import SkillForge
from klomboagi.memory.working_memory import MissionMemoryManager
from klomboagi.reasoning.critic import Critic
from klomboagi.reasoning.planner import Planner
from klomboagi.reasoning.verifier import Verifier
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.config import load_config
from klomboagi.world.model import WorldModel


class AutonomousMode:
    """Container for autonomous mission execution subsystems."""

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self.executive = Executive(storage)
        self.missions = MissionManager(storage)
        self.scheduler = Scheduler(storage)
        self.executor = ActionExecutor(storage)
        self.mission_wm = MissionMemoryManager(storage)
        self.planner = Planner(storage)
        self.verifier = Verifier(storage)
        self.critic = Critic(storage)
        self.reflection = ReflectionEngine(storage)
        self.consolidator = MemoryConsolidator(storage)
        self.semantic_memory = SemanticMemory(storage)
        self.autonomy = AutonomyEvaluator(storage)
        self.world_model = WorldModel(storage)
        self.execution_auditor = ExecutionAuditor(storage)
        self.skill_forge = SkillForge(storage)
        self.episode_indexer = EpisodeIndexer(storage)
        self.max_cycle_steps = int(load_config()["runtime"]["max_cycle_steps"])
