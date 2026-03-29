from __future__ import annotations

from dataclasses import asdict

from klomboagi.action.executor import ActionExecutor
from klomboagi.core.executive import Executive
from klomboagi.core.mission import MissionManager
from klomboagi.core.scheduler import Scheduler
from klomboagi.evals.autonomy import AutonomyEvaluator
from klomboagi.learning.consolidation import MemoryConsolidator
from klomboagi.evals.execution_auditor import ExecutionAuditor, ActionTelemetry
from klomboagi.learning.skill_forge import SkillForge
from klomboagi.learning.episode_indexer import EpisodeIndexer
from klomboagi.learning.reflection import ReflectionEngine
from klomboagi.learning.semantic import SemanticMemory
from klomboagi.memory.working_memory import WorkingMemoryManager
from klomboagi.reasoning.critic import Critic
from klomboagi.reasoning.planner import Planner
from klomboagi.reasoning.verifier import Verifier
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.config import load_config
from klomboagi.world.model import WorldModel


class RuntimeLoop:
    def __init__(self, storage: StorageManager, genesis=None) -> None:
        self.storage = storage
        self.genesis = genesis  # Bridge to the Genesis brain — shared knowledge
        self.executive = Executive(storage)
        self.missions = MissionManager(storage)
        self.scheduler = Scheduler(storage)
        self.executor = ActionExecutor(storage)
        self.working_memory = WorkingMemoryManager(storage)
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

    def initialize(self) -> dict[str, object]:
        existing = self.storage.world_state.load(default={})
        world_state = self.world_model.bootstrap()
        if not existing:
            self.storage.event_log.append("runtime.initialized", {"initialized_at": world_state["initialized_at"]})
        return world_state

    def status(self) -> dict[str, object]:
        snapshot = asdict(self.executive.snapshot())
        world_state = self.storage.world_state.load(default={})
        plans = self.storage.plans.load(default={})
        verifications = self.storage.verifications.load(default={})
        critiques = self.storage.critiques.load(default={})
        mission_queue = self.storage.mission_queue.load(default=[])
        cycle_traces = self.storage.cycle_traces.load(default=[])
        semantic_memory = self.storage.semantic_memory.load(default=[])
        procedures = self.storage.procedures.load(default=[])
        working_memory = self.storage.working_memory.load(default={})
        reflections = self.storage.reflections.load(default=[])
        autonomy_runs = self.storage.autonomy_evals.load(default=[])
        autonomy_report = self.storage.autonomy_reports.load(default={})
        world_entities = self.storage.world_entities.load(default={})
        world_relations = self.storage.world_relations.load(default=[])
        active_mission_id = world_state.get("active_mission_id")
        return {
            "snapshot": snapshot,
            "world_state": world_state,
            "world_entities": world_entities,
            "world_relations": world_relations,
            "mission_queue": mission_queue,
            "latest_cycle_trace": cycle_traces[-1] if cycle_traces else None,
            "active_plan": plans.get(active_mission_id) if active_mission_id else None,
            "latest_verification": verifications.get(active_mission_id) if active_mission_id else None,
            "latest_critique": critiques.get(active_mission_id) if active_mission_id else None,
            "semantic_memory": semantic_memory,
            "procedures": procedures,
            "working_memory": working_memory.get(active_mission_id) if active_mission_id else None,
            "latest_reflection": reflections[-1] if reflections else None,
            "latest_autonomy_eval": autonomy_runs[-1] if autonomy_runs else None,
            "autonomy_report": autonomy_report,
            "shared_skills": self.skill_forge.list_shared_skills(),
            "runtime_root": str(self.storage.paths.runtime_root),
            "long_term_root": str(self.storage.paths.long_term_root),
        }

    def run_cycle(self) -> dict[str, object]:
        world_state = self.initialize()
        missions = self.missions.list_missions()
        tasks_by_mission = {mission["id"]: self.missions.list_tasks(mission["id"]) for mission in missions}
        active_missions = [mission for mission in missions if mission["status"] == "active"]
        mission_queue = self.scheduler.build_queue(missions, tasks_by_mission)
        if not active_missions:
            world_state = self.world_model.refresh(
                missions=[],
                tasks=[],
                active_mission_id=None,
                current_focus=None,
                last_plan_id=None,
                last_reflection_id=None,
                notes=["Runtime is idle; no active missions available."],
            )
            idle = {
                "status": "idle",
                "reason": "No active missions available.",
                "world_state": world_state,
            }
            self.storage.event_log.append("runtime.cycle.idle", idle)
            return idle

        mission = self.scheduler.select_next(missions, tasks_by_mission) or active_missions[0]
        step_history: list[dict[str, object]] = []
        final_plan = None
        final_verification = None
        final_critique = None
        final_action = None
        final_outcome = None
        final_memory = None
        final_reflection = None
        final_procedure = None
        final_semantic_fact = None
        final_promoted_skills: list[dict[str, object]] = []
        stop_reason = "budget_exhausted"
        replan_count = 0
        max_replans = 2
        failure_count = 0

        for step_number in range(1, self.max_cycle_steps + 1):
            tasks = self.missions.list_tasks(mission["id"])
            active_task = self._select_active_task(tasks)
            learned_procedures = self.consolidator.retrieve(str(mission["description"]))
            learned_facts = self.semantic_memory.retrieve(str(mission["description"]))

            # Bridge: consult Genesis brain for relevant knowledge
            genesis_knowledge = self._query_genesis(str(mission["description"]))
            memory = self.working_memory.load_or_create(
                str(mission["id"]),
                str(mission["description"]),
                active_task_id=active_task["id"] if active_task else None,
            )
            self.world_model.refresh(
                missions=active_missions,
                tasks=tasks,
                active_mission_id=str(mission["id"]),
                current_focus=str(memory["current_focus"]),
                last_plan_id=None,
                last_reflection_id=None,
                notes=[f"Loaded mission '{mission['description']}' into world state."],
            )
            plan = self.planner.build_plan(mission, tasks)
            proposed_action = self.planner.next_action(plan)
            world_entities = self.storage.world_entities.load(default={})
            world_relations = self.storage.world_relations.load(default=[])
            verification = self.verifier.verify(
                mission=mission,
                tasks=tasks,
                plan=plan,
                next_action=proposed_action,
                world_entities=world_entities,
                world_relations=world_relations,
            )
            critique = self.critic.critique(
                mission=mission,
                tasks=tasks,
                plan=plan,
                proposed_action=proposed_action,
                verification=verification,
            )
            next_action = critique["final_action"]

            blockers = self._collect_blockers(tasks)
            focus = next_action["description"] if next_action["type"] != "mission_complete" else str(mission["description"])
            memory = self.working_memory.update(
                str(mission["id"]),
                current_focus=focus,
                active_task_id=next_action.get("task_id"),
                active_plan_id=str(plan["id"]),
                blockers=blockers,
                relevant_memories=[f"Mission priority is {mission['priority']}.", *learned_procedures, *learned_facts, *genesis_knowledge],
                verification_alerts=list(verification.get("issues", [])) + list(verification.get("warnings", [])),
                critique_notes=list(critique.get("notes", [])),
                last_action=str(next_action["type"]),
                hypothesis=f"Next action selected: {next_action['description']}",
            )
            action_outcome = self._apply_action_outcome(mission, next_action)
            tasks = self.missions.list_tasks(mission["id"])
            mission = self.missions.get_mission(mission["id"]) or mission
            reflection = self.reflection.reflect(mission, tasks, memory, next_action, action_outcome)
            procedure = self.consolidator.consolidate(
                mission=mission,
                reflection=reflection,
                action_outcome=action_outcome,
            )
            semantic_fact = self.semantic_memory.remember(
                mission=mission,
                reflection=reflection,
                action_outcome=action_outcome,
            )

            # SkillForge: promote high-confidence procedures to shared skill store
            promoted_skills = self.skill_forge.scan_and_promote()

            step_history.append(
                {
                    "step_number": step_number,
                    "plan_id": plan["id"],
                    "next_action": next_action,
                    "action_outcome": action_outcome,
                    "reflection_id": reflection["id"],
                }
            )

            final_plan = plan
            final_verification = verification
            final_critique = critique
            final_action = next_action
            final_outcome = action_outcome
            final_memory = memory
            final_reflection = reflection
            final_procedure = procedure
            final_semantic_fact = semantic_fact
            final_promoted_skills = promoted_skills

            if next_action["type"] == "replan":
                if replan_count < max_replans:
                    replan_count += 1
                    self.storage.event_log.append(
                        "runtime.replan.recovery",
                        {"mission_id": mission["id"], "replan_count": replan_count},
                    )
                    # Refresh tasks and rebuild plan on next iteration
                    continue
                stop_reason = "replan_requested"
                break
            if action_outcome["status"] in {"failed", "blocked"}:
                failure_count += 1
                if failure_count < max_replans and replan_count < max_replans:
                    replan_count += 1
                    self.storage.event_log.append(
                        "runtime.failure.replan",
                        {
                            "mission_id": mission["id"],
                            "failure_count": failure_count,
                            "replan_count": replan_count,
                        },
                    )
                    continue
                stop_reason = action_outcome["status"]
                break
            if mission["status"] == "completed" or action_outcome.get("mission_status") == "completed":
                stop_reason = "mission_completed"
                break
            if step_number >= self.max_cycle_steps:
                stop_reason = "budget_exhausted"
                break

        world_state = self.world_model.refresh(
            missions=[item if item["id"] != mission["id"] else mission for item in active_missions],
            tasks=self.missions.list_tasks(mission["id"]),
            active_mission_id=str(mission["id"]),
            current_focus=str(final_memory["current_focus"]),
            last_plan_id=str(final_plan["id"]),
            last_reflection_id=str(final_reflection["id"]),
            notes=[f"Current mission focus: {final_memory['current_focus']}"],
        )
        cycle_trace = self._record_cycle_trace(
            mission=mission,
            step_history=step_history,
            stop_reason=stop_reason,
        )
        autonomy_eval = self.autonomy.record(
            mission=mission,
            plan=final_plan,
            verification=final_verification,
            critique=final_critique,
            next_action=final_action,
            action_outcome=final_outcome,
            working_memory=final_memory,
            cycle_trace=cycle_trace,
        )
        autonomy_report = self.autonomy.summarize()
        payload = {
            "status": "active",
            "mission": mission,
            "plan": final_plan,
            "verification": final_verification,
            "critique": final_critique,
            "next_action": final_action,
            "action_outcome": final_outcome,
            "semantic_fact": final_semantic_fact,
            "procedure": final_procedure,
            "working_memory": final_memory,
            "reflection": final_reflection,
            "cycle_trace": cycle_trace,
            "autonomy_eval": autonomy_eval,
            "autonomy_report": autonomy_report,
            "world_state": world_state,
            "world_entities": self.storage.world_entities.load(default={}),
            "world_relations": self.storage.world_relations.load(default=[]),
            "mission_queue": self.storage.mission_queue.load(default=[]),
        }
        self.storage.event_log.append(
            "runtime.cycle.completed",
            {
                "mission_id": mission["id"],
                "plan_id": final_plan["id"],
                "verification_valid": final_verification["valid"],
                "critique_approved": final_critique["approved"],
                "reflection_id": final_reflection["id"],
                "action_outcome": final_outcome["status"],
                "next_action": final_action["type"],
                "cycle_steps": cycle_trace["step_count"],
                "stop_reason": cycle_trace["stop_reason"],
            },
        )

        # Cross-session learning: record episode after each cycle
        try:
            tool_calls = [
                {"tool": step["next_action"]["type"], "success": step["action_outcome"]["status"] == "completed"}
                for step in step_history
                if "next_action" in step and "action_outcome" in step
            ]
            episode = self.episode_indexer.build_episode(
                session_id=str(mission["id"]),
                project_root=str(self.storage.paths.runtime_root),
                goal=str(mission.get("description", "")),
                tool_calls=tool_calls,
                success=stop_reason == "mission_completed",
                outcomes=[str(final_outcome.get("status", "unknown"))],
            )
            self.episode_indexer.record_episode(episode)
        except Exception:
            pass  # Episode recording should never crash the cycle

        return payload

    def _select_active_task(self, tasks: list[dict[str, object]]) -> dict[str, object] | None:
        for task in tasks:
            if task["status"] == "active":
                return task
        completed_ids = {str(task["id"]) for task in tasks if task["status"] == "completed"}
        for task in tasks:
            if task["status"] not in {"queued", "active"}:
                continue
            dependencies = {str(dep) for dep in task.get("dependencies", [])}
            if dependencies.issubset(completed_ids):
                return task
        return None

    def _collect_blockers(self, tasks: list[dict[str, object]]) -> list[str]:
        blockers = []
        for task in tasks:
            if task["status"] == "blocked" and task.get("blocked_reason"):
                blockers.append(str(task["blocked_reason"]))
        return blockers

    def _query_genesis(self, description: str) -> list[str]:
        """
        Bridge: query the Genesis brain for relevant knowledge.

        Returns a list of relevant facts/beliefs from Genesis's knowledge graph.
        If Genesis is not connected, returns empty list (graceful degradation).
        """
        if not self.genesis:
            return []
        try:
            # Search Genesis beliefs for anything relevant to this mission
            relevant = []
            desc_words = set(description.lower().split())
            stop = {"the", "a", "an", "to", "for", "and", "or", "in", "of", "is", "are",
                    "with", "that", "this", "from", "by", "on", "at"}
            desc_words -= stop

            # Check beliefs
            for stmt, belief in self.genesis.base._beliefs.items():
                stmt_words = set(stmt.lower().split()) - stop
                if desc_words & stmt_words and len(relevant) < 5:
                    relevant.append(f"[brain] {stmt}")

            # Check concept store (studied knowledge)
            for word in desc_words:
                facts = self.genesis.base.memory.concepts.get(word, {}).get("facts", [])
                for fact in facts[:2]:
                    if isinstance(fact, str) and len(fact) > 20 and fact not in relevant:
                        relevant.append(f"[brain] {fact[:150]}")
                        if len(relevant) >= 10:
                            break

            return relevant
        except Exception:
            return []

    def _teach_genesis(self, description: str, outcome: str) -> None:
        """
        Bridge: feed learnings back to Genesis brain.

        After completing a task, teach Genesis what we learned.
        """
        if not self.genesis:
            return
        try:
            statement = f"task '{description[:50]}' resulted in {outcome}"
            self.genesis.hear(statement)
        except Exception:
            pass

    def _apply_action_outcome(self, mission: dict[str, object], next_action: dict[str, object]) -> dict[str, object]:
        result = self.executor.execute(mission, next_action)
        # Feed outcome back to Genesis brain
        self._teach_genesis(
            str(next_action.get("description", "")),
            str(result.get("status", "unknown")))
        return result

    def _record_cycle_trace(
        self,
        *,
        mission: dict[str, object],
        step_history: list[dict[str, object]],
        stop_reason: str,
    ) -> dict[str, object]:
        traces = self.storage.cycle_traces.load(default=[])
        trace = {
            "mission_id": mission["id"],
            "step_count": len(step_history),
            "stop_reason": stop_reason,
            "steps": step_history,
        }
        traces.append(trace)
        self.storage.cycle_traces.save(traces)
        self.storage.event_log.append(
            "runtime.cycle.trace",
            {"mission_id": mission["id"], "step_count": len(step_history), "stop_reason": stop_reason},
        )
        return trace
