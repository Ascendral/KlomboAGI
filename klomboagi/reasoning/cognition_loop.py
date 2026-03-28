"""
Cognition Loop — the integration algorithm.

This is the unpublished piece: a single loop that orchestrates
abstraction, comparison, causal reasoning, inquiry, self-evaluation,
and transfer into one coherent thinking process.

Every existing system does ONE of these well.
Nobody has wired them together into a loop that actually thinks.

The loop:
1. PERCEIVE — receive and decompose the problem
2. REMEMBER — search for structurally similar past experiences
3. TRANSFER — if similar experience found, map its approach
4. INQUIRE — if knowledge gaps exist, identify them
5. HYPOTHESIZE — generate a candidate answer/approach
6. EVALUATE — check the hypothesis against known facts and structure
7. REVISE — if evaluation fails, shift approach (or accept human nudge)
8. ACT — execute the chosen approach
9. OBSERVE — record what happened
10. LEARN — abstract the pattern, update causal model, strengthen/weaken

This is not a pipeline. Steps can loop back. The evaluator can
send you back to step 2. A human nudge can send you back to step 1
with a different decomposition. The inquiry engine can pause everything
to ask a question before proceeding.

No LLM required for the orchestration. The LLM is an optional tool
that can be called from step 5 (hypothesize) if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

from klomboagi.reasoning.abstraction import AbstractionEngine
from klomboagi.reasoning.comparator import StructuralComparator, ComparisonResult
from klomboagi.reasoning.causal import CausalModel
from klomboagi.reasoning.inquiry import InquiryEngine, KnowledgeGap
from klomboagi.reasoning.self_eval import SelfEvaluator, ReasoningAttempt, EvaluationResult
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now

# Optional import — trait system is not required
try:
    from klomboagi.core.traits import TraitSystem, TraitInfluence
except ImportError:
    TraitSystem = None  # type: ignore[assignment,misc]
    TraitInfluence = None  # type: ignore[assignment,misc]


class CognitionPhase(Enum):
    PERCEIVE = "perceive"
    REMEMBER = "remember"
    TRANSFER = "transfer"
    INQUIRE = "inquire"
    HYPOTHESIZE = "hypothesize"
    EVALUATE = "evaluate"
    REVISE = "revise"
    ACT = "act"
    OBSERVE = "observe"
    LEARN = "learn"
    PAUSED = "paused"       # Waiting for input (human nudge or answer to inquiry)
    COMPLETE = "complete"


@dataclass
class CognitionState:
    """The full state of a thinking process."""
    problem: dict                               # The original problem
    phase: CognitionPhase = CognitionPhase.PERCEIVE
    decomposition: list = field(default_factory=list)    # Structural elements
    similar_experiences: list = field(default_factory=list)  # Past matches
    transfer_plan: dict | None = None           # Mapped approach from past experience
    knowledge_gaps: list = field(default_factory=list)   # What we don't know
    hypothesis: dict | None = None              # Current candidate answer
    evaluation: dict | None = None              # Self-evaluation result
    action_result: dict | None = None           # What happened when we acted
    learned: dict | None = None                 # What was abstracted
    nudges: list[str] = field(default_factory=list)  # Human corrections received
    trait_influence: dict | None = None         # Trait system influence (if active)
    attempts: int = 0                           # How many times through the loop
    max_attempts: int = 3
    trace: list[dict] = field(default_factory=list)  # Full reasoning trace
    created_at: str = ""
    completed_at: str | None = None

    def log(self, phase: str, message: str, data: Any = None) -> None:
        """Record a step in the reasoning trace."""
        self.trace.append({
            "phase": phase,
            "message": message,
            "data": data,
            "timestamp": utc_now(),
            "attempt": self.attempts,
        })

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "phase": self.phase.value,
            "decomposition": [str(d) for d in self.decomposition],
            "similar_count": len(self.similar_experiences),
            "has_transfer": self.transfer_plan is not None,
            "gap_count": len(self.knowledge_gaps),
            "hypothesis": self.hypothesis,
            "evaluation": self.evaluation,
            "action_result": self.action_result,
            "learned": self.learned,
            "nudges": self.nudges,
            "trait_influence": self.trait_influence,
            "attempts": self.attempts,
            "trace": self.trace,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class CognitionLoop:
    """
    The integration algorithm. Orchestrates all reasoning components
    into one coherent thinking process.

    This is the piece nobody has published.
    """

    def __init__(self, storage: StorageManager,
                 trait_system: TraitSystem | None = None) -> None:
        self.storage = storage
        self.abstraction = AbstractionEngine(storage)
        self.comparator = StructuralComparator(self.abstraction)
        self.causal = CausalModel(storage)
        self.inquiry = InquiryEngine(storage)
        self.evaluator = SelfEvaluator()
        self.trait_system = trait_system

        # Optional callbacks
        self.on_phase: Callable[[CognitionPhase, str], None] | None = None
        self.on_inquiry: Callable[[KnowledgeGap], Any] | None = None  # Called when system needs an answer
        self.on_hypothesis: Callable[[dict], dict] | None = None  # Optional LLM for hypothesis generation
        self.on_action: Callable[[dict], dict] | None = None  # Execute an action in the real world

    def think(self, problem: dict) -> CognitionState:
        """
        Main entry point. Give it a problem, get back a full reasoning trace.

        The problem dict should have at minimum:
        - "description": what needs to be solved
        - Optional: "context", "constraints", "known_facts"
        """
        state = CognitionState(
            problem=problem,
            created_at=utc_now(),
        )

        while state.phase != CognitionPhase.COMPLETE and state.attempts < state.max_attempts:
            self._step(state)

        state.completed_at = utc_now()
        return state

    def _step(self, state: CognitionState) -> None:
        """Execute one phase of the cognition loop."""
        phase = state.phase

        if self.on_phase:
            self.on_phase(phase, f"Entering {phase.value}")

        if phase == CognitionPhase.PERCEIVE:
            self._perceive(state)
        elif phase == CognitionPhase.REMEMBER:
            self._remember(state)
        elif phase == CognitionPhase.TRANSFER:
            self._transfer(state)
        elif phase == CognitionPhase.INQUIRE:
            self._inquire(state)
        elif phase == CognitionPhase.HYPOTHESIZE:
            self._hypothesize(state)
        elif phase == CognitionPhase.EVALUATE:
            self._evaluate(state)
        elif phase == CognitionPhase.REVISE:
            self._revise(state)
        elif phase == CognitionPhase.ACT:
            self._act(state)
        elif phase == CognitionPhase.OBSERVE:
            self._observe(state)
        elif phase == CognitionPhase.LEARN:
            self._learn(state)
        elif phase == CognitionPhase.PAUSED:
            # Waiting for external input — don't advance
            pass

    # ── Phase 1: PERCEIVE ──

    def _perceive(self, state: CognitionState) -> None:
        """Decompose the problem into structural elements."""
        state.log("perceive", "Decomposing problem into structural elements")

        # Consult trait system if available
        if self.trait_system is not None:
            influence = self.trait_system.influence(state.problem)
            state.trait_influence = influence.to_dict()
            if influence.active_traits:
                state.log("perceive",
                          f"Traits activated: {influence.reasoning}",
                          state.trait_influence)
                # Persistence trait grants extra attempts
                if influence.persistence_modifier > 0:
                    state.max_attempts += influence.persistence_modifier

        # Use the problem as an episode-like structure
        episode = {
            "description": state.problem.get("description", ""),
            "actions": state.problem.get("steps", state.problem.get("actions", [])),
            "outcome": state.problem.get("expected_outcome", "unknown"),
        }

        state.decomposition = self.abstraction.decompose(episode)
        state.log("perceive", f"Found {len(state.decomposition)} structural elements",
                  {"elements": [e.to_dict() for e in state.decomposition]})

        state.phase = CognitionPhase.REMEMBER

    # ── Phase 2: REMEMBER ──

    def _remember(self, state: CognitionState) -> None:
        """Search for structurally similar past experiences."""
        state.log("remember", "Searching for similar past experiences")

        # Load past episodes from storage
        past_episodes = self.storage.load_json("episodes", default=[])

        if not past_episodes:
            state.log("remember", "No past experiences to compare against")
            state.phase = CognitionPhase.INQUIRE
            return

        # Find structurally similar experiences
        results = self.comparator.find_most_similar(state.problem, past_episodes)
        state.similar_experiences = [
            {"episode": ep, "comparison": comp.to_dict()}
            for ep, comp in results
            if comp.similarity > 0.3
        ]

        if state.similar_experiences:
            best = state.similar_experiences[0]
            state.log("remember", f"Found {len(state.similar_experiences)} similar experiences. "
                      f"Best match: {best['comparison']['similarity']:.0%} similar",
                      {"best_match": best})
            state.phase = CognitionPhase.TRANSFER
        else:
            state.log("remember", "No similar experiences found — this is novel")
            state.phase = CognitionPhase.INQUIRE

    # ── Phase 3: TRANSFER ──

    def _transfer(self, state: CognitionState) -> None:
        """Apply approach from similar experience to current problem."""
        state.log("transfer", "Attempting to transfer from past experience")

        if not state.similar_experiences:
            state.phase = CognitionPhase.INQUIRE
            return

        best_match = state.similar_experiences[0]["episode"]
        transfer_result = self.comparator.transfer(best_match, state.problem)

        if transfer_result.get("transfer_possible"):
            state.transfer_plan = transfer_result
            state.log("transfer", f"Transfer successful from '{transfer_result.get('source_domain', '?')}'",
                      transfer_result)
            state.phase = CognitionPhase.HYPOTHESIZE
        else:
            state.log("transfer", "Transfer not viable — structure too different")
            state.phase = CognitionPhase.INQUIRE

    # ── Phase 4: INQUIRE ──

    def _inquire(self, state: CognitionState) -> None:
        """Identify knowledge gaps before attempting a hypothesis."""
        state.log("inquire", "Checking for knowledge gaps")

        context = {
            "description": state.problem.get("description", ""),
            "known_entities": state.problem.get("known_entities", []),
            "referenced_entities": state.problem.get("referenced_entities", []),
            "task_confidence": 0.5 if not state.similar_experiences else 0.7,
        }

        gaps = self.inquiry.assess(context)
        state.knowledge_gaps = gaps

        if gaps:
            critical_gaps = [g for g in gaps if g.priority > 0.8]
            if critical_gaps and self.on_inquiry:
                # Ask for help with critical gaps
                state.log("inquire", f"Found {len(critical_gaps)} critical knowledge gaps — asking for help",
                          [g.to_dict() for g in critical_gaps])
                for gap in critical_gaps:
                    answer = self.on_inquiry(gap)
                    if answer is not None:
                        self.inquiry.resolve(gap.id, answer, "external")
                        state.log("inquire", f"Gap resolved: {gap.question}", {"answer": answer})
            else:
                state.log("inquire", f"Found {len(gaps)} knowledge gaps (non-critical — proceeding)",
                          [g.to_dict() for g in gaps])

        state.phase = CognitionPhase.HYPOTHESIZE

    # ── Phase 5: HYPOTHESIZE ──

    def _hypothesize(self, state: CognitionState) -> None:
        """Generate a candidate answer or approach."""
        state.log("hypothesize", "Generating candidate hypothesis")

        # Build hypothesis from available information
        approach = "unknown"
        answer = None

        if state.transfer_plan:
            # Use transferred approach
            approach = f"transfer from {state.transfer_plan.get('source_domain', 'past experience')}"
            answer = {
                "type": "transferred",
                "actions": state.transfer_plan.get("transferred_actions", []),
                "confidence_basis": "structural similarity with past experience",
            }
        elif self.on_hypothesis:
            # Use external hypothesis generator (e.g., LLM)
            approach = "external generation"
            answer = self.on_hypothesis(state.problem)
        else:
            # No transfer, no LLM — use causal model predictions
            description = state.problem.get("description", "")
            predictions = self.causal.predict_outcome(description)
            if predictions:
                approach = "causal prediction"
                answer = {
                    "type": "predicted",
                    "predictions": [{"effect": e, "probability": p, "confidence": c} for e, p, c in predictions],
                }
            else:
                approach = "no basis"
                answer = {"type": "uncertain", "note": "No past experience, no causal model, no external help"}

        # Record as a reasoning attempt
        assumptions = []
        if state.transfer_plan:
            assumptions.append(f"Past experience in '{state.transfer_plan.get('source_domain')}' applies here")
        if state.knowledge_gaps:
            assumptions.append(f"{len(state.knowledge_gaps)} knowledge gaps remain unresolved")

        reasoning_chain = [entry["message"] for entry in state.trace]

        attempt = self.evaluator.attempt(
            approach=approach,
            answer=answer,
            reasoning_chain=reasoning_chain,
            assumptions=assumptions,
        )

        state.hypothesis = {
            "attempt_id": attempt.id,
            "approach": approach,
            "answer": answer,
            "confidence": attempt.confidence,
        }

        # Apply trait confidence modifier if available
        if state.trait_influence and state.trait_influence.get("confidence_modifier"):
            attempt.confidence = max(0.0, min(1.0,
                attempt.confidence + state.trait_influence["confidence_modifier"]))

        state.log("hypothesize", f"Hypothesis generated via {approach} (confidence: {attempt.confidence:.0%})",
                  state.hypothesis)

        state.phase = CognitionPhase.EVALUATE

    # ── Phase 6: EVALUATE ──

    def _evaluate(self, state: CognitionState) -> None:
        """Self-evaluate the hypothesis."""
        state.log("evaluate", "Evaluating hypothesis")

        known_facts = state.problem.get("known_facts", [])
        attempt = self.evaluator.attempts[-1] if self.evaluator.attempts else None

        if not attempt:
            state.log("evaluate", "No attempt to evaluate")
            state.phase = CognitionPhase.COMPLETE
            return

        result = self.evaluator.evaluate(attempt, known_facts=known_facts)

        state.evaluation = result.to_dict()
        state.log("evaluate",
                  f"Evaluation: {'PASSED' if result.passed else 'FAILED'} — {len(result.issues)} issues",
                  result.to_dict())

        if result.passed:
            state.phase = CognitionPhase.ACT
        elif result.should_retry:
            state.phase = CognitionPhase.REVISE
        else:
            # Can't do better — go with what we have
            state.log("evaluate", "Cannot improve further — proceeding with best attempt")
            state.phase = CognitionPhase.ACT

    # ── Phase 7: REVISE ──

    def _revise(self, state: CognitionState) -> None:
        """Shift approach based on evaluation or human nudge."""
        state.attempts += 1
        state.log("revise", f"Revising approach (attempt {state.attempts})")

        if state.nudges:
            # Apply human nudge
            last_nudge = state.nudges[-1]
            strategy = self.evaluator.nudge(last_nudge)
            state.log("revise", f"Applying human nudge: '{last_nudge}' → {strategy}")
            state.nudges.pop()  # Consume the nudge
        elif state.evaluation:
            # Use evaluation hint
            hint = state.evaluation.get("retry_hint", "restructure")
            strategy = self.evaluator.nudge(hint)
            state.log("revise", f"Self-directed revision: {hint} → {strategy}")

        # Go back to hypothesize with new approach
        state.phase = CognitionPhase.HYPOTHESIZE

    # ── Phase 8: ACT ──

    def _act(self, state: CognitionState) -> None:
        """Execute the chosen approach."""
        state.log("act", "Executing hypothesis")

        if self.on_action and state.hypothesis:
            try:
                result = self.on_action(state.hypothesis)
                state.action_result = result
                state.log("act", "Action executed", result)
            except Exception as e:
                state.action_result = {"error": str(e), "success": False}
                state.log("act", f"Action failed: {e}")
        else:
            # No action callback — this is a pure reasoning problem
            state.action_result = {"type": "reasoning_only", "answer": state.hypothesis}
            state.log("act", "No action needed — reasoning problem")

        state.phase = CognitionPhase.OBSERVE

    # ── Phase 9: OBSERVE ──

    def _observe(self, state: CognitionState) -> None:
        """Record what happened."""
        state.log("observe", "Recording outcome")

        success = False
        if state.action_result:
            success = state.action_result.get("success", True)
            if "error" in state.action_result:
                success = False

        # Build episode for memory
        episode = {
            "id": f"ep_{state.created_at}",
            "description": state.problem.get("description", ""),
            "actions": state.trace,
            "outcome": "success" if success else "failure",
            "success": success,
            "hypothesis": state.hypothesis,
            "attempts": state.attempts,
            "nudges_received": len(state.nudges),
            "similar_experiences_found": len(state.similar_experiences),
            "knowledge_gaps": len(state.knowledge_gaps),
            "timestamp": utc_now(),
        }

        # Save to episodic memory
        episodes = self.storage.load_json("episodes", default=[])
        episodes.append(episode)
        # Keep last 1000 episodes
        if len(episodes) > 1000:
            episodes = episodes[-1000:]
        self.storage.save_json("episodes", episodes)

        # Update causal model
        self.causal.learn_from_episode(episode)

        # Update trait system with outcome
        if self.trait_system is not None and state.trait_influence:
            for trait_name in state.trait_influence.get("active_traits", []):
                trait = self.trait_system.get_trait(trait_name)
                if trait:
                    for ability in trait.abilities.values():
                        for skill in ability.skills.values():
                            self.trait_system.record_outcome(
                                trait_name, ability.name, skill.name, success)
            self.trait_system.tick()

        state.log("observe", f"Episode recorded (success={success})")
        state.phase = CognitionPhase.LEARN

    # ── Phase 10: LEARN ──

    def _learn(self, state: CognitionState) -> None:
        """Abstract the pattern from this experience."""
        state.log("learn", "Abstracting pattern from experience")

        # Try to form or strengthen an abstraction
        episodes = self.storage.load_json("episodes", default=[])

        if len(episodes) >= 2:
            # Compare this episode with recent ones
            recent = episodes[-5:]  # Last 5 episodes
            abstraction = self.abstraction.abstract(recent)
            if abstraction:
                state.learned = {
                    "abstraction_id": abstraction.id,
                    "name": abstraction.name,
                    "instance_count": abstraction.instance_count,
                    "confidence": abstraction.confidence,
                }
                state.log("learn", f"Pattern formed: {abstraction.name} "
                          f"(confidence: {abstraction.confidence:.0%}, instances: {abstraction.instance_count})",
                          state.learned)
            else:
                state.log("learn", "No new pattern detected")
        else:
            state.log("learn", "Not enough episodes yet for abstraction")

        state.phase = CognitionPhase.COMPLETE

    # ── External interface ──

    def nudge(self, state: CognitionState, direction: str) -> None:
        """
        Human nudge — shift how the system is thinking.

        This doesn't just change the next output.
        It changes the DECOMPOSITION — how the problem is being structured.
        """
        state.nudges.append(direction)
        state.log("nudge", f"Human nudge received: '{direction}'")

        # If we're past hypothesize, go back to revise
        if state.phase in (CognitionPhase.EVALUATE, CognitionPhase.ACT, CognitionPhase.COMPLETE):
            state.phase = CognitionPhase.REVISE
        # If we're still early, go back to perceive with new lens
        elif state.phase in (CognitionPhase.PERCEIVE, CognitionPhase.REMEMBER):
            state.phase = CognitionPhase.PERCEIVE  # Re-decompose with nudge

    def get_trace(self, state: CognitionState) -> list[dict]:
        """Get the full reasoning trace — every step the system took."""
        return state.trace

    def explain(self, state: CognitionState) -> str:
        """Explain the reasoning in plain language."""
        lines = []
        for entry in state.trace:
            lines.append(f"[{entry['phase']}] {entry['message']}")
        return "\n".join(lines)
