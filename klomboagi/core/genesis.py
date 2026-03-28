"""
Cognitive Genesis — bootstrapping cognition from zero.

This is the integration layer that wires together:
- Conversation interface (parsing, teaching, questioning)
- Trait system (personality that develops through use)
- Dialog context (multi-turn coherence)
- Surprise detection (contradictions trigger deeper learning)
- Proactive curiosity (system asks what IT wants to know)

The loop:
  hear → perceive → check beliefs → surprise? → learn → respond → curiosity check → ask

No LLM. No API. Pure algorithm + knowledge graph + traits + curiosity.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field

from klomboagi.interface.conversation import Baby, Memory
from klomboagi.core.traits import TraitSystem, Trait, Ability, Skill
from klomboagi.reasoning.truth import TruthValue, Belief, EvidenceStamp
from klomboagi.reasoning.cognition_loop import CognitionLoop, CognitionPhase
from klomboagi.reasoning.engine import ReasoningEngine
from klomboagi.core.curriculum import (
    get_curriculum, get_all_domains, curriculum_stats,
    get_relation_curriculum, get_all_relation_domains,
)
from klomboagi.core.relations import RelationStore, RelationType, INVERSE_RELATIONS
from klomboagi.reasoning.compute import ComputeEngine
from klomboagi.reasoning.activation import ActivationNetwork
from klomboagi.reasoning.hypothesis import HypothesisEngine
from klomboagi.reasoning.synthesizer import Synthesizer
from klomboagi.reasoning.self_test import SelfTester
from klomboagi.reasoning.working_mem import WorkingMemory
from klomboagi.reasoning.metacognition import MetacognitionEngine
from klomboagi.reasoning.focus import FocusEngine
from klomboagi.reasoning.learning_planner import LearningPlanner
from klomboagi.core.drive import LearningDrive
from klomboagi.reasoning.self_model import SelfModel
from klomboagi.reasoning.inner_state import InnerStateEngine
from klomboagi.reasoning.behavioral_loop import BehavioralLoop, BehaviorMode
from klomboagi.reasoning.counterfactual import CounterfactualEngine
from klomboagi.reasoning.nlu import NLU


@dataclass
class DialogContext:
    """
    Tracks what we're talking about RIGHT NOW.

    Without this, every utterance is independent. With it, the system
    can resolve "it", "that", track topic continuity, and know when
    the conversation shifts.
    """
    current_topic: str = ""
    previous_topic: str = ""
    entities_mentioned: list[str] = field(default_factory=list)
    turn_count: int = 0
    last_intent_type: str = ""
    topic_depth: int = 0              # how many turns on this topic
    pending_questions: list[str] = field(default_factory=list)  # system wants to ask

    def update(self, intent: dict, message: str) -> None:
        """Update context from a new turn."""
        self.turn_count += 1

        # Extract entities (nouns from the message)
        words = re.findall(r'\b([a-z]{3,})\b', message.lower())
        common = {"the", "and", "for", "are", "but", "not", "you", "all",
                  "can", "had", "was", "one", "has", "how", "its", "may",
                  "new", "now", "see", "way", "who", "did", "get", "let",
                  "say", "she", "too", "use", "with", "that", "this", "from",
                  "they", "been", "have", "many", "some", "them", "than",
                  "each", "make", "like", "long", "look", "come", "could",
                  "people", "into", "just", "about", "would", "there",
                  "their", "which", "very", "also", "more", "other",
                  "what", "tell", "know", "does"}
        entities = [w for w in words if w not in common]
        self.entities_mentioned = entities[:10]

        # Detect topic from intent
        new_topic = ""
        if intent["type"] == "teach":
            new_topic = intent.get("subject", "")
        elif intent["type"] == "question":
            new_topic = intent.get("query", "").split()[-1] if intent.get("query") else ""
        elif intent["type"] == "command":
            new_topic = intent.get("target", "")
        elif entities:
            new_topic = entities[0]

        if new_topic and new_topic != self.current_topic:
            self.previous_topic = self.current_topic
            self.current_topic = new_topic
            self.topic_depth = 1
        else:
            self.topic_depth += 1

        self.last_intent_type = intent["type"]

    def resolve_pronoun(self, message: str) -> str:
        """Replace 'it', 'that' with current topic if applicable."""
        if not self.current_topic:
            return message
        # Only replace standalone pronouns, not inside words
        msg = re.sub(r'\bit\b', self.current_topic, message, count=1)
        msg = re.sub(r'\bthat\b', self.current_topic, msg, count=1)
        return msg

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "previous_topic": self.previous_topic,
            "entities": self.entities_mentioned,
            "turn_count": self.turn_count,
            "topic_depth": self.topic_depth,
        }


@dataclass
class Surprise:
    """
    A contradiction between new input and existing belief.

    Surprises are the most valuable learning signal — they mean
    the system's model of the world was WRONG about something.
    """
    statement: str
    old_belief: str
    new_input: str
    old_confidence: float
    surprise_magnitude: float   # 0-1, how unexpected
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "old_belief": self.old_belief,
            "new_input": self.new_input,
            "old_confidence": self.old_confidence,
            "surprise_magnitude": self.surprise_magnitude,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class Genesis:
    """
    Cognitive Genesis — bootstraps cognition from zero.

    Extends the base conversation interface with:
    - Dialog context (multi-turn coherence, pronoun resolution)
    - Trait system (personality develops through use)
    - Surprise detection (contradictions trigger deeper learning)
    - Proactive curiosity (system asks what it wants to know)
    """

    def __init__(self, memory_path: str = "/tmp/klomboagi_genesis.json") -> None:
        # Base conversation system — already handles teaching, questions, commands
        self.base = Baby(memory_path=memory_path)

        # CognitionLoop — the full 10-phase reasoning engine
        self._init_cognition_loop()

        # Reasoning engine — for direct fact derivation
        self.reasoning_engine = ReasoningEngine()

        # Relation store — multi-directional reasoning
        self.relations = RelationStore()

        # Computation engine — can DO math, not just know about it
        self.compute = ComputeEngine()

        # Activation network — spreading activation like real neurons
        self.activation = ActivationNetwork(self.relations, self.base._beliefs)

        # Hypothesis engine — reason about unknowns
        self.hypothesizer = HypothesisEngine(self.relations, self.base._beliefs)

        # Explanation synthesizer — coherent paragraphs not fact dumps
        self.synthesizer = Synthesizer(self.relations, self.base._beliefs)

        # Self-tester — verify own beliefs
        self.self_tester = SelfTester(self.compute)

        # Working memory — short-term focus buffer (~7 items)
        self.working_memory = WorkingMemory()

        # Metacognition — thinking about thinking
        self.metacognition = MetacognitionEngine()

        # Focus engine — filter noise, find relevance
        self.focus = FocusEngine()

        # Learning planner — autonomous goal-directed learning
        self.planner = LearningPlanner()

        # Learning drive — persistent, never stops
        self.drive = LearningDrive(self)

        # Self-model — mathematical understanding of own existence
        self.self_model = SelfModel()

        # Inner state — mathematical emotions derived from cognitive metrics
        self.inner = InnerStateEngine()

        # Behavioral loop — inner state drives actual decisions
        self.behavior = BehavioralLoop()

        # Counterfactual engine — "what if X were different?"
        self.counterfactual = CounterfactualEngine(self.relations)

        # NLU — real language understanding beyond regex
        self.nlu = NLU()

        # Dialog context — multi-turn tracking
        self.context = DialogContext()

        # Trait system — personality that develops
        self.traits = TraitSystem()
        self._init_default_traits()

        # Surprise tracking
        self.surprises: list[Surprise] = []

        # Proactive curiosity queue — what the system WANTS to ask
        self.proactive_questions: list[str] = []

        # Metrics
        self.total_turns = 0
        self.total_surprises = 0
        self.total_proactive = 0
        self.deep_thinks = 0

        # Load saved state if it exists
        self.load_state()

    def _init_cognition_loop(self) -> None:
        """Wire the CognitionLoop with a lightweight mock storage."""
        from unittest.mock import MagicMock

        # Lightweight storage that satisfies CognitionLoop's interface
        # Uses in-memory dicts instead of the full file system
        self._cognition_data: dict[str, object] = {
            "episodes": [],
            "abstractions": [],
            "knowledge_gaps": [],
            "causal_graph": {"nodes": [], "edges": {}},
        }

        storage = MagicMock()
        storage.load_json = MagicMock(
            side_effect=lambda key, default=None:
                self._cognition_data.get(key, default if default is not None else [])
        )
        def _save(key, data):
            self._cognition_data[key] = data
        storage.save_json = MagicMock(side_effect=_save)
        storage.event_log = MagicMock()

        self.cognition = CognitionLoop(storage, trait_system=None)  # traits wired later

        # Wire inquiry callback — when the brain has a question, ask the human
        self.cognition.on_inquiry = self._handle_cognition_inquiry

    def _init_default_traits(self) -> None:
        """
        Seed minimal traits. These are TENDENCIES, not knowledge.
        They develop through use — start weak, grow with practice.
        """
        # Curiosity: the drive to learn
        curiosity = self.traits.add_trait(
            "curiosity", "drive to understand the unknown", 0.4,
            keywords=["unknown", "what", "why", "how", "explore", "learn", "new", "curious"],
        )
        investigate = Ability(name="investigate", description="look into unknowns")
        investigate.add_skill(Skill(name="search", description="search for information", proficiency=0.3))
        investigate.add_skill(Skill(name="ask_question", description="formulate good questions", proficiency=0.2))
        curiosity.add_ability(investigate)

        # Persistence: the drive to keep going
        persistence = self.traits.add_trait(
            "persistence", "drive to not give up", 0.3,
            keywords=["retry", "fail", "wrong", "again", "stuck", "hard", "difficult", "persist"],
        )
        retry = Ability(name="retry", description="try again with different approach")
        retry.add_skill(Skill(name="reframe", description="reframe the problem", proficiency=0.2))
        persistence.add_ability(retry)

        # Analysis: the drive to break things down
        analysis = self.traits.add_trait(
            "analysis", "drive to decompose and understand structure", 0.3,
            keywords=["because", "therefore", "means", "structure", "pattern", "compare", "analyze"],
        )
        decompose = Ability(name="decompose", description="break into parts")
        decompose.add_skill(Skill(name="find_parts", description="identify components", proficiency=0.2))
        decompose.add_skill(Skill(name="find_relations", description="identify relationships", proficiency=0.2))
        analysis.add_ability(decompose)

        # Accuracy: the drive to be correct
        accuracy = self.traits.add_trait(
            "accuracy", "drive to be right, not just fast", 0.3,
            keywords=["correct", "wrong", "actually", "precise", "exactly", "sure", "verify"],
        )
        verify = Ability(name="verify", description="check own understanding")
        verify.add_skill(Skill(name="self_check", description="check for contradictions", proficiency=0.2))
        accuracy.add_ability(verify)

    def hear(self, message: str) -> str:
        """
        Main entry point. Process a message through the full Genesis pipeline.

        Pipeline:
        1. Resolve pronouns from dialog context
        2. Parse intent (via base Baby)
        3. Check for surprises (contradictions with beliefs)
        4. Consult traits (which personality aspects activate?)
        5. Process through base conversation system
        6. Update dialog context
        7. Check proactive curiosity (anything we want to ask?)
        8. Return response
        """
        self.total_turns += 1

        # 0. Update working memory with this input
        self.working_memory.add_context(message)

        # 1. Resolve pronouns
        resolved_message = self.context.resolve_pronoun(message)

        # 2. Parse intent
        intent = self.base._parse_intent(resolved_message)

        # 3. Working memory — attend to mentioned concepts
        for word in resolved_message.lower().split():
            if len(word) > 3 and word not in self.base.COMMON_WORDS:
                self.working_memory.attend(word, "concept", "input")

        # 4. Check for surprise BEFORE learning
        surprise = self._check_surprise(intent)

        # 5. Consult traits
        trait_influence = self.traits.influence({
            "description": resolved_message,
            "known_entities": self.context.entities_mentioned,
        })

        # 6. Route: deep think for questions, base system for everything else
        if intent["type"] == "question":
            self.metacognition.record_question("knowledge")
            response = self._think_deep(resolved_message, intent)
        elif intent["type"] == "command" and intent.get("command") == "learn":
            response = self._active_learn(intent.get("target", ""))
        elif intent["type"] == "correction":
            self.metacognition.record_correction()
            self.inner.record_failure()  # correction = we were wrong
            response = self.base.hear(resolved_message)
        else:
            response = self.base.hear(resolved_message)
            self._extract_relations(resolved_message)

        # 6. Update dialog context
        self.context.update(intent, resolved_message)

        # 7. Handle surprise — append to response
        if surprise:
            self.total_surprises += 1
            response = self._handle_surprise(surprise, response)
            self.traits.record_outcome("accuracy", "verify", "self_check", True)
            self.inner.record_surprise(surprise.surprise_magnitude)

        # 8. Record trait outcome
        if trait_influence.active_traits:
            for t_name in trait_influence.active_traits:
                trait = self.traits.get_trait(t_name)
                if trait:
                    trait.strengthen(0.01)

        # 9. Check proactive curiosity — anything we want to ask?
        proactive = self._check_proactive_curiosity()
        if proactive:
            self.total_proactive += 1
            response += f"\n\nBy the way — {proactive}"

        # 10. Working memory tick — decay unused items
        self.working_memory.tick()

        # 11. Self-model snapshot — track own trajectory
        gaps = len([g for g in self.base.curiosity.gaps if not g.resolved])
        self.self_model.snapshot(
            self.base._beliefs, self.relations,
            self.base.memory.concepts, gaps)

        # 12. Inner state — compute how we "feel" based on real metrics
        self.inner.record_success()  # made it through a cycle
        self.inner.compute(
            beliefs_in_focus=len(self.working_memory.get_active_items()),
            active_gaps=gaps,
            total_beliefs=len(self.base._beliefs),
            working_memory_items=len(self.working_memory._items),
        )

        # 13. Auto-save state
        self.save_state()

        return response

    def _check_surprise(self, intent: dict) -> Surprise | None:
        """
        Check if new input contradicts an existing belief.

        Surprise = the system expected X but got NOT X.
        This is the most valuable learning signal.
        """
        if intent["type"] != "teach":
            return None

        subject = intent.get("subject", "")
        new_predicate = intent.get("predicate", "")

        if not subject or not new_predicate:
            return None

        # Check existing beliefs about this subject
        for statement, belief in self.base._beliefs.items():
            if belief.subject != subject:
                continue
            if belief.predicate == new_predicate:
                continue  # Same thing, not a surprise

            # Check if new predicate contradicts old predicate
            # Simple heuristic: if we already know "X is A" and now hear "X is B",
            # and A != B, that MIGHT be a surprise (or just additional info)
            #
            # It's a surprise if A and B are in the same category but different values
            # For now: flag if confidence is high and predicates look contradictory
            if self._predicates_conflict(belief.predicate, new_predicate):
                magnitude = belief.truth.confidence  # Higher confidence = bigger surprise
                if magnitude > 0.3:
                    return Surprise(
                        statement=f"{subject} is {new_predicate}",
                        old_belief=f"{subject} is {belief.predicate}",
                        new_input=f"{subject} is {new_predicate}",
                        old_confidence=belief.truth.confidence,
                        surprise_magnitude=magnitude,
                    )

        return None

    def _predicates_conflict(self, old: str, new: str) -> bool:
        """
        Do two predicates conflict?

        Uses the knowledge graph to detect conflicts:
        if both predicates share a parent category, they conflict.
        "green" and "red" both → "a color" → conflict.
        "reptile" and "mammal" both → "an animal class" → conflict.

        Falls back to hardcoded opposites for basic cases.
        """
        def strip_article(s: str) -> str:
            for art in ("a ", "an ", "the "):
                if s.startswith(art):
                    return s[len(art):]
            return s

        old_clean = strip_article(old.lower().strip())
        new_clean = strip_article(new.lower().strip())

        # Negation
        if new_clean.startswith("not ") and strip_article(new_clean[4:]) == old_clean:
            return True
        if old_clean.startswith("not ") and strip_article(old_clean[4:]) == new_clean:
            return True

        # Graph-based: check if both predicates share a parent category
        old_concept = self.base.graph.get(old_clean)
        new_concept = self.base.graph.get(new_clean)
        if old_concept and new_concept:
            old_parents = set(old_concept.is_a)
            new_parents = set(new_concept.is_a)
            shared = old_parents & new_parents
            if shared and old_clean != new_clean:
                return True

        # Also check beliefs for shared categories
        old_categories = set()
        new_categories = set()
        for statement, belief in self.base._beliefs.items():
            if belief.subject == old_clean:
                old_categories.add(belief.predicate)
            if belief.subject == new_clean:
                new_categories.add(belief.predicate)
        shared_beliefs = old_categories & new_categories
        if shared_beliefs and old_clean != new_clean:
            return True

        # Hardcoded opposites (minimal — these are logical, not domain-specific)
        opposites = [
            ("true", "false"), ("alive", "dead"), ("male", "female"),
            ("positive", "negative"), ("open", "closed"), ("on", "off"),
        ]
        for a, b in opposites:
            if (old_clean == a and new_clean == b) or (old_clean == b and new_clean == a):
                return True

        return False

    def _handle_surprise(self, surprise: Surprise, base_response: str) -> str:
        """
        Handle a surprise — this is where deep learning happens.

        When something contradicts what we believed, we need to:
        1. Flag it explicitly
        2. Weaken the old belief
        3. Strengthen the new one
        4. Record the surprise for learning
        """
        self.surprises.append(surprise)

        # Weaken the old belief
        old_statement = surprise.old_belief
        if old_statement in self.base._beliefs:
            old = self.base._beliefs[old_statement]
            old.truth.frequency = max(0.0, old.truth.frequency - 0.2)
            self.base.memory.beliefs[old_statement] = old.to_dict()

        surprise.resolved = True
        surprise.resolution = "revised belief based on new teaching"

        # Prepend surprise notification
        surprise_msg = (
            f"Wait — I thought {surprise.old_belief} "
            f"(confidence: {surprise.old_confidence:.0%}), "
            f"but you're telling me {surprise.new_input}. "
            f"Updating my understanding.\n\n"
        )

        return surprise_msg + base_response

    def _check_proactive_curiosity(self) -> str | None:
        """
        After responding, check if the system has its own questions.

        Proactive = the system INITIATES learning, not just reacts.
        This fires based on:
        1. Open knowledge gaps from curiosity driver
        2. Topic depth (deep conversation = more questions)
        3. Curiosity trait strength
        """
        curiosity_trait = self.traits.get_trait("curiosity")
        if not curiosity_trait or curiosity_trait.drive_strength < 0.3:
            return None

        # Only ask proactively every few turns, not every turn
        if self.context.turn_count % 3 != 0:
            return None

        # Check for open gaps related to current topic
        if self.context.current_topic:
            for gap in self.base.curiosity.gaps:
                if gap.resolved:
                    continue
                if (self.context.current_topic.lower() in gap.concept.lower() or
                        gap.concept.lower() in self.context.current_topic.lower()):
                    self.traits.record_outcome("curiosity", "investigate", "ask_question", True)
                    return f"I'm curious about '{gap.concept}'. Can you tell me more?"

        # Check general high-priority gaps
        next_gap = self.base.curiosity.get_next_gap()
        if next_gap and not next_gap.resolved:
            self.traits.record_outcome("curiosity", "investigate", "ask_question", True)
            return f"I've been wondering — what is '{next_gap.concept}'?"

        return None

    # ── Deep Thinking — Parallel Cognition ──

    def _think_deep(self, message: str, intent: dict) -> str:
        """
        Parallel cognition — ALL systems fire simultaneously, results merge.

        Like real neurons: every subsystem activates at once.
        The final response is the COMBINED output, weighted by confidence.

        Systems firing in parallel:
        1. Math engine (computation)
        2. Relation engine (structural connections)
        3. Activation network (spreading neural activation)
        4. Belief system (stored knowledge)
        5. Reasoning engine (logical derivation)
        6. CognitionLoop (pattern detection + transfer)
        """
        self.deep_thinks += 1
        query = intent.get("query", message)

        # ── Specialized question handlers (short-circuit) ──

        # Math
        math_result = self.compute.compute(query)
        if math_result.success and math_result.result != 0:
            steps = "\n  ".join(math_result.steps) if math_result.steps else ""
            return f"{math_result.result}\n  {steps}" if steps else str(math_result.result)

        # Connection questions
        connection = self._parse_connection_question(query)
        if connection:
            return self.relations.explain_connection(connection[0], connection[1])

        # Why questions
        why_result = self._parse_why_question(query)
        if why_result:
            return self._answer_why(why_result)

        # Relational questions
        relational = self._parse_relational_question(query)
        if relational:
            return self._answer_relational(relational)

        # Analogy questions
        analogy = self._parse_analogy_question(query)
        if analogy:
            return self._answer_analogy(analogy)

        # Counterfactual questions: "what if there were no gravity?"
        if query.lower().startswith("what if") or query.lower().startswith("without"):
            result = self.counterfactual.what_if(query)
            if result.total_affected > 0:
                return result.explain()

        # ── BEHAVIORAL DECISION — inner state drives approach ──

        decision = self.behavior.decide(
            self.inner.state, self.traits,
            self.working_memory, self.self_model)

        # BOREDOM → trigger exploration instead of answering
        if decision.mode == BehaviorMode.EXPLORE:
            self.inner.record_learning(0)
            explore_result = self._auto_explore()
            if explore_result:
                return f"[{decision.reason}]\n{explore_result}"

        # ASK HUMAN → admit uncertainty
        if decision.should_ask and decision.mode in (BehaviorMode.ASK_HUMAN, BehaviorMode.SEARCH_FIRST):
            prefix = f"I'm not confident enough to answer this well ({self.inner.state.confidence:.0%}). "
        else:
            prefix = ""

        # ── PARALLEL FIRE — all systems at once, shaped by behavior ──

        # System 0: FOCUS — filter noise, find what's relevant
        focus_result = self.focus.focus(query, self.base._beliefs,
                                        self.relations, self.working_memory,
                                        max_results=decision.chain_length)
        known_facts = focus_result.top_beliefs()
        relation_lines = [f"  {r}" for r in focus_result.top_relations()]
        query_words = set(focus_result.focus_concepts)

        # Update working memory with focus concepts
        for concept in focus_result.focus_concepts:
            self.working_memory.attend(concept, "focus", "attention")

        # SEARCH FIRST → go search before answering from beliefs
        if decision.mode == BehaviorMode.SEARCH_FIRST and not known_facts:
            search_result = self.base._curious_lookup(query)
            if search_result and "couldn't find" not in search_result.lower():
                return f"{prefix}{search_result}"

        # SWITCH APPROACH → try the question from a different angle
        if decision.mode == BehaviorMode.SWITCH_APPROACH:
            # Try activation-first approach instead of belief-first
            activation_result = self.activation.activate(list(query_words))
            if activation_result.convergence_points:
                alt_query = activation_result.convergence_points[0]
                alt_facts = []
                for stmt, belief in self.base._beliefs.items():
                    if hasattr(belief, 'subject') and belief.subject == alt_query:
                        alt_facts.append(stmt)
                if alt_facts:
                    known_facts = alt_facts[:decision.chain_length] + known_facts

        # System 3: ACTIVATION — spreading neural fire
        activation_result = self.activation.activate(list(query_words))

        # System 4: COGNITION — pattern detection with REAL structured actions
        cognition_actions = []
        for f in known_facts[:5]:
            b = self.base._beliefs.get(f)
            if b and hasattr(b, 'subject'):
                cognition_actions.append({
                    "type": "recall_belief",
                    "target": b.subject,
                    "result": b.predicate,
                })
        for r_line in relation_lines[:5]:
            cognition_actions.append({"type": "traverse_relation", "target": r_line.strip()})
        if activation_result.convergence_points:
            cognition_actions.append({
                "type": "neural_convergence",
                "target": activation_result.convergence_points[:3],
                "result": "convergence",
            })

        state = self.cognition.think({
            "description": query,
            "known_facts": known_facts[:20],
            "known_entities": list(query_words),
            "actions": cognition_actions,
            "expected_outcome": f"answer: {query}",
        })

        # System 5: REASONING — logical derivation (use max 5 facts to avoid noise)
        reasoning_result = ""
        if known_facts:
            try:
                chain = self.reasoning_engine.reason(query, known_facts[:5])
                if chain.confidence > 0.3 and chain.conclusion:
                    reasoning_result = chain.conclusion
            except Exception:
                pass

        # ── MERGE — combine all signals into unified response ──

        parts = []

        # Core answer (beliefs, sorted by relevance)
        if known_facts:
            parts.append("What I know:")
            for f in known_facts[:8]:
                b = self.base._beliefs.get(f)
                conf = f" ({b.truth.confidence:.0%})" if b else ""
                parts.append(f"  {f}{conf}")

        # Structural connections
        if relation_lines:
            unique_rels = list(dict.fromkeys(relation_lines))
            parts.append("\nConnections:")
            for line in unique_rels[:8]:
                parts.append(line)

        # Logical reasoning
        if reasoning_result:
            parts.append(f"\nReasoning: {reasoning_result}")

        # Neural associations — convergence points
        if activation_result.convergence_points:
            parts.append("\nNeural associations:")
            for cp in activation_result.convergence_points[:5]:
                node = next((n for n in activation_result.activated if n.name == cp), None)
                if node:
                    parts.append(f"  * {cp} (from: {', '.join(node.sources)})")
        elif activation_result.activated:
            top = activation_result.top(3)
            if top:
                parts.append("\nAssociated concepts:")
                for node in top:
                    parts.append(f"  ~ {node.name} ({node.activation:.2f})")

        # Pattern from CognitionLoop
        for entry in state.trace:
            if entry["phase"] == "transfer" and "successful" in entry.get("message", "").lower():
                parts.append(f"\nTransfer: {entry['message']}")
                break

        # 8. Synthesized explanation (if we have enough)
        if known_facts and relation_lines:
            for word in query_words:
                synth = self.synthesizer.explain(word)
                if synth and len(synth) > 20:
                    parts.insert(0, synth)
                    break

        # 9. If we don't know much — form a hypothesis
        if not known_facts and not relation_lines:
            hypothesis = self.hypothesizer.hypothesize(query, list(query_words))
            if hypothesis:
                return prefix + hypothesis.explain()
            # INVESTIGATE mode → search harder
            if decision.mode == BehaviorMode.INVESTIGATE:
                search_result = self.base._curious_lookup(query)
                if search_result:
                    return prefix + search_result
            return self.base.hear(message)

        return prefix + "\n".join(parts)

    def _parse_relational_question(self, query: str) -> dict | None:
        """
        Parse relational questions:
          "what causes acceleration?" → {relation: CAUSES, target: "acceleration", direction: "backward"}
          "what does gravity cause?" → {relation: CAUSES, source: "gravity", direction: "forward"}
          "what uses mathematics?" → {relation: USES, target: "mathematics", direction: "backward"}
          "what is energy part of?" → {relation: PART_OF, source: "energy", direction: "forward"}
        """
        q = query.lower().strip().rstrip("?").strip()

        # Pattern: "what causes X" / "what causes X"
        patterns = [
            # Backward: "what causes X?" → find things that cause X
            (r"what (?:causes|cause) (.+)", RelationType.CAUSES, "backward"),
            (r"what (?:requires|require) (.+)", RelationType.REQUIRES, "backward"),
            (r"what (?:enables|enable) (.+)", RelationType.ENABLES, "backward"),
            (r"what (?:uses|use) (.+)", RelationType.USES, "backward"),
            (r"what (?:measures|measure) (.+)", RelationType.MEASURES, "backward"),
            (r"what is (.+) part of", RelationType.PART_OF, "forward"),
            (r"what is (.+) opposite of", RelationType.OPPOSITE_OF, "forward"),
            (r"what is opposite of (.+)", RelationType.OPPOSITE_OF, "backward"),
            (r"what are the parts of (.+)", RelationType.PART_OF, "backward"),

            # Forward: "what does X cause?" → find things X causes
            (r"what does (.+) cause", RelationType.CAUSES, "forward"),
            (r"what does (.+) require", RelationType.REQUIRES, "forward"),
            (r"what does (.+) enable", RelationType.ENABLES, "forward"),
            (r"what does (.+) use", RelationType.USES, "forward"),
            (r"what does (.+) measure", RelationType.MEASURES, "forward"),
        ]

        for pattern, rel_type, direction in patterns:
            m = re.match(pattern, q)
            if m:
                concept = m.group(1).strip()
                return {"relation": rel_type, "concept": concept, "direction": direction}

        return None

    def _answer_relational(self, parsed: dict) -> str:
        """Answer a relational question using the relation store."""
        rel_type = parsed["relation"]
        concept = parsed["concept"]
        direction = parsed["direction"]

        if direction == "forward":
            rels = self.relations.get_forward(concept, rel_type)
            if not rels:
                # Try without article
                for prefix in ("a ", "an ", "the "):
                    rels = self.relations.get_forward(prefix + concept, rel_type)
                    if rels:
                        break
            if rels:
                lines = [f"{concept} {rel_type.value}:"]
                for r in rels:
                    lines.append(f"  → {r.target} ({r.confidence:.0%})"
                                + (" [inferred]" if r.derived else ""))
                return "\n".join(lines)
        else:  # backward
            rels = self.relations.get_backward(concept, rel_type)
            if not rels:
                for prefix in ("a ", "an ", "the "):
                    rels = self.relations.get_backward(prefix + concept, rel_type)
                    if rels:
                        break
            if rels:
                inverse = INVERSE_RELATIONS.get(rel_type, f"inverse of {rel_type.value}")
                lines = [f"{concept} {inverse}:"]
                for r in rels:
                    lines.append(f"  ← {r.source} ({r.confidence:.0%})"
                                + (" [inferred]" if r.derived else ""))
                return "\n".join(lines)

        # For symmetric relations (opposite_of, analogous_to), try the other direction too
        if rel_type in (RelationType.OPPOSITE_OF, RelationType.ANALOGOUS_TO):
            other_dir = self.relations.get_forward(concept, rel_type) if direction == "backward" else self.relations.get_backward(concept, rel_type)
            if not other_dir:
                for prefix in ("a ", "an ", "the "):
                    other_dir = self.relations.get_forward(prefix + concept, rel_type) if direction == "backward" else self.relations.get_backward(prefix + concept, rel_type)
                    if other_dir:
                        break
            if other_dir:
                lines = [f"{concept} {rel_type.value}:"]
                for r in other_dir:
                    other = r.target if r.source.endswith(concept) or r.source == concept else r.source
                    lines.append(f"  ↔ {other} ({r.confidence:.0%})")
                return "\n".join(lines)

        return f"I don't know what {rel_type.value} {concept}. Teach me?"

    def _handle_cognition_inquiry(self, gap) -> str | None:
        """
        Called when CognitionLoop has a knowledge gap.
        Try to fill it from beliefs or curiosity search.
        """
        concept = gap.question if hasattr(gap, 'question') else str(gap)
        concept_words = set(concept.lower().split()) - {"what", "is", "a", "an", "the", "are"}
        # Check existing beliefs — match on subject/predicate or keyword overlap
        for statement, belief in self.base._beliefs.items():
            stmt_words = set(statement.lower().split())
            if concept_words & stmt_words:
                return statement
        return None

    # ── Active Learning ──

    def _active_learn(self, topic: str) -> str:
        """
        'Learn about X' — triggers the full pipeline:
        1. Search multiple sources
        2. Extract structured concepts from raw text
        3. Store as beliefs with truth values
        4. Run CognitionLoop to connect to existing knowledge
        5. Identify follow-up gaps
        """
        if not topic:
            return "What should I learn about?"

        clean_topic = topic.strip()
        if clean_topic.startswith("about "):
            clean_topic = clean_topic[6:].strip()

        lines = [f"Learning about '{clean_topic}'..."]

        # 1. Search
        raw_info = self.base.searcher.search(clean_topic)
        if not raw_info or "Could not find" in raw_info:
            return f"I couldn't find information about '{clean_topic}'. Can you teach me?"

        lines.append(f"\nFound information ({len(raw_info)} chars).")

        # 2. Extract structured concepts
        extracted = self._extract_concepts(clean_topic, raw_info)
        if extracted:
            lines.append(f"\nExtracted {len(extracted)} facts:")
            for subj, pred in extracted[:10]:
                lines.append(f"  {subj} is {pred}")

                # 3. Store as beliefs
                statement = f"{subj} is {pred}"
                if statement not in self.base._beliefs:
                    self.base._evidence_counter += 1
                    belief = Belief(
                        statement=statement,
                        truth=TruthValue.from_single_observation(True),
                        stamp=EvidenceStamp.new(self.base._evidence_counter),
                        subject=subj,
                        predicate=pred,
                        source="discovery",
                    )
                    self.base._beliefs[statement] = belief
                    self.base.memory.beliefs[statement] = belief.to_dict()
                    self.base.graph.add(subj, is_a=[pred])

        # 3.5. Extract relations from the raw text too
        self._extract_relations(raw_info)

        # 4. Store raw discovery
        self.base._process_discovery(clean_topic, raw_info)

        # 5. Run CognitionLoop to connect to existing knowledge
        known_facts = [f"{s} is {p}" for s, p in extracted[:10]] if extracted else []
        problem = {
            "description": f"understand {clean_topic}",
            "known_facts": known_facts,
            "known_entities": [clean_topic],
        }
        state = self.cognition.think(problem)

        # Report what the brain figured out
        if state.learned:
            lines.append(f"\nPattern detected: {state.learned.get('name', 'unnamed')}")
        if state.similar_experiences:
            lines.append(f"Connected to {len(state.similar_experiences)} past experiences.")

        # 6. Identify follow-up gaps
        new_gaps = []
        for fact_subj, fact_pred in (extracted or [])[:5]:
            unknowns = self.base._find_unknowns_in(fact_pred)
            for u in unknowns:
                if u != clean_topic and u not in new_gaps:
                    new_gaps.append(u)
                    self.base.curiosity.notice_gap(u, context=f"discovered while learning about {clean_topic}")

        if new_gaps:
            lines.append(f"\nNew questions: {', '.join(new_gaps[:5])}")

        # Save memory
        self.base.memory.save(self.base.memory_path)

        self.traits.record_outcome("curiosity", "investigate", "search", True)
        self.inner.record_learning(len(extracted) if extracted else 0)
        return "\n".join(lines)

    # Words that should never be a subject in an extracted fact
    _BAD_SUBJECTS = frozenset({
        "it", "this", "that", "which", "they", "he", "she", "we", "these",
        "those", "there", "here", "what", "who", "where", "how", "such",
        "its", "their", "his", "her", "our", "one", "each", "both",
        "some", "many", "most", "all", "any", "other", "another",
        "above", "below", "however", "therefore", "thus", "hence",
    })

    def _extract_concepts(self, topic: str, raw_text: str) -> list[tuple[str, str]]:
        """
        Extract structured (subject, predicate) pairs from raw text.

        Quality filters:
        - Subject must be 1-5 words, not a pronoun
        - Predicate must be 5-80 chars, not start with junk
        - No duplicate subjects
        - First sentence of article = topic definition
        """
        concepts: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Clean the text
        text = re.sub(r'\[(?:Wikipedia|DuckDuckGo|Wikidata|Open Library)[^\]]*\]', '', raw_text)
        text = re.sub(r'\([^)]{0,100}\)', '', text)  # Remove parentheticals
        text = re.sub(r'\s+', ' ', text)

        def _valid_subject(s: str) -> bool:
            s = s.strip().lower()
            words = s.split()
            if not words or len(words) > 5:
                return False
            if words[0] in self._BAD_SUBJECTS:
                return False
            if len(s) < 2:
                return False
            return True

        def _valid_predicate(p: str) -> bool:
            p = p.strip().lower()
            if len(p) < 5 or len(p) > 80:
                return False
            # Must start with a real word
            if p[0].isdigit():
                return False
            return True

        def _add(subj: str, pred: str) -> None:
            subj = subj.strip().lower()
            pred = pred.strip().lower().rstrip(".")
            if not _valid_subject(subj) or not _valid_predicate(pred):
                return
            key = f"{subj}:{pred}"
            if key not in seen:
                seen.add(key)
                concepts.append((subj, pred))

        # Pattern 1: "X is/are Y" (strongest signal)
        for m in re.finditer(
            r'(?:^|[.;]\s+)(?:A |An |The )?([\w][\w\s]{0,30}?)\s+'
            r'(?:is|are)\s+(?:a |an |the )?([\w][\w\s,]{4,60}?)(?:[.]|$)',
            text, re.IGNORECASE | re.MULTILINE
        ):
            _add(m.group(1), m.group(2))

        # Pattern 2: "X, a type of/kind of/known as/also called Y"
        for m in re.finditer(
            r'([\w][\w\s]{1,25}?),\s+(?:a type of|a kind of|known as|also called|'
            r'also known as|referred to as|defined as)\s+([\w\s]{2,40})',
            text, re.IGNORECASE
        ):
            _add(m.group(1), m.group(2))

        # Pattern 3: "X belongs to / is part of / is a branch of Y"
        for m in re.finditer(
            r'([\w][\w\s]{1,25}?)\s+(?:belongs? to|is part of|is a branch of|'
            r'is a form of|is a subset of|is classified as)\s+([\w\s]{2,40})',
            text, re.IGNORECASE
        ):
            _add(m.group(1), m.group(2))

        # Pattern 4: "X contains / includes / consists of Y"
        for m in re.finditer(
            r'([\w][\w\s]{1,25}?)\s+(?:contains?|includes?|consists? of|'
            r'comprises?|encompasses?)\s+([\w\s]{2,60})',
            text, re.IGNORECASE
        ):
            _add(m.group(1), m.group(2))

        # Topic definition: first sentence often defines the topic
        if concepts and topic.lower() not in [c[0] for c in concepts]:
            first_pred = concepts[0][1]
            if first_pred:
                concepts.insert(0, (topic.lower(), first_pred))

        return concepts[:20]

    # ── Teaching Protocol ──

    def teach_domain(self, domain: str) -> str:
        """
        Teach an entire domain at machine speed.

        Feeds structured (subject, predicate) facts through the learning
        pipeline. Each fact becomes a belief with NARS truth values
        and a knowledge graph entry.

        Returns a summary of what was learned.
        """
        facts = get_curriculum(domain)
        if not facts:
            available = ", ".join(get_all_domains())
            return f"Unknown domain '{domain}'. Available: {available}"

        learned = 0
        for subject, predicate in facts:
            intent = {"subject": subject, "predicate": predicate, "raw": f"{subject} is {predicate}"}
            self.base._learn_from_teaching_quiet(intent)
            learned += 1

        # Run CognitionLoop to form patterns from the new knowledge
        known = [f"{s} is {p}" for s, p in facts[:20]]
        self.cognition.think({
            "description": f"understand {domain}",
            "known_facts": known,
            "known_entities": [domain],
        })

        self.base.memory.save(self.base.memory_path)

        return (
            f"Learned {learned} facts about {domain}.\n"
            f"Concepts: {len(self.base.memory.concepts)} | "
            f"Beliefs: {len(self.base._beliefs)}"
        )

    def teach_all(self) -> str:
        """Teach all available fact curricula."""
        results = []
        for domain in get_all_domains():
            result = self.teach_domain(domain)
            results.append(f"  {domain}: {result}")
        stats = curriculum_stats()
        return (
            f"Taught {stats['total_facts']} facts across {stats['domains']} domains:\n"
            + "\n".join(results)
        )

    def teach_relations(self, domain: str = "all") -> str:
        """
        Teach relations between concepts — causes, requires, part_of, uses, etc.

        This is what enables spherical reasoning: every concept connects
        to others in multiple directions, not just "is_a".
        """
        relation_type_map = {
            "is_a": RelationType.IS_A,
            "causes": RelationType.CAUSES,
            "requires": RelationType.REQUIRES,
            "part_of": RelationType.PART_OF,
            "uses": RelationType.USES,
            "opposite_of": RelationType.OPPOSITE_OF,
            "enables": RelationType.ENABLES,
            "measures": RelationType.MEASURES,
            "example_of": RelationType.EXAMPLE_OF,
            "analogous_to": RelationType.ANALOGOUS_TO,
        }

        domains_to_teach = get_all_relation_domains() if domain == "all" else [domain]
        total_taught = 0

        for d in domains_to_teach:
            triples = get_relation_curriculum(d)
            if not triples:
                continue
            for source, rel_str, target in triples:
                rel_type = relation_type_map.get(rel_str)
                if rel_type:
                    self.relations.add(source, rel_type, target, confidence=0.5, domain=d)
                    total_taught += 1

        # Run inference to derive new relations from what we taught
        inferred = self.relations.run_inference()

        stats = self.relations.stats()
        return (
            f"Taught {total_taught} relations. "
            f"Inferred {len(inferred)} new facts.\n"
            f"Total: {stats['total_relations']} relations across "
            f"{stats['unique_concepts']} concepts.\n"
            f"By type: {stats['by_type']}"
        )

    def teach_everything(self) -> str:
        """Teach ALL facts AND relations, then run inference."""
        lines = []

        # 1. Teach all fact curricula
        lines.append("Phase 1: Teaching facts...")
        lines.append(self.teach_all())

        # 2. Teach all relation curricula
        lines.append("\nPhase 2: Teaching relations...")
        lines.append(self.teach_relations("all"))

        # 3. Run inference again (now with more data)
        lines.append("\nPhase 3: Running global inference...")
        inferred = self.relations.run_inference()
        lines.append(f"Derived {len(inferred)} additional facts from cross-referencing.")

        # 4. Summary
        r_stats = self.relations.stats()
        lines.append(
            f"\nTotal knowledge: {len(self.base._beliefs)} beliefs, "
            f"{r_stats['total_relations']} relations, "
            f"{r_stats['unique_concepts']} connected concepts."
        )

        self.base.memory.save(self.base.memory_path)
        return "\n".join(lines)

    def what_connects(self, concept: str) -> str:
        """Show everything connected to a concept — all directions."""
        concept = concept.lower().strip()
        rels = self.relations.get_all_about(concept)
        if not rels:
            return f"No relations found for '{concept}'."

        lines = [f"Relations for '{concept}':"]
        for r in rels:
            direction = "→" if r.source == concept else "←"
            if r.source == concept:
                lines.append(f"  {direction} {r.relation.value} {r.target} ({r.confidence:.0%})")
            else:
                lines.append(f"  {direction} {r.source} {r.relation.value} this ({r.confidence:.0%})")
        return "\n".join(lines)

    # ── Document Learning ──

    def read_and_learn(self, source: str) -> str:
        """
        Read an entire document (URL, file, or Wikipedia topic) and learn from it.

        Pipeline:
        1. Read the document
        2. Split into sentences
        3. Extract (subject, predicate) facts
        4. Extract relations (causes, requires, part_of, uses)
        5. Store as beliefs + knowledge graph entries
        6. Run inference to derive new connections
        7. Report what was learned
        """
        # 1. Read — detect source type
        if source.startswith(("http://", "https://")):
            content = self.base.reader.read(source)
        elif source.startswith("wiki:"):
            # wiki:Quantum_mechanics → read full Wikipedia article
            topic = source[5:].strip()
            content = self.base.reader.read_wikipedia(topic)
        elif not source.startswith("/") and not source.startswith(".") and " " not in source:
            # Bare word like "Gravity" → try Wikipedia first, then file
            content = self.base.reader.read_wikipedia(source)
            if not content or len(content) < 50:
                content = self.base.reader.read(source)
        else:
            content = self.base.reader.read(source)

        if not content or len(content) < 50:
            return f"Could not read '{source}' or content too short."

        lines = [f"Reading document ({len(content)} chars)..."]

        # 2. Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        lines.append(f"Found {len(sentences)} sentences.")

        # 3. Extract facts
        facts_learned = 0
        relations_learned = 0

        for sentence in sentences[:100]:  # Cap at 100 sentences
            # Extract "X is Y" facts — case insensitive
            for m in re.finditer(
                r'(?:A |An |The )?(\b[\w][\w\s]{1,30}?\b)\s+'
                r'(?:is|are)\s+(?:a |an |the )?'
                r'([\w\s,]{2,60}?)(?:[.]|$)',
                sentence, re.IGNORECASE
            ):
                subj = m.group(1).strip().lower()
                pred = m.group(2).strip().lower()
                # Quality filter: subject max 5 words, predicate meaningful
                subj_words = subj.split()
                if (len(subj) > 2 and len(pred) > 5 and len(pred) <= 80
                        and len(subj_words) <= 5 and not pred.startswith(("same", "no ", "not "))):
                    statement = f"{subj} is {pred}"
                    if statement not in self.base._beliefs:
                        self.base._evidence_counter += 1
                        belief = Belief(
                            statement=statement,
                            truth=TruthValue.from_single_observation(True),
                            stamp=EvidenceStamp.new(self.base._evidence_counter),
                            subject=subj, predicate=pred, source="document",
                        )
                        self.base._beliefs[statement] = belief
                        self.base.memory.beliefs[statement] = belief.to_dict()
                        self.base.graph.add(subj, is_a=[pred])
                        if subj not in self.base.memory.concepts:
                            self.base.memory.concepts[subj] = {"facts": [], "taught_by": "document"}
                        self.base.memory.concepts[subj]["facts"].append(pred)
                        facts_learned += 1

            # 4. Extract relations
            self._extract_relations(sentence)

        # Count new relations
        r_stats_before = self.relations.stats()["total_relations"]

        # Also extract from the full text for cross-sentence patterns
        self._extract_relations(content[:5000])

        r_stats_after = self.relations.stats()["total_relations"]
        relations_learned = r_stats_after - r_stats_before

        # 5. Run inference
        inferred = self.relations.run_inference()

        # 6. Save
        self.base.memory.save(self.base.memory_path)
        self.save_state()

        lines.append(f"\nLearned:")
        lines.append(f"  Facts extracted: {facts_learned}")
        lines.append(f"  Relations extracted: {relations_learned}")
        lines.append(f"  Inferred: {len(inferred)} new connections")
        lines.append(f"  Total beliefs: {len(self.base._beliefs)}")
        lines.append(f"  Total relations: {self.relations.stats()['total_relations']}")

        # Show some of what we learned
        if facts_learned > 0:
            lines.append(f"\nSample facts:")
            recent = list(self.base._beliefs.items())[-min(facts_learned, 10):]
            for stmt, belief in recent:
                lines.append(f"  {stmt}")

        return "\n".join(lines)

    # ── Auto-Exploration (triggered by boredom) ──

    def _auto_explore(self) -> str | None:
        """
        Triggered when the system is bored — go learn something on its own.

        Picks the highest-priority knowledge gap and investigates it.
        """
        # Check learning priorities from metacognition
        priorities = self.metacognition.identify_learning_priorities(
            self.base._beliefs, self.relations)

        # Try curiosity gaps first
        next_gap = self.base.curiosity.get_next_gap()
        if next_gap and not next_gap.resolved:
            result = self.read_and_learn(next_gap.concept)
            if "Could not read" not in result:
                self.inner.record_learning(3)
                return f"I was bored, so I went and learned about '{next_gap.concept}'.\n{result}"

        # Try a topic from learning priorities
        for p in priorities:
            words = [w for w in p.lower().split() if len(w) > 4 and w not in {"study", "learn", "more", "currently"}]
            if words:
                topic = words[0]
                result = self.read_and_learn(topic)
                if "Could not read" not in result:
                    self.inner.record_learning(2)
                    return f"I decided to study '{topic}' because: {p}\n{result}"

        return None

    # ── Analogical Reasoning ──

    def _parse_analogy_question(self, query: str) -> dict | None:
        """Parse "X is to Y as A is to what?" style questions."""
        q = query.lower().strip().rstrip("?")
        patterns = [
            r"(.+?) is to (.+?) as (.+?) is to what",
            r"(.+?) is to (.+?) as (.+?) is to (.+)",  # verify analogy
            r"what is (.+?) to (.+?) as (.+?) to (.+)",
        ]
        for i, pattern in enumerate(patterns):
            m = re.match(pattern, q)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    return {"a": groups[0].strip(), "b": groups[1].strip(),
                            "c": groups[2].strip(), "d": None}
                elif len(groups) == 4:
                    return {"a": groups[0].strip(), "b": groups[1].strip(),
                            "c": groups[2].strip(), "d": groups[3].strip()}
        return None

    def _answer_analogy(self, analogy: dict) -> str:
        """
        Solve analogies by finding structural parallels in the relation graph.

        "addition is to subtraction as multiplication is to what?"
        → Find: addition --opposite_of--> subtraction
        → Apply same relation: multiplication --opposite_of--> ?
        → Answer: division
        """
        a, b, c = analogy["a"], analogy["b"], analogy["c"]
        d = analogy.get("d")

        # Find relations between A and B
        ab_relations = []
        for r in self.relations.get_forward(a):
            if r.target == b:
                ab_relations.append(r)
        for r in self.relations.get_backward(a):
            if r.source == b:
                ab_relations.append(r)
        # Also try with articles
        if not ab_relations:
            for prefix in ("a ", "an ", "the "):
                for r in self.relations.get_forward(prefix + a):
                    if r.target == b or r.target == prefix + b:
                        ab_relations.append(r)

        if not ab_relations:
            return (f"I don't know the relationship between '{a}' and '{b}'. "
                    f"Teach me how they're related?")

        # Apply the same relation(s) to C to find D
        candidates = []
        for ab_rel in ab_relations:
            # Look for C having the same relation type
            c_rels = self.relations.get_forward(c, ab_rel.relation)
            if not c_rels:
                for prefix in ("a ", "an ", "the "):
                    c_rels = self.relations.get_forward(prefix + c, ab_rel.relation)
                    if c_rels:
                        break
            for cr in c_rels:
                candidates.append((cr.target, ab_rel.relation.value, cr.confidence))

        if d:
            # Verify mode
            match = any(cand[0] == d or cand[0].endswith(d) for cand in candidates)
            if match:
                rel = candidates[0][1] if candidates else "related"
                return f"Yes! {a} is to {b} as {c} is to {d} (both {rel})"
            elif candidates:
                best = candidates[0]
                return (f"Not quite. {a} {ab_relations[0].relation.value} {b}, "
                        f"so {c} {best[1]} {best[0]}, not {d}")
            return f"I can't verify this analogy — I don't know enough about {c}."

        if candidates:
            best = candidates[0]
            rel = ab_relations[0].relation.value
            return (f"{a} is to {b} as {c} is to {best[0]}\n"
                    f"  ({a} {rel} {b}, so {c} {rel} {best[0]})")

        return f"I know {a} {ab_relations[0].relation.value} {b}, but I don't know what {c} {ab_relations[0].relation.value}."

    # ── Evidence Accumulation ──

    def strengthen_belief(self, statement: str, source: str = "confirmation") -> None:
        """
        Strengthen a belief when multiple sources agree.

        If the system discovers something through search AND the human
        confirms it, the belief should be stronger than either alone.
        """
        if statement in self.base._beliefs:
            belief = self.base._beliefs[statement]
            self.base._evidence_counter += 1
            confirming = Belief(
                statement=statement,
                truth=TruthValue.from_single_observation(True),
                stamp=EvidenceStamp.new(self.base._evidence_counter),
                subject=belief.subject,
                predicate=belief.predicate,
                source=source,
            )
            revised = belief.revise_with(confirming)
            if revised:
                self.base._beliefs[statement] = revised
                self.base.memory.beliefs[statement] = revised.to_dict()

    # ── Connection & Why Questions ──

    def _parse_connection_question(self, query: str) -> tuple[str, str] | None:
        """Parse "how does X connect to Y?" / "how are X and Y related?" """
        q = query.lower().strip().rstrip("?")
        patterns = [
            r"how does (.+?) connect to (.+)",
            r"how is (.+?) connected to (.+)",
            r"how are (.+?) and (.+?) (?:connected|related|linked)",
            r"what connects (.+?) (?:to|and|with) (.+)",
            r"is (.+?) related to (.+)",
            r"does (.+?) connect to (.+)",
            r"link between (.+?) and (.+)",
        ]
        for pattern in patterns:
            m = re.match(pattern, q)
            if m:
                return (m.group(1).strip(), m.group(2).strip())
        return None

    def _parse_why_question(self, query: str) -> str | None:
        """Parse "why does X happen?" → trace causal chain backward."""
        q = query.lower().strip().rstrip("?")
        patterns = [
            r"why does (.+?) happen",
            r"why is there (.+)",
            r"why does (.+?) occur",
            r"why (.+)",
        ]
        for pattern in patterns:
            m = re.match(pattern, q)
            if m:
                return m.group(1).strip()
        return None

    def _answer_why(self, concept: str) -> str:
        """
        Answer "why" by tracing causal chains backward.

        "Why does acceleration happen?" → because force causes acceleration,
        and gravity causes force.
        """
        # Find everything that causes this concept
        causes = self.relations.get_backward(concept, RelationType.CAUSES)
        if not causes:
            # Try with prefixes
            for prefix in ("a ", "an ", "the "):
                causes = self.relations.get_backward(prefix + concept, RelationType.CAUSES)
                if causes:
                    break

        if not causes:
            return f"I don't know why {concept} happens. Teach me what causes it?"

        lines = [f"Why {concept}? Because:"]
        for r in causes:
            lines.append(f"  {r.source} causes {r.target} ({r.confidence:.0%})")
            # Trace one more level back
            deeper = self.relations.get_backward(r.source, RelationType.CAUSES)
            for d in deeper[:2]:
                lines.append(f"    ← {d.source} causes {d.target} ({d.confidence:.0%})")

        return "\n".join(lines)

    # ── Auto-Relation Extraction ──

    def _extract_relations(self, message: str) -> None:
        """
        Auto-extract relations from natural language using NLU.

        Parses sentence structure to find:
        "gravity causes acceleration" → relation(gravity, CAUSES, acceleration)
        "because heat increases, molecules move faster" → causal relation
        "the dog, which is a mammal, has fur" → relative clause extraction
        """
        rel_type_map = {
            "causes": RelationType.CAUSES,
            "requires": RelationType.REQUIRES,
            "part_of": RelationType.PART_OF,
            "uses": RelationType.USES,
            "enables": RelationType.ENABLES,
            "measures": RelationType.MEASURES,
        }

        # Use NLU to extract structured relations
        nlu_relations = self.nlu.extract_relations(message)
        for source, rel_str, target in nlu_relations:
            rel_type = rel_type_map.get(rel_str)
            if rel_type and len(source) > 1 and len(target) > 1:
                self.relations.add(source, rel_type, target,
                                  confidence=0.5, domain="conversation")

        # Also extract facts as beliefs
        nlu_facts = self.nlu.extract_facts(message)
        for subj, pred in nlu_facts:
            if len(subj) > 1 and len(pred) > 3:
                statement = f"{subj} is {pred}"
                if statement not in self.base._beliefs:
                    self.base._evidence_counter += 1
                    belief = Belief(
                        statement=statement,
                        truth=TruthValue.from_single_observation(True),
                        stamp=EvidenceStamp.new(self.base._evidence_counter),
                        subject=subj, predicate=pred, source="nlu",
                    )
                    self.base._beliefs[statement] = belief
                    self.base.memory.beliefs[statement] = belief.to_dict()
                    self.base.graph.add(subj, is_a=[pred])

    # ── State Persistence ──

    def save_state(self) -> None:
        """
        Save the full Genesis state — traits, relations, cognition data.

        The base Baby already saves beliefs/concepts. This saves everything else
        so personality and relations persist across sessions.
        """
        import json
        from pathlib import Path

        state_path = Path(self.base.memory_path).parent / "genesis_state.json"

        # Serialize traits
        trait_data = {}
        for name, trait in self.traits.traits.items():
            trait_data[name] = trait.to_dict()

        # Serialize relations
        relation_data = [r.to_dict() for r in self.relations._all]

        state = {
            "traits": trait_data,
            "relations": relation_data,
            "cognition_episodes": self._cognition_data.get("episodes", []),
            "metrics": {
                "total_turns": self.total_turns,
                "deep_thinks": self.deep_thinks,
                "total_surprises": self.total_surprises,
                "total_proactive": self.total_proactive,
            },
        }

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str))

    def load_state(self) -> bool:
        """
        Load saved Genesis state. Returns True if state was loaded.
        """
        import json
        from pathlib import Path

        state_path = Path(self.base.memory_path).parent / "genesis_state.json"
        if not state_path.exists():
            return False

        try:
            state = json.loads(state_path.read_text())

            # Restore traits
            for name, data in state.get("traits", {}).items():
                if name in self.traits.traits:
                    self.traits.traits[name].drive_strength = data.get("drive_strength", 0.5)
                    self.traits.traits[name].activation_count = data.get("activation_count", 0)

            # Restore relations
            rel_type_map = {rt.value: rt for rt in RelationType}
            for r_data in state.get("relations", []):
                rel_type = rel_type_map.get(r_data.get("relation"))
                if rel_type:
                    self.relations.add(
                        r_data["source"], rel_type, r_data["target"],
                        confidence=r_data.get("confidence", 0.5),
                        domain=r_data.get("source_domain", "loaded"),
                    )

            # Restore cognition episodes
            self._cognition_data["episodes"] = state.get("cognition_episodes", [])

            # Restore metrics
            metrics = state.get("metrics", {})
            self.total_turns = metrics.get("total_turns", 0)
            self.deep_thinks = metrics.get("deep_thinks", 0)
            self.total_surprises = metrics.get("total_surprises", 0)
            self.total_proactive = metrics.get("total_proactive", 0)

            return True
        except Exception:
            return False

    def status(self) -> str:
        """Full system status."""
        base_status = self.base._status()
        personality = self.traits.personality_vector()
        trait_stats = self.traits.stats()

        lines = [
            base_status,
            "",
            "Cognitive Genesis Status:",
            f"  Total turns: {self.total_turns}",
            f"  Deep thinks: {self.deep_thinks}",
            f"  Cognition episodes: {len(self._cognition_data.get('episodes', []))}",
            f"  Surprises detected: {self.total_surprises}",
            f"  Proactive questions: {self.total_proactive}",
            f"  Dialog topic: {self.context.current_topic or '(none)'}",
            f"  Topic depth: {self.context.topic_depth} turns",
            "",
            "Personality:",
        ]
        for trait_name, strength in personality.items():
            bar = "█" * int(strength * 20) + "░" * (20 - int(strength * 20))
            lines.append(f"  {trait_name:15s} [{bar}] {strength:.0%}")

        lines.append(f"\n  Active traits: {trait_stats['active_traits']}")
        lines.append(f"  Total skills: {trait_stats['total_skills']}")

        # Relation stats
        r_stats = self.relations.stats()
        if r_stats["total_relations"] > 0:
            lines.append(f"\nRelation Graph:")
            lines.append(f"  Total relations: {r_stats['total_relations']}")
            lines.append(f"  Connected concepts: {r_stats['unique_concepts']}")
            lines.append(f"  Taught: {r_stats['taught']} | Inferred: {r_stats['derived']}")
            for rtype, count in sorted(r_stats["by_type"].items(), key=lambda x: -x[1]):
                lines.append(f"    {rtype:15s} {count}")

        return "\n".join(lines)
