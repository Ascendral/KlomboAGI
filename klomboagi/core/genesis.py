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

        # 1. Resolve pronouns
        resolved_message = self.context.resolve_pronoun(message)

        # 2. Parse intent
        intent = self.base._parse_intent(resolved_message)

        # 3. Check for surprise BEFORE learning
        surprise = self._check_surprise(intent)

        # 4. Consult traits
        trait_influence = self.traits.influence({
            "description": resolved_message,
            "known_entities": self.context.entities_mentioned,
        })

        # 5. Route: deep think for questions, base system for everything else
        if intent["type"] == "question":
            response = self._think_deep(resolved_message, intent)
        elif intent["type"] == "command" and intent.get("command") == "learn":
            response = self._active_learn(intent.get("target", ""))
        else:
            response = self.base.hear(resolved_message)

        # 6. Update dialog context
        self.context.update(intent, resolved_message)

        # 7. Handle surprise — append to response
        if surprise:
            self.total_surprises += 1
            response = self._handle_surprise(surprise, response)
            self.traits.record_outcome("accuracy", "verify", "self_check", True)

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

        "green" and "red" conflict (both colors).
        "green" and "large" don't (different properties).
        "reptile" and "mammal" conflict (both animal classes).
        """
        # Negation patterns — strip articles for comparison
        def strip_article(s: str) -> str:
            for art in ("a ", "an ", "the "):
                if s.startswith(art):
                    return s[len(art):]
            return s

        old_clean = strip_article(old.lower().strip())
        new_clean = strip_article(new.lower().strip())

        if new_clean.startswith("not ") and strip_article(new_clean[4:]) == old_clean:
            return True
        if old_clean.startswith("not ") and strip_article(old_clean[4:]) == new_clean:
            return True

        # Known conflicting categories (bootstrap knowledge)
        conflict_groups = [
            {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"},
            {"hot", "cold", "warm", "cool", "freezing", "boiling"},
            {"big", "small", "large", "tiny", "huge", "little"},
            {"fast", "slow", "quick"},
            {"mammal", "reptile", "bird", "fish", "insect", "amphibian", "arachnid"},
            {"solid", "liquid", "gas", "plasma"},
            {"true", "false"},
            {"alive", "dead"},
            {"male", "female"},
        ]
        old_lower = old.lower().strip()
        new_lower = new.lower().strip()
        for group in conflict_groups:
            if old_lower in group and new_lower in group and old_lower != new_lower:
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

    # ── Deep Thinking ──

    def _think_deep(self, message: str, intent: dict) -> str:
        """
        Fire the full CognitionLoop for a question.

        Instead of simple belief lookup, runs the 10-phase pipeline:
        perceive → remember → transfer → inquire → hypothesize → evaluate → act → observe → learn

        Falls back to base system if CognitionLoop adds nothing.
        """
        self.deep_thinks += 1
        query = intent.get("query", message)

        # Gather known facts about the topic
        known_facts = []
        query_words = set(query.lower().split())
        for statement, belief in self.base._beliefs.items():
            stmt_words = set(statement.lower().split())
            if stmt_words & query_words:
                known_facts.append(statement)

        # Build problem for CognitionLoop
        problem = {
            "description": query,
            "known_facts": known_facts[:20],
            "known_entities": self.context.entities_mentioned,
            "referenced_entities": [w for w in query_words if len(w) > 2],
            "expected_outcome": f"answer to: {query}",
        }

        # Run CognitionLoop
        state = self.cognition.think(problem)

        # Extract insights from the trace
        insights = []
        for entry in state.trace:
            if entry["phase"] == "learn" and entry.get("data"):
                insights.append(entry["message"])
            if entry["phase"] == "transfer" and "successful" in entry.get("message", "").lower():
                insights.append(f"Transfer: {entry['message']}")

        # Also try ReasoningEngine for dimensional/property questions
        reasoning_result = ""
        if known_facts:
            try:
                chain = self.reasoning_engine.reason(query, known_facts[:10])
                if chain.confidence > 0.3:
                    reasoning_result = f"\nReasoning: {chain.conclusion} (confidence: {chain.confidence:.0%})"
                    if chain.frameworks_used:
                        reasoning_result += f" [frameworks: {', '.join(chain.frameworks_used)}]"
            except Exception:
                pass

        # Get base response (belief lookup + curiosity search)
        base_response = self.base.hear(message)

        # Combine if CognitionLoop added value
        if insights or reasoning_result:
            parts = [base_response]
            if reasoning_result:
                parts.append(reasoning_result)
            if insights:
                parts.append("\nDeep analysis:")
                for insight in insights[:3]:
                    parts.append(f"  {insight}")
            # Record to cognition episodes for future transfer
            parts.append(f"\n[CognitionLoop: {len(state.trace)} steps, "
                        f"{state.attempts} attempts, "
                        f"{'transferred' if state.transfer_plan else 'novel'}]")
            return "\n".join(parts)

        return base_response

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
        return "\n".join(lines)

    def _extract_concepts(self, topic: str, raw_text: str) -> list[tuple[str, str]]:
        """
        Extract structured (subject, predicate) pairs from raw text.

        Parses natural language into knowledge graph entries.
        This is concept OWNERSHIP — the system forms its own understanding.
        """
        concepts: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Clean the text — remove source tags
        text = re.sub(r'\[(?:Wikipedia|DuckDuckGo|Wikidata|Open Library)[^\]]*\]', '', raw_text)

        # Pattern 1: "X is/are Y" statements
        for m in re.finditer(
            r'(?:^|[.;])\s*(?:A |An |The )?(\b[A-Z][\w\s]{1,30}?\b)\s+'
            r'(?:is|are)\s+'
            r'(?:a |an |the )?'
            r'([\w\s]{2,40}?)(?:[.,;]|$)',
            text, re.MULTILINE
        ):
            subj = m.group(1).strip().lower()
            pred = m.group(2).strip().lower()
            key = f"{subj}:{pred}"
            if key not in seen and len(subj) > 1 and len(pred) > 1:
                seen.add(key)
                concepts.append((subj, pred))

        # Pattern 2: "X, a type of Y" or "X, known as Y"
        for m in re.finditer(
            r'(\b[\w\s]{2,25}?\b),\s+(?:a type of|a kind of|known as|also called)\s+([\w\s]{2,30})',
            text, re.IGNORECASE
        ):
            subj = m.group(1).strip().lower()
            pred = m.group(2).strip().lower()
            key = f"{subj}:{pred}"
            if key not in seen:
                seen.add(key)
                concepts.append((subj, pred))

        # Pattern 3: "X belong(s) to Y" or "X are members of Y"
        for m in re.finditer(
            r'(\b[\w\s]{2,25}?\b)\s+(?:belongs? to|are members of|is part of)\s+([\w\s]{2,30})',
            text, re.IGNORECASE
        ):
            subj = m.group(1).strip().lower()
            pred = m.group(2).strip().lower()
            key = f"{subj}:{pred}"
            if key not in seen:
                seen.add(key)
                concepts.append((subj, pred))

        # Always add the topic itself with any extracted description
        if concepts:
            # The first sentence often defines the topic
            first_pred = concepts[0][1] if concepts else ""
            if first_pred and (topic.lower(), first_pred) not in concepts:
                concepts.insert(0, (topic.lower(), first_pred))

        return concepts[:15]  # Cap at 15 facts per discovery

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

        return "\n".join(lines)
