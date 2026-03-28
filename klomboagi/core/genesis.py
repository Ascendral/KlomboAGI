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

        # 5. Process through base system
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
