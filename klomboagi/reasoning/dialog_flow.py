"""
Dialog Flow — natural back-and-forth conversation that builds on itself.

Current system: Q&A. Human asks, system answers. No flow.

Real conversation:
  Human: "What is gravity?"
  System: "Gravity is a force that causes acceleration." ← knows this
  System: "What about gravity interests you?" ← FOLLOW-UP
  Human: "How does it work at the quantum level?"
  System: "That's actually unsolved — quantum gravity..." ← BUILDS ON CONTEXT
  System: "Want me to learn more about that?" ← OFFERS TO GROW

Dialog state tracks:
  - What we just talked about (immediate context)
  - What the human seems to be building toward (conversation arc)
  - What follow-up questions are natural
  - When to offer to go deeper vs move on
  - When the human is confused (repeat/rephrase signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class DialogState:
    """Full state of an ongoing conversation."""
    turn_count: int = 0
    current_topic: str = ""
    topic_history: list[str] = field(default_factory=list)
    depth_on_topic: int = 0          # how deep into current topic
    human_sentiment: str = "neutral"  # "curious", "confused", "satisfied", "frustrated"
    pending_followups: list[str] = field(default_factory=list)
    last_answer_type: str = ""       # "fact", "explanation", "unknown", "correction"
    conversation_arc: str = ""       # "exploring", "deep_diving", "teaching", "testing"


class DialogFlowEngine:
    """
    Manages natural conversation flow — not just Q&A.

    Generates follow-up questions, detects conversation patterns,
    offers to go deeper, notices when human is confused.
    """

    def __init__(self) -> None:
        self.state = DialogState()
        self._topic_transitions: list[tuple[str, str]] = []

    def update(self, human_msg: str, system_response: str,
               intent_type: str, topic: str) -> None:
        """Update dialog state after each exchange."""
        self.state.turn_count += 1

        # Track topic transitions
        if topic and topic != self.state.current_topic:
            if self.state.current_topic:
                self._topic_transitions.append(
                    (self.state.current_topic, topic))
            self.state.topic_history.append(topic)
            self.state.current_topic = topic
            self.state.depth_on_topic = 1
        else:
            self.state.depth_on_topic += 1

        # Detect human sentiment from message
        self.state.human_sentiment = self._detect_sentiment(human_msg)

        # Track answer type
        self.state.last_answer_type = self._classify_answer(system_response)

        # Detect conversation arc
        self.state.conversation_arc = self._detect_arc()

        # Generate follow-ups
        self.state.pending_followups = self._generate_followups(
            topic, intent_type, system_response)

    def get_followup(self) -> str | None:
        """Get a natural follow-up to append to the response."""
        if not self.state.pending_followups:
            return None

        # Pick based on sentiment
        if self.state.human_sentiment == "confused":
            return "Would you like me to explain that differently?"
        if self.state.human_sentiment == "curious" and self.state.depth_on_topic < 3:
            return self.state.pending_followups[0]
        if self.state.depth_on_topic >= 3:
            return "Want to explore something else, or go deeper?"

        # Default: first follow-up every 2 turns
        if self.state.turn_count % 2 == 0 and self.state.pending_followups:
            return self.state.pending_followups[0]

        return None

    def should_offer_depth(self) -> bool:
        """Should we offer to go deeper on the current topic?"""
        return (self.state.depth_on_topic >= 2
                and self.state.human_sentiment in ("curious", "neutral")
                and self.state.last_answer_type == "fact")

    def should_change_topic(self) -> bool:
        """Has the conversation stalled?"""
        return self.state.depth_on_topic > 5

    def _detect_sentiment(self, msg: str) -> str:
        """Detect human's emotional state from their words."""
        m = msg.lower()

        # Confusion signals
        if any(w in m for w in ["huh", "what do you mean", "confused",
                                "don't understand", "explain again", "what?"]):
            return "confused"

        # Curiosity signals
        if any(w in m for w in ["interesting", "tell me more", "how does",
                                "why does", "what about", "curious",
                                "go deeper", "elaborate"]):
            return "curious"

        # Satisfaction signals
        if any(w in m for w in ["thanks", "got it", "makes sense",
                                "good", "perfect", "ok cool", "nice"]):
            return "satisfied"

        # Frustration signals
        if any(w in m for w in ["wrong", "no", "that's not",
                                "stupid", "useless", "bad"]):
            return "frustrated"

        return "neutral"

    def _classify_answer(self, response: str) -> str:
        """Classify what type of answer we gave."""
        r = response.lower()
        if "don't know" in r or "can you teach" in r:
            return "unknown"
        if "because" in r or "means" in r or "therefore" in r:
            return "explanation"
        if "corrected" in r or "updating" in r:
            return "correction"
        return "fact"

    def _detect_arc(self) -> str:
        """What pattern is the conversation following?"""
        if self.state.depth_on_topic > 3:
            return "deep_diving"
        if len(set(self.state.topic_history[-5:])) > 3:
            return "exploring"
        if self.state.last_answer_type == "correction":
            return "teaching"
        return "conversing"

    def _generate_followups(self, topic: str, intent: str,
                           response: str) -> list[str]:
        """Generate natural follow-up questions."""
        followups = []

        if topic:
            followups.append(f"What else would you like to know about {topic}?")

            if intent == "teach":
                followups.append(f"How does {topic} connect to what I already know?")
            elif intent == "question":
                followups.append(f"Should I learn more about {topic}?")

        if "don't know" in response.lower():
            followups.insert(0, "Can you teach me about that?")

        return followups[:3]
