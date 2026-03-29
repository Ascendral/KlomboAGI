"""
Emotional Intelligence — read the human's emotional state from their words.

Not just "detect keywords." Understand the EMOTIONAL CONTEXT.

"I don't understand this at all" → frustrated, needs simpler explanation
"This is amazing!" → excited, wants to go deeper
"Whatever" → disengaged, needs to reconnect
"I'm worried about..." → anxious, needs reassurance
"Can you explain like I'm five?" → confused, wants basics

The system should ADAPT its response based on the human's state:
  - Frustrated → simplify, be patient, offer help
  - Curious → go deeper, provide more detail
  - Confused → rephrase, use analogies, check understanding
  - Excited → match energy, explore further
  - Skeptical → provide evidence, be precise
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HumanEmotion(Enum):
    CURIOUS = "curious"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    SKEPTICAL = "skeptical"
    SAD = "sad"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"
    GRATEFUL = "grateful"
    IMPATIENT = "impatient"
    PLAYFUL = "playful"


@dataclass
class EmotionalReading:
    """Assessment of the human's emotional state."""
    primary: HumanEmotion
    confidence: float
    signals: list[str]           # what words/patterns indicated this
    suggested_approach: str      # how to respond
    tone: str                    # "warm", "precise", "simple", "enthusiastic"

    def to_dict(self) -> dict:
        return {
            "emotion": self.primary.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals[:3],
            "approach": self.suggested_approach,
            "tone": self.tone,
        }


class EmotionalIntelligence:
    """
    Reads the human's emotional state and adapts responses.
    """

    # Emotion indicators — words/patterns that signal emotional state
    INDICATORS: dict[HumanEmotion, list[str]] = {
        HumanEmotion.CURIOUS: [
            "interesting", "tell me more", "how does", "why does",
            "what about", "curious", "fascinating", "go deeper",
            "elaborate", "explain more", "what else", "really?",
        ],
        HumanEmotion.CONFUSED: [
            "huh", "what do you mean", "confused", "don't understand",
            "explain again", "what?", "i don't get", "lost me",
            "explain like", "simpler", "too complex", "over my head",
        ],
        HumanEmotion.FRUSTRATED: [
            "wrong", "that's not right", "no!", "stupid", "useless",
            "terrible", "awful", "you're not helping", "still wrong",
            "again?", "come on", "seriously?", "ugh",
        ],
        HumanEmotion.EXCITED: [
            "amazing", "awesome", "incredible", "wow", "love it",
            "brilliant", "perfect", "exactly", "that's it", "mind blown",
            "oh my god", "holy", "yes!",
        ],
        HumanEmotion.SKEPTICAL: [
            "are you sure", "prove it", "source?", "how do you know",
            "doubt", "seems wrong", "citation", "evidence",
            "i don't think so", "really?", "that can't be right",
        ],
        HumanEmotion.SAD: [
            "sad", "depressed", "lost", "alone", "miss", "grief",
            "painful", "hurts", "hopeless", "gave up",
        ],
        HumanEmotion.ANXIOUS: [
            "worried", "scared", "afraid", "nervous", "anxious",
            "what if", "concerned", "risky", "dangerous",
        ],
        HumanEmotion.GRATEFUL: [
            "thank you", "thanks", "appreciate", "helpful",
            "grateful", "that helped", "you're great",
        ],
        HumanEmotion.IMPATIENT: [
            "hurry", "faster", "just tell me", "skip", "bottom line",
            "tldr", "get to the point", "too long", "short answer",
        ],
        HumanEmotion.PLAYFUL: [
            "haha", "lol", "funny", "joke", "play", "game",
            "fun", "silly", "let's try", "what if we",
        ],
    }

    # How to respond to each emotion
    RESPONSE_APPROACHES: dict[HumanEmotion, tuple[str, str]] = {
        HumanEmotion.CURIOUS: ("Go deeper, provide detail and connections.", "enthusiastic"),
        HumanEmotion.CONFUSED: ("Simplify. Use analogies. Check understanding.", "warm"),
        HumanEmotion.FRUSTRATED: ("Be patient. Acknowledge the frustration. Try different approach.", "calm"),
        HumanEmotion.EXCITED: ("Match energy. Explore further. Build on momentum.", "enthusiastic"),
        HumanEmotion.SKEPTICAL: ("Provide evidence. Be precise. Show reasoning chain.", "precise"),
        HumanEmotion.SAD: ("Be empathetic. Don't try to fix. Acknowledge the feeling.", "warm"),
        HumanEmotion.ANXIOUS: ("Be reassuring. Provide facts. Reduce uncertainty.", "calm"),
        HumanEmotion.NEUTRAL: ("Be clear and informative.", "balanced"),
        HumanEmotion.GRATEFUL: ("Accept graciously. Offer to continue.", "warm"),
        HumanEmotion.IMPATIENT: ("Be concise. Get to the point. No filler.", "direct"),
        HumanEmotion.PLAYFUL: ("Be light. Match playfulness. Keep it fun.", "playful"),
    }

    def __init__(self) -> None:
        self._history: list[EmotionalReading] = []

    def read(self, message: str) -> EmotionalReading:
        """Read the human's emotional state from their message."""
        msg_lower = message.lower()
        scores: dict[HumanEmotion, tuple[float, list[str]]] = {}

        for emotion, indicators in self.INDICATORS.items():
            matches = [ind for ind in indicators if ind in msg_lower]
            if matches:
                score = len(matches) / len(indicators)
                scores[emotion] = (score, matches)

        if not scores:
            reading = EmotionalReading(
                primary=HumanEmotion.NEUTRAL,
                confidence=0.5,
                signals=[],
                suggested_approach="Be clear and informative.",
                tone="balanced",
            )
        else:
            best_emotion = max(scores, key=lambda e: scores[e][0])
            score, signals = scores[best_emotion]
            approach, tone = self.RESPONSE_APPROACHES[best_emotion]

            reading = EmotionalReading(
                primary=best_emotion,
                confidence=min(0.95, score * 2),
                signals=signals,
                suggested_approach=approach,
                tone=tone,
            )

        self._history.append(reading)
        return reading

    def emotional_trend(self) -> str:
        """What's the emotional trend of the conversation?"""
        if len(self._history) < 3:
            return "Not enough data for trend."

        recent = self._history[-5:]
        emotions = [r.primary.value for r in recent]

        from collections import Counter
        counts = Counter(emotions)
        dominant = counts.most_common(1)[0]

        if dominant[1] >= 3:
            return f"Consistently {dominant[0]}."
        return f"Mixed: {', '.join(set(emotions))}."
