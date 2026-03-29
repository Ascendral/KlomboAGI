"""
Conversational Memory — remember past conversations, build on them.

Right now every session starts fresh. A real mind remembers:
  "Last time we talked about gravity and you taught me about
   general relativity. Before that, we discussed consciousness."

Stores conversation SUMMARIES (not raw text) organized by:
  - Topic discussed
  - What was learned
  - What questions remained unanswered
  - What the human seemed to care about
  - When it happened

Next conversation can reference past ones:
  "You asked about gravity before. I've learned more since then."
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversationSummary:
    """Summary of a past conversation."""
    session_id: str
    timestamp: str
    topics: list[str]
    facts_learned: int
    questions_asked: list[str]
    questions_unanswered: list[str]
    human_interests: list[str]      # what the human seemed to care about
    surprises: int                  # contradictions detected
    corrections: int                # times human corrected us
    turns: int

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "topics": self.topics,
            "facts_learned": self.facts_learned,
            "questions_asked": self.questions_asked[:10],
            "questions_unanswered": self.questions_unanswered[:5],
            "human_interests": self.human_interests[:5],
            "surprises": self.surprises,
            "corrections": self.corrections,
            "turns": self.turns,
        }


class ConversationMemory:
    """
    Persistent memory of past conversations.

    Saves summaries to disk. Loads them on startup.
    Can reference past conversations in current responses.
    """

    def __init__(self, memory_dir: str) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[ConversationSummary] = []
        self._current_topics: list[str] = []
        self._current_questions: list[str] = []
        self._current_unanswered: list[str] = []
        self._load()

    def _load(self) -> None:
        """Load past conversation summaries from disk."""
        history_file = self.memory_dir / "conversation_history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text())
                for entry in data:
                    self.history.append(ConversationSummary(**entry))
            except Exception:
                pass

    def save(self) -> None:
        """Save conversation history to disk."""
        history_file = self.memory_dir / "conversation_history.json"
        data = [s.to_dict() for s in self.history[-100:]]  # Keep last 100
        history_file.write_text(json.dumps(data, indent=2))

    def record_topic(self, topic: str) -> None:
        """Record a topic discussed in current conversation."""
        if topic and topic not in self._current_topics and len(topic) > 2:
            self._current_topics.append(topic)

    def record_question(self, question: str, answered: bool) -> None:
        """Record a question asked in current conversation."""
        self._current_questions.append(question[:80])
        if not answered:
            self._current_unanswered.append(question[:80])

    def end_session(self, facts_learned: int, surprises: int,
                    corrections: int, turns: int,
                    human_interests: list[str] = None) -> ConversationSummary:
        """End current session and save summary."""
        summary = ConversationSummary(
            session_id=f"session_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%d %H:%M"),
            topics=self._current_topics[:10],
            facts_learned=facts_learned,
            questions_asked=self._current_questions[:10],
            questions_unanswered=self._current_unanswered[:5],
            human_interests=human_interests or [],
            surprises=surprises,
            corrections=corrections,
            turns=turns,
        )
        self.history.append(summary)
        self.save()

        # Reset current session
        self._current_topics = []
        self._current_questions = []
        self._current_unanswered = []

        return summary

    def recall_about(self, topic: str) -> str | None:
        """Recall past conversations about a topic."""
        topic_lower = topic.lower()
        relevant = []
        for session in reversed(self.history):
            if any(topic_lower in t.lower() for t in session.topics):
                relevant.append(session)

        if not relevant:
            return None

        last = relevant[0]
        return (
            f"We talked about {topic} before ({last.timestamp}). "
            f"That session covered {', '.join(last.topics[:3])} "
            f"and I learned {last.facts_learned} facts."
        )

    def get_unanswered(self) -> list[str]:
        """Questions from past sessions that were never answered."""
        all_unanswered = []
        for session in self.history[-10:]:
            all_unanswered.extend(session.questions_unanswered)
        return all_unanswered[:10]

    def get_human_interests(self) -> list[str]:
        """What topics the human tends to ask about."""
        all_topics = []
        for session in self.history:
            all_topics.extend(session.topics)
        # Count frequency
        from collections import Counter
        counts = Counter(all_topics)
        return [topic for topic, _ in counts.most_common(10)]

    def summary(self) -> str:
        """Summary of conversation history."""
        if not self.history:
            return "No past conversations recorded."
        lines = [f"Conversation history: {len(self.history)} sessions"]
        for s in self.history[-5:]:
            lines.append(f"  [{s.timestamp}] {', '.join(s.topics[:3])} "
                        f"({s.turns} turns, +{s.facts_learned} facts)")
        unanswered = self.get_unanswered()
        if unanswered:
            lines.append(f"\n  Still unanswered: {', '.join(unanswered[:3])}")
        return "\n".join(lines)
