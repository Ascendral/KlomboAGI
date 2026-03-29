"""
Context-Dependent Answers — same question, different context = different answer.

"What is energy?" in a physics conversation → thermodynamics, kinetic, potential
"What is energy?" in an economics conversation → labor force, productivity
"What is energy?" in a biology conversation → cellular respiration, ATP

The system checks what we've been talking about (dialog context,
working memory) and weights beliefs from that domain higher.
"""

from __future__ import annotations


class ContextualAnswerer:
    """
    Adjusts answers based on conversation context.
    """

    def __init__(self) -> None:
        pass

    def contextualize(self, beliefs: list[tuple[str, float]],
                      context_topics: list[str],
                      working_memory_items: list[str]) -> list[tuple[str, float]]:
        """
        Re-rank beliefs based on conversation context.

        Beliefs that relate to current topics get boosted.
        """
        if not context_topics and not working_memory_items:
            return beliefs

        context_words = set()
        for topic in context_topics:
            context_words.update(topic.lower().split())
        for item in working_memory_items:
            context_words.update(item.lower().split())

        # Remove stop words from context
        stop = {"is", "a", "an", "the", "of", "and", "or", "in", "to",
                "for", "with", "that", "this", "by", "from"}
        context_words -= stop

        if not context_words:
            return beliefs

        # Re-score beliefs based on context overlap
        rescored = []
        for stmt, score in beliefs:
            stmt_words = set(stmt.lower().split()) - stop
            overlap = len(context_words & stmt_words)
            if overlap > 0:
                # Context boost: more overlap = bigger boost
                boost = overlap * 0.15
                rescored.append((stmt, score + boost))
            else:
                rescored.append((stmt, score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored
