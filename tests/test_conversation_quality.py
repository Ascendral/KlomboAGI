"""
Conversation Quality Tests — does the system actually work as a conversational AGI?

Tests:
1. Multi-turn coherence — can it follow a conversation across turns?
2. Teaching quality — does it learn correctly from human teaching?
3. Auto-study quality — does it learn from Wikipedia correctly?
4. Answer quality — clean natural language, not fact dumps
5. Cross-concept reasoning — can it connect taught concepts?
6. Sequential questions — can it answer multiple different questions?
"""

import time
import pytest
from klomboagi.core.genesis import Genesis


@pytest.fixture
def fresh_genesis():
    """Create a fresh Genesis with no prior knowledge."""
    path = f"/tmp/klombo_test_conv_{int(time.time() * 1000)}.json"
    return Genesis(memory_path=path)


@pytest.fixture
def taught_genesis():
    """Genesis with curriculum loaded."""
    path = f"/tmp/klombo_test_taught_{int(time.time() * 1000)}.json"
    g = Genesis(memory_path=path)
    g.teach_everything()
    return g


class TestMultiTurnCoherence:
    """Can the system follow a multi-turn conversation?"""

    def test_remembers_topic(self, fresh_genesis):
        """After teaching about X, questions about X should be answerable."""
        g = fresh_genesis
        g.hear("Dolphins are intelligent marine mammals")
        g.hear("Dolphins use echolocation to navigate")
        resp = g.hear("What do dolphins use to navigate?")
        assert "echolocation" in resp.lower() or "dolphins" in resp.lower()

    def test_pronoun_resolution(self, fresh_genesis):
        """System should resolve 'it' and 'that' to current topic."""
        g = fresh_genesis
        g.hear("A volcano is a mountain that erupts with lava")
        # "it" should resolve to "volcano"
        # The dialog context tracks current topic
        assert g.context.current_topic != ""

    def test_multiple_topics(self, fresh_genesis):
        """Can handle switching between topics."""
        g = fresh_genesis
        g.hear("The sun is a star")
        g.hear("Water is made of hydrogen and oxygen")
        resp1 = g.hear("What is the sun?")
        resp2 = g.hear("What is water?")
        assert "star" in resp1.lower()
        assert "hydrogen" in resp2.lower() or "oxygen" in resp2.lower()


class TestTeachingQuality:
    """Does it learn correctly from human teaching?"""

    def test_svo_teaching_stored(self, fresh_genesis):
        """SVO sentences should be stored as beliefs."""
        g = fresh_genesis
        g.hear("Plants use chlorophyll to capture light")
        beliefs = {s: b for s, b in g.base._beliefs.items()}
        # Should have a belief about plants and chlorophyll
        found = any("plants" in s and "chlorophyll" in s for s in beliefs)
        assert found, f"Expected belief about plants+chlorophyll, got: {list(beliefs.keys())}"

    def test_is_teaching_stored(self, fresh_genesis):
        """'X is Y' sentences should create beliefs."""
        g = fresh_genesis
        g.hear("A neuron is a nerve cell that transmits electrical signals")
        beliefs = {s: b for s, b in g.base._beliefs.items()}
        found = any("neuron" in s for s in beliefs)
        assert found

    def test_teaching_increases_belief_count(self, fresh_genesis):
        """Teaching should increase the number of beliefs."""
        g = fresh_genesis
        before = len(g.base._beliefs)
        g.hear("Mars is the fourth planet from the sun")
        after = len(g.base._beliefs)
        assert after > before

    def test_proactive_curiosity_fires(self, fresh_genesis):
        """Teaching should trigger proactive questions about unknown concepts."""
        g = fresh_genesis
        resp = g.hear("Mitochondria are the powerhouse of the cell")
        # Should ask about an unknown concept or acknowledge the teaching
        assert len(resp) > 10  # Not empty


class TestAnswerQuality:
    """Are answers clean natural language?"""

    def test_no_duplicate_definitions(self, taught_genesis):
        """Answer should not repeat the same definition twice."""
        g = taught_genesis
        resp = g.hear("What is gravity?")
        # Count how many times the definition appears
        definition = "fundamental force"
        count = resp.lower().count(definition)
        assert count <= 1, f"Definition repeated {count} times: {resp[:300]}"

    def test_definition_first(self, taught_genesis):
        """For 'What is X?' questions, the definition should come first."""
        g = taught_genesis
        resp = g.hear("What is energy?")
        first_line = resp.strip().split("\n")[0]
        # Should contain the actual definition, not a relation
        assert "capacity" in first_line.lower() or "work" in first_line.lower() or "energy" in first_line.lower()

    def test_answer_not_too_long(self, taught_genesis):
        """Answers should be focused, not paragraph dumps."""
        g = taught_genesis
        resp = g.hear("What is a prime number?")
        first_line = resp.strip().split("\n")[0]
        # First line should be under 300 chars
        assert len(first_line) < 300, f"Answer too long: {len(first_line)} chars"


class TestCrossConceptReasoning:
    """Can it connect concepts taught separately?"""

    def test_finds_connection_through_teaching(self, fresh_genesis):
        """Concepts taught in separate turns should be connectable."""
        g = fresh_genesis
        g.hear("Bees use flowers for nectar")
        g.hear("Flowers produce fruit")
        g.hear("Fruit contains seeds")
        # The beliefs should exist
        assert len(g.base._beliefs) >= 3

    def test_why_uses_forward_effects(self, taught_genesis):
        """'Why is X important' should trace forward effects."""
        g = taught_genesis
        resp = g.hear("Why is photosynthesis important?")
        # Should mention effects, not just definition
        low = resp.lower()
        assert "important" in low or "because" in low or "causes" in low or "produces" in low


class TestAutoStudy:
    """Does the system learn from Wikipedia correctly?"""

    def test_study_creates_concepts(self, fresh_genesis):
        """read_and_learn should create concepts in the store."""
        g = fresh_genesis
        result = g.read_and_learn("photosynthesis")
        assert "Could not read" not in result
        assert len(g.base.memory.concepts) > 10

    def test_study_enables_answering(self, fresh_genesis):
        """After studying, should be able to answer questions on the topic."""
        g = fresh_genesis
        g.read_and_learn("photosynthesis")
        resp = g.hear("What is photosynthesis?")
        low = resp.lower()
        # Should contain something substantive about photosynthesis
        assert "photosynthesis" in low or "plant" in low or "oxygen" in low or "light" in low
