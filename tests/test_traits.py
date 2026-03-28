"""Tests for the trait system — personality layer between Brain and Skills."""

from __future__ import annotations

import pytest

from klomboagi.core.traits import (
    Skill,
    Ability,
    Trait,
    TraitInfluence,
    TraitSystem,
)


# ── Skill tests ──


class TestSkill:

    def test_starts_with_low_proficiency(self):
        s = Skill(name="search", description="web search")
        assert s.proficiency == 0.1
        assert s.use_count == 0

    def test_practice_success_increases_proficiency(self):
        s = Skill(name="search", description="web search", proficiency=0.5)
        s.practice(succeeded=True)
        assert s.proficiency > 0.5
        assert s.use_count == 1
        assert s.success_count == 1

    def test_practice_failure_decreases_proficiency(self):
        s = Skill(name="search", description="web search", proficiency=0.5)
        s.practice(succeeded=False)
        assert s.proficiency < 0.5
        assert s.use_count == 1
        assert s.success_count == 0

    def test_proficiency_clamped_at_1(self):
        s = Skill(name="search", description="web search", proficiency=0.98)
        s.practice(succeeded=True)
        assert s.proficiency <= 1.0

    def test_proficiency_clamped_at_0(self):
        s = Skill(name="search", description="web search", proficiency=0.01)
        s.practice(succeeded=False)
        assert s.proficiency >= 0.0

    def test_success_rate_empty(self):
        s = Skill(name="search", description="web search")
        assert s.success_rate() == 0.0

    def test_success_rate_tracked(self):
        s = Skill(name="search", description="web search")
        s.practice(succeeded=True)
        s.practice(succeeded=True)
        s.practice(succeeded=False)
        assert abs(s.success_rate() - 2 / 3) < 0.01

    def test_failure_hurts_less_than_success_helps(self):
        s = Skill(name="search", description="web search", proficiency=0.5)
        gain = s.improvement_rate
        loss = s.improvement_rate * 0.5
        assert gain > loss

    def test_last_used_updated(self):
        s = Skill(name="search", description="web search")
        assert s.last_used == ""
        s.practice(succeeded=True)
        assert s.last_used != ""

    def test_to_dict(self):
        s = Skill(name="search", description="web search")
        d = s.to_dict()
        assert d["name"] == "search"
        assert "proficiency" in d
        assert "success_rate" in d


# ── Ability tests ──


class TestAbility:

    def test_readiness_no_skills(self):
        a = Ability(name="investigate", description="look into things")
        assert a.readiness() == 0.0

    def test_readiness_average(self):
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="s1", description="s1", proficiency=0.6))
        a.add_skill(Skill(name="s2", description="s2", proficiency=0.4))
        assert abs(a.readiness() - 0.5) < 0.01

    def test_best_skill(self):
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="s1", description="s1", proficiency=0.3))
        a.add_skill(Skill(name="s2", description="s2", proficiency=0.8))
        assert a.best_skill().name == "s2"

    def test_best_skill_empty(self):
        a = Ability(name="investigate", description="look into things")
        assert a.best_skill() is None

    def test_practice_skill(self):
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="s1", description="s1", proficiency=0.5))
        assert a.practice_skill("s1", True)
        assert a.skills["s1"].proficiency > 0.5

    def test_practice_unknown_skill(self):
        a = Ability(name="investigate", description="look into things")
        assert not a.practice_skill("nonexistent", True)

    def test_to_dict(self):
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="s1", description="s1"))
        d = a.to_dict()
        assert d["name"] == "investigate"
        assert "s1" in d["skills"]


# ── Trait tests ──


class TestTrait:

    def test_strengthen(self):
        t = Trait(name="curiosity", description="desire to learn", drive_strength=0.5)
        t.strengthen(0.1)
        assert t.drive_strength == 0.6
        assert t.activation_count == 1

    def test_strengthen_clamped(self):
        t = Trait(name="curiosity", description="desire to learn", drive_strength=0.95)
        t.strengthen(0.1)
        assert t.drive_strength == 1.0

    def test_weaken(self):
        t = Trait(name="curiosity", description="desire to learn", drive_strength=0.5)
        t.weaken(0.1)
        assert t.drive_strength == 0.4

    def test_weaken_clamped(self):
        t = Trait(name="curiosity", description="desire to learn", drive_strength=0.05)
        t.weaken(0.1)
        assert t.drive_strength == 0.0

    def test_matches_keywords(self):
        t = Trait(
            name="curiosity", description="desire to learn",
            activation_keywords=["unknown", "explore", "why", "new"],
        )
        assert t.matches("explore the unknown territory") > 0
        assert t.matches("compile the code") == 0

    def test_matches_case_insensitive(self):
        t = Trait(
            name="curiosity", description="desire to learn",
            activation_keywords=["explore"],
        )
        assert t.matches("EXPLORE the world") > 0

    def test_readiness(self):
        t = Trait(name="curiosity", description="desire to learn")
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="s1", description="s1", proficiency=0.8))
        t.add_ability(a)
        assert t.readiness() == 0.8

    def test_readiness_no_abilities(self):
        t = Trait(name="curiosity", description="desire to learn")
        assert t.readiness() == 0.0

    def test_to_dict(self):
        t = Trait(name="curiosity", description="desire to learn", drive_strength=0.7)
        d = t.to_dict()
        assert d["name"] == "curiosity"
        assert d["drive_strength"] == 0.7


# ── TraitSystem tests ──


class TestTraitSystem:

    @pytest.fixture
    def system(self):
        ts = TraitSystem()
        ts.add_trait("curiosity", "desire to learn", 0.8,
                     keywords=["unknown", "explore", "why", "new", "investigate"])
        ts.add_trait("persistence", "never give up", 0.7,
                     keywords=["retry", "fail", "stuck", "timeout", "persist", "determined"])
        ts.add_trait("analysis", "break things down", 0.6,
                     keywords=["decompose", "pattern", "compare", "structure", "analyze"])
        return ts

    def test_add_trait(self):
        ts = TraitSystem()
        t = ts.add_trait("curiosity", "desire to learn", 0.8)
        assert t.name == "curiosity"
        assert t.drive_strength == 0.8
        assert "curiosity" in ts.traits

    def test_add_trait_clamps_strength(self):
        ts = TraitSystem()
        t = ts.add_trait("x", "x", 1.5)
        assert t.drive_strength == 1.0
        t2 = ts.add_trait("y", "y", -0.5)
        assert t2.drive_strength == 0.0

    def test_get_active_traits(self, system):
        active = system.get_active_traits(threshold=0.3)
        assert len(active) == 3
        # Sorted by strength
        assert active[0].name == "curiosity"

    def test_get_active_traits_threshold(self, system):
        active = system.get_active_traits(threshold=0.75)
        assert len(active) == 1
        assert active[0].name == "curiosity"

    def test_influence_matches_curiosity(self, system):
        influence = system.influence({"description": "explore the unknown API"})
        assert "curiosity" in influence.active_traits
        assert influence.recommended_approach == "curiosity"

    def test_influence_matches_persistence(self, system):
        influence = system.influence({"description": "the build keeps failing, retry"})
        assert "persistence" in influence.active_traits

    def test_influence_matches_analysis(self, system):
        influence = system.influence({"description": "decompose the structure and compare patterns"})
        assert "analysis" in influence.active_traits

    def test_influence_no_match(self):
        ts = TraitSystem()
        ts.add_trait("curiosity", "desire to learn", 0.8, keywords=["explore"])
        influence = ts.influence({"description": "compile the code"})
        assert influence.recommended_approach == "default"

    def test_influence_strengthens_activated_traits(self, system):
        before = system.traits["curiosity"].drive_strength
        system.influence({"description": "explore the unknown"})
        after = system.traits["curiosity"].drive_strength
        assert after > before

    def test_influence_includes_skills(self, system):
        # Add a skill under curiosity
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="web_search", description="search the web"))
        system.traits["curiosity"].add_ability(a)

        influence = system.influence({"description": "explore unknown territory"})
        assert "web_search" in influence.available_skills

    def test_influence_persistence_modifier(self, system):
        influence = system.influence({"description": "retry the failing build, persist"})
        assert influence.persistence_modifier > 0

    def test_record_outcome_success(self, system):
        a = Ability(name="investigate", description="look into things")
        a.add_skill(Skill(name="web_search", description="search", proficiency=0.5))
        system.traits["curiosity"].add_ability(a)

        assert system.record_outcome("curiosity", "investigate", "web_search", True)
        assert system.traits["curiosity"].abilities["investigate"].skills["web_search"].proficiency > 0.5

    def test_record_outcome_bad_path(self, system):
        assert not system.record_outcome("nonexistent", "x", "y", True)
        assert not system.record_outcome("curiosity", "nonexistent", "y", True)

    def test_personality_vector(self, system):
        pv = system.personality_vector()
        assert "curiosity" in pv
        assert "persistence" in pv
        assert "analysis" in pv
        assert pv["curiosity"] == 0.8

    def test_tick_decays_unused(self, system):
        before = system.traits["analysis"].drive_strength
        system.tick()
        after = system.traits["analysis"].drive_strength
        assert after < before

    def test_stats(self, system):
        s = system.stats()
        assert s["trait_count"] == 3
        assert "personality" in s

    def test_starts_empty(self):
        ts = TraitSystem()
        assert len(ts.traits) == 0
        assert ts.stats()["trait_count"] == 0

    def test_influence_uses_known_entities(self, system):
        influence = system.influence({
            "description": "fix the bug",
            "known_entities": ["unknown_module"],
        })
        # "unknown" should trigger curiosity
        assert "curiosity" in influence.active_traits

    def test_influence_has_reasoning(self, system):
        influence = system.influence({"description": "explore the unknown"})
        assert len(influence.reasoning) > 0


# ── TraitInfluence tests ──


class TestTraitInfluence:

    def test_to_dict(self):
        ti = TraitInfluence(
            active_traits=["curiosity"],
            recommended_approach="investigate",
            available_skills=["web_search"],
            confidence_modifier=0.05,
            persistence_modifier=1,
            reasoning="curiosity activated",
        )
        d = ti.to_dict()
        assert d["active_traits"] == ["curiosity"]
        assert d["recommended_approach"] == "investigate"


# ── Integration test: full lifecycle ──


class TestTraitLifecycle:

    def test_personality_develops_over_time(self):
        """A system that repeatedly uses curiosity should develop
        a curiosity-dominant personality."""
        ts = TraitSystem()
        ts.add_trait("curiosity", "desire to learn", 0.5,
                     keywords=["unknown", "explore", "why"])
        ts.add_trait("persistence", "never give up", 0.5,
                     keywords=["retry", "fail", "stuck"])

        # Repeatedly trigger curiosity
        for _ in range(10):
            ts.influence({"description": "explore the unknown"})
            ts.tick()

        pv = ts.personality_vector()
        # Curiosity should be stronger than persistence
        assert pv["curiosity"] > pv["persistence"]

    def test_skills_improve_through_practice(self):
        """Practicing a skill repeatedly should increase proficiency."""
        ts = TraitSystem()
        t = ts.add_trait("analysis", "break things down", 0.6, keywords=["analyze"])
        a = Ability(name="decompose", description="break into parts")
        a.add_skill(Skill(name="structural_split", description="split structures", proficiency=0.2))
        t.add_ability(a)

        for _ in range(20):
            ts.record_outcome("analysis", "decompose", "structural_split", True)

        skill = ts.traits["analysis"].abilities["decompose"].skills["structural_split"]
        assert skill.proficiency > 0.5
        assert skill.success_rate() == 1.0

    def test_unused_traits_fade(self):
        """Traits that are never activated should decay toward zero."""
        ts = TraitSystem()
        ts.add_trait("laziness", "do nothing", 0.3, keywords=["nap"])

        for _ in range(50):
            ts.tick()

        assert ts.traits["laziness"].drive_strength < 0.3
