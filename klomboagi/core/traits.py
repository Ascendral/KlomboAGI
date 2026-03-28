"""
Trait System — the personality layer between Brain and Skills.

Architecture (from the whiteboard):
  Brain (center) → Traits (branches) → Abilities (per trait) → Skills (per ability)

Traits are DRIVES — persistence, curiosity, analysis. They determine HOW
the brain approaches problems. Each trait enables abilities, and each ability
has specific skills that improve through practice.

Starts empty. The human teaches the system what to value.
Every instance develops its own personality over time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Skill:
    """
    Leaf node — a specific capability that improves through practice.

    Each skill has its own learning algorithm: practice(succeeded) adjusts
    proficiency based on outcome. Success rates are tracked honestly.
    """
    name: str
    description: str
    proficiency: float = 0.1          # 0-1, starts low
    use_count: int = 0
    success_count: int = 0
    last_used: str = ""
    improvement_rate: float = 0.05    # how fast proficiency changes per practice

    def practice(self, succeeded: bool) -> None:
        """Adjust proficiency based on outcome."""
        self.use_count += 1
        self.last_used = time.strftime("%Y-%m-%dT%H:%M:%S")
        if succeeded:
            self.success_count += 1
            self.proficiency = min(1.0, self.proficiency + self.improvement_rate)
        else:
            # Failures teach less than successes reward
            self.proficiency = max(0.0, self.proficiency - self.improvement_rate * 0.5)

    def success_rate(self) -> float:
        """Honest success rate."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "proficiency": round(self.proficiency, 3),
            "use_count": self.use_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate(), 3),
            "last_used": self.last_used,
            "improvement_rate": self.improvement_rate,
        }


@dataclass
class Ability:
    """
    Groups related skills. An ability is "ready" when its skills
    have sufficient proficiency.
    """
    name: str
    description: str
    skills: dict[str, Skill] = field(default_factory=dict)

    def add_skill(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def readiness(self) -> float:
        """Average proficiency across skills. 0 if no skills."""
        if not self.skills:
            return 0.0
        return sum(s.proficiency for s in self.skills.values()) / len(self.skills)

    def best_skill(self) -> Skill | None:
        """Highest proficiency skill."""
        if not self.skills:
            return None
        return max(self.skills.values(), key=lambda s: s.proficiency)

    def practice_skill(self, skill_name: str, succeeded: bool) -> bool:
        """Practice a specific skill. Returns False if skill not found."""
        if skill_name not in self.skills:
            return False
        self.skills[skill_name].practice(succeeded)
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "readiness": round(self.readiness(), 3),
            "skills": {n: s.to_dict() for n, s in self.skills.items()},
        }


@dataclass
class Trait:
    """
    A drive — persistence, curiosity, analysis.

    Traits have drive_strength (how strongly they manifest) and contain
    abilities. They strengthen through use and decay when unused.
    """
    name: str
    description: str
    drive_strength: float = 0.5       # 0-1, how strongly this manifests
    abilities: dict[str, Ability] = field(default_factory=dict)
    activation_keywords: list[str] = field(default_factory=list)
    activation_count: int = 0
    last_activated: str = ""

    def add_ability(self, ability: Ability) -> None:
        self.abilities[ability.name] = ability

    def strengthen(self, amount: float = 0.02) -> None:
        """Trait gets stronger through use."""
        self.drive_strength = min(1.0, self.drive_strength + amount)
        self.activation_count += 1
        self.last_activated = time.strftime("%Y-%m-%dT%H:%M:%S")

    def weaken(self, amount: float = 0.01) -> None:
        """Trait decays when unused."""
        self.drive_strength = max(0.0, self.drive_strength - amount)

    def readiness(self) -> float:
        """Average ability readiness."""
        if not self.abilities:
            return 0.0
        return sum(a.readiness() for a in self.abilities.values()) / len(self.abilities)

    def matches(self, text: str) -> float:
        """How strongly does this trait match the given text? 0-1."""
        if not self.activation_keywords:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in self.activation_keywords if kw.lower() in text_lower)
        return min(1.0, hits / max(len(self.activation_keywords), 1))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "drive_strength": round(self.drive_strength, 3),
            "readiness": round(self.readiness(), 3),
            "activation_count": self.activation_count,
            "activation_keywords": self.activation_keywords,
            "abilities": {n: a.to_dict() for n, a in self.abilities.items()},
        }


@dataclass
class TraitInfluence:
    """What the brain gets back when it consults the trait system."""
    active_traits: list[str]
    recommended_approach: str          # e.g. "investigate", "persist", "decompose"
    available_skills: list[str]        # skills from active traits
    confidence_modifier: float = 0.0   # positive = boost, negative = reduce
    persistence_modifier: int = 0      # extra attempts to grant
    reasoning: str = ""                # why these traits activated

    def to_dict(self) -> dict:
        return {
            "active_traits": self.active_traits,
            "recommended_approach": self.recommended_approach,
            "available_skills": self.available_skills,
            "confidence_modifier": self.confidence_modifier,
            "persistence_modifier": self.persistence_modifier,
            "reasoning": self.reasoning,
        }


class TraitSystem:
    """
    The hub connecting Brain to Traits.

    Starts empty. Traits are added by the human through teaching.
    The system develops personality over time based on which traits
    get activated and practiced.
    """

    DECAY_RATE = 0.005  # per tick, unused traits decay this much

    def __init__(self) -> None:
        self.traits: dict[str, Trait] = {}
        self._tick_count = 0

    def add_trait(self, name: str, description: str,
                  drive_strength: float = 0.5,
                  keywords: list[str] | None = None) -> Trait:
        """Register a new trait. Returns the trait."""
        trait = Trait(
            name=name,
            description=description,
            drive_strength=max(0.0, min(1.0, drive_strength)),
            activation_keywords=keywords or [],
        )
        self.traits[name] = trait
        return trait

    def get_trait(self, name: str) -> Trait | None:
        return self.traits.get(name)

    def get_active_traits(self, threshold: float = 0.3) -> list[Trait]:
        """Traits above the drive threshold, sorted by strength."""
        active = [t for t in self.traits.values() if t.drive_strength >= threshold]
        return sorted(active, key=lambda t: t.drive_strength, reverse=True)

    def influence(self, problem_context: dict) -> TraitInfluence:
        """
        Given a problem, determine which traits activate and what they recommend.

        This is the key integration point with CognitionLoop.
        """
        description = problem_context.get("description", "")
        known_entities = problem_context.get("known_entities", [])
        text = f"{description} {' '.join(known_entities)}"

        # Score each trait against the problem
        scored: list[tuple[Trait, float]] = []
        for trait in self.traits.values():
            keyword_match = trait.matches(text)
            if keyword_match == 0:
                continue  # No keyword match = trait doesn't activate
            # Combine keyword relevance with drive strength
            activation = keyword_match * 0.6 + trait.drive_strength * 0.4
            scored.append((trait, activation))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return TraitInfluence(
                active_traits=[],
                recommended_approach="default",
                available_skills=[],
                reasoning="No traits matched this problem",
            )

        # Activate matching traits
        active_traits = []
        all_skills = []
        for trait, score in scored:
            trait.strengthen()
            active_traits.append(trait.name)
            for ability in trait.abilities.values():
                for skill in ability.skills.values():
                    all_skills.append(skill.name)

        # The strongest trait determines the recommended approach
        top_trait = scored[0][0]

        # Calculate modifiers from active traits
        avg_readiness = sum(t.readiness() for t, _ in scored) / len(scored) if scored else 0
        confidence_mod = (avg_readiness - 0.5) * 0.2  # readiness above 0.5 boosts confidence

        # Persistence modifier: if persistence-like traits are active, grant extra attempts
        persistence_bonus = 0
        for trait, score in scored:
            if any(kw in trait.activation_keywords for kw in ["persist", "retry", "stubborn", "determined"]):
                persistence_bonus = max(persistence_bonus, int(trait.drive_strength * 2))

        approach_parts = [t.name for t, _ in scored[:3]]
        reasoning = f"Traits [{', '.join(approach_parts)}] activated (top: {top_trait.name} at {scored[0][1]:.0%})"

        return TraitInfluence(
            active_traits=active_traits,
            recommended_approach=top_trait.name,
            available_skills=all_skills,
            confidence_modifier=round(confidence_mod, 3),
            persistence_modifier=persistence_bonus,
            reasoning=reasoning,
        )

    def record_outcome(self, trait_name: str, ability_name: str,
                       skill_name: str, succeeded: bool) -> bool:
        """
        Record the outcome of using a skill.

        Updates: skill proficiency, trait drive_strength (strengthen on success).
        Returns False if trait/ability/skill not found.
        """
        trait = self.traits.get(trait_name)
        if not trait:
            return False

        ability = trait.abilities.get(ability_name)
        if not ability:
            return False

        if not ability.practice_skill(skill_name, succeeded):
            return False

        if succeeded:
            trait.strengthen(0.01)
        return True

    def tick(self) -> None:
        """
        Periodic maintenance. Decay unused traits slightly.

        Call this at the end of each cognition cycle or on a timer.
        """
        self._tick_count += 1
        current = time.strftime("%Y-%m-%dT%H:%M:%S")
        for trait in self.traits.values():
            if trait.last_activated != current:
                trait.weaken(self.DECAY_RATE)

    def personality_vector(self) -> dict[str, float]:
        """Snapshot of all trait strengths — the system's personality."""
        return {name: round(t.drive_strength, 3) for name, t in self.traits.items()}

    def stats(self) -> dict:
        """Full system stats."""
        total_skills = sum(
            len(skill)
            for t in self.traits.values()
            for skill in [a.skills for a in t.abilities.values()]
        )
        return {
            "trait_count": len(self.traits),
            "active_traits": len(self.get_active_traits()),
            "total_abilities": sum(len(t.abilities) for t in self.traits.values()),
            "total_skills": total_skills,
            "personality": self.personality_vector(),
            "ticks": self._tick_count,
        }
