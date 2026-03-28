"""
Skill Growth — discovered patterns become self-improving capabilities.

When concept formation discovers "I keep needing to chain causal relations",
that becomes a SKILL under the analysis trait. When it discovers "multiple
domains use mathematics", that becomes an ability under the curiosity trait.

Each branch has its own ecosystem of learning:
  Brain
  └── Trait (curiosity)
      └── Ability (investigate)
          └── Skill (causal_chaining)     ← CREATED from pattern discovery
              └── improvement_algorithm    ← SELF-IMPROVING from practice

The skill tree GROWS from experience. It's not hardcoded.
Concept formation feeds INTO trait/ability/skill creation.
Each skill has its own improvement rate tracked by practice outcomes.
"""

from __future__ import annotations

from klomboagi.core.traits import TraitSystem, Trait, Ability, Skill
from klomboagi.reasoning.concept_formation import FormedConcept


# Maps concept formation basis types to which trait should own the new capability
BASIS_TO_TRAIT = {
    "shared_parent": "analysis",          # noticing shared categories = analytical
    "shared_target": "curiosity",         # noticing shared dependencies = curiosity about connections
    "shared_source": "analysis",          # noticing shared origins = analytical
    "causal_chain": "analysis",           # tracing causation = analytical
}

# Maps concept formation basis types to ability names
BASIS_TO_ABILITY = {
    "shared_parent": "categorize",
    "shared_target": "find_dependencies",
    "shared_source": "trace_origins",
    "causal_chain": "chain_reasoning",
}


class SkillGrowth:
    """
    Grows the trait/ability/skill tree from discovered concepts.

    Each discovery → new or strengthened skill.
    Each skill → its own improvement ecosystem.
    """

    def __init__(self, traits: TraitSystem) -> None:
        self.traits = traits
        self._created_skills: list[str] = []

    def integrate(self, concepts: list[FormedConcept]) -> list[str]:
        """
        Integrate discovered concepts into the trait/ability/skill tree.

        Returns list of new skills created or strengthened.
        """
        results = []

        for concept in concepts:
            result = self._integrate_one(concept)
            if result:
                results.append(result)

        return results

    def _integrate_one(self, concept: FormedConcept) -> str | None:
        """Integrate a single discovered concept into the skill tree."""
        # Determine which trait this belongs to
        trait_name = BASIS_TO_TRAIT.get(concept.basis, "analysis")
        ability_name = BASIS_TO_ABILITY.get(concept.basis, "pattern_recognition")

        trait = self.traits.get_trait(trait_name)
        if not trait:
            return None

        # Get or create the ability
        ability = trait.abilities.get(ability_name)
        if not ability:
            ability = Ability(
                name=ability_name,
                description=f"ability to {ability_name.replace('_', ' ')}",
            )
            trait.add_ability(ability)

        # Create a skill for this specific pattern
        skill_name = self._concept_to_skill_name(concept)
        if skill_name in ability.skills:
            # Skill already exists — strengthen it (more evidence)
            ability.skills[skill_name].practice(True)
            return f"Strengthened: {trait_name}/{ability_name}/{skill_name}"

        # Create new skill
        # Proficiency starts based on evidence count
        initial_proficiency = min(0.5, concept.evidence_count * 0.05)
        skill = Skill(
            name=skill_name,
            description=concept.description[:80],
            proficiency=initial_proficiency,
            improvement_rate=0.03 + concept.confidence * 0.02,  # confidence → faster improvement
        )
        ability.add_skill(skill)
        self._created_skills.append(skill_name)

        return f"Created: {trait_name}/{ability_name}/{skill_name} (proficiency: {initial_proficiency:.0%})"

    def _concept_to_skill_name(self, concept: FormedConcept) -> str:
        """Generate a skill name from a concept."""
        # Clean the concept name into a valid skill name
        name = concept.name.lower()
        name = name.replace(" ", "_").replace("-", "_")
        # Shorten
        parts = name.split("_")[:4]
        return "_".join(parts)

    def report(self) -> str:
        """Report the current skill tree structure."""
        lines = ["Skill Tree:"]
        for trait_name, trait in self.traits.traits.items():
            lines.append(f"\n  [{trait.drive_strength:.0%}] {trait_name}: {trait.description}")
            for ab_name, ability in trait.abilities.items():
                lines.append(f"    └── {ab_name} (readiness: {ability.readiness():.0%})")
                for sk_name, skill in ability.skills.items():
                    bar = "█" * int(skill.proficiency * 10) + "░" * (10 - int(skill.proficiency * 10))
                    lines.append(f"        └── [{bar}] {sk_name} "
                               f"(prof: {skill.proficiency:.0%}, "
                               f"used: {skill.use_count}, "
                               f"rate: {skill.success_rate():.0%})")
        return "\n".join(lines)
