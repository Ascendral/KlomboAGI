"""
Marathon Teaching — 2+ hours of continuous knowledge acquisition.

Reads hundreds of Wikipedia articles across every domain of human knowledge.
Extracts facts and relations from each. Runs inference after each batch.
Forms new concepts. Grows the skill tree. Never stops.

Domains covered:
- Sciences: physics, chemistry, biology, astronomy, geology, ecology
- Mathematics: algebra, calculus, statistics, topology, number theory
- Technology: CS, AI, networking, databases, cryptography
- Humanities: philosophy, history, psychology, linguistics, sociology
- Arts: music theory, architecture, literature
- Applied: engineering, medicine, economics, law
- World knowledge: geography, politics, religion, culture
"""

from __future__ import annotations

import time


# ═══════════════════════════════════════════════════════════════
# EVERY DOMAIN OF HUMAN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════

ALL_TOPICS: dict[str, list[str]] = {
    # ── SCIENCES ──
    "physics_core": [
        "Physics", "Classical_mechanics", "Quantum_mechanics",
        "General_relativity", "Special_relativity", "Thermodynamics",
        "Electromagnetism", "Optics", "Acoustics", "Fluid_mechanics",
    ],
    "physics_advanced": [
        "Nuclear_physics", "Particle_physics", "Plasma_(physics)",
        "Condensed_matter_physics", "Astrophysics", "Cosmology",
        "String_theory", "Quantum_field_theory", "Standard_Model",
        "Dark_matter", "Dark_energy", "Higgs_boson",
    ],
    "chemistry": [
        "Chemistry", "Organic_chemistry", "Inorganic_chemistry",
        "Biochemistry", "Physical_chemistry", "Analytical_chemistry",
        "Atom", "Molecule", "Chemical_element", "Chemical_bond",
        "Periodic_table", "Acid", "Oxidation", "Catalysis",
        "Polymer", "Crystal", "Solution_(chemistry)",
    ],
    "biology_core": [
        "Biology", "Cell_(biology)", "DNA", "RNA", "Protein",
        "Evolution", "Natural_selection", "Genetics", "Gene",
        "Mutation", "Genome", "Chromosome",
    ],
    "biology_advanced": [
        "Ecology", "Neuroscience", "Neuron", "Brain",
        "Photosynthesis", "Cellular_respiration", "Mitosis", "Meiosis",
        "Virus", "Bacterium", "Immune_system", "Hormone",
        "Taxonomy_(biology)", "Biodiversity", "Ecosystem",
    ],
    "earth_science": [
        "Geology", "Plate_tectonics", "Earthquake", "Volcano",
        "Atmosphere_of_Earth", "Climate", "Climate_change",
        "Ocean", "Water_cycle", "Mineral", "Fossil",
    ],
    "astronomy": [
        "Astronomy", "Solar_System", "Sun", "Moon", "Planet",
        "Star", "Galaxy", "Black_hole", "Neutron_star",
        "Big_Bang", "Universe", "Exoplanet", "Nebula",
    ],

    # ── MATHEMATICS ──
    "math_foundations": [
        "Mathematics", "Arithmetic", "Algebra", "Geometry",
        "Calculus", "Linear_algebra", "Statistics",
        "Probability_theory", "Number_theory", "Trigonometry",
    ],
    "math_advanced": [
        "Set_theory", "Graph_theory", "Topology", "Group_theory",
        "Real_analysis", "Complex_analysis", "Differential_equation",
        "Combinatorics", "Game_theory", "Information_theory",
        "Category_theory", "Mathematical_logic", "Chaos_theory",
    ],

    # ── COMPUTER SCIENCE ──
    "cs_core": [
        "Computer_science", "Algorithm", "Data_structure",
        "Turing_machine", "Computational_complexity_theory",
        "Programming_language", "Compiler", "Operating_system",
        "Computer_network", "Database", "Software_engineering",
    ],
    "cs_ai": [
        "Artificial_intelligence", "Machine_learning", "Neural_network",
        "Deep_learning", "Natural_language_processing",
        "Computer_vision", "Reinforcement_learning",
        "Expert_system", "Knowledge_representation_and_reasoning",
    ],
    "cs_applied": [
        "Cryptography", "Cybersecurity", "Distributed_computing",
        "Cloud_computing", "Internet", "World_Wide_Web",
        "Blockchain", "Quantum_computing",
    ],

    # ── PHILOSOPHY ──
    "philosophy": [
        "Philosophy", "Epistemology", "Metaphysics", "Ethics",
        "Logic", "Ontology", "Aesthetics", "Philosophy_of_mind",
        "Philosophy_of_science", "Political_philosophy",
        "Existentialism", "Rationalism", "Empiricism",
    ],

    # ── PSYCHOLOGY ──
    "psychology": [
        "Psychology", "Cognitive_psychology", "Behavioral_psychology",
        "Developmental_psychology", "Social_psychology",
        "Consciousness", "Memory", "Emotion", "Intelligence",
        "Motivation", "Perception", "Learning",
    ],

    # ── HISTORY ──
    "history": [
        "History", "Ancient_history", "Middle_Ages", "Renaissance",
        "Industrial_Revolution", "World_War_I", "World_War_II",
        "Cold_War", "Ancient_Egypt", "Ancient_Greece", "Roman_Empire",
        "Scientific_Revolution", "Age_of_Enlightenment",
    ],

    # ── ECONOMICS ──
    "economics": [
        "Economics", "Microeconomics", "Macroeconomics",
        "Supply_and_demand", "Inflation", "Monetary_policy",
        "Fiscal_policy", "Gross_domestic_product", "Game_theory",
        "Capitalism", "Socialism", "Free_market",
    ],

    # ── ENGINEERING ──
    "engineering": [
        "Engineering", "Mechanical_engineering", "Electrical_engineering",
        "Civil_engineering", "Chemical_engineering",
        "Aerospace_engineering", "Robotics", "Nanotechnology",
        "Renewable_energy", "Nuclear_power", "Semiconductor",
    ],

    # ── MEDICINE ──
    "medicine": [
        "Medicine", "Anatomy", "Physiology", "Pathology",
        "Pharmacology", "Surgery", "Immunology", "Epidemiology",
        "Vaccine", "Antibiotic", "Cancer",
    ],

    # ── LINGUISTICS ──
    "linguistics": [
        "Linguistics", "Syntax", "Semantics", "Phonetics",
        "Morphology_(linguistics)", "Pragmatics",
        "Language", "Grammar", "Alphabet",
    ],

    # ── GEOGRAPHY ──
    "geography": [
        "Geography", "Continent", "Africa", "Asia", "Europe",
        "North_America", "South_America", "Antarctica", "Australia",
        "Pacific_Ocean", "Atlantic_Ocean",
    ],

    # ── RELIGION & CULTURE ──
    "religion": [
        "Religion", "Christianity", "Islam", "Buddhism", "Hinduism",
        "Judaism", "Bible", "Quran", "Mythology",
    ],

    # ── MUSIC & ARTS ──
    "arts": [
        "Music_theory", "Harmony", "Rhythm", "Melody",
        "Architecture", "Painting", "Sculpture", "Literature",
        "Poetry", "Film", "Photography",
    ],

    # ── SOCIOLOGY ──
    "sociology": [
        "Sociology", "Culture", "Society", "Social_class",
        "Globalization", "Urbanization", "Democracy", "Law",
        "Human_rights", "Education",
    ],
}


def count_topics() -> int:
    return sum(len(topics) for topics in ALL_TOPICS.values())


def run_marathon(genesis, max_hours: float = 2.0,
                 on_progress=None) -> dict:
    """
    Run the marathon teaching session.

    Reads Wikipedia articles continuously for up to max_hours.
    """
    start = time.time()
    max_seconds = max_hours * 3600
    total_articles = 0
    total_facts = 0
    total_relations = 0
    failed = 0
    domains_completed = []

    beliefs_start = len(genesis.base._beliefs)
    relations_start = genesis.relations.stats()["total_relations"]

    for domain, topics in ALL_TOPICS.items():
        # Check time limit
        elapsed = time.time() - start
        if elapsed > max_seconds:
            break

        domain_facts = 0
        domain_start = len(genesis.base._beliefs)

        for topic in topics:
            elapsed = time.time() - start
            if elapsed > max_seconds:
                break

            try:
                result = genesis.read_and_learn(topic)
                if "Could not read" not in result:
                    total_articles += 1
                    # Count new facts
                    new_facts = len(genesis.base._beliefs) - (beliefs_start + total_facts)
                    if new_facts > 0:
                        total_facts += new_facts
                        domain_facts += new_facts
                else:
                    failed += 1
            except Exception:
                failed += 1

            if on_progress:
                on_progress(domain, topic, total_articles, total_facts, elapsed)

        # After each domain: run inference + concept formation
        genesis.relations.run_inference()

        domain_new = len(genesis.base._beliefs) - domain_start
        domains_completed.append((domain, domain_new))

        # Save periodically
        genesis.base.memory.save(genesis.base.memory_path)
        genesis.save_state()

    # Final concept formation + skill growth
    formed = genesis.concept_former.scan()
    growth = genesis.skill_growth.integrate(formed)

    # Final save
    genesis.base.memory.save(genesis.base.memory_path)
    genesis.save_state()

    elapsed = time.time() - start
    beliefs_end = len(genesis.base._beliefs)
    relations_end = genesis.relations.stats()["total_relations"]

    return {
        "duration_seconds": elapsed,
        "duration_minutes": elapsed / 60,
        "articles_read": total_articles,
        "articles_failed": failed,
        "facts_gained": beliefs_end - beliefs_start,
        "relations_gained": relations_end - relations_start,
        "concepts_formed": len(formed),
        "skills_grown": len(growth),
        "domains_completed": domains_completed,
        "total_beliefs": beliefs_end,
        "total_relations": relations_end,
    }


def marathon_report(results: dict) -> str:
    lines = [
        f"Marathon Teaching Complete",
        f"══════════════════════════",
        f"  Duration: {results['duration_minutes']:.1f} minutes",
        f"  Articles read: {results['articles_read']}",
        f"  Articles failed: {results['articles_failed']}",
        f"  Facts gained: +{results['facts_gained']}",
        f"  Relations gained: +{results['relations_gained']}",
        f"  Concepts formed: {results['concepts_formed']}",
        f"  Skills grown: {results['skills_grown']}",
        f"  Total beliefs: {results['total_beliefs']}",
        f"  Total relations: {results['total_relations']}",
        f"",
        f"  Domains:",
    ]
    for domain, facts in results.get("domains_completed", []):
        lines.append(f"    {domain:25s} +{facts} facts")
    return "\n".join(lines)
