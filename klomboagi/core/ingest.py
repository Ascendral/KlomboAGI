"""
Mass Wikipedia Ingest — read dozens of articles and build a knowledge base.

Usage:
    from klomboagi.core.ingest import TOPICS, ingest_all
    results = ingest_all(genesis)
"""

from __future__ import annotations


# Topics organized by domain — each is a Wikipedia article title
TOPICS: dict[str, list[str]] = {
    "mathematics": [
        "Mathematics", "Arithmetic", "Algebra", "Geometry",
        "Calculus", "Linear_algebra", "Statistics",
        "Probability_theory", "Number_theory", "Trigonometry",
        "Set_theory", "Graph_theory", "Topology",
    ],
    "physics": [
        "Physics", "Classical_mechanics", "Quantum_mechanics",
        "General_relativity", "Thermodynamics", "Electromagnetism",
        "Optics", "Nuclear_physics", "Particle_physics",
        "Gravity", "Wave", "Energy",
    ],
    "chemistry": [
        "Chemistry", "Atom", "Molecule", "Chemical_element",
        "Chemical_bond", "Periodic_table", "Organic_chemistry",
        "Biochemistry", "Acid", "Oxidation",
    ],
    "biology": [
        "Biology", "Cell_(biology)", "DNA", "Evolution",
        "Genetics", "Ecology", "Neuron", "Photosynthesis",
        "Protein", "Virus",
    ],
    "computer_science": [
        "Computer_science", "Algorithm", "Data_structure",
        "Turing_machine", "Computational_complexity_theory",
        "Artificial_intelligence", "Machine_learning",
        "Cryptography", "Operating_system", "Computer_network",
    ],
    "economics": [
        "Economics", "Microeconomics", "Macroeconomics",
        "Supply_and_demand", "Inflation", "Game_theory",
        "Monetary_policy", "Gross_domestic_product",
    ],
}


def get_all_topics() -> list[str]:
    """All topics across all domains."""
    all_topics = []
    for topics in TOPICS.values():
        all_topics.extend(topics)
    return all_topics


def ingest_all(genesis, domains: list[str] | None = None,
               max_per_domain: int = 15) -> dict:
    """
    Read all Wikipedia articles and learn from them.

    Returns stats about what was learned.
    """
    target_domains = domains or list(TOPICS.keys())
    results = {
        "domains_processed": 0,
        "articles_read": 0,
        "articles_failed": 0,
        "facts_before": len(genesis.base._beliefs),
        "facts_after": 0,
        "relations_before": genesis.relations.stats()["total_relations"],
        "relations_after": 0,
        "per_domain": {},
    }

    for domain in target_domains:
        topics = TOPICS.get(domain, [])[:max_per_domain]
        if not topics:
            continue

        domain_facts_before = len(genesis.base._beliefs)
        domain_rels_before = genesis.relations.stats()["total_relations"]
        articles_read = 0

        for topic in topics:
            try:
                result = genesis.read_and_learn(topic)
                if "Could not read" not in result:
                    articles_read += 1
                else:
                    results["articles_failed"] += 1
            except Exception:
                results["articles_failed"] += 1

        domain_facts_after = len(genesis.base._beliefs)
        domain_rels_after = genesis.relations.stats()["total_relations"]

        results["per_domain"][domain] = {
            "articles": articles_read,
            "new_facts": domain_facts_after - domain_facts_before,
            "new_relations": domain_rels_after - domain_rels_before,
        }
        results["articles_read"] += articles_read
        results["domains_processed"] += 1

    results["facts_after"] = len(genesis.base._beliefs)
    results["relations_after"] = genesis.relations.stats()["total_relations"]

    # Run inference after all ingestion
    inferred = genesis.relations.run_inference()
    results["inferred"] = len(inferred)

    # Save
    genesis.base.memory.save(genesis.base.memory_path)
    genesis.save_state()

    return results


def ingest_report(results: dict) -> str:
    """Human-readable report of ingest results."""
    lines = [
        f"Wikipedia Ingest Complete",
        f"  Domains: {results['domains_processed']}",
        f"  Articles read: {results['articles_read']}",
        f"  Articles failed: {results['articles_failed']}",
        f"  Facts: {results['facts_before']} → {results['facts_after']} "
        f"(+{results['facts_after'] - results['facts_before']})",
        f"  Relations: {results['relations_before']} → {results['relations_after']} "
        f"(+{results['relations_after'] - results['relations_before']})",
        f"  Inferred: {results.get('inferred', 0)}",
        "",
        "Per domain:",
    ]
    for domain, stats in results.get("per_domain", {}).items():
        lines.append(
            f"  {domain:20s} {stats['articles']} articles, "
            f"+{stats['new_facts']} facts, +{stats['new_relations']} relations"
        )
    return "\n".join(lines)
