#!/usr/bin/env python3
"""
KlomboAGI Interactive Shell — Cognitive Genesis.

Usage: python3 -m klomboagi.shell
"""

import sys
from klomboagi.core.genesis import Genesis


def main():
    print("\n  KlomboAGI — Cognitive Genesis")
    print("  ═══════════════════════════════")
    print("  Bootstrapping cognition from zero.")
    print("  Teach me. Ask me. Correct me. I learn.\n")

    genesis = Genesis()

    n_concepts = len(genesis.base.memory.concepts)
    n_beliefs = len(genesis.base._beliefs)
    n_relations = genesis.relations.stats()["total_relations"]
    if n_concepts > 0 or n_relations > 0:
        print(f"  Loaded {n_concepts} concepts, {n_beliefs} beliefs, {n_relations} relations.\n")
    else:
        print("  Starting empty — I know nothing yet.\n")

    print("  Commands: status, personality, teach <domain>, teach all, teach everything")
    print("           domains, connect <concept>, quit\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"\n  Session: {genesis.total_turns} turns, "
                  f"{genesis.total_surprises} surprises, "
                  f"{genesis.total_proactive} proactive questions.")
            print("  Goodbye.")
            break
        if user_input.lower() == "status":
            print(f"\n{genesis.status()}\n")
            continue
        if user_input.lower() == "personality":
            pv = genesis.traits.personality_vector()
            print("\n  Personality:")
            for name, strength in pv.items():
                bar = "█" * int(strength * 20) + "░" * (20 - int(strength * 20))
                print(f"    {name:15s} [{bar}] {strength:.0%}")
            print()
            continue
        if user_input.lower() == "domains":
            from klomboagi.core.curriculum import get_all_domains, curriculum_stats
            stats = curriculum_stats()
            print(f"\n  Available domains ({stats['total_facts']} total facts):")
            for domain, count in stats["per_domain"].items():
                print(f"    {domain:20s} {count} facts")
            print()
            continue
        if user_input.lower().startswith("teach "):
            domain = user_input[6:].strip()
            if domain == "everything":
                print("\n  Teaching everything...")
                print(f"  {genesis.teach_everything()}\n")
            elif domain == "all":
                print("\n  Teaching all fact domains...")
                print(f"  {genesis.teach_all()}\n")
            elif domain == "relations":
                print("\n  Teaching all relations...")
                print(f"  {genesis.teach_relations('all')}\n")
            else:
                print(f"\n  Teaching {domain}...")
                print(f"  {genesis.teach_domain(domain)}\n")
            continue
        if user_input.lower().startswith("connect "):
            concept = user_input[8:].strip()
            print(f"\n{genesis.what_connects(concept)}\n")
            continue

        response = genesis.hear(user_input)
        # Indent response for readability
        for line in response.split("\n"):
            print(f"  KlomboAGI: {line}")
        print()


if __name__ == "__main__":
    main()
