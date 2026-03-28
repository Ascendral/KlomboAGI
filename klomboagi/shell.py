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
    if n_concepts > 0:
        print(f"  Loaded {n_concepts} concepts, {n_beliefs} beliefs from memory.\n")
    else:
        print("  Starting empty — I know nothing yet.\n")

    print("  Commands: 'status', 'personality', 'quit'\n")

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

        response = genesis.hear(user_input)
        # Indent response for readability
        for line in response.split("\n"):
            print(f"  KlomboAGI: {line}")
        print()


if __name__ == "__main__":
    main()
