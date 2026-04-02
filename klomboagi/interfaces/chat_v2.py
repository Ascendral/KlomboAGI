"""
KlomboAGI Chat v2 -- talks directly to the Brain, no HTTP.

Usage: python3 -m klomboagi brain
"""

from __future__ import annotations

import sys
from klomboagi.core.brain import Brain


def chat_v2():
    """Direct conversation with the Brain. No server needed."""
    brain = Brain()

    print(f"\n  KlomboAGI Brain -- {brain.reasoner.total_facts} facts loaded")
    print(f"  Type 'quit' to exit.\n")

    while True:
        try:
            msg = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break

        response = brain.hear(msg)
        print(f"\nklombo: {response}\n")

    brain._save()
    print("Brain saved.")


if __name__ == "__main__":
    chat_v2()
