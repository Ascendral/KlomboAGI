#!/usr/bin/env python3
"""
KlomboAGI Interactive Shell — talk to the baby.

Usage: python3 -m klomboagi.shell
"""

import sys
from klomboagi.interface.conversation import Baby

def main():
    baby = Baby()
    
    print("\n  KlomboAGI — Learning System")
    print("  ─────────────────────────────")
    print(f"  Concepts: {len(baby.memory.concepts)}")
    print(f"  Beliefs: {len(baby._beliefs)}")
    print(f"  Type 'quit' to exit, 'status' for stats\n")
    
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("  Goodbye.")
            break
        if user_input.lower() == "status":
            print(baby.hear("What do you know?"))
            continue
        
        response = baby.hear(user_input)
        print(f"  Baby: {response}\n")

if __name__ == "__main__":
    main()
