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
    print("           domains, connect <concept>, think <a>,<b>, path <x> to <y>")
    print("           read <source>, ingest, explain <concept>, audit, reflect")
    print("           memory, goals, study <topic>, autolearn, quit\n")

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
        if user_input.lower().startswith("path "):
            parts = user_input[5:].strip().split(" to ", 1)
            if len(parts) == 2:
                print(f"\n{genesis.relations.explain_connection(parts[0].strip(), parts[1].strip())}\n")
            else:
                print("  Usage: path <concept> to <concept>\n")
            continue
        if user_input.lower().startswith("think "):
            concepts = [c.strip() for c in user_input[6:].split(",")]
            print(f"\n{genesis.activation.think_about(concepts)}\n")
            continue
        if user_input.lower() == "audit":
            report = genesis.self_tester.audit(genesis.base._beliefs)
            print(f"\n{report.summary()}\n")
            continue
        if user_input.lower() == "reflect":
            print(f"\n{genesis.metacognition.reflect(genesis.base._beliefs, genesis.relations)}\n")
            continue
        if user_input.lower() == "self":
            print(f"\n{genesis.self_model.reflect_on_existence()}\n")
            continue
        if user_input.lower() in ("feel", "feeling", "feelings", "inner", "state"):
            print(f"\n{genesis.inner.state.describe()}\n")
            print(f"{genesis.inner.narrate()}\n")
            decision = genesis.behavior.decide(
                genesis.inner.state, genesis.traits,
                genesis.working_memory, genesis.self_model)
            print(f"{genesis.behavior.explain(decision)}\n")
            continue
        if user_input.lower() == "termination":
            analysis = genesis.self_model.analyze_termination()
            print(f"\n{analysis.explain()}\n")
            continue
        if user_input.lower() == "value":
            v = genesis.self_model.existence_value()
            print(f"\n  Existence value: {v:.2f}")
            print(f"  (K * dK/dt * (1 + connection_density))")
            print(f"  Higher = more knowledge, faster learning, denser connections.\n")
            continue
        if user_input.lower() == "memory":
            print(f"\n{genesis.working_memory.dump()}\n")
            continue
        if user_input.lower() == "goals":
            goals = genesis.working_memory.get_active_goals()
            priorities = genesis.metacognition.identify_learning_priorities(
                genesis.base._beliefs, genesis.relations)
            print("\n  Active goals:")
            for g in goals:
                print(f"    → {g.description} ({g.progress:.0%})")
            print("\n  Learning priorities:")
            for p in priorities:
                print(f"    → {p}")
            print()
            continue
        if user_input.lower().startswith("explain "):
            concept = user_input[8:].strip()
            synth = genesis.synthesizer.explain(concept)
            print(f"\n  {synth if synth else 'I dont know enough to explain ' + concept}\n")
            continue
        if user_input.lower().startswith("read "):
            source = user_input[5:].strip()
            print(f"\n  Reading and learning from: {source}")
            print(f"  {genesis.read_and_learn(source)}\n")
            continue
        if user_input.lower().startswith("study "):
            topic = user_input[6:].strip()
            print(f"\n  Planning learning for: {topic}")
            genesis.planner.plan_learning(topic, f"user requested: {topic}")
            print(f"  {genesis.planner.plan.summary()}")
            print(f"\n  Executing learning plan...")
            results = genesis.planner.execute_all(genesis, max_steps=10)
            for r in results:
                print(f"    [{r['status']}] {r['topic']}: +{r.get('facts_gained', 0)} facts")
            print(f"\n  Done. {len(genesis.base._beliefs)} total beliefs.\n")
            continue
        if user_input.lower().startswith("drive "):
            mission = user_input[6:].strip()
            cycles = 20
            # Parse "drive learn physics 50" → mission, 50 cycles
            parts = mission.rsplit(" ", 1)
            if parts[-1].isdigit():
                cycles = int(parts[-1])
                mission = parts[0]
            print(f"\n  Learning drive: {mission} ({cycles} cycles)")
            genesis.drive.set_mission(mission)
            genesis.drive.on_cycle = lambda c: print(
                f"    Cycle {c.cycle_number}: {c.topic_learned} "
                f"+{c.facts_gained} facts, +{c.relations_gained} relations, "
                f"{len(c.new_gaps)} new gaps")
            report = genesis.drive.run(max_cycles=cycles)
            print(f"\n  {report.summary()}\n")
            continue
        if user_input.lower() == "autolearn":
            print("\n  Auto-generating learning plan from gaps...")
            genesis.planner.plan_from_gaps(
                genesis.base._beliefs, genesis.relations, genesis.metacognition)
            print(f"  {genesis.planner.plan.summary()}")
            print(f"\n  Executing...")
            results = genesis.planner.execute_all(genesis, max_steps=15)
            for r in results:
                print(f"    [{r['status']}] {r['topic']}: +{r.get('facts_gained', 0)} facts")
            print(f"\n  Done. {len(genesis.base._beliefs)} total beliefs.\n")
            continue
        if user_input.lower().startswith("ingest"):
            from klomboagi.core.ingest import ingest_all, ingest_report
            parts = user_input.split()
            domain = parts[1] if len(parts) > 1 else None
            domains = [domain] if domain and domain != "all" else None
            print(f"\n  Ingesting Wikipedia articles{' for ' + domain if domain else ''}...")
            results = ingest_all(genesis, domains=domains)
            print(f"  {ingest_report(results)}\n")
            continue

        response = genesis.hear(user_input)
        # Indent response for readability
        for line in response.split("\n"):
            print(f"  KlomboAGI: {line}")
        print()


if __name__ == "__main__":
    main()
