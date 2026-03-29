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

    genesis = Genesis(memory_path="/Volumes/AIStorage/AI/klomboagi/memory/brain.json")

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
            # Save conversation memory
            genesis.conversation_memory.end_session(
                facts_learned=genesis.total_turns,
                surprises=genesis.total_surprises,
                corrections=genesis.metacognition.metrics.corrections_received,
                turns=genesis.total_turns,
            )
            print(f"\n  Session: {genesis.total_turns} turns, "
                  f"{genesis.total_surprises} surprises, "
                  f"{genesis.total_proactive} proactive questions.")
            print("  Session saved to memory. Goodbye.")
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
        if user_input.lower().startswith("what if ") or user_input.lower().startswith("without "):
            result = genesis.counterfactual.what_if(user_input)
            print(f"\n{result.explain()}\n")
            continue
        if user_input.lower().startswith("generate "):
            concept = user_input[9:].strip()
            exp = genesis.generator.explain(concept)
            print(f"\n  {exp.text}")
            print(f"  [{exp.facts_used} facts, {exp.relations_used} relations, {'novel' if exp.novel else 'retrieved'}]\n")
            continue
        if user_input.lower().startswith("compare "):
            parts = user_input[8:].strip().split(" and ", 1)
            if len(parts) == 2:
                print(f"\n{genesis.generator.compare(parts[0].strip(), parts[1].strip())}\n")
            else:
                print("  Usage: compare X and Y\n")
            continue
        if user_input.lower() in ("concepts", "formed", "formed concepts"):
            formed = genesis.concept_former.scan()
            growth = genesis.skill_growth.integrate(formed)
            print(f"\n  {genesis.concept_former.report()}")
            if growth:
                print(f"\n  Skills grown: {len(growth)}")
                for g in growth[:10]:
                    print(f"    {g}")
            print()
            continue
        if user_input.lower() in ("skills", "skill tree", "tree"):
            print(f"\n{genesis.skill_growth.report()}\n")
            continue
        if user_input.lower() == "eval":
            from klomboagi.evals.genesis_eval import run_evals
            print("\n  Running evals...")
            report = run_evals(genesis)
            print(f"\n{report.summary()}\n")
            continue
        if user_input.lower() == "llm on":
            key = input("  API key: ").strip()
            if key:
                genesis.translator.api_key = key
                genesis.translator.enabled = True
                print("  LLM translator ENABLED. Parsing only. No reasoning. All tagged.\n")
            continue
        if user_input.lower() == "llm off":
            genesis.translator.enabled = False
            print("  LLM translator DISABLED. Using rule-based NLU.\n")
            continue
        if user_input.lower() in ("llm", "llm status"):
            print(f"\n{genesis.translator.audit_report()}\n")
            continue
        if user_input.lower() == "sources":
            # Show where beliefs came from
            sources = {}
            for b in genesis.base._beliefs.values():
                src = b.source if hasattr(b, 'source') else "unknown"
                sources[src] = sources.get(src, 0) + 1
            print("\n  Belief sources:")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"    {src}: {count}")
            print()
            continue
        if user_input.lower().startswith("simulate "):
            trigger = user_input[9:].strip()
            sim = genesis.simulator.simulate(trigger)
            print(f"\n{sim.explain()}\n")
            continue
        if user_input.lower().startswith("similar "):
            concept = user_input[8:].strip()
            similar = genesis.semantic.similar_to(concept, top_n=10)
            print(f"\n  Similar to '{concept}':")
            for c, score in similar:
                print(f"    {c}: {score:.2f}")
            print()
            continue
        if user_input.lower() in ("calibration", "calibrate"):
            print(f"\n{genesis.calibrator.report()}\n")
            continue
        if user_input.lower() in ("autonomous", "auto-goals", "formulate"):
            goals = genesis.goal_autonomy.formulate_goals()
            print(f"\n{genesis.goal_autonomy.report()}\n")
            continue
        if user_input.lower() == "pursue":
            result = genesis.goal_autonomy.pursue_next()
            print(f"\n  {result}\n")
            print(f"{genesis.goal_autonomy.report()}\n")
            continue
        if user_input.lower() in ("principles", "patterns", "generalize"):
            principles = genesis.pattern_gen.discover_all()
            print(f"\n{genesis.pattern_gen.report()}\n")
            continue
        if user_input.lower() in ("costs", "cost", "efficiency"):
            print(f"\n{genesis.cost_tracker.report()}\n")
            continue
        if user_input.lower() == "infer":
            print("\n  Running global inference...")
            derived = genesis.inference_engine.run(max_derivations=200)
            print(f"  Derived {len(derived)} new beliefs from chains.")
            for d in derived[:10]:
                print(f"    {d}")
            print()
            continue
        if user_input.lower().startswith("decompose "):
            question = user_input[10:].strip()
            subs = genesis.decomposer.decompose(question)
            print(f"\n  Sub-questions:")
            for s in subs:
                print(f"    {s}")
            print()
            continue
        if user_input.lower().startswith("solve "):
            question = user_input[6:].strip()
            sol = genesis.solver.solve(question)
            print(f"\n{sol.explain()}\n")
            continue
        if user_input.lower() in ("improve", "self-improve"):
            print("\n  Running self-improvement cycle...")
            cycle = genesis.self_improver.improve()
            print(f"  {genesis.self_improver.report()}\n")
            continue
        if user_input.lower() in ("history", "past", "conversations"):
            print(f"\n{genesis.conversation_memory.summary()}\n")
            continue
        if user_input.lower() in ("transfers", "transfer"):
            transfers = genesis.deep_transfer.scan_all()
            print(f"\n{genesis.deep_transfer.report()}\n")
            continue
        if user_input.lower() in ("lessons", "learned"):
            stats = genesis.experiential.stats()
            print(f"\n  Experiential learning: {stats['total_attempts']} attempts, {stats['lessons_learned']} lessons")
            for l in genesis.experiential.lessons[-5:]:
                print(f"    {l.insight[:70]}")
            print()
            continue
        if user_input.lower() in ("modulators", "mods", "mode"):
            print(f"\n{genesis.modulator.explain()}\n")
            continue
        if user_input.lower() in ("chunks", "compiled"):
            stats = genesis.chunker.stats()
            print(f"\n  Compiled chunks: {stats['total_chunks']}")
            for cond, conc, uses in stats['most_used']:
                print(f"    {cond} → {conc} (used {uses}x)")
            print()
            continue
        if user_input.lower() in ("workspace", "broadcast"):
            recent = genesis.workspace.recent_broadcasts()
            print(f"\n  Recent broadcasts: {recent}")
            print(f"  Stats: {genesis.workspace.stats()}\n")
            continue
        if user_input.lower() in ("cleanup", "clean", "purge"):
            print(f"\n  {genesis.cleanup_memory()}\n")
            continue
        if user_input.lower() in ("identity", "who am i", "what am i", "archetype"):
            print(f"\n  {genesis.archetype.identity()}\n")
            print(f"  {genesis.archetype.values()}\n")
            continue
        if user_input.lower() == "failures":
            s = genesis.failure_memory.stats()
            print(f"\n  Failures: {s['total_failures']} total, {s['repeated_mistakes']} repeated")
            for f in genesis.failure_memory.worst_mistakes(5):
                print(f"    x{f.times_repeated}: {f.description[:60]}")
            print()
            continue
        if user_input.lower().startswith("timeline"):
            print(f"\n{genesis.temporal.timeline()}\n")
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
        if user_input.lower().startswith("marathon"):
            from klomboagi.core.marathon import run_marathon, marathon_report
            parts = user_input.split()
            hours = float(parts[1]) if len(parts) > 1 else 2.0
            print(f"\n  Starting {hours}h marathon ({263} articles across 23 domains)...")
            import sys
            def prog(d, t, a, f, e):
                sys.stdout.write(f'\r  [{e/60:.0f}m] {d}/{t} | {a} articles | +{f} facts    ')
                sys.stdout.flush()
            results = run_marathon(genesis, max_hours=hours, on_progress=prog)
            print(f"\n\n{marathon_report(results)}\n")
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
