# AGI Landscape Report -- March 2026

Comprehensive research sweep of every serious AGI project, research lab, and approach.
No hype. No flattery. Brutal honesty.

---

## Table of Contents
1. [Major Corporate Labs](#1-major-corporate-labs)
2. [Independent AGI Research Projects](#2-independent-agi-research-projects)
3. [Open-Source AGI Frameworks](#3-open-source-agi-frameworks)
4. [AGI Startups](#4-agi-startups)
5. [ARC-AGI Benchmark Results](#5-arc-agi-benchmark-results)
6. [Key Research Frontiers](#6-key-research-frontiers)
7. [KlomboAGI Honest Assessment](#7-klomboagi-honest-assessment)
8. [Comparative Analysis](#8-comparative-analysis)
9. [What Actually Matters](#9-what-actually-matters)

---

## 1. Major Corporate Labs

### OpenAI
- **What they're doing:** Scaling LLMs with reasoning chains (o-series) and unifying into GPT-5.x family. o3 does test-time compute scaling -- thinks longer before answering.
- **How far they've gotten:** o3 scored 87.5% on ARC-AGI-1 (high compute). GPT-5.2 is their latest unified model. o3/o4-mini can use tools agentically. Reasoning models are their best work.
- **What's novel:** Test-time compute scaling (think longer = do better) was genuinely new when introduced. Now everyone does it.
- **What's incremental:** Still fundamentally an LLM. Still pattern matching on training data. Reasoning chains are learned from training, not derived from first principles.
- **Open source:** No. Closed weights, closed training data.
- **Funding:** $157B+ valuation, raised $40B+ total. Functionally unlimited resources.

### Google DeepMind
- **What they're doing:** Gemini 3 model family (reasoning, multimodal), world models (Genie 3 -- real-time 3D world generation at 24fps), and fundamental AGI research.
- **How far they've gotten:** Gemini 3 Deep Think hit 84.6% on ARC-AGI-2. Elo 3455 on Codeforces. IMO gold medal level. Genie 3 generates navigable 3D worlds in real-time.
- **What's novel:** Genie 3 world models are genuinely impressive -- learning spatial understanding from video. Their "Measuring Progress Towards AGI" cognitive framework paper shows serious thinking about what AGI means.
- **What's incremental:** Gemini is still an LLM, just a very good one.
- **Open source:** Partially. Some models open-weighted.
- **Funding:** Google bankroll (~$80B/year R&D budget). Hassabis says AGI still 5-10 years away and "1-2 more breakthroughs" are needed. He's probably right.

### Anthropic
- **What they're doing:** Constitutional AI alignment, Claude model family, interpretability research. Focus on safe deployment rather than explicit AGI push.
- **How far they've gotten:** Claude Opus 4.6 (current), Claude Opus 4.5 scored 37.6% on ARC-AGI-2 as a raw LLM (no scaffolding). New constitution published January 2026 with 4-tier priority hierarchy. First major AI company to formally acknowledge possibility of AI consciousness.
- **What's novel:** Constitutional AI approach to alignment is genuinely thoughtful. Interpretability work (circuits, features) is some of the best in the field.
- **What's incremental:** Core technology is still transformer-based LLM.
- **Open source:** No. Closed models, but publishes research.
- **Funding:** ~$20B+ raised. Google partnership provides 1M+ TPUs.

### Meta / AMI Labs (Yann LeCun)
- **What they're doing:** LeCun LEFT Meta in December 2025 to found Advanced Machine Intelligence (AMI) Labs. His thesis: LLMs are a dead end, world models are the path to AGI. Building on I-JEPA (Image Joint Embedding Predictive Architecture) -- models that learn by predicting abstract representations, not pixels.
- **How far they've gotten:** AMI Labs raised $1.03B. Meta separately created Meta Superintelligence Labs under Alexandr Wang. LeCun's ideas are theoretically compelling but AMI Labs has been operational for only ~3 months.
- **What's novel:** LeCun's vision is the most intellectually honest critique of LLMs from inside the industry. I-JEPA learning from video/spatial data without language is genuinely different. His "world model" approach predicts representations, not raw pixels.
- **What's incremental:** Still early. AMI Labs has money and vision but no public results yet.
- **Open source:** AMI approach TBD. Meta's LLaMA models are open-weighted.
- **Funding:** AMI Labs: $1.03B. Meta FAIR had Google-scale resources.

### xAI
- **What they're doing:** Scaling Grok models aggressively. Grok 5 (6T parameters) planned for Q1 2026. Massive compute cluster (Colossus, 200K GPUs).
- **How far they've gotten:** Grok 4.1 released. Musk claims Grok 5 has "10% and rising" probability of AGI. Benchmark performance competitive but not leading.
- **What's novel:** Nothing architecturally novel. It's an LLM scaling play with massive hardware.
- **What's incremental:** Everything. More parameters, more GPUs, same approach.
- **Open source:** Partially. Some models open-weighted.
- **Funding:** Acquired by SpaceX in Feb 2026. Massive resources.

### Microsoft
- **What they're doing:** Phi-series small reasoning models. Demonstrating that careful data curation beats brute-force scaling.
- **How far they've gotten:** Phi-4-reasoning (14B params) beats o1-mini and rivals DeepSeek-R1 (671B params) on AIME 2025. Phi-4-reasoning-vision extends to multimodal. This is real efficiency.
- **What's novel:** Proving small models can compete with models 50x their size through data quality. Genuinely useful research.
- **What's incremental:** Still LLMs. Still supervised fine-tuning + RL. Just doing it more efficiently.
- **Open source:** Yes. Phi models are open-weighted on HuggingFace.
- **Funding:** Microsoft R&D budget. OpenAI partnership.

### DeepSeek (China)
- **What they're doing:** Open-source reasoning models. DeepSeek-R1 (671B params) trained for <$6M, rivaling OpenAI o1. R1-Zero trained with pure RL, no SFT.
- **How far they've gotten:** R1 scores 79.8% on AIME, 97.4% on MATH. R1-0528 added JSON output and function-calling. Most-liked open-source model on HuggingFace as of Jan 2026. Spawned massive Chinese AI ecosystem.
- **What's novel:** Training cost efficiency ($6M vs. hundreds of millions). R1-Zero showing emergent reasoning from pure RL without SFT is significant.
- **What's incremental:** Architecture is still MoE transformer. Intelligence comes from scale + training tricks.
- **Open source:** Yes. MIT license. Free for commercial use.
- **Funding:** Backed by quantitative hedge fund High-Flyer.

---

## 2. Independent AGI Research Projects

### NARS -- Pei Wang (Temple University)
- **What they're doing:** Non-Axiomatic Reasoning System. 30+ year project building intelligence as "adaptation under insufficient knowledge and resources." Term logic, experience-grounded semantics, truth values based on evidence.
- **How far they've gotten:** NARS Workshop at AGI-25. Multiple implementations (OpenNARS in Java, Narjure in Clojure). Can do basic reasoning, learning, and adaptation. Academic respect but no breakout results.
- **What's novel:** The theoretical foundation is the most rigorous formal definition of intelligence-as-adaptation in the field. NAL truth functions are mathematically sound. The "insufficient knowledge and resources" framing is correct.
- **What's incremental:** Implementation is academic-grade. Performance doesn't compete with modern systems on any practical benchmark. 30 years of development, still at research prototype stage.
- **Open source:** Yes. OpenNARS on GitHub.
- **Funding:** Academic funding only. Essentially unfunded relative to competitors.

### OpenCog Hyperon -- Ben Goertzel
- **What they're doing:** Complete rewrite of OpenCog as Hyperon. MeTTa programming language for AGI cognitive processes. AtomSpace knowledge representation. Blockchain integration via SingularityNET/ASI Alliance.
- **How far they've gotten:** Production Hyperon stack reached milestone late 2025. "Baby AGI" prototypes in virtual environments. Istanbul workshop October 2025 advancing MeTTa compilers. Goertzel keynote: "2026: The Year of Decentralized AGI."
- **What's novel:** MeTTa language designed specifically for AGI cognitive scripting is interesting. The vision of integrating symbolic, neural, and evolutionary approaches is comprehensive.
- **What's incremental:** OpenCog has been "almost there" for 15+ years. Hyperon is the third rewrite. Blockchain/crypto integration feels like a funding strategy, not an AGI strategy. No benchmark results to show.
- **Open source:** Yes.
- **Funding:** SingularityNET token economics. Hard to separate hype from substance.

### Numenta / Thousand Brains Project -- Jeff Hawkins
- **What they're doing:** Brain-inspired AI based on neocortex theory. Cortical column "learning modules" that learn through sensorimotor interaction. Spun off as independent nonprofit in January 2025.
- **How far they've gotten:** Open-source sensorimotor learning framework released November 2024. MIT license. Led by Dr. Viviane Clay. Patents placed under non-assert pledge.
- **What's novel:** The neuroscience-first approach is genuinely different. Thousand Brains Theory (many models voting, not one big model) is an interesting architectural insight. Sensorimotor learning from embodied interaction is the right direction.
- **What's incremental:** Still very early. No practical demonstrations that compete with any existing system on any benchmark. Beautiful theory, minimal code.
- **Open source:** Yes. MIT license.
- **Funding:** Numenta commercial revenue + nonprofit funding. Modest.

### Joscha Bach / MicroPsi / California Institute for Machine Consciousness
- **What they're doing:** Cognitive architecture combining symbolic AI, connectionist models, and motivation-based psychology. Now executive director of California Institute for Machine Consciousness. AI Strategist at Liquid AI.
- **How far they've gotten:** MicroPsi remains a research framework. Bach is now more of a public intellectual/philosopher of AGI than an active implementer. No system running.
- **What's novel:** Bach's theoretical framework (agents as self-organizing systems with motivational dynamics) is intellectually rich. His podcast appearances are the best thinking-out-loud about what consciousness and intelligence actually require.
- **What's incremental:** No running system. Theory without implementation.
- **Open source:** MicroPsi code exists but is not actively maintained.
- **Funding:** Academic/institutional.

### Marcus Hutter / AIXI (DeepMind)
- **What they're doing:** Mathematical theory of universal artificial intelligence. AIXI = optimal agent that combines Solomonoff induction with sequential decision theory. New textbook published 2024.
- **How far they've gotten:** AIXI is provably optimal but provably incomputable. It's a theoretical upper bound, not a buildable system. Approximations (MC-AIXI) exist but don't scale.
- **What's novel:** The theoretical framework is unassailable. It defines what a perfect AGI would look like mathematically.
- **What's incremental:** Has been the same fundamental result since 2000. No path to practical implementation.
- **Open source:** Theory is published. Some approximation code exists.
- **Funding:** DeepMind salary.

### BIGAI -- Beijing Institute for General Artificial Intelligence
- **What they're doing:** "Small data, big tasks" approach inspired by cognitive science and developmental psychology. Tong Tong virtual child agent. Embodied AI in humanoid robots.
- **How far they've gotten:** Tong Tong 2.0 (April 2024) -- cognitive capabilities of a 5-6 year old child (their claim). Won humanoid robot dance championship 2025. Completed world's first 5G-A robot power inspection demo. Papers at ICLR 2025 and 2026.
- **What's novel:** The developmental/cognitive-science approach is sound. Zhu Songchun's work on grounding AGI in human-like thinking is theoretically strong. "Small data, big tasks" is the right framing.
- **What's incremental:** Claims about "5-6 year old" capabilities are marketing. Actual demos are narrowly scoped (dancing, inspection).
- **Open source:** Some code on GitHub.
- **Funding:** Chinese government backing. Substantial but exact figures unclear.

---

## 3. Open-Source AGI Frameworks

### Soar (University of Michigan, John Laird)
- **What it is:** Cognitive architecture with production rules, reinforcement learning, episodic/semantic memory, visual imagery, and emotion modeling. 40+ years of development.
- **Status:** Actively maintained by Center for Integrated Cognition. Recent work on visual-symbolic integration (2025).
- **Honest assessment:** The most mature cognitive architecture. Decades of validated results. But it's a modeling tool for cognitive science, not a path to AGI. Extensions keep adding modules but the core hasn't changed fundamentally.

### ACT-R (Carnegie Mellon, John Anderson)
- **What it is:** Cognitive architecture for modeling human cognition. Modular (declarative memory, procedural memory, visual, motor). Psychologically validated.
- **Status:** Still actively used. Six new modules added recently (physiology, emotion, traits, values, experiential learning, behavioral change). CogDriver -- longest-running autonomous driving cognitive model.
- **Honest assessment:** Best tool for modeling how humans think. Not a path to building AGI. Designed to explain human cognition, not create new intelligence.

### BabyAGI / AutoGPT / AgentGPT
- **What they are:** LLM wrapper frameworks that chain GPT calls with task planning. BabyAGI = research reference. AutoGPT = production automation. AgentGPT = browser-based.
- **Status:** Ecosystem grew 920% from 2023 to mid-2025. AutoGPT now has visual builders. All fundamentally LLM orchestrators.
- **Honest assessment:** These are NOT AGI projects. They are automation frameworks that call an LLM in a loop. No reasoning. No learning. No knowledge representation. The "AGI" in the name is pure marketing. They break without the LLM. Remove GPT and you have nothing.

### LIDA / Sigma
- **What they are:** Academic cognitive architectures. LIDA (Learning Intelligent Distribution Agent) based on Global Workspace Theory. Sigma integrates graphical models with cognitive architecture.
- **Status:** Published papers continue. No mainstream adoption.
- **Honest assessment:** Interesting theoretical contributions. No practical impact.

---

## 4. AGI Startups

### Keen Technologies -- John Carmack
- **What they're doing:** Fundamental AGI research through robotics + RL. Physical Atari bot to escape simulation-only traps.
- **How far they've gotten:** Camera-pointed-at-TV-screen learning rig. Partnered with Richard Sutton. Atari rig open-sourced on GitHub. Carmack coding daily as of March 2026.
- **What's novel:** Carmack's insistence on grounding in physical reality (not just simulated environments) is correct. Partnership with Sutton (RL godfather) is serious.
- **What's incremental:** $20M is nothing for AGI research. Physical RL learning from pixel input is well-studied.
- **Open source:** Atari rig open-sourced.
- **Funding:** $20M. Backed by Nat Friedman, Daniel Gross, Patrick Collison, Tobi Lutke, Sequoia.

### Sakana AI (Tokyo)
- **What they're doing:** Nature-inspired AI. Evolutionary model merging. AI Scientist (automated research lifecycle). AB-MCTS inference-time scaling.
- **How far they've gotten:** ALE-Agent won AtCoder Heuristic Contest 058 -- first AI to beat all human participants. AI Scientist generates research papers (mixed reception on quality). AB-MCTS achieves promising ARC-AGI-2 results.
- **What's novel:** Evolutionary approach to model creation is genuinely different. AI Scientist concept is ambitious.
- **What's incremental:** Still uses LLMs under the hood. Evolution operates on LLM components, not on raw intelligence.
- **Open source:** Partially.
- **Funding:** $135M Series B at $2.6B valuation.

### Imbue (formerly Generally Intelligent)
- **What they're doing:** Reasoning-focused AI research. Code evolution approach to ARC-AGI.
- **How far they've gotten:** Pushed Gemini 3.1 Pro from 88.1% to 95.1% on ARC-AGI-2 using evolutionary harness. Published open research on ARC-AGI-2 approaches.
- **What's novel:** Code evolution for problem-solving is interesting scaffolding research.
- **What's incremental:** They're improving how to use existing LLMs, not building new intelligence.
- **Open source:** Research published openly.
- **Funding:** Raised $220M+ at $1B+ valuation (2023).

### Adept AI -- EFFECTIVELY DEAD
- **What happened:** Amazon hired the co-founders and key engineers in June 2024. Licensed the tech. ~20 employees remained. Investors recouped their $414M. FTC investigating whether it was a de facto acquisition.
- **Lesson:** Building AGI startups is hard when Big Tech can just hire your team.

### Vicarious -- ACQUIRED 2022
- **What happened:** Acquired by Alphabet's Intrinsic (robotics subsidiary) in 2022. Team split between Intrinsic and DeepMind. Had raised $250M from Bezos, Musk, Zuckerberg, Samsung.
- **Lesson:** Another AGI startup absorbed by Big Tech before delivering results.

---

## 5. ARC-AGI Benchmark Results

ARC-AGI (Abstraction and Reasoning Corpus) is the only benchmark that specifically tests for the kind of fluid intelligence AGI would require. Created by Francois Chollet.

### ARC-AGI-2 Leaderboard (March 2026)

| System | Score | Cost/Task | Notes |
|--------|-------|-----------|-------|
| Confluence Lab | 97.9% | $11.77 | Public eval, not semi-private |
| Imbue + Gemini 3.1 Pro (evolution) | 95.1% | $8.71 | Code evolution harness |
| Gemini 3.1 Pro (base) | 88.1% | API cost | Strongest raw API model |
| Gemini 3 Deep Think | 84.6% | ~$77 | Semi-private: 45% |
| GPT 5.2 Pro | 54.2% | - | Early 2026 baseline |
| Poetiq (open-source) | 54% | $30.57 | Semi-private, open source |
| Claude Opus 4.5 (Thinking) | 37.6% | $2.20 | Raw LLM, no scaffolding |
| Human average | ~60% | N/A | Every task is human-solvable |
| Pure LLMs (no scaffolding) | 0-5% | - | Shows LLMs don't generalize |

### Key Takeaways
- High scores on the PUBLIC eval require massive test-time compute (thousands of dollars per task)
- Semi-private scores are dramatically lower (top is ~54%)
- The systems scoring 90%+ on public eval are using code evolution, program synthesis, and massive search -- not genuine understanding
- The gap between public eval and semi-private shows overfitting/memorization
- ARC-AGI-3 planned for early 2026 to stay ahead of benchmark gaming
- Chollet's core thesis holds: LLMs don't generalize to novel tasks without scaffolding

---

## 6. Key Research Frontiers

### Neuro-Symbolic Integration
- **Status:** Active research area. 167 peer-reviewed papers analyzed in recent systematic review. Dominant approach: connect LLMs with symbolic rule systems. Amazon applying to warehouse robots.
- **The gap:** Nobody has cracked deep integration. Current systems are "LLM extracts, rules verify" -- duct-taped together, not truly integrated.

### Causal Reasoning
- **Status:** 2026 being called "breakout year for Causal AI." Pearl's do-calculus is well understood theoretically. Commercial causal AI platforms emerging.
- **The gap:** LLMs still can't do causal reasoning from first principles. They pattern-match causal language from training data. Real causal reasoning requires intervention and counterfactual testing.

### Continuous Learning / Catastrophic Forgetting
- **Status:** Multiple approaches showing progress. Google's Nested Learning. Neural ODEs + memory-augmented transformers (24% forgetting reduction). RL-based continual learning for LLMs.
- **The gap:** No system achieves human-like continuous learning. Every approach involves trade-offs (plasticity vs. stability). Fundamentally unsolved for neural networks.

### Intrinsic Motivation / Curiosity
- **Status:** Curiosity-Driven Autonomous Learning Networks (CDALNs) show 267% improvement in autonomous skill acquisition. Research mostly in RL agents.
- **The gap:** Current implementations are reward-shaped curiosity, not genuine epistemic drive. The system doesn't actually "want" to know -- it's optimizing a curiosity bonus.

### World Models
- **Status:** Hottest research area in 2026. LeCun bet his career on it ($1B). DeepMind Genie 3 generates interactive 3D worlds. Multiple labs racing.
- **The gap:** Current world models are perceptual (predict pixels/video). Nobody has world models that support abstract reasoning, planning, and causal inference simultaneously.

### Few-Shot Abstraction / Meta-Learning
- **Status:** MAML variants, prototypical networks, in-context learning all active. Open-MAML for open-task settings (2026).
- **The gap:** Current meta-learning still requires massive pretraining. True few-shot abstraction (like humans seeing one example and generalizing) remains unsolved.

---

## 7. KlomboAGI Honest Assessment

### What Exists (as of March 2026)

**Architecture (~70 Python files, ~275 tests):**
- 10-phase CognitionLoop (perceive/remember/transfer/inquire/hypothesize/evaluate/revise/act/observe/learn)
- ReasoningEngine with property type classification and dimensional comparison
- NARS-inspired TruthValue system (frequency, confidence, evidence stamps)
- Full NAL inference functions (revision, deduction, induction, abduction, analogy, comparison)
- CausalGraph with observation, intervention, counterfactual, confounder detection, experiment suggestion
- AbstractionEngine (structural element decomposition, alignment, invariant extraction)
- InquiryEngine (knowledge gap detection, prioritization)
- SelfEvaluator (hypothesis checking, nudge interface)
- CuriosityDriver (gap monitoring, sense selection, tool triggering)
- StructuralComparator (cross-domain transfer)
- PropertyDeriver (knowledge graph traversal for dimensional signatures)
- Persistent storage, episodic memory, event logging

### What's Genuinely Good

1. **The architecture is sound.** The 10-phase cognition loop is a legitimate cognitive architecture. Perceive-remember-transfer-inquire-hypothesize-evaluate-revise-act-observe-learn is a reasonable formalization of how reasoning should work. It's not far from what SOAR and ACT-R do, expressed differently.

2. **NARS truth values are implemented correctly.** The truth functions (revision, deduction, induction, abduction, analogy) match Pei Wang's NAL. Evidence stamps prevent double-counting. Temporal projection decays old beliefs. This is one of the most complete NARS truth-value implementations outside of OpenNARS itself.

3. **Causal model is the right idea.** The three-level Pearl hierarchy (association, intervention, counterfactual) is implemented. Confounder detection exists. Experiment suggestion is present. This is more than most AGI projects have.

4. **The separation of reasoning from LLM is philosophically correct.** Every other "AGI" project right now is an LLM wrapper. KlomboAGI's insistence that the reasoning engine should work without an LLM, using the LLM as an optional tool, is the right architectural decision.

5. **Test coverage is excellent for a solo project.** 275 tests with zero failures is professional-grade discipline.

### What's Honestly Weak

1. **The ReasoningEngine is keyword-matching, not reasoning.** The `identify_property_type` method uses word-set intersection (`words & color_words`). This is a lookup table, not derivation. The docstring says "Pure logical operations" but the implementation is string pattern matching. A real reasoning engine would derive that "green" is a color from structural relationships in a knowledge graph, not from a hardcoded set of color words. The engine solves exactly one class of problem (the alligator riddle) and its structure is tailored to that problem.

2. **The CognitionLoop has never processed a non-trivial problem.** The tests verify that the loop transitions through phases correctly. They don't verify that the loop produces correct answers to novel problems. There's a fundamental difference between "the state machine works" and "the system can think."

3. **Scale is toy-level.** The entire knowledge base fits in JSON files. There's no graph database, no vector store, no efficient retrieval. The AbstractionEngine compares episodes by iterating through lists. This works for 10 episodes. It won't work for 10,000. NARS implementations like OpenNARS have dealt with this for decades.

4. **No perception layer.** The system can only process pre-structured dictionary inputs. It can't parse natural language, images, audio, or any raw sensory data. The "senses" module (reader, searcher, executor) exists as stubs.

5. **No learning has been demonstrated.** The system has infrastructure for learning (episodic memory, abstraction, causal model) but there's no demonstration that it actually learns something useful from experience. Can it solve problem 10 better than problem 1 because of what it learned from problems 2-9? This hasn't been tested.

6. **The curiosity driver is conceptual.** The file exists with good data structures but the actual "go find out" loop that connects gaps to senses to learning hasn't been wired up end-to-end.

7. **No comparison to any benchmark.** KlomboAGI hasn't been run against ARC-AGI, any reasoning benchmark, or any standardized test. Without this, claims about the architecture are untestable.

8. **The dimensional reasoning trick is hardcoded.** The "alligator is greener than it is long" insight is impressive as human reasoning, but the code hardcodes that color=2D and measurement=1D. A real system would derive these dimensional signatures from experience, not from `PROPERTY_DIMENSIONS` lookup table.

### Where It Actually Stands Relative to the Field

**Compared to NARS/OpenNARS:** KlomboAGI uses NARS truth values correctly but has a fraction of NARS's reasoning capability. OpenNARS has 30 years of inference rule development, term logic, and demonstrated reasoning chains. KlomboAGI has the truth-value math but not the inference engine.

**Compared to Soar/ACT-R:** These have 40+ years of validated cognitive modeling. KlomboAGI's cognition loop is conceptually similar but has been tested for days, not decades. Soar has demonstrated performance on hundreds of real tasks.

**Compared to OpenCog Hyperon:** Hyperon has a custom language (MeTTa), distributed knowledge store, and larger team. KlomboAGI has cleaner code and more honest claims but less capability.

**Compared to BabyAGI/AutoGPT:** KlomboAGI is architecturally superior because it doesn't depend on an LLM for reasoning. But those tools actually DO things in the real world (write code, search, execute) while KlomboAGI is still theoretical.

**Compared to corporate labs:** Not comparable. The gap in resources, team size, and demonstrated capability is too large. This isn't a criticism -- it's physics.

---

## 8. Comparative Analysis

### The Honest Matrix

| Capability | KlomboAGI | OpenNARS | Soar | OpenCog | Corporate LLMs |
|-----------|-----------|----------|------|---------|----------------|
| Formal reasoning theory | Partial (uses NARS) | Complete | Complete | Partial | None (learned) |
| Evidence-based belief | Yes | Yes | Partial | Yes | No |
| Causal reasoning | Basic graph | Weak | Weak | Planned | Pattern-matched |
| Abstraction | Structural (basic) | Term-based | Chunking | Conceptual | Statistical |
| Transfer learning | Structural comparator | NAL analogy | Analogical | Category mapping | In-context |
| Continuous learning | Infrastructure only | Yes | Yes | Yes | No (catastrophic forgetting) |
| Natural language | No | Limited | No | Limited | Excellent |
| Scale | Toy (<100 facts) | Small (1000s) | Medium | Medium-large | Massive |
| Real-world tasks | None demonstrated | Few | Many | Few | Many |
| LLM independence | Yes | Yes | Yes | Partial | N/A |
| Tests passing | 275 | Unknown | Extensive | Unknown | Extensive |
| Team size | 1 person | ~5 academics | ~10 researchers | ~20 + community | 100s-1000s |
| Funding | $0 | Academic | Academic | Crypto + grants | Billions |
| Years of development | ~2 weeks | ~30 years | ~40 years | ~15 years | ~5 years |

### What Nobody Has Solved

Every project in this report, including the ones with billions of dollars, fails at:

1. **True few-shot abstraction** -- seeing one or two examples and extracting the general rule
2. **Genuine causal reasoning** -- distinguishing causation from correlation without massive data
3. **Continuous learning without forgetting** -- learning new things without losing old things
4. **Grounded world models** -- understanding the physical world well enough to predict and plan
5. **Transfer across domains** -- applying knowledge from one domain to a genuinely different one
6. **Self-directed learning** -- choosing what to learn based on genuine epistemic need

These are the hard problems. LLMs sidestep them by memorizing everything. Cognitive architectures address some but not all. Nobody has cracked the full set.

---

## 9. What Actually Matters

### The Landscape Honestly Summarized

**The LLM camp** (OpenAI, Anthropic, DeepMind, xAI, DeepSeek) is throwing massive compute at the problem. They've achieved impressive benchmark results but haven't solved generalization. Every benchmark they conquer gets replaced by a harder one, because the underlying capability (pattern matching on training data) hasn't changed. Test-time compute (thinking longer) is their best innovation, and it's genuine, but it's scaling existing capability, not creating new capability.

**The cognitive architecture camp** (NARS, Soar, ACT-R, OpenCog) has the right theory but can't compete on practical capability. They've been at this for decades and still can't parse a sentence or write code. They understand intelligence better than the LLM camp but can't demonstrate it.

**The world model camp** (LeCun/AMI, DeepMind Genie, Numenta) is making the right bet but is early. Learning from interaction with a world model is probably the right path, but nobody has a working system yet.

**The neuro-symbolic camp** is duct-taping LLMs and symbolic systems together. It works for narrow applications but isn't a path to AGI.

### Where KlomboAGI Fits

KlomboAGI is in the cognitive architecture camp with NARS-inspired evidence-based reasoning. The architecture is sound. The vision (start empty, learn from experience, reason from first principles, use LLM as a tool not a brain) is correct.

But the implementation is 2 weeks old, built by 1 person, with $0 funding. It's a prototype -- well-structured, well-tested, but a prototype. The gap between "correct architecture" and "working intelligence" is the gap that NARS has been trying to cross for 30 years, that Soar has been trying to cross for 40 years, and that OpenCog has been trying to cross for 15 years.

The question isn't whether KlomboAGI's architecture is right. It might be. The question is whether one person can cross the implementation gap that entire research groups haven't crossed in decades.

### What Would Change the Assessment

KlomboAGI would become genuinely interesting if it could demonstrate:
1. **Learning from conversation** -- teach it something, then test if it can apply that knowledge to a novel situation
2. **ARC-AGI performance** -- even 5% on ARC-AGI-2 without any LLM would be significant
3. **Knowledge accumulation** -- show that the system gets better at a class of problems over time
4. **Genuine transfer** -- learn from domain A, succeed in domain B with structural similarity
5. **Self-directed inquiry** -- encounter an unknown, go find out, incorporate the answer, use it

None of these have been demonstrated yet. The infrastructure is there. The tests verify the plumbing works. But the water hasn't flowed through the pipes yet.

---

*Report compiled March 2026. Sources include ARC Prize leaderboard, company blogs, arxiv papers, GitHub repositories, and direct codebase inspection.*
