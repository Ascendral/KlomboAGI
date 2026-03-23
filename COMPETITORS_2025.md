# Open-Source AGI Projects: Comprehensive Survey (2025-2026)

Compiled 2026-03-22. Every serious open-source attempt at AGI or AGI-adjacent work.

---

## TIER 1: Classical Cognitive Architectures (Decades of Research)

### 1. Soar
- **URL:** https://github.com/SoarGroup/Soar
- **What it does:** General cognitive architecture (since 1983) based on production rules, chunking, and reinforcement learning. Models human-like problem solving with working memory, long-term memory (procedural, semantic, episodic), and a decision cycle. Written in C/C++.
- **Activity:** Actively maintained by U of Michigan. Soar 9.6.4 released. 45th Soar Workshop held May 2025. BSD license.
- **Language:** C/C++, with Java port (JSoar)
- **Does it work?** Yes -- one of the most battle-tested cognitive architectures. Used in military simulations, robotics, game AI. Thousands of published papers.
- **vs KlomboAGI:** Soar is rule-based at its core. It does not reason from first principles -- it matches patterns against hand-coded production rules. No curiosity-driven learning. No knowledge graph in the KlomboAGI sense. KlomboAGI's "ask and learn" approach is fundamentally different from Soar's "recognize and act" cycle.

### 2. ACT-R
- **URL:** http://act-r.psy.cmu.edu/ (official), https://github.com/jakdot/pyactr (Python), https://github.com/CarletonCognitiveModelingLab/python_actr
- **What it does:** Cognitive architecture from CMU modeling human cognition -- declarative memory (facts) + procedural memory (rules) with activation-based retrieval. Models how humans actually think (reaction times, errors), not how to build AGI.
- **Activity:** Official version in Common Lisp, maintained by CMU. Python ports (pyactr, python_actr) less actively maintained.
- **Language:** Common Lisp (official), Python (ports), Swift (partial)
- **Does it work?** Yes, for cognitive modeling. Not designed as an AGI system per se -- it's a theory of human cognition implemented as software.
- **vs KlomboAGI:** ACT-R models human memory retrieval and learning curves. It doesn't do structural reasoning or abstraction. KlomboAGI's decompose/compare/abstract pipeline has no equivalent in ACT-R. Different goals entirely.

### 3. Sigma
- **URL:** https://cogarch.ict.usc.edu/ , https://github.com/TownesZhou/PySigma
- **What it does:** Cognitive architecture from USC ICT based on factor graphs (graphical models). Tries to unify perception, reasoning, learning, and motor control in one formalism. Built in Lisp, with a Python/PyTorch port (PySigma).
- **Activity:** Active research at USC. PySigma is the newer Python implementation. BSD 2-clause license.
- **Language:** Lisp (original), Python/PyTorch (PySigma)
- **Does it work?** Research prototype. Demonstrated on various cognitive tasks but not deployed at scale.
- **vs KlomboAGI:** Sigma's graphical model approach is mathematically elegant but very different from KlomboAGI's explicit reasoning pipeline. Sigma doesn't have curiosity-driven learning or causal modeling in the KlomboAGI sense.

### 4. LIDA (Learning Intelligent Decision Agent)
- **URL:** https://github.com/CognitiveComputingResearchGroup/lida-framework , https://github.com/mindpixel20/lida
- **What it does:** Cognitive architecture from U of Memphis based on Global Workspace Theory (consciousness). Implements full cognitive cycles: perception, attention, action selection, learning, multiple memory systems. Models conscious and unconscious processing.
- **Activity:** Academic project. The mindpixel20/lida repo is a Python implementation. Java framework from CognitiveComputingResearchGroup. Limited recent activity.
- **Language:** Java (original framework), Python (newer implementation)
- **Does it work?** Research prototype. Interesting theoretical foundation but limited practical demonstrations.
- **vs KlomboAGI:** LIDA is consciousness-first (Global Workspace Theory). KlomboAGI is reasoning-first. Different entry points to AGI. LIDA doesn't have KlomboAGI's structural pattern extraction or inquiry engine.

---

## TIER 2: Symbolic/Hybrid AGI Systems (Active Development)

### 5. OpenCog Hyperon + MeTTa
- **URL:** https://github.com/trueagi-io/hyperon-experimental , https://github.com/opencog/atomspace
- **What it does:** The most ambitious open-source AGI project. Hyperon is the next-gen rewrite of OpenCog, backed by SingularityNET (Ben Goertzel). Features: AtomSpace (hypergraph knowledge base), MeTTa (purpose-built AGI programming language), PLN (Probabilistic Logic Networks), MOSES (evolutionary program learning). Tries to integrate symbolic reasoning, neural nets, evolutionary learning, and attention allocation.
- **Activity:** 207 stars, 77 forks on hyperon-experimental. Pre-alpha. Last release v0.2.8 (Sep 2025). Funded by SingularityNET/ASI Alliance. ~20+ contributors.
- **Language:** Rust (core), Python (bindings), MeTTa (scripting)
- **Does it work?** Pre-alpha. The original OpenCog had working demos (robot Sophia, natural language). Hyperon is a ground-up rewrite that's still finding its feet. MeTTa language is novel but immature.
- **vs KlomboAGI:** OpenCog is the closest competitor in spirit. Both use knowledge graphs, both aim for reasoning from first principles. Key differences: OpenCog tries to be everything at once (perception, language, reasoning, learning). KlomboAGI is more focused on the 7-piece reasoning core. OpenCog's PLN is probabilistic logic; KlomboAGI's approach is structural pattern extraction. OpenCog has 20+ years of theoretical backing but struggles with practical results. KlomboAGI is younger but more opinionated.

### 6. OpenNARS (Non-Axiomatic Reasoning System)
- **URL:** https://github.com/opennars/OpenNARS-for-Applications (C, practical), https://github.com/opennars/opennars (Java, research)
- **What it does:** Based on Pei Wang's NARS theory -- reasoning under insufficient knowledge and resources. Uses "experience-grounded semantics" where meaning comes from experience, not definitions. Handles uncertainty, learning, planning in a unified framework. Non-monotonic reasoning (can change beliefs based on new evidence).
- **Activity:** ONA (C version) is the practical one, used in NASA JPL collaboration. Funded by Digital Futures, Cisco, NASA. NARS-GPT bridges it with LLMs.
- **Language:** C (ONA), Java (research version), Clojure (Narjure)
- **Does it work?** Yes, within its domain. ONA has real applications (NASA first responder assistance). The reasoning is genuinely novel -- it handles partial knowledge better than most systems.
- **vs KlomboAGI:** NARS is the closest philosophical match. Both reject the assumption that you need complete knowledge. Both learn from experience. Key difference: NARS uses a formal logic (Non-Axiomatic Logic) with truth values. KlomboAGI uses structural decomposition and pattern matching. NARS has a more complete theoretical foundation (published since 1990s). KlomboAGI's "3 dogs -- it's about the 3, not the dogs" insight maps well to NARS's abstraction, but the implementation is different.

### 7. DANEEL
- **URL:** https://github.com/mollendorff-ai/daneel
- **What it does:** Experimental cognitive architecture in Rust implementing Global Workspace Theory, Hebbian learning, sleep consolidation, and criticality dynamics. Named after Asimov's robot. Uses hybrid Actor + Event-Driven architecture with microsecond latency and competing thought streams.
- **Activity:** Early stage. Solo developer project.
- **Language:** Rust
- **Does it work?** Experimental. Interesting architecture choices (Rust for performance, Erlang-style supervision) but very early.
- **vs KlomboAGI:** DANEEL focuses on consciousness/attention mechanisms. KlomboAGI focuses on reasoning. Complementary approaches. DANEEL's "competing thought streams" is interesting but different from KlomboAGI's explicit reasoning pipeline.

---

## TIER 3: AGI Frameworks & Evolutionary Approaches

### 8. AGI Laboratory
- **URL:** https://github.com/Dan23RR/AGI_Laboratory
- **What it does:** PyTorch framework for evolving a "society" of specialized AIs through hierarchical evolution. Starts from a primordial genome, creates domain experts (finance, cybersecurity, science). Claims to work on consumer hardware.
- **Activity:** Maintained by Daniel Culotta. Apache 2.0 license. Small community.
- **Language:** Python/PyTorch
- **Does it work?** Unclear. Interesting concept but limited evidence of results.
- **vs KlomboAGI:** Evolutionary approach vs. KlomboAGI's reasoning-from-principles approach. AGI Lab evolves specialists; KlomboAGI builds a general reasoner. Very different philosophies.

### 9. Cerenaut (formerly ProjectAGI)
- **URL:** https://github.com/Cerenaut/agi
- **What it does:** Australian research group combining neuroscience, psychology, and AI. Framework for AGI experiments with Docker-based execution, full logging, graphical UI. Implements various algorithms from AI/ML literature.
- **Activity:** Research group, limited recent commits. Docker-based.
- **Language:** Java (framework), various (algorithms)
- **Does it work?** Research platform. Good experiment infrastructure but results are academic.
- **vs KlomboAGI:** Cerenaut is more of a research platform than a specific AGI approach. KlomboAGI has a specific theory (7-piece reasoning core). Cerenaut is a tool for trying different approaches.

### 10. MBLS-3.0 (Meaningful Based Learning System)
- **URL:** https://github.com/howard8888/MBLS-3.0
- **What it does:** Implementation of Meaningful Based Cognitive Architecture. Tries to combine neural network sensory processing with symbolic logical abilities. Presented at BICA 2018.
- **Activity:** Limited recent activity.
- **Language:** Python
- **Does it work?** Academic prototype. Novel ideas about meaning-based learning but limited demonstration.
- **vs KlomboAGI:** Both try to combine sub-symbolic and symbolic processing. MBLS focuses on "meaning"; KlomboAGI focuses on structural reasoning. Different but overlapping concerns.

---

## TIER 4: LLM-Era "AGI" Projects (2023-2026)

### 11. BabyAGI
- **URL:** https://github.com/yoheinakajima/babyagi , https://github.com/yoheinakajima/babyagi-2o
- **What it does:** Task-driven autonomous agent using LLMs. Creates, executes, and prioritizes tasks in a loop. Latest version (babyagi-2o) is a self-building agent that creates its own tools. MIT license.
- **Activity:** Original archived Sep 2024. babyagi-2o is the current version. High stars (~20k original).
- **Language:** Python
- **Does it work?** As an LLM wrapper, yes. As AGI, no. It's task automation, not general intelligence. No real reasoning, no knowledge representation, no learning beyond context window.
- **vs KlomboAGI:** Fundamentally different. BabyAGI is an LLM prompt chain. KlomboAGI is building actual reasoning machinery. BabyAGI has no structural understanding, no causal modeling, no curiosity. It's "fake AGI" -- the name is aspirational.

### 12. SuperAGI
- **URL:** https://github.com/TransformerOptimus/SuperAGI
- **What it does:** Dev-first autonomous AI agent framework. Build, manage, run autonomous agents. Multi-model support, tool ecosystem, resource management.
- **Activity:** High stars (~15k+). Active development.
- **Language:** Python
- **Does it work?** As an agent framework, yes. As AGI, no. It's agent orchestration, not intelligence.
- **vs KlomboAGI:** Same critique as BabyAGI. It's infrastructure for LLM agents, not actual AGI. No reasoning, no knowledge graph, no learning.

### 13. OpenR (OpenReasoner)
- **URL:** https://github.com/openreasoner/openr
- **What it does:** Open-source framework inspired by OpenAI's o1 model. Integrates test-time compute, reinforcement learning, and process supervision for LLM reasoning. Supports beam search, best-of-N, Monte Carlo Tree Search.
- **Activity:** 1.8k stars, 135 forks. Last updated Jan 2025. MIT license.
- **Language:** Python
- **Does it work?** Yes, for improving LLM reasoning on math/logic benchmarks. Not AGI -- it's a technique for making LLMs reason better within their existing paradigm.
- **vs KlomboAGI:** OpenR improves LLM reasoning through search. KlomboAGI builds reasoning from scratch. OpenR still depends on an LLM's implicit knowledge. KlomboAGI builds explicit knowledge structures. Completely different approaches.

### 14. ROMA (Recursive-Open-Meta-Agent)
- **URL:** https://github.com/sentient-agi/ROMA
- **What it does:** Meta-agent framework using recursive hierarchical structures for complex problem-solving. From Sentient Foundation. Apache 2.0.
- **Activity:** Early stage. Backed by Sentient Foundation.
- **Language:** Python
- **Does it work?** Too early to tell. Interesting recursive architecture concept.
- **vs KlomboAGI:** ROMA is multi-agent orchestration. KlomboAGI is single-agent deep reasoning. Different levels of the stack.

---

## TIER 5: Domain-Specific AGI-Adjacent Projects

### 15. Numenta / HTM (Hierarchical Temporal Memory)
- **URL:** https://github.com/numenta , https://github.com/htm-community/nupic.py
- **What it does:** Biologically-constrained theory of intelligence based on neocortex neuroscience (Jeff Hawkins). Models spatial/temporal patterns, anomaly detection. Key concepts: sparse distributed representations, minicolumns, temporal sequences.
- **Activity:** NuPIC (original) is legacy. Community forks exist. Numenta pivoted to applying HTM theory to deep learning (sparse networks). htm.java maintained.
- **Language:** Python (NuPIC), Java (htm.java), various community ports
- **Does it work?** HTM works well for anomaly detection in streaming data. As AGI, it's incomplete -- it models cortical columns but not the full brain. The theory is strong but the implementation gap is large.
- **vs KlomboAGI:** HTM is bottom-up (neuroscience to intelligence). KlomboAGI is top-down (reasoning principles to implementation). HTM's sparse distributed representations could complement KlomboAGI's knowledge graph. HTM has no explicit reasoning -- it's pattern recognition. No causal modeling, no inquiry.

### 16. ARC-AGI Solvers
- **URLs:**
  - https://github.com/fchollet/ARC-AGI (benchmark)
  - https://github.com/poetiq-ai/poetiq-arc-agi-solver (77.1% ARC-1, 26% ARC-2)
  - https://github.com/1ytic/NVARC (NVIDIA solver)
  - https://github.com/iliao2345/CompressARC (no pretraining)
  - https://github.com/NalishJain/ARC-AGI-Meta-Learning (meta-learning + LLM)
  - https://github.com/TrelisResearch/arc-agi-2025
- **What they do:** Solve Francois Chollet's Abstraction and Reasoning Corpus -- visual pattern puzzles that test core human reasoning. ARC-AGI-2 is the harder version (2025).
- **Activity:** Very active. ARC Prize competition drives development. Poetiq is current SOTA (77.1% ARC-1).
- **Language:** Python (mostly)
- **Do they work?** Yes, on the benchmark. But most use LLMs or brute-force search, not genuine abstraction. CompressARC (no pretraining) is most interesting.
- **vs KlomboAGI:** ARC is the benchmark KlomboAGI should target. KlomboAGI's decompose/compare/abstract pipeline is exactly what ARC requires. The "3 dogs" insight is directly applicable. Most ARC solvers throw compute at the problem; KlomboAGI could potentially solve it with genuine structural reasoning. This is where KlomboAGI could prove itself.

### 17. Causal Reasoning Libraries
- **URLs:**
  - https://github.com/py-why/dowhy (Microsoft, Judea Pearl-inspired)
  - https://github.com/salesforce/causalai (Salesforce)
  - https://github.com/microsoft/causica (Microsoft, deep learning)
- **What they do:** Causal discovery and inference. DoWhy implements Pearl's do-calculus. CausalAI handles time series. Causica uses deep learning for causal structure.
- **Activity:** DoWhy is very active (5k+ stars). Well-maintained.
- **Language:** Python
- **Do they work?** Yes, for statistical causal inference. Not AGI -- they're tools for data analysis.
- **vs KlomboAGI:** These are complementary tools. KlomboAGI's causal model (component 6) could use DoWhy's theory but needs to go beyond statistical inference to structural causal understanding ("X causes Y" not just "X correlates with Y").

### 18. Continual Learning Frameworks
- **URLs:**
  - https://github.com/ContinualAI/avalanche (main framework, MIT)
  - https://github.com/aimagelab/mammoth (70+ methods, 20+ datasets)
  - https://github.com/AGI-Labs/continual_rl (RL-focused)
- **What they do:** Frameworks for training models that learn continuously without forgetting. Implement EWC, SI, progressive nets, etc.
- **Activity:** Avalanche is very active. Mammoth has 70+ methods.
- **Language:** Python/PyTorch
- **Do they work?** Yes, for their specific problem (catastrophic forgetting mitigation). Not AGI themselves.
- **vs KlomboAGI:** KlomboAGI needs continual learning. These frameworks solve a piece of the puzzle but not the whole thing. KlomboAGI's knowledge graph naturally handles continual learning (add to graph, don't overwrite).

### 19. World Model Projects
- **URLs:**
  - https://github.com/UMass-Embodied-AGI/TesserAct (4D world model, robotics)
  - https://github.com/OpenDriveLab/AgiBot-World (manipulation platform)
  - https://github.com/World-In-World/world-in-world (ICLR 2026 Oral)
- **What they do:** Learn internal models of how the world works -- predict what happens next given actions. Used for robotics and embodied AI.
- **Activity:** Very active area. TesserAct and AgiBot-World are well-funded.
- **Language:** Python/PyTorch
- **Do they work?** Yes, in constrained domains (robotics, video prediction). Not general world models.
- **vs KlomboAGI:** World models are about prediction; KlomboAGI is about understanding. A world model predicts the next frame; KlomboAGI asks "why did that happen?" Different but both needed for AGI.

---

## TIER 6: Chinese AGI Projects

### 20. BIGAI (Beijing Institute for General Artificial Intelligence)
- **URL:** https://github.com/bigai-ai (34 repositories)
- **What it does:** Led by Song-Chun Zhu (ex-UCLA). "Small data, big tasks" approach inspired by cognitive science and developmental psychology. Focuses on AGI through understanding human cognition rather than scaling.
- **Activity:** 34 repos on GitHub. Backed by Chinese government. Peking + Tsinghua affiliated.
- **Language:** Python (mostly)
- **Does it work?** Research outputs, publications. Most complete Chinese AGI research institute.
- **vs KlomboAGI:** BIGAI's "small data, big tasks" philosophy aligns with KlomboAGI's approach. Both reject the "just scale it" paradigm. BIGAI has institutional backing and dozens of researchers. Worth watching closely.

### 21. Zhipu AI / GLM (Tsinghua)
- **URL:** Tsinghua spinoff, VisualGLM-6B open-sourced
- **What it does:** LLM development, not AGI per se. But represents Tsinghua's AI research output.
- **vs KlomboAGI:** LLM-based, not comparable in approach.

---

## TIER 7: Neuro-Symbolic Approaches

### 22. Nucleoid
- **URL:** https://github.com/NucleoidAI/Nucleoid
- **What it does:** Declarative, logic-based runtime that integrates with neural networks. Knowledge graph with contextual reasoning.
- **Language:** JavaScript/TypeScript
- **vs KlomboAGI:** Similar knowledge graph + reasoning concept but from a web/enterprise angle.

### 23. David Shapiro's NLCA (Natural Language Cognitive Architecture)
- **URL:** https://github.com/daveshap/NaturalLanguageCognitiveArchitecture
- **What it does:** Architecture using natural language as the primary representation for cognitive processes. Book + implementation.
- **Language:** Python
- **vs KlomboAGI:** NLCA uses language as the medium of thought. KlomboAGI uses structural representations. KlomboAGI's approach is more principled -- language is ambiguous; structure is precise.

---

## Summary: Competitive Landscape for KlomboAGI

### Direct Competitors (same philosophical space):
1. **OpenCog Hyperon** -- Knowledge graph + reasoning, most similar vision, but 20 years in and still pre-alpha
2. **OpenNARS** -- Reasoning under insufficient knowledge, closest theory match, actually works (NASA)
3. **BIGAI** -- Same "small data, big tasks" philosophy, institutional backing

### Complementary (could inform KlomboAGI):
4. **Soar/ACT-R** -- Decades of cognitive architecture lessons learned
5. **DoWhy** -- Causal reasoning theory for component 6
6. **ARC-AGI** -- The benchmark to prove KlomboAGI works
7. **Avalanche** -- Continual learning techniques

### Not Real Competition (different paradigm):
8. **BabyAGI, SuperAGI, AutoGen, etc.** -- LLM wrappers, not AGI
9. **HTM** -- Bottom-up neuroscience, no reasoning

### KlomboAGI's Unique Position:
- **7-piece explicit reasoning core** -- No one else has this specific decompose/compare/abstract/transfer/inquiry/causal/self-eval pipeline
- **"It's about the 3, not the dogs"** -- Structural abstraction as the core insight
- **Curiosity-driven learning** -- InquiryEngine detecting knowledge gaps is rare in the field
- **Honest about scope** -- Not pretending to be everything at once

### Key Risks:
- OpenCog has 20+ years of theory and SingularityNET funding
- NARS has a more complete formal logic foundation
- BIGAI has institutional resources
- LLM-based approaches might "fake it till they make it" and reduce interest in principled AGI

### Key Advantages:
- KlomboAGI is focused and opinionated (vs. OpenCog's kitchen sink)
- Python-native (vs. Soar's C++, ACT-R's Lisp, OpenCog's Rust/MeTTa)
- Modern architecture (not carrying 20 years of legacy code)
- ARC-AGI is the perfect proving ground
