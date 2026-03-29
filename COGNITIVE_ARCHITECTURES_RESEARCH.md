# Cognitive Architecture Research for KlomboAGI

Research date: 2026-03-28
Purpose: Extract concrete implementable mechanisms from major cognitive architectures and recent papers.

---

## PART 1: ARCHITECTURE-BY-ARCHITECTURE ANALYSIS

### 1. SOAR (Newell, Laird — University of Michigan)

**Best idea to steal: CHUNKING (automatic skill compilation)**

When SOAR hits an impasse (can't decide what to do), it drops into a substate and reasons through the problem. Once it finds the answer, it automatically compiles that entire reasoning chain into a single production rule (a "chunk"). Next time it sees the same situation, it fires the chunk instantly — no reasoning needed. This is how deliberate thinking becomes automatic reflexes.

The algorithm:
1. Agent hits an impasse (operator tie, operator no-change, state no-change)
2. Architecture creates a substate automatically
3. Agent reasons in substate until it produces a result
4. Chunking mechanism traces back through the reasoning to find which superstate conditions led to this result
5. Creates a new rule: IF [those conditions] THEN [that result]
6. Rule fires instantly in future, preventing the impasse from ever occurring again

**What SOAR missed:** No real learning from the environment. Chunking only compiles what you already reasoned through — it doesn't discover NEW knowledge. It's a speed optimization, not a knowledge acquisition mechanism. SOAR agents are brittle when facing genuinely novel situations because chunking can only compress existing reasoning, never generate new insights. Also: no grounded sensory learning, no motivation system, no emotional modulation.

**KlomboAGI relevance:** Your CognitionLoop already has phases for this (hypothesize -> evaluate -> learn). What's missing is the AUTOMATIC COMPILATION step — when the loop solves something through multi-step reasoning, it should compress that into a direct rule for next time. This is not in your codebase yet.

---

### 2. ACT-R (Anderson — Carnegie Mellon)

**Best idea to steal: BASE-LEVEL ACTIVATION with decay**

ACT-R's memory retrieval equation is the most empirically validated formula in cognitive science:

```
B_i = ln(sum(t_j^(-d))) for j=1..n
```

Where:
- B_i = base-level activation of chunk i
- n = number of times chunk was accessed
- t_j = time since the j-th access
- d = decay rate (typically 0.5)

Each access adds a trace that decays as a power function of time. Frequently AND recently accessed memories are easy to retrieve. Old, rarely-used memories fade. This single equation captures the power law of forgetting AND the spacing effect AND the testing effect.

On top of this, SPREADING ACTIVATION from the current context adds to base-level:

```
A_i = B_i + sum(W_j * S_ji) + noise
```

Where W_j is attention weight from source j, and S_ji is associative strength from j to i.

**What ACT-R missed:** It's a psychology modeling tool, not an intelligence system. It models how humans DO think, not how an agent SHOULD think. No curiosity, no self-directed learning, no goal generation. The architecture has no way to notice its own ignorance — it just fails to retrieve and moves on.

**KlomboAGI relevance:** Your `activation.py` has spreading activation but I don't see the temporal decay component. You should add ACT-R's base-level learning equation to your activation system so that knowledge graph nodes that haven't been accessed recently naturally fade, and frequently-used concepts stay hot. This interacts beautifully with your curiosity driver — faded concepts that suddenly get reactivated are "surprising" and should trigger curiosity.

---

### 3. NARS (Pei Wang — Temple University)

**Best idea to steal: EVIDENTIAL TRUTH VALUES (already implemented)**

You already have this in `truth.py`. The (frequency, confidence) pair with:
- f = w+ / w
- c = w / (w + k)

And the revision rule for combining evidence. This is the right foundation.

**What NARS missed:** Pei Wang spent 30 years on the logic and not enough on grounding. NARS can reason about symbols beautifully but has almost no mechanism for acquiring those symbols from raw experience. It also has no real attention mechanism — it processes judgments in a bag with random selection, which means it can get stuck processing irrelevant things when under time pressure.

**KlomboAGI relevance:** You have the truth values. What you should add from NARS is the **revision rule priority** — when two pieces of evidence about the same thing arrive, revision should be triggered IMMEDIATELY and with HIGH PRIORITY, because conflicting evidence is the most informative signal the system can receive. Currently your system likely treats revision as just another operation.

---

### 4. OpenCog (Ben Goertzel)

**Best idea to steal: ECONOMIC ATTENTION NETWORK (ECAN)**

OpenCog treats attention as a scarce currency with conservation laws. Every atom (concept) has:
- STI (Short-Term Importance): how relevant right now
- LTI (Long-Term Importance): how useful historically

STI spreads between connected atoms along HebbianLinks, but the TOTAL STI in the system is CONSERVED. When one concept gets more attention, others must lose it. This creates natural competition — concepts literally compete for cognitive resources.

The algorithm:
1. Each atom has STI and LTI values
2. Agents (reasoning processes) pay STI "rent" — if your STI drops below the attentional focus boundary, you get forgotten (moved to long-term storage)
3. STI spreads along links with a spreading rate proportional to link strength
4. Total STI is conserved — attention is zero-sum
5. Atoms with high STI form the "attentional focus" — only these are actively reasoned about

**What OpenCog missed:** Too complex. Goertzel built a system with 20+ interacting subsystems but never got them all working together reliably. The architecture diagram looks like a city power grid. Individual components (PLN, MOSES, ECAN) are interesting but the integration never solidified. Also: the Atomspace graph database turned into a performance bottleneck.

**KlomboAGI relevance:** Your `attention.py` and `focus.py` likely have some attention mechanism. The key insight to steal is CONSERVATION OF ATTENTION — make it zero-sum. When the system focuses on X, it should literally lose focus on Y. This prevents the "everything is important" failure mode that kills most AGI systems.

---

### 5. Sigma (Rosenbloom — USC)

**Best idea to steal: FACTOR GRAPHS AS UNIVERSAL SUBSTRATE**

Sigma encodes ALL cognitive operations (memory, learning, perception, action, reasoning) as message-passing on factor graphs. One formalism, one algorithm (belief propagation / variational inference), handles everything. This achieves "functional elegance" — the architecture is simple but the behaviors that emerge are complex.

The key: instead of having separate modules for declarative memory, procedural memory, perception, etc., Sigma represents them ALL as conditional distributions over variables, encoded as factors in a graphical model. Reasoning = inference. Learning = parameter updates. Memory = variable states.

**What Sigma missed:** Too elegant for its own good. The factor graph formalism is mathematically beautiful but makes it hard to implement specialized fast-paths for common operations. Everything goes through the same message-passing bottleneck. Also, Sigma remains an academic project with limited real-world testing.

**KlomboAGI relevance:** The insight for you is not to USE factor graphs, but to notice what Sigma gets right about unification. Your cognition loop already tries to be universal — the lesson is to keep the number of fundamental operations SMALL. If you can express new capabilities as compositions of existing primitives rather than new modules, the system stays coherent. Avoid the OpenCog trap of 20 subsystems.

---

### 6. CLARION (Ron Sun — RPI)

**Best idea to steal: IMPLICIT/EXPLICIT DUAL REPRESENTATION with bottom-up learning**

CLARION maintains TWO representations of every piece of knowledge:
- **Top level (explicit):** Rules you can inspect and explain ("if X then Y")
- **Bottom level (implicit):** Neural network weights that do the same thing but can't be inspected

The key mechanism is BOTTOM-UP LEARNING (Rule Extraction Refinement / RER): the implicit network learns from experience first, then the system extracts explicit rules from the trained network. This models how humans develop intuitions first, then later articulate what they know.

The algorithm for RER:
1. Bottom level learns via reinforcement (Q-learning on neural nets)
2. When bottom level achieves reliable performance, trigger rule extraction
3. Extract candidate rules by testing which input conditions correlate with successful outputs
4. Add extracted rules to top level
5. Top level rules can then be inspected, communicated, and refined
6. Top-down: explicit rules can also train the implicit level

**What CLARION missed:** The neural networks used are simple MLPs, not modern architectures. The rule extraction is brittle. And the dual system doesn't truly interact in a deep way — it's more like two parallel tracks than a genuine dual-process system.

**KlomboAGI relevance:** This maps directly to your vision. Your knowledge graph (explicit) and any future pattern-matching layers (implicit) should have a bottom-up extraction mechanism. When the system successfully navigates a situation multiple times, it should extract a RULE from the pattern. This is the opposite direction from SOAR's chunking — SOAR goes top-down (reasoning -> rule), CLARION goes bottom-up (experience -> rule). You should do BOTH.

---

### 7. LIDA (Stan Franklin — University of Memphis)

**Best idea to steal: THE COGNITIVE CYCLE with competitive attention**

LIDA's cognitive cycle runs at ~10Hz and has three phases:
1. **Understanding:** Perception builds a situational model from sensory input + memory
2. **Consciousness:** Attention codelets compete to bring content to the Global Workspace. Winner gets BROADCAST to all modules simultaneously
3. **Action selection:** All modules that received the broadcast respond with proposed actions. Best action wins.

The critical mechanism is the ATTENTION CODELET COMPETITION:
- Small, specialized agents (codelets) each monitor for specific conditions
- When their conditions are met, they form coalitions with relevant content
- Coalitions compete for access to the Global Workspace
- The winning coalition's content gets broadcast system-wide
- This broadcast IS the moment of "consciousness" — it's when the whole system becomes aware of something

**What LIDA missed:** The cognitive cycle is fixed at ~10Hz, which is biologically inspired but computationally arbitrary. The codelets are hand-coded, not learned. And LIDA has limited learning — it models cognition well but doesn't grow much from experience.

**KlomboAGI relevance:** Your CognitionLoop has phases but lacks the COMPETITIVE BROADCASTING mechanism. The key insight: not everything the system perceives should get equal processing. You need a bottleneck — a narrow channel where different perceptions, memories, and reasoning results compete for system-wide attention. The winner gets broadcast to ALL subsystems simultaneously. This is how you avoid the "everything is equally important" problem.

---

### 8. MicroPsi / Joscha Bach

**Best idea to steal: COGNITIVE MODULATORS**

Bach's key insight: cognition isn't just reasoning — it's MODULATED reasoning. The same reasoning engine behaves differently depending on:
- **Resolution level:** How much detail to process (high when calm, low when stressed)
- **Certainty threshold:** How sure you need to be before acting (high when safe, low when urgent)
- **Arousal:** How fast to process (high urgency = fast but sloppy)

These modulators are driven by the MOTIVATIONAL SYSTEM (urges/needs):
1. The system has needs (knowledge acquisition, task completion, self-preservation)
2. Unsatisfied needs create urges
3. Urges modulate the cognitive parameters
4. A frustrated system becomes more exploratory (lower certainty threshold, lower resolution)
5. A satisfied system becomes more careful (higher certainty, higher resolution)

The algorithm:
```
resolution = baseline_resolution * (1 - arousal * arousal_sensitivity)
certainty_threshold = baseline_certainty * (1 - urgency * urgency_sensitivity)
exploration_rate = frustration / (frustration + satisfaction)
```

**What Bach missed:** MicroPsi remains a research framework, not a deployed system. The motivational model is well-designed but tested only in simple virtual environments. The connection between motivation and actual capability is thin.

**KlomboAGI relevance:** Your `inner_state.py` and `traits.py` might have some emotional/state modeling. The concrete mechanism to add: cognitive modulators that change HOW the reasoning engine operates based on the system's current state. When the system is frustrated (many failed attempts), it should automatically increase exploration (lower certainty threshold, try wider search). When it's succeeding, it should narrow focus (raise resolution, process more carefully). This is a simple set of multipliers but it makes the system adaptive at the meta-level.

---

### 9. Joscha Bach's Broader Work (Beyond MicroPsi)

**Best idea to steal: CONSTRUCTIVE MEMORY**

Bach's key philosophical contribution: memory isn't storage, it's CONSTRUCTION. You don't retrieve a memory — you reconstruct it from compressed patterns every time. This means:
1. Every recall is slightly different (based on current context)
2. Memory naturally generalizes (the compression removes specifics)
3. False memories are a feature, not a bug (they're what generalization looks like)
4. Memory IS reasoning — there's no separate "recall" step

**What Bach missed/hasn't finished:** He articulates the vision better than anyone but the implementations lag behind the philosophy. MicroPsi 2 was incomplete when Bach moved on to other work.

**KlomboAGI relevance:** Your memory system should reconstruct, not just retrieve. When the system "remembers" a past experience, it should reconstruct it through the same reasoning engine that processes new input, using the current context to fill in gaps. This means stored memories are compressed/abstract, and recall is an active inference process.

---

## PART 2: RECENT RESEARCH (2024-2025)

### Neurosymbolic Reasoning
Key systems: DomiKnowS (domain knowledge constraints in PyTorch), Scallop (differentiable Datalog), AlphaGeometry (LLM + symbolic deduction). The trend is using neural networks for PATTERN RECOGNITION and symbolic systems for REASONING, with a clean interface between them. This is exactly KlomboAGI's architecture: knowledge graph (symbolic) + LLM-as-library-card (neural when needed).

### World Models (LeCun's JEPA)
Joint Embedding Predictive Architecture predicts in REPRESENTATION SPACE, not pixel space. It doesn't try to predict every detail — it predicts abstract representations of what will happen. This matches KlomboAGI's approach: predict structural patterns, not surface features. AMI Labs raised $1B to build this. The insight for you: your world model should predict at the level of RELATIONS and PROPERTIES, not raw observations.

### Active Inference / Free Energy Principle
The Expected Free Energy (EFE) decomposes into:
- **Pragmatic value:** Does this action get me what I want? (exploitation)
- **Epistemic value:** Does this action teach me something? (exploration)

```
G(policy) = pragmatic_value + epistemic_value
          = E[log P(preferred_outcomes | policy)] + E[information_gain(policy)]
```

An agent should take the action that EITHER gets it closer to its goal OR teaches it the most. Exploration is not random — it's directed at reducing uncertainty about things that MATTER for the goal. This resolves the exploration-exploitation dilemma without any temperature parameter or epsilon-greedy heuristic.

### Predictive Processing / Predictive Coding
The brain maintains a hierarchical generative model:
- Each layer predicts the activity of the layer below
- Only PREDICTION ERRORS propagate upward
- Precision weighting controls how much to trust prediction errors vs. priors

The algorithm:
```
For each layer L:
  prediction = top_down_weights * layer_above_state
  error = actual_input - prediction
  weighted_error = precision_L * error
  update = learning_rate * weighted_error
  layer_state += update  (inference)
  top_down_weights += learning_rate * weighted_error * layer_above_state  (learning)
```

Key insight: PRECISION WEIGHTING is how the system decides what to pay attention to. High precision on a prediction error means "this surprise is important, update beliefs." Low precision means "this is noise, ignore it." Attention IS precision control.

### Global Workspace Theory (Computational)
2024-2025 results: GWT-based architectures OUTPERFORM Transformers on causal/sequential reasoning and out-of-distribution generalization. The competitive ignition + broadcast mechanism forces the system to commit to one interpretation at a time, which prevents the "soft attention over everything" problem that plagues Transformers.

### Dual Process Theory (System 1 / System 2)
The computational challenge: System 1 (fast/automatic) and System 2 (slow/deliberate) need to INTERACT, not just coexist. Recent work on LLMs shows they can vary computation depth — spending more tokens on hard problems — which is a crude System 2. But true dual process requires System 1 to RECOGNIZE when it's out of its depth and hand off to System 2.

---

## PART 3: TOP 10 CONCRETE MECHANISMS TO IMPLEMENT

Ordered by impact for KlomboAGI. Each one is a specific algorithm, not vague theory.

---

### #1. EXPECTED FREE ENERGY FOR ACTION SELECTION (from Active Inference)

**What:** Replace any ad-hoc action selection with a single formula that naturally balances doing useful things vs. learning new things.

**Algorithm:**
```python
def select_action(possible_actions, beliefs, preferences):
    best_action = None
    best_score = float('-inf')
    for action in possible_actions:
        # Pragmatic: how likely does this action lead to preferred outcomes?
        pragmatic = expected_utility(action, beliefs, preferences)
        # Epistemic: how much would this action reduce uncertainty?
        epistemic = expected_information_gain(action, beliefs)
        score = pragmatic + epistemic
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def expected_information_gain(action, beliefs):
    """How much would beliefs change if we took this action?"""
    predicted_outcomes = beliefs.predict(action)
    # High entropy in predicted outcomes = high info gain
    return entropy(predicted_outcomes)
```

**Why #1:** This single mechanism replaces your entire curiosity priority system with something principled. When the system doesn't know enough, epistemic value dominates and it explores. When it knows enough, pragmatic value dominates and it exploits. No hand-tuned thresholds. Your curiosity driver (`curiosity.py`) becomes a special case of this.

---

### #2. TEMPORAL DECAY ACTIVATION (from ACT-R)

**What:** Add time-based decay to your activation network so memories naturally fade unless reinforced.

**Algorithm:**
```python
import math

def base_level_activation(access_times: list[float], current_time: float, decay: float = 0.5) -> float:
    """ACT-R's base-level learning equation.

    access_times: list of timestamps when this concept was accessed
    current_time: now
    decay: rate of forgetting (0.5 is standard)
    """
    if not access_times:
        return float('-inf')  # never accessed = irretrievable

    total = 0.0
    for t in access_times:
        age = current_time - t
        if age > 0:
            total += age ** (-decay)

    return math.log(total) if total > 0 else float('-inf')
```

**Why #2:** Your `activation.py` has spreading activation but no temporal component. Without decay, your knowledge graph will bloat — everything stays equally accessible forever. With decay, the system naturally develops a "working set" of recently-relevant concepts, and old knowledge fades to long-term storage. Reactivation of faded concepts becomes a SURPRISE signal that triggers learning.

---

### #3. COMPETITIVE BROADCAST (GLOBAL WORKSPACE) (from LIDA + GWT)

**What:** Add a bottleneck to your CognitionLoop where perceptions, memories, and reasoning results COMPETE for system-wide attention. Only one thing at a time gets broadcast.

**Algorithm:**
```python
class GlobalWorkspace:
    def __init__(self):
        self.coalitions = []  # competing items
        self.broadcast_threshold = 0.5
        self.subscribers = []  # all subsystems

    def submit_coalition(self, content, salience, source):
        """Any module can submit content for broadcasting."""
        self.coalitions.append({
            'content': content,
            'salience': salience,
            'source': source,
            'timestamp': now()
        })

    def compete_and_broadcast(self):
        """Run competition. Winner gets broadcast to all subscribers."""
        if not self.coalitions:
            return None

        # Competitive inhibition: strongest signal wins
        winner = max(self.coalitions, key=lambda c: c['salience'])

        if winner['salience'] < self.broadcast_threshold:
            return None  # nothing important enough

        # BROADCAST: every subsystem receives the winning content
        for subscriber in self.subscribers:
            subscriber.receive_broadcast(winner['content'], winner['source'])

        # Clear competition for next cycle
        self.coalitions.clear()
        return winner
```

**Why #3:** Your CognitionLoop processes phases sequentially. The Global Workspace pattern adds a FILTER — not everything that enters the perceive phase makes it to the hypothesize phase. This prevents the system from wasting reasoning cycles on irrelevant input. Recent research (2024) shows GWT architectures beat Transformers on causal reasoning because they COMMIT to one interpretation instead of soft-attending over everything.

---

### #4. COGNITIVE MODULATORS (from MicroPsi / Joscha Bach)

**What:** Add 3-4 scalar parameters that modulate HOW your reasoning engine operates based on the system's current emotional/motivational state.

**Algorithm:**
```python
@dataclass
class CognitiveModulators:
    resolution: float = 0.7    # [0,1] how much detail to process
    certainty_threshold: float = 0.6  # [0,1] how sure before acting
    arousal: float = 0.5      # [0,1] processing speed vs. accuracy
    exploration_rate: float = 0.3  # [0,1] novel vs. familiar paths

    def update_from_state(self, frustration: float, urgency: float,
                          curiosity: float, success_rate: float):
        """Modulate cognition based on current internal state."""
        # High frustration → increase exploration, decrease resolution
        self.exploration_rate = min(1.0, 0.2 + frustration * 0.6)
        self.resolution = max(0.3, 0.9 - frustration * 0.4)

        # High urgency → lower certainty threshold, increase arousal
        self.certainty_threshold = max(0.2, 0.8 - urgency * 0.5)
        self.arousal = min(1.0, 0.3 + urgency * 0.5)

        # High curiosity → increase resolution, moderate exploration
        self.resolution = min(1.0, self.resolution + curiosity * 0.2)

        # Recent success → increase certainty, decrease exploration
        self.certainty_threshold = min(0.95, self.certainty_threshold + success_rate * 0.1)
        self.exploration_rate = max(0.1, self.exploration_rate - success_rate * 0.2)
```

**Why #4:** This makes the entire reasoning engine ADAPTIVE. Same engine, different behavior depending on context. A frustrated system becomes more creative (lower certainty, higher exploration). A system under time pressure becomes faster but sloppier. This is the simplest mechanism that produces genuinely different cognitive "modes" without separate System 1 / System 2 implementations.

---

### #5. SURPRISE-DRIVEN LEARNING (from Predictive Processing)

**What:** Only learn from things that VIOLATE predictions. Expected outcomes teach nothing.

**Algorithm:**
```python
def process_observation(self, observation, context):
    """Predictive processing: only learn from surprises."""
    # Generate prediction from current beliefs
    prediction = self.world_model.predict(context)

    # Calculate prediction error
    error = self.compute_difference(observation, prediction)
    error_magnitude = self.magnitude(error)

    # Precision: how reliable is this error signal?
    precision = self.estimate_precision(context)

    # Weighted prediction error
    weighted_error = error_magnitude * precision

    if weighted_error < self.surprise_threshold:
        return None  # Expected outcome, nothing to learn

    # SURPRISE! This is worth learning from
    learning_signal = {
        'observation': observation,
        'prediction': prediction,
        'error': error,
        'precision': precision,
        'weighted_error': weighted_error,
    }

    # Update beliefs proportional to precision-weighted error
    self.world_model.update(context, observation, learning_rate=weighted_error)

    # High surprise → trigger curiosity about WHY prediction was wrong
    if weighted_error > self.high_surprise_threshold:
        self.curiosity_driver.add_gap(
            concept=context,
            reason=f"Prediction error: expected {prediction}, got {observation}",
            priority='high'
        )

    return learning_signal
```

**Why #5:** This is the most efficient learning signal possible. Instead of trying to learn from everything (which overwhelms the system), you only learn from VIOLATIONS of expectations. This connects directly to your curiosity driver — prediction errors ARE the curiosity signal. Your "surprise detection" in the baby phase roadmap should be exactly this mechanism.

---

### #6. AUTOMATIC SKILL COMPILATION (from SOAR Chunking)

**What:** When the CognitionLoop solves a problem through multi-step reasoning, automatically compile that reasoning chain into a direct rule for next time.

**Algorithm:**
```python
def compile_skill(self, reasoning_trace: list[CognitionPhase], result):
    """After successful reasoning, compile into a direct rule.

    Like SOAR chunking: trace back through the reasoning to find
    the minimal conditions that led to the result.
    """
    # Find which initial conditions actually mattered
    relevant_conditions = []
    for phase in reasoning_trace:
        if phase.contributed_to_result:
            relevant_conditions.extend(phase.input_conditions)

    # Remove redundant conditions
    minimal_conditions = self.minimize(relevant_conditions, result)

    # Create compiled rule
    new_rule = CompiledRule(
        conditions=minimal_conditions,
        action=result,
        confidence=TruthValue(frequency=1.0, confidence=0.5),  # Single observation
        source='compiled',
        reasoning_trace_id=trace_id  # For debugging
    )

    # Add to fast-path rules
    self.compiled_rules.add(new_rule)

    return new_rule
```

**Why #6:** Without this, your system reasons through the same problem the same way every time. With it, solved problems become instant reactions. This is how expertise develops: deliberate reasoning becomes automatic skill. Combined with truth values, compiled rules get stronger (higher confidence) each time they're confirmed, and weaker when they fail.

---

### #7. NARS-STYLE EVIDENCE REVISION WITH PRIORITY BOOST (from NARS)

**What:** When conflicting evidence arrives for the same belief, IMMEDIATELY prioritize processing it. Conflicting evidence is the most informative signal.

**Algorithm:**
```python
def detect_and_prioritize_conflict(self, new_evidence, belief_store):
    """NARS revision + priority boost for conflicting evidence."""
    subject = new_evidence.subject
    existing = belief_store.get(subject)

    if existing is None:
        belief_store.add(new_evidence)
        return 'new_belief'

    # Calculate how much the new evidence conflicts
    frequency_delta = abs(new_evidence.truth.frequency - existing.truth.frequency)

    if frequency_delta > 0.3 and existing.truth.confidence > 0.3:
        # CONFLICT DETECTED — this is the most valuable signal
        # Boost priority: conflicting evidence should be processed IMMEDIATELY
        self.global_workspace.submit_coalition(
            content={
                'type': 'belief_conflict',
                'belief': subject,
                'old_truth': existing.truth,
                'new_truth': new_evidence.truth,
                'delta': frequency_delta,
            },
            salience=frequency_delta * existing.truth.confidence * 2.0,  # Priority boost
            source='revision_detector'
        )

    # Standard NARS revision
    revised = nars_revision(existing.truth, new_evidence.truth)
    belief_store.update(subject, revised)
    return 'revised'
```

**Why #7:** You have the truth value system from NARS but conflict detection is probably not special-cased. In most systems, conflicting evidence is processed like any other input. But conflicting evidence is where the MOST learning happens — it means your model of the world is wrong somewhere. Making conflict a high-priority signal means the system self-corrects quickly.

---

### #8. CONSERVATION OF ATTENTION (from OpenCog ECAN)

**What:** Make attention zero-sum. The system has a fixed "attention budget" that gets allocated across concepts. When something gets more attention, everything else gets less.

**Algorithm:**
```python
class AttentionBudget:
    def __init__(self, total_sti=1000.0, rent_rate=0.01, focus_threshold=0.5):
        self.total_sti = total_sti  # Total attention in the system (conserved)
        self.rent_rate = rent_rate  # Cost of staying in focus per cycle
        self.focus_threshold = focus_threshold
        self.sti_map = {}  # concept -> STI value

    def boost(self, concept: str, amount: float, source: str):
        """Give attention to a concept. Must take from somewhere."""
        # Tax everything else proportionally
        tax_per_concept = amount / max(1, len(self.sti_map))
        for c in self.sti_map:
            if c != concept:
                self.sti_map[c] = max(0, self.sti_map[c] - tax_per_concept)

        self.sti_map[concept] = self.sti_map.get(concept, 0) + amount

    def collect_rent(self):
        """Every cycle, concepts in focus pay rent. Can't pay = evicted."""
        evicted = []
        for concept, sti in list(self.sti_map.items()):
            self.sti_map[concept] -= self.rent_rate * sti
            if self.sti_map[concept] < self.focus_threshold:
                evicted.append(concept)

        for c in evicted:
            # Move to long-term storage, out of active processing
            self.archive(c)
            del self.sti_map[c]

        return evicted

    def get_focus(self) -> list[str]:
        """What concepts are currently in the attentional focus?"""
        return [c for c, sti in self.sti_map.items()
                if sti >= self.focus_threshold]
```

**Why #8:** Without attention conservation, your knowledge graph will try to process everything equally. With a fixed budget, the system is FORCED to prioritize. This is what makes human cognition efficient — we can't attend to everything, so we attend to what matters. The "rent" mechanism naturally evicts stale concepts.

---

### #9. DUAL-LEVEL KNOWLEDGE (IMPLICIT + EXPLICIT) (from CLARION)

**What:** Maintain two representations: explicit rules in the knowledge graph (inspectable, shareable) and implicit pattern weights (fast, approximate, can't be explained).

**Algorithm:**
```python
class DualKnowledge:
    def __init__(self):
        self.explicit_rules = []   # IF-THEN rules, inspectable
        self.implicit_weights = {} # pattern -> outcome weights, opaque

    def decide(self, situation):
        """Both levels vote, weighted by track record."""
        # Explicit: check rules
        explicit_vote = self.explicit_decide(situation)
        explicit_confidence = explicit_vote.confidence if explicit_vote else 0

        # Implicit: pattern match
        implicit_vote = self.implicit_decide(situation)
        implicit_confidence = self.implicit_confidence(situation)

        # Weighted combination
        if explicit_confidence > implicit_confidence:
            return explicit_vote  # We have a clear rule
        elif implicit_confidence > explicit_confidence:
            return implicit_vote  # Intuition is stronger
        else:
            return explicit_vote  # Tie goes to explainable

    def bottom_up_extract(self, situation, outcome):
        """CLARION's key: extract explicit rules from implicit patterns.
        When the implicit level consistently succeeds, extract a rule."""
        self.implicit_weights[situation] = outcome

        # Check if implicit has been consistently right
        pattern = self.find_consistent_pattern()
        if pattern and pattern.success_rate > 0.8 and pattern.count > 5:
            new_rule = ExplicitRule(
                conditions=pattern.conditions,
                action=pattern.action,
                confidence=TruthValue(pattern.success_rate,
                                     pattern.count / (pattern.count + 1))
            )
            self.explicit_rules.append(new_rule)
            return new_rule
        return None
```

**Why #9:** This gives you System 1 / System 2 without the complexity. The implicit level is fast pattern matching (System 1). The explicit level is your knowledge graph reasoning (System 2). The bottom-up extraction is how intuitions become articulable knowledge — the system can eventually EXPLAIN why it made a decision, even if it initially just "felt right."

---

### #10. PREDICTIVE WORLD MODEL WITH PRECISION WEIGHTING (from Predictive Coding)

**What:** Your `world/model.py` should be a hierarchical prediction engine where each level predicts the level below, and only prediction errors propagate.

**Algorithm:**
```python
class PredictiveLayer:
    def __init__(self, level):
        self.level = level
        self.state = {}           # Current beliefs at this level
        self.precision = {}       # How reliable each signal is
        self.predictions_down = {} # What we predict the level below will show

    def process(self, bottom_up_error, top_down_prediction):
        """Core predictive coding update."""
        # Weight the error by precision (attention)
        weighted_error = {}
        for key in bottom_up_error:
            p = self.precision.get(key, 0.5)
            weighted_error[key] = bottom_up_error[key] * p

        # Update state: move toward reducing weighted prediction error
        for key in weighted_error:
            self.state[key] += self.learning_rate * weighted_error[key]

        # Generate new predictions for level below
        self.predictions_down = self.generate_predictions()

        # Compute error to send up: what I see vs. what layer above predicted
        error_up = {}
        for key in self.state:
            if key in top_down_prediction:
                error_up[key] = self.state[key] - top_down_prediction[key]

        return error_up, self.predictions_down

    def update_precision(self, key, prediction_error_history):
        """Precision = inverse variance of recent prediction errors.
        Consistent errors → high precision (pay attention).
        Noisy errors → low precision (ignore)."""
        if len(prediction_error_history) < 3:
            return
        variance = np.var(prediction_error_history[-10:])
        self.precision[key] = 1.0 / (variance + 0.01)  # Inverse variance
```

**Why #10:** This gives your world model a principled way to decide WHAT TO ATTEND TO. Precision weighting is the mechanism: reliable signals get high precision (the system listens), noisy signals get low precision (the system ignores). This replaces hand-tuned attention heuristics with a learned, adaptive attention system. Combined with #5 (surprise-driven learning), it means the system only learns from RELIABLE surprises, not noise.

---

## SUMMARY TABLE

| # | Mechanism | Source | What It Replaces/Adds |
|---|-----------|--------|----------------------|
| 1 | Expected Free Energy | Active Inference | Principled exploration vs. exploitation |
| 2 | Temporal Decay Activation | ACT-R | Memory fading + recency + frequency effects |
| 3 | Competitive Broadcast | LIDA/GWT | Attentional bottleneck for the CognitionLoop |
| 4 | Cognitive Modulators | MicroPsi/Bach | Adaptive reasoning parameters (meta-cognition) |
| 5 | Surprise-Driven Learning | Predictive Processing | Only learn from prediction violations |
| 6 | Automatic Skill Compilation | SOAR | Convert deliberate reasoning into fast rules |
| 7 | Conflict Priority Boost | NARS | Fast self-correction from contradictory evidence |
| 8 | Attention Conservation | OpenCog ECAN | Zero-sum attention budget |
| 9 | Dual-Level Knowledge | CLARION | Implicit intuitions + explicit rules |
| 10 | Predictive Precision Weighting | Predictive Coding | Learned attention through error reliability |

---

## WHAT'S ALREADY IN KLOMBOAGI

- NARS truth values (truth.py) -- #7 extends this
- Spreading activation (activation.py) -- #2 adds temporal decay to this
- Curiosity driver (curiosity.py) -- #1 subsumes this with EFE
- Cognition loop (cognition_loop.py) -- #3 adds competitive broadcast to this
- Inner state (inner_state.py) -- #4 formalizes this
- World model (world/model.py) -- #10 restructures this
- Self-evaluation (self_eval.py) -- #5 adds prediction error framework

## IMPLEMENTATION ORDER

1. **#2 Temporal Decay** — Simplest, biggest immediate effect on activation.py
2. **#5 Surprise-Driven Learning** — Connects to existing curiosity.py
3. **#4 Cognitive Modulators** — Simple multipliers, huge behavioral impact
4. **#7 Conflict Priority** — Small extension to existing truth.py
5. **#3 Global Workspace** — Moderate refactor of cognition_loop.py
6. **#1 Expected Free Energy** — Replaces ad-hoc curiosity prioritization
7. **#6 Skill Compilation** — Requires reasoning trace infrastructure
8. **#8 Attention Conservation** — Enhances activation.py further
9. **#9 Dual Knowledge** — Requires second representation layer
10. **#10 Predictive Precision** — Full world model restructure

---

## SOURCES

### Cognitive Architectures
- [SOAR Architecture](https://soar.eecs.umich.edu/home/About/)
- [SOAR Chunking](https://link.springer.com/article/10.1007/BF00116249)
- [ACT-R Architecture](https://act-r.psy.cmu.edu/about/)
- [ACT-R Base-Level Activation Tutorial](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/unit4.htm)
- [NARS Introduction](https://cis.temple.edu/~pwang/NARS-Intro.html)
- [OpenCog Attention Allocation](https://wiki.opencog.org/w/Attention_Allocation)
- [OpenCog ECAN](https://wiki.opencog.org/w/OpenCogPrime:EconomicAttentionAllocation)
- [Sigma Architecture](https://www.researchgate.net/publication/305423635_The_Sigma_Cognitive_Architecture_and_System_Towards_Functionally_Elegant_Grand_Unification)
- [CLARION Architecture](https://en.wikipedia.org/wiki/CLARION_(cognitive_architecture))
- [LIDA Architecture](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [LIDA Cognitive Cycle Timing](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014803)
- [MicroPsi Motivation](https://agi-conf.org/2015/wp-content/uploads/2015/07/agi15_bach.pdf)

### Recent Research (2024-2025)
- [Neurosymbolic AI for Knowledge Graphs Survey](https://arxiv.org/pdf/2302.07200)
- [JEPA and World Models (Meta AI)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [AMI Labs Launch](https://www.latent.space/p/ainews-yann-lecuns-ami-labs-launches)
- [Expected Free Energy Planning as Variational Inference](https://arxiv.org/abs/2504.14898)
- [Active Inference and Epistemic Value](https://www.fil.ion.ucl.ac.uk/~karl/Active%20inference%20and%20epistemic%20value.pdf)
- [Predictive Coding Networks for Temporal Prediction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183)
- [GWT-based Agent in Multimodal Environment (2024)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/full)
- [GWT Selection-Broadcast Cycle (2025)](https://arxiv.org/html/2505.13969v1)
- [Dual Process Theory for AI Architectures](https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2024.1356941/pdf)
- [Dual Process Theory in LLMs (Nature, 2025)](https://www.nature.com/articles/s44159-025-00506-1)
