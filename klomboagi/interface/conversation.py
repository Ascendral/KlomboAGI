"""
Conversation Interface — talk to the baby.

This is where human meets algorithm. You type, it processes through
the reasoning engine, builds its knowledge graph, and responds from
what it KNOWS — not from an LLM.

The conversation is the teaching method. Everything you say becomes
knowledge the system OWNS. Over time it needs less teaching and
does more on its own.

No LLM. No API. Pure algorithm + knowledge graph + curiosity.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path

from klomboagi.reasoning.deriver import PropertyDeriver, KnowledgeGraph
from klomboagi.reasoning.engine import ReasoningEngine, ReasoningChain
from klomboagi.reasoning.curiosity import CuriosityDriver, GapPriority, SenseType
from klomboagi.reasoning.truth import TruthValue, EvidenceStamp, Belief, revision
from klomboagi.senses.reader import Reader
from klomboagi.senses.searcher import Searcher
from klomboagi.senses.executor import Executor


@dataclass
class Memory:
    """Everything the system has learned, persistently."""
    concepts: dict[str, dict] = field(default_factory=dict)  # concept_name → facts about it
    beliefs: dict[str, dict] = field(default_factory=dict)      # statement → serialized Belief
    conversations: list[dict] = field(default_factory=list)   # full conversation history
    teachings: list[dict] = field(default_factory=list)        # things human explicitly taught
    discoveries: list[dict] = field(default_factory=list)      # things system found on its own
    corrections: list[dict] = field(default_factory=list)      # times human said "no, that's wrong"

    def save(self, path: str) -> None:
        data = {
            "concepts": self.concepts,
            "conversations": self.conversations[-1000:],  # Keep last 1000 exchanges
            "teachings": self.teachings[-500:],
            "discoveries": self.discoveries[-500:],
            "corrections": self.corrections[-200:],
            "beliefs": self.beliefs,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            self.concepts = data.get("concepts", {})
            self.conversations = data.get("conversations", [])
            self.teachings = data.get("teachings", [])
            self.discoveries = data.get("discoveries", [])
            self.corrections = data.get("corrections", [])
            self.beliefs = data.get("beliefs", {})


class Baby:
    """
    The conversational AGI interface.

    Starts empty. Learns from conversation.
    Uses curiosity to fill gaps on its own.
    No LLM. Pure algorithm.

    Usage:
        baby = Baby()
        response = baby.hear("What's an alligator?")
        print(response)
        # → "I don't know what an alligator is. Let me find out..."
        # → [searches Wikipedia]
        # → "I found out: an alligator is a large reptile..."
    """

    def __init__(self, memory_path: str = "/Volumes/AIStorage/AI/klomboagi/memory/brain.json") -> None:
        self.memory_path = str(Path(memory_path).expanduser())
        self.memory = Memory()
        self.graph = KnowledgeGraph()
        self.deriver = PropertyDeriver(self.graph)
        self.engine = ReasoningEngine()
        self.curiosity = CuriosityDriver()

        # Wire senses
        self.reader = Reader()
        self.searcher = Searcher()
        self.executor = Executor()

        self.curiosity.on_search = self.searcher.search
        self.curiosity.on_read = self.reader.read
        self.curiosity.on_execute = self.executor.execute
        self.curiosity.on_ask_human = lambda q: q  # Will be handled in conversation

        # Load persistent memory
        self.memory.load(self.memory_path)
        self._rebuild_graph_from_memory()

        # Belief system — NARS truth values
        self._evidence_counter = 0
        self._beliefs: dict[str, Belief] = {}
        self._rebuild_beliefs_from_memory()

        # Conversation state
        self.pending_questions: list[str] = []  # Questions to ask the human

    def hear(self, message: str) -> str:
        """
        Process a message from the human.
        This is the main entry point.

        Returns the system's response — from its own understanding.
        """
        # If the message is long (multi-sentence), split and process each sentence
        if len(message) > 200:
            return self._hear_long(message)

        # Record conversation
        self.memory.conversations.append({"role": "human", "content": message})

        # Parse intent
        intent = self._parse_intent(message)

        if intent["type"] == "teach":
            response = self._learn_from_teaching(intent)
        elif intent["type"] == "question":
            response = self._answer_question(intent)
        elif intent["type"] == "command":
            response = self._handle_command(intent)
        elif intent["type"] == "correction":
            response = self._handle_correction(intent)
        else:
            response = self._process_general(message)

        # Record response
        self.memory.conversations.append({"role": "system", "content": response})

        # Save memory
        self.memory.save(self.memory_path)

        return response

    def _hear_long(self, message: str) -> str:
        """
        Process long multi-sentence input by splitting into sentences.
        Each sentence is parsed independently to avoid paragraph-as-predicate.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', message.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return self._process_general(message)

        learned = []
        for sentence in sentences[:20]:  # Cap at 20 sentences
            intent = self._parse_intent(sentence)
            if intent["type"] == "teach":
                subj = intent["subject"]
                pred = intent["predicate"]
                # Cap predicate length
                if len(pred) > 80:
                    pred = pred[:80].rsplit(" ", 1)[0]
                    intent["predicate"] = pred
                self._learn_from_teaching_quiet(intent)
                learned.append(f"{subj} → {pred}")
            elif intent["type"] == "general":
                # Try to extract any "X is Y" from general text
                for m in re.finditer(
                    r'(?:A |An |The )?(\b[\w\s]{2,25}?\b)\s+(?:is|are)\s+(?:a |an |the )?([\w\s]{2,60})',
                    sentence, re.IGNORECASE
                ):
                    subj = m.group(1).strip().lower()
                    pred = m.group(2).strip().lower()
                    if len(subj) > 1 and len(pred) > 1 and subj != pred and len(pred) <= 80:
                        self.graph.add(subj, is_a=[pred])
                        if subj not in self.memory.concepts:
                            self.memory.concepts[subj] = {"facts": [], "taught_by": "human"}
                        if pred not in self.memory.concepts[subj]["facts"]:
                            self.memory.concepts[subj]["facts"].append(pred)
                        learned.append(f"{subj} → {pred}")

        self.memory.conversations.append({"role": "human", "content": message[:500]})

        if learned:
            # Deduplicate
            seen = set()
            unique = []
            for item in learned:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            response = f"I absorbed {len(unique)} facts from that:\n"
            for item in unique[:15]:
                response += f"  {item}\n"
            if len(unique) > 15:
                response += f"  ...and {len(unique) - 15} more."
        else:
            response = "I heard you, but I couldn't extract specific facts from that. Try shorter sentences like 'X is Y'."

        self.memory.conversations.append({"role": "system", "content": response})
        self.memory.save(self.memory_path)
        return response

    def _learn_from_teaching_quiet(self, intent: dict) -> None:
        """Learn without generating a full response (for batch processing)."""
        subject = intent["subject"]
        predicate = intent["predicate"]
        statement = f"{subject} is {predicate}"

        self._evidence_counter += 1
        new_belief = Belief(
            statement=statement,
            truth=TruthValue.from_single_observation(True),
            stamp=EvidenceStamp.new(self._evidence_counter),
            subject=subject,
            predicate=predicate,
            source="human",
        )

        existing = self._beliefs.get(statement)
        if existing:
            revised = existing.revise_with(new_belief)
            if revised:
                self._beliefs[statement] = revised
        else:
            self._beliefs[statement] = new_belief

        self.graph.add(subject, is_a=[predicate])

        if subject not in self.memory.concepts:
            self.memory.concepts[subject] = {"facts": [], "taught_by": "human"}
        if predicate not in self.memory.concepts[subject]["facts"]:
            self.memory.concepts[subject]["facts"].append(predicate)

        self.memory.beliefs[statement] = self._beliefs[statement].to_dict()
        self.memory.teachings.append({"subject": subject, "predicate": predicate})

    def _parse_intent(self, message: str) -> dict:
        """
        Figure out what the human is doing.
        Order matters: correction → command → question → teach → general
        """
        msg = message.strip().lower()

        # 1. Correction: "no,", "wrong", "actually"
        correction_starters = ("no,", "no ", "wrong", "actually", "that's not",
                               "thats not", "incorrect", "not quite")
        if any(msg.startswith(c) for c in correction_starters):
            return {"type": "correction", "content": message, "raw": message}

        # 2. Command: "learn", "search", "look up", "status"
        command_words = {"learn ": "learn", "look up ": "lookup", "search ": "search",
                         "search for ": "search", "find out about ": "search",
                         "read ": "read", "forget ": "forget", "what do you know": "status",
                         "show me": "show", "status": "status", "what do you": "status"}
        for phrase, cmd in command_words.items():
            if msg.startswith(phrase):
                rest = msg[len(phrase):].strip()
                return {"type": "command", "command": cmd, "target": rest, "raw": message}

        # 3. Question: starts with question word or ends with ?
        question_starters = ("what ", "who ", "where ", "how ", "is ", "are ",
                             "can ", "do ", "does ", "why ", "which ", "when ",
                             "tell me about ", "explain ")
        if any(msg.startswith(q) for q in question_starters) or msg.endswith("?"):
            return {"type": "question", "query": message.strip().rstrip("?"), "raw": message}

        # 4. Teaching: "X is Y", "X are Y", "X means Y", "X has Y"
        teach_patterns = [
            r"^(?:a |an |the )?(.+?) (?:is|are) (?:a |an |the )?(.+?)\.?$",
            r"^(.+?) means (.+?)\.?$",
            r"^(.+?) has (.+?)\.?$",
        ]
        for pattern in teach_patterns:
            m = re.match(pattern, msg)
            if m:
                subject = m.group(1).strip()
                predicate = m.group(2).strip()
                # Skip if subject is a question word
                if subject in ("what", "who", "where", "how", "when", "which", "why"):
                    continue
                # Cap predicate length — don't store paragraphs
                if len(predicate) > 80:
                    predicate = predicate[:80].rsplit(" ", 1)[0]
                return {"type": "teach", "subject": subject,
                        "predicate": predicate, "raw": message}

        # 5. General statement
        return {"type": "general", "content": message, "raw": message}

    def _learn_from_teaching(self, intent: dict) -> str:
        """Human is teaching us something. Learn it with evidence-based truth."""
        subject = intent["subject"]
        predicate = intent["predicate"]
        statement = f"{subject} is {predicate}"

        # Create or revise belief with NARS truth values
        self._evidence_counter += 1
        new_belief = Belief(
            statement=statement,
            truth=TruthValue.from_single_observation(True),
            stamp=EvidenceStamp.new(self._evidence_counter),
            subject=subject,
            predicate=predicate,
            source="human",
        )

        existing = self._beliefs.get(statement)
        if existing:
            revised = existing.revise_with(new_belief)
            if revised:
                self._beliefs[statement] = revised
                tv = revised.truth
            else:
                tv = existing.truth
        else:
            self._beliefs[statement] = new_belief
            tv = new_belief.truth

        # Add to knowledge graph
        self.graph.add(subject, is_a=[predicate])

        # Store in memory
        if subject not in self.memory.concepts:
            self.memory.concepts[subject] = {"facts": [], "taught_by": "human"}
        if predicate not in self.memory.concepts[subject]["facts"]:
            self.memory.concepts[subject]["facts"].append(predicate)

        # Persist belief
        self.memory.beliefs[statement] = self._beliefs[statement].to_dict()

        self.memory.teachings.append({
            "subject": subject,
            "predicate": predicate,
            "raw": intent["raw"],
        })

        # Try to connect to what we already know
        connections = self._find_connections(subject)

        # Build response with truth value
        response = f"Got it. {subject} is {predicate}. (confidence: {tv.confidence:.0%})"
        if tv.confidence > 0.7:
            response = f"I'm now quite sure: {subject} is {predicate}. (confidence: {tv.confidence:.0%})"

        if connections:
            response += f" Connects to: {', '.join(connections)}."

        # Check deductions — can we derive new knowledge?
        deductions = self._try_deductions(subject, predicate)
        if deductions:
            response += "\n" + "\n".join(deductions)

        # Check if this fills any curiosity gaps
        resolved = self._check_gaps_resolved(subject, predicate)
        if resolved:
            response += f" That answers my question about {resolved}!"

        # Be curious about new concepts
        unknowns = self._find_unknowns_in(predicate)
        if unknowns:
            gap = unknowns[0]
            self.curiosity.notice_gap(gap, context=f"Learning about {subject}")
            response += f"\n\nI don't know what '{gap}' is yet. Let me find out..."
            event = self.curiosity.investigate()
            if event and event.learned:
                self._process_discovery(gap, event.result)
                response += f"\n{self._summarize_discovery(gap, event.result)}"
            else:
                response += f"\nI couldn't find information about '{gap}'. Can you tell me?"
                self.pending_questions.append(f"What is '{gap}'?")

        return response

    def _answer_question(self, intent: dict) -> str:
        """Human is asking a question. Answer from what we know."""
        query = intent["query"]

        # First: do we know anything about this?
        known = self._what_do_i_know(query)

        if known:
            # Check beliefs for relevant deductions
            relevant_beliefs = []
            stop_words = {"is", "a", "an", "the", "what", "who", "where", "how",
                          "when", "which", "why", "are", "was", "were", "do", "does",
                          "can", "could", "about", "tell", "me", "explain"}
            query_words = set(query.lower().split()) - stop_words
            for statement, belief in self._beliefs.items():
                stmt_words = set(statement.lower().split()) - stop_words
                if query_words & stmt_words:
                    relevant_beliefs.append(belief)

            if relevant_beliefs:
                # Sort by confidence
                relevant_beliefs.sort(key=lambda b: b.truth.confidence, reverse=True)
                lines = ["Based on what I know:"]
                for b in relevant_beliefs[:10]:
                    source_tag = f"[{b.source}]" if b.source != "human" else ""
                    lines.append(f"  • {b.statement} (confidence: {b.truth.confidence:.0%}) {source_tag}")

                # Check for deduction chains we can make
                chains = self._find_deduction_chains(query_words)
                if chains:
                    lines.append("")
                    lines.append("I can also derive:")
                    for chain in chains:
                        lines.append(f"  → {chain}")

                return "\n".join(lines)

            # Fallback to engine reasoning
            facts = [f"{k}: {', '.join(str(f)[:100] for f in v['facts'])}" for k, v in known.items()]
            chain = self.engine.reason(query, facts)

            if chain.confidence > 0.3:
                return f"Based on what I know:\n{chain.conclusion}\n\nMy reasoning:\n{chain.explain()}"
            else:
                response = f"I know a little: {'; '.join(facts[:3])}\n"
                response += "But I'm not confident. Let me look for more..."
                return response + self._curious_lookup(query)
        else:
            # We don't know — go find out
            return self._curious_lookup(query)

    def _find_deduction_chains(self, query_words: set) -> list[str]:
        """Find transitive deductions: A→B, B→C ∴ A→C"""
        chains = []
        for stmt1, b1 in list(self._beliefs.items()):
            if not b1.predicate:
                continue
            # If this belief's predicate is another belief's subject, we can chain
            for stmt2, b2 in list(self._beliefs.items()):
                if b2.subject and b2.subject == b1.predicate and b1.subject != b2.predicate:
                    derived_stmt = f"{b1.subject} is {b2.predicate}"
                    if derived_stmt not in self._beliefs:
                        # Only include if relevant to query
                        all_words = set(b1.subject.split()) | set(b2.predicate.split())
                        if all_words & query_words:
                            from klomboagi.reasoning.truth import deduction as nars_ded
                            tv = nars_ded(b1.truth, b2.truth)
                            chains.append(
                                f"{b1.subject} is {b2.predicate} "
                                f"(because {b1.subject}→{b1.predicate}→{b2.predicate}, "
                                f"confidence: {tv.confidence:.0%})"
                            )
                            # Store the derived belief
                            self._evidence_counter += 1
                            derived = Belief(
                                statement=derived_stmt,
                                truth=tv,
                                stamp=b1.stamp.merge(b2.stamp),
                                subject=b1.subject,
                                predicate=b2.predicate,
                                source="deduction",
                            )
                            self._beliefs[derived_stmt] = derived
                            self.memory.beliefs[derived_stmt] = derived.to_dict()
        return chains

    def _curious_lookup(self, topic: str) -> str:
        """Don't know something? Let's find out."""
        # Clean the topic — strip question words
        clean = topic.lower().strip()
        for prefix in ('what is ', 'what are ', 'who is ', 'where is ', 'tell me about ', 'explain ', 'about '):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        clean = clean.strip()
        if not clean:
            clean = topic
        self.curiosity.notice_gap(clean, context="answering a question",
                                  priority=GapPriority.HIGH)
        event = self.curiosity.investigate()

        if event and event.learned:
            self._process_discovery(topic, event.result)
            return f"\nI found out:\n{self._summarize_discovery(topic, event.result)}"
        else:
            # Try a different sense
            event2 = self.curiosity.investigate()
            if event2 and event2.learned:
                self._process_discovery(topic, event2.result)
                return f"\nOn second try, I found:\n{self._summarize_discovery(topic, event2.result)}"
            return f"\nI couldn't find information about '{topic}'. Can you teach me?"

    def _handle_command(self, intent: dict) -> str:
        """Handle explicit commands."""
        cmd = intent["command"]
        target = intent.get("target", "")

        if cmd == "learn":
            clean_target = target
            if clean_target.startswith("about "):
                clean_target = clean_target[6:]
            return self._curious_lookup(clean_target)

        elif cmd == "lookup" or cmd == "search":
            result = self.searcher.search(target)
            if result and len(result) > 50:
                self._process_discovery(target, result)
                return f"Here's what I found:\n{self._summarize_discovery(target, result)}"
            return f"Couldn't find much about '{target}'."

        elif cmd == "read":
            content = self.reader.read(target)
            if content and len(content) > 20:
                summary = content[:500] + "..." if len(content) > 500 else content
                return f"I read it. Content ({len(content)} chars):\n{summary}"
            return f"Couldn't read '{target}'."

        elif cmd == "status":
            return self._status()

        elif cmd == "forget":
            if target in self.memory.concepts:
                del self.memory.concepts[target]
                return f"Forgot everything about '{target}'."
            return f"I didn't know about '{target}' anyway."

        return f"I don't understand the command '{cmd}'."

    def _handle_correction(self, intent: dict) -> str:
        """Human is correcting us. Update understanding."""
        content = intent["content"]
        self.memory.corrections.append({"content": content})

        # Try to extract what's being corrected
        # Remove correction starters
        cleaned = re.sub(r"^(no,?|wrong,?|actually,?|that'?s not right,?)\s*",
                         "", content, flags=re.IGNORECASE).strip()

        if cleaned:
            # Try to parse as a teaching
            teach_intent = self._parse_intent(cleaned)
            if teach_intent["type"] == "teach":
                response = self._learn_from_teaching(teach_intent)
                return f"I stand corrected. {response}"

        return "Understood — I was wrong. Can you tell me what the right answer is?"

    def _process_general(self, message: str) -> str:
        """Process a general statement — extract any knowledge."""
        # Try to find "X is Y" patterns even in longer text
        patterns = [
            r"(\w[\w\s]*?) (?:is|are) (?:a |an |the )?(\w[\w\s]*?)(?:\.|,|$)",
        ]
        learned = []
        for pattern in patterns:
            for m in re.finditer(pattern, message, re.IGNORECASE):
                subject = m.group(1).strip().lower()
                obj = m.group(2).strip().lower()
                if len(subject) > 1 and len(obj) > 1 and subject != obj:
                    self.graph.add(subject, is_a=[obj])
                    if subject not in self.memory.concepts:
                        self.memory.concepts[subject] = {"facts": [], "taught_by": "human"}
                    self.memory.concepts[subject]["facts"].append(obj)
                    learned.append(f"{subject} → {obj}")

        if learned:
            return f"I picked up: {'; '.join(learned)}. Tell me more?"

        return "I'm listening. Teach me something or ask me a question."

    # ── Internal helpers ──

    def _try_deductions(self, subject: str, predicate: str) -> list[str]:
        """Try to derive new knowledge from what we just learned."""
        results = []

        # If we know A→B and B→C, derive A→C
        # Check if predicate has its own facts
        pred_info = self.memory.concepts.get(predicate, {})
        pred_facts = pred_info.get("facts", [])

        for fact in pred_facts:
            chain_statement = f"{subject} is {fact}"
            if chain_statement not in self._beliefs:
                # Derive through deduction
                from klomboagi.reasoning.truth import deduction as nars_deduction
                ab = self._beliefs.get(f"{subject} is {predicate}")
                bc = self._beliefs.get(f"{predicate} is {fact}")
                if ab and bc:
                    derived_tv = nars_deduction(ab.truth, bc.truth)
                    self._evidence_counter += 1
                    derived = Belief(
                        statement=chain_statement,
                        truth=derived_tv,
                        stamp=ab.stamp.merge(bc.stamp),
                        subject=subject,
                        predicate=fact,
                        source="deduction",
                    )
                    self._beliefs[chain_statement] = derived
                    self.memory.beliefs[chain_statement] = derived.to_dict()
                    results.append(
                        f"  → I derived: {subject} is {fact} "
                        f"(confidence: {derived_tv.confidence:.0%}, via {predicate})"
                    )

        return results

    def _rebuild_beliefs_from_memory(self) -> None:
        """Rebuild belief objects from persisted memory."""
        for statement, data in self.memory.beliefs.items():
            try:
                self._beliefs[statement] = Belief.from_dict(data)
                # Track highest evidence counter
                for src in data.get("stamp", {}).get("sources", []):
                    if isinstance(src, int) and src > self._evidence_counter:
                        self._evidence_counter = src
            except Exception:
                pass

    def _find_connections(self, concept: str) -> list[str]:
        """Find what this concept connects to in the graph."""
        connections = []
        c = self.graph.get(concept)
        if c:
            for parent in c.is_a:
                if parent in self.memory.concepts:
                    connections.append(parent)
        # Also check if other concepts reference this one
        for name, info in self.memory.concepts.items():
            if concept in info.get("facts", []) and name != concept:
                connections.append(name)
        return connections[:5]

    # Common English words that should NEVER trigger curiosity gaps.
    # These are function words, common verbs, adjectives, and adverbs
    # that don't represent learnable concepts.
    COMMON_WORDS = frozenset({
        # Articles, pronouns, prepositions, conjunctions
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
        "she", "too", "use", "with", "that", "this", "from", "they", "been",
        "have", "many", "some", "them", "than", "each", "make", "like", "long",
        "look", "come", "could", "people", "into", "just", "about", "would",
        "there", "their", "which", "large", "small", "very", "also", "more",
        "other", "what", "when", "where", "then", "only", "most", "such",
        "both", "even", "well", "back", "much", "will", "still", "should",
        "after", "before", "between", "under", "over", "while", "being",
        "through", "during", "without", "within", "along", "every", "those",
        "same", "another", "because", "however", "never", "always", "often",
        "here", "these", "thing", "things", "really", "need", "right",
        # Common verbs
        "know", "think", "take", "give", "tell", "call", "try", "ask", "work",
        "seem", "feel", "leave", "keep", "put", "run", "read", "set", "turn",
        "show", "hear", "play", "move", "live", "believe", "hold", "bring",
        "happen", "write", "provide", "stand", "lose", "pay", "meet", "include",
        "continue", "start", "begin", "might", "must", "goes", "went", "done",
        "made", "found", "known", "said", "used", "called", "based", "given",
        "does", "using", "means", "says", "told", "takes", "comes", "goes",
        "became", "become", "makes", "follows", "describes", "explains",
        "exists", "contains", "requires", "allows", "creates", "produces",
        "involves", "includes", "provides", "remains", "appears", "occurs",
        "behaves", "combines", "defines", "measures", "observed", "measured",
        "fired", "powered", "composed", "formed", "called", "named",
        # Common adjectives/adverbs
        "good", "great", "first", "last", "little", "own", "important",
        "different", "possible", "able", "certain", "sure", "real", "whole",
        "true", "false", "high", "low", "best", "better", "enough", "far",
        "yet", "quite", "rather", "almost", "already", "actually", "simply",
        "exactly", "especially", "extremely", "incredibly", "completely",
        "essentially", "approximately", "typically", "generally", "basically",
        "probably", "perhaps", "famous", "specific", "particular", "various",
        "several", "modern", "early", "later", "above", "below",
        # Common nouns that aren't learnable concepts
        "way", "part", "case", "fact", "time", "year", "day", "number",
        "point", "place", "world", "hand", "example", "state", "kind",
        "type", "form", "level", "side", "area", "name", "result", "end",
        "feature", "system", "group", "set", "order", "process", "idea",
        "issue", "question", "answer", "problem", "reason", "word", "words",
        "something", "anything", "everything", "nothing", "someone",
        "anyone", "everyone", "others", "rest", "term", "terms",
        # Technical common words
        "data", "value", "values", "model", "method", "function", "rule",
        "rules", "step", "steps", "unit", "units", "version", "discrete",
        "continuous", "fundamental", "standard", "classical", "advanced",
    })

    def _find_unknowns_in(self, text: str) -> list[str]:
        """Find concepts mentioned in text that we don't know about."""
        words = set(re.findall(r'\b([a-z]{4,})\b', text.lower()))  # min 4 chars
        unknowns = []
        for word in words - self.COMMON_WORDS:
            if word not in self.memory.concepts and not self.graph.get(word):
                unknowns.append(word)
        return unknowns[:3]

    def _what_do_i_know(self, query: str) -> dict:
        """Find what we know that's relevant to a query."""
        query_words = set(query.lower().split())
        relevant = {}
        for concept, info in self.memory.concepts.items():
            concept_words = set(concept.lower().split())
            if concept_words & query_words:
                relevant[concept] = info
        return relevant

    def _check_gaps_resolved(self, subject: str, predicate: str) -> str | None:
        """Check if a teaching resolves any curiosity gaps."""
        for gap in self.curiosity.gaps:
            if not gap.resolved and (
                gap.concept.lower() in subject.lower() or
                gap.concept.lower() in predicate.lower()
            ):
                gap.resolved = True
                gap.resolution = f"{subject} is {predicate}"
                return gap.concept
        return None

    def _process_discovery(self, topic: str, raw_info: str) -> None:
        """Process something we discovered on our own."""
        # Extract facts from raw info
        if topic not in self.memory.concepts:
            self.memory.concepts[topic] = {"facts": [], "taught_by": "self"}
        self.memory.concepts[topic]["facts"].append(raw_info[:500])

        self.memory.discoveries.append({
            "topic": topic,
            "info": raw_info[:500],
        })

        # Try to add to knowledge graph
        self.graph.add(topic)

    def _summarize_discovery(self, topic: str, raw_info: str) -> str:
        """Summarize what we discovered — first 300 chars."""
        clean = raw_info.strip()
        if len(clean) > 300:
            clean = clean[:300] + "..."
        return f"About '{topic}': {clean}"

    def _rebuild_graph_from_memory(self) -> None:
        """Rebuild the knowledge graph from persistent memory."""
        for concept, info in self.memory.concepts.items():
            facts = info.get("facts", [])
            for fact in facts:
                if len(fact) < 50:  # Only short facts are likely "is_a" relationships
                    self.graph.add(concept, is_a=[fact])

    def _status(self) -> str:
        """Report what I know."""
        n_concepts = len(self.memory.concepts)
        n_beliefs = len(self._beliefs)
        n_teachings = len(self.memory.teachings)
        n_discoveries = len(self.memory.discoveries)
        n_corrections = len(self.memory.corrections)
        n_conversations = len(self.memory.conversations)
        curiosity_stats = self.curiosity.stats()

        lines = [
            "Here's what I know:",
            f"  Concepts: {n_concepts}",
            f"  Beliefs: {n_beliefs}",
            f"  Taught by human: {n_teachings}",
            f"  Discovered myself: {n_discoveries}",
            f"  Times corrected: {n_corrections}",
            f"  Conversations: {n_conversations}",
            f"  Curiosity gaps: {curiosity_stats['unresolved']} open, {curiosity_stats['resolved']} resolved",
            "",
            "My beliefs (strongest first):",
        ]

        # Sort beliefs by confidence
        sorted_beliefs = sorted(
            self._beliefs.values(),
            key=lambda b: b.truth.confidence,
            reverse=True,
        )
        for b in sorted_beliefs[:20]:
            lines.append(f"  {b.statement} {b.truth} [{b.source}]")

        if len(sorted_beliefs) > 20:
            lines.append(f"  ... and {len(sorted_beliefs) - 20} more beliefs")

        return "\n".join(lines)
