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

    def __init__(self, memory_path: str = "~/.klomboagi/memory.json") -> None:
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

    def _parse_intent(self, message: str) -> dict:
        """
        Figure out what the human is doing.
        Not NLU — just structural pattern matching.
        """
        msg = message.strip().lower()

        # Teaching: "X is Y", "X are Y", "a X is a Y"
        teach_patterns = [
            r"^(?:a |an |the )?(.+?) (?:is|are) (?:a |an |the )?(.+?)\.?$",
            r"^(.+?) means (.+?)\.?$",
            r"^(.+?) has (.+?)\.?$",
        ]
        for pattern in teach_patterns:
            m = re.match(pattern, msg)
            if m:
                return {"type": "teach", "subject": m.group(1).strip(),
                        "predicate": m.group(2).strip(), "raw": message}

        # Command: check FIRST — commands take priority over questions
        command_words = {"learn": "learn", "look up": "lookup", "search": "search",
                         "read": "read", "forget": "forget", "what do you know": "status",
                         "show me": "show", "status": "status", "what do you": "status"}
        for phrase, cmd in command_words.items():
            if msg.startswith(phrase):
                rest = msg[len(phrase):].strip()
                return {"type": "command", "command": cmd, "target": rest, "raw": message}

        # Question: starts with what/who/where/how/is/are/can/do/why
        question_starters = ("what", "who", "where", "how", "is ", "are ",
                             "can ", "do ", "does ", "why", "which", "when")
        if any(msg.startswith(q) for q in question_starters) or msg.endswith("?"):
            return {"type": "question", "query": message.strip().rstrip("?"), "raw": message}

        # Correction: "no,", "wrong", "actually", "that's not right"
        correction_starters = ("no,", "no ", "wrong", "actually", "that's not",
                               "thats not", "incorrect", "not quite")
        if any(msg.startswith(c) for c in correction_starters):
            return {"type": "correction", "content": message, "raw": message}

        # General statement — try to extract knowledge
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
            # We know something — reason about it
            facts = [f"{k}: {', '.join(v['facts'])}" for k, v in known.items()]
            chain = self.engine.reason(query, facts)

            if chain.confidence > 0.5:
                return f"Based on what I know:\n{chain.conclusion}\n\nMy reasoning:\n{chain.explain()}"
            else:
                # Low confidence — supplement with search
                response = f"I know a little: {'; '.join(facts)}\n"
                response += "But I'm not confident. Let me look for more..."
                return response + self._curious_lookup(query)
        else:
            # We don't know — go find out
            return self._curious_lookup(query)

    def _curious_lookup(self, topic: str) -> str:
        """Don't know something? Let's find out."""
        self.curiosity.notice_gap(topic, context="answering a question",
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
            return self._curious_lookup(target)

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

    def _find_unknowns_in(self, text: str) -> list[str]:
        """Find concepts mentioned in text that we don't know about."""
        words = set(re.findall(r'\b([a-z]{3,})\b', text.lower()))
        common = {"the", "and", "for", "are", "but", "not", "you", "all",
                  "can", "had", "her", "was", "one", "our", "out", "has",
                  "his", "how", "its", "may", "new", "now", "old", "see",
                  "way", "who", "did", "get", "let", "say", "she", "too",
                  "use", "with", "that", "this", "from", "they", "been",
                  "have", "many", "some", "them", "than", "each", "make",
                  "like", "long", "look", "come", "could", "people", "into",
                  "just", "about", "would", "there", "their", "which",
                  "large", "small", "very", "also", "more", "other"}
        unknowns = []
        for word in words - common:
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
