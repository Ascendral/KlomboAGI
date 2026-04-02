"""
KlomboAGI Brain -- the real one.

This replaces the 3400-line genesis.py with something honest.
No 60 imports. No 50 subsystems that do nothing. Just:

1. A Prolog-style inference engine (CoreReasoner)
2. Natural language parsing (in/out)
3. Learning from text and conversation
4. Persistent memory
5. Self-awareness (what it knows, what it doesn't, what it wants to learn)

If it can't do something, it says so. If it can, it shows its work.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from klomboagi.reasoning.core_reasoner import CoreReasoner, Rel, Fact, ReasoningResult


class Brain:
    """The actual thinking engine. No scaffolding."""

    def __init__(self, data_dir: str | None = None):
        # Where to persist knowledge
        if data_dir is None:
            data_dir = os.environ.get("KLOMBOAGI_HOME", "/opt/klomboagi/data")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # The reasoner IS the brain
        self.reasoner = CoreReasoner()

        # Conversation state
        self.turns = 0
        self.history: list[dict] = []  # [{role, content, timestamp}]

        # Load persisted knowledge
        self._load()

        # Seed boot knowledge if empty
        if self.reasoner.total_facts == 0:
            self._seed()

    # ---- Conversation ----

    def hear(self, message: str) -> str:
        """Process a message. Returns a response.

        This is the only entry point. Everything goes through here.
        """
        self.turns += 1
        message = message.strip()
        if not message:
            return ""

        self.history.append({
            "role": "human",
            "content": message,
            "timestamp": time.time(),
        })

        # Parse what the human is doing
        action = self._parse(message)

        if action["type"] == "teach":
            response = self._handle_teach(action)
        elif action["type"] == "question":
            response = self._handle_question(action)
        elif action["type"] == "command":
            response = self._handle_command(action)
        else:
            response = self._handle_statement(action)

        self.history.append({
            "role": "klombo",
            "content": response,
            "timestamp": time.time(),
        })

        # Auto-save every 5 turns
        if self.turns % 5 == 0:
            self._save()

        return response

    def _parse(self, message: str) -> dict:
        """Determine what the human is doing."""
        m = message.lower().strip()

        # Questions
        if m.endswith("?") or m.startswith(("what ", "who ", "where ", "when ",
            "why ", "how ", "is ", "are ", "can ", "does ", "do ", "will ")):
            return {"type": "question", "query": message}

        # Teaching: "X is Y", "X are Y"
        teach = re.match(
            r'^(?:a |an |the )?(.+?)\s+(?:is|are)\s+(.+?)\.?$', m, re.IGNORECASE)
        if teach:
            subject = teach.group(1).strip()
            predicate = teach.group(2).strip()
            if len(subject) > 1 and len(predicate) > 1:
                return {"type": "teach", "subject": subject, "predicate": predicate,
                        "raw": message}

        # Commands
        if m.startswith(("open ", "run ", "exec ", "kill ", "show ", "list ")):
            return {"type": "command", "raw": message}

        # Negation teaching: "X cannot Y", "X is not Y"
        deny = re.match(r'^(.+?)\s+(?:cannot|can\'t|is\s+not|are\s+not)\s+(.+?)\.?$', m)
        if deny:
            return {"type": "deny", "subject": deny.group(1).strip(),
                    "predicate": deny.group(2).strip()}

        return {"type": "statement", "raw": message}

    # ---- Handlers ----

    def _handle_teach(self, action: dict) -> str:
        """Human is teaching a fact."""
        subject = action["subject"].lower()
        predicate = action["predicate"].lower()

        # Determine relation type
        rel = Rel.HAS_PROP
        clean_pred = predicate
        for article in ("a ", "an "):
            if predicate.startswith(article):
                rel = Rel.IS_A
                clean_pred = predicate[len(article):]
                break

        # Check for contradiction with existing knowledge
        existing = self.reasoner.ask(subject, rel, "?")
        contradiction = None
        if existing.known and rel == Rel.IS_A:
            for f in (self.reasoner.facts | self.reasoner.derived):
                if f.subject == subject and f.relation == rel and f.obj != clean_pred:
                    # Check if old and new share a parent (real contradiction)
                    old_parents = {p.obj for p in (self.reasoner.facts | self.reasoner.derived)
                                   if p.subject == f.obj and p.relation == Rel.IS_A}
                    new_parents = {p.obj for p in (self.reasoner.facts | self.reasoner.derived)
                                   if p.subject == clean_pred and p.relation == Rel.IS_A}
                    # Not a contradiction if one is parent of the other
                    if f.obj == clean_pred or clean_pred == f.obj:
                        continue
                    if f.obj in new_parents or clean_pred in {p.obj for p in (self.reasoner.facts | self.reasoner.derived) if p.subject == f.obj}:
                        continue
                    if old_parents & new_parents:
                        contradiction = f

        # Store the fact
        self.reasoner.tell(subject, rel, clean_pred, confidence=0.8, source="taught")
        self.reasoner.forward_chain(max_iterations=3)

        response = f"Learned: {subject} {rel.value} {clean_pred}."

        if contradiction:
            response = (f"Wait -- I previously knew '{contradiction}'. "
                       f"You're telling me {subject} {rel.value} {clean_pred}. "
                       f"Updating.\n\n{response}")

        # Show what was derived
        new_derived = [f for f in self.reasoner.derived
                       if f.subject == subject and f.source == "derived"]
        if new_derived:
            response += "\nDerived:"
            for f in new_derived[:3]:
                response += f"\n  {f}"

        return response

    def _handle_question(self, action: dict) -> str:
        """Human is asking a question. Reason about it."""
        query = action["query"].lower().strip().rstrip("?").strip()

        # Identity
        if query in ("what are you", "who are you"):
            n = self.reasoner.total_facts
            return (f"I am KlomboAGI. A reasoning engine with {n} facts. "
                    f"I derive conclusions through logical inference, not pattern matching. "
                    f"I know what I know and I know what I don't know.")

        # Status
        if query in ("how are you", "how are you doing", "status"):
            return self.status()

        # What do you know
        if query in ("what do you know", "what have you learned"):
            return self._knowledge_summary()

        # What don't you know
        if query in ("what don't you know", "what are you curious about",
                     "what do you want to learn"):
            gaps = self.reasoner.find_gaps()
            if gaps:
                return "I want to know:\n" + "\n".join(f"  {g}" for g in gaps[:10])
            return "I don't have specific gaps right now."

        # Route to reasoner
        answer = self._reason(query)
        if answer is not None:
            return answer

        # Reasoner doesn't know -- try to learn
        learned = self._try_learn(query)
        if learned:
            # Try reasoning again with new knowledge
            answer = self._reason(query)
            if answer is not None:
                return f"I just learned about this.\n\n{answer}"

        return f"I don't know. Teach me?"

    def _handle_command(self, action: dict) -> str:
        """System commands."""
        raw = action["raw"].lower().strip()

        # Hardware
        if raw in ("show hardware", "show system", "system info"):
            try:
                from klomboagi.senses.hardware import HardwareSense
                return HardwareSense().scan().summary()
            except Exception:
                return "Can't read hardware."

        # Processes
        if raw in ("show processes", "list processes", "top"):
            try:
                from klomboagi.senses.system_control import SystemControl
                procs = SystemControl().list_processes()[:10]
                lines = ["Top processes:"]
                for p in procs:
                    lines.append(f"  {p['name']:20s} PID {p['pid']:>7} CPU {p['cpu']:>5.1f}% MEM {p['mem']:>5.1f}%")
                return "\n".join(lines)
            except Exception:
                return "Can't list processes."

        # Open app
        m = re.match(r"open (.+)", raw)
        if m:
            try:
                from klomboagi.senses.system_control import SystemControl
                result = SystemControl().open_app(m.group(1).strip())
                return f"Opened {m.group(1).strip()}." if result.allowed else f"Blocked: {result.blocked_reason}"
            except Exception as e:
                return f"Error: {e}"

        # Safe exec
        m = re.match(r"(?:run|exec) (.+)", raw)
        if m:
            try:
                from klomboagi.senses.system_control import SystemControl
                result = SystemControl().execute(m.group(1).strip())
                if result.allowed:
                    return result.stdout or "(no output)"
                return f"Blocked: {result.blocked_reason}"
            except Exception as e:
                return f"Error: {e}"

        return f"Unknown command: {raw}"

    def _handle_statement(self, action: dict) -> str:
        """Human said something that isn't a question or teach."""
        # Try to extract facts from it
        new = self.reasoner.learn_from_text(action["raw"])
        if new:
            return f"Noted. Extracted {len(new)} facts."
        return "I'm listening. Teach me something or ask me a question."

    # ---- Reasoning ----

    def _reason(self, query: str) -> str | None:
        """Parse a natural language question and run inference."""

        # "what is X"
        m = re.match(r"what (?:is|are) (?:a |an |the )?(.+)", query)
        if m:
            subject = m.group(1).strip()
            result = self.reasoner.ask(subject, Rel.IS_A, "?")
            if result.known:
                return result.explain()
            # Try properties
            result = self.reasoner.ask(subject, Rel.HAS_PROP, "?")
            if result.known:
                return result.explain()
            return None

        # "is X a Y"
        m = re.match(r"is (?:a |an |the )?(\w[\w\s]*?)\s+(?:a |an |the )?([\w\s]+)", query)
        if m:
            s, o = m.group(1).strip(), m.group(2).strip()
            result = self.reasoner.ask(s, Rel.IS_A, o)
            if result.known:
                return result.explain()
            return None

        # "can X Y"
        m = re.match(r"can (?:a |an |the )?(\w+)\s+(.+)", query)
        if m:
            s, o = m.group(1).strip(), m.group(2).strip()
            result = self.reasoner.ask(s, Rel.CAN, o)
            if result.known:
                return result.explain()
            return None

        # "what causes X" / "why does X happen"
        m = re.match(r"(?:what causes|why does?\s+\w+\s+\w*\s*(?:happen)?)\s*(.+)", query)
        if m:
            obj = m.group(1).strip()
            # Search backward for causes
            all_facts = self.reasoner.facts | self.reasoner.derived
            causes = [f.subject for f in all_facts if f.relation == Rel.CAUSES and f.obj == obj]
            if causes:
                return f"{obj} is caused by: {', '.join(causes)}"
            # Try gerund
            if not obj.endswith("ing"):
                gerund = obj.rstrip("e") + "ing" if obj.endswith("e") else obj + "ing"
                causes = [f.subject for f in all_facts if f.relation == Rel.CAUSES and f.obj == gerund]
                if causes:
                    return f"{gerund} is caused by: {', '.join(causes)}"
            return None

        # "what does X cause"
        m = re.match(r"what does (?:a |an |the )?(.+?)\s+cause", query)
        if m:
            result = self.reasoner.ask(m.group(1).strip(), Rel.CAUSES, "?")
            if result.known:
                return result.explain()
            return None

        # "is X bigger/heavier/etc than Y"
        m = re.match(r"is (?:a |an |the )?(.+?)\s+(bigger|heavier|longer|taller|larger|smaller|lighter|shorter)\s+than\s+(?:a |an |the )?(.+)", query)
        if m:
            a, comp, b = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            prop_map = {"bigger": "weight", "larger": "weight", "smaller": "weight",
                        "lighter": "weight", "heavier": "weight",
                        "longer": "length", "shorter": "length", "taller": "height"}
            result = self.reasoner.ask_compare(a, b, prop_map.get(comp, "weight"))
            return result.answer

        # "where is X"
        m = re.match(r"where (?:is|does|do)\s+(?:a |an |the )?(.+?)(?:\s+live)?$", query)
        if m:
            result = self.reasoner.ask(m.group(1).strip(), Rel.LOCATED, "?")
            if result.known:
                return result.explain()
            return None

        # "what properties does X have"
        m = re.match(r"what (?:properties|traits|features)\s+does\s+(?:a |an |the )?(.+?)\s+have", query)
        if m:
            result = self.reasoner.ask(m.group(1).strip(), Rel.HAS_PROP, "?")
            if result.known:
                return result.explain()
            return None

        return None

    def _try_learn(self, query: str) -> bool:
        """Try to learn about a topic from the internet."""
        try:
            from klomboagi.senses.searcher import Searcher
            searcher = Searcher()
            # Extract topic from query
            words = [w for w in query.split()
                     if w not in ("what", "is", "a", "an", "the", "does", "do",
                                  "can", "how", "where", "why", "are", "has", "have")
                     and len(w) > 2]
            if not words:
                return False
            topic = " ".join(words[:3])
            raw = searcher.search(topic)
            if raw and "Could not find" not in raw:
                new_facts = self.reasoner.learn_from_text(raw)
                return len(new_facts) > 0
        except Exception:
            pass
        return False

    # ---- Status ----

    def status(self) -> str:
        given = len(self.reasoner.facts)
        derived = len(self.reasoner.derived)
        numeric = len(self.reasoner.numeric_facts)
        total = self.reasoner.total_facts
        gaps = len(self.reasoner.find_gaps())

        lines = [
            f"Facts: {total} ({given} given, {derived} derived, {numeric} numeric)",
            f"Rules: {len(self.reasoner.rules)}",
            f"Knowledge gaps: {gaps}",
            f"Conversation turns: {self.turns}",
        ]

        try:
            from klomboagi.senses.hardware import HardwareSense
            hw = HardwareSense().scan()
            lines.append(f"Running on: {hw.cpu.model}, {hw.ram.total_gb:.0f}GB RAM")
        except Exception:
            pass

        return "\n".join(lines)

    def _knowledge_summary(self) -> str:
        all_facts = self.reasoner.facts | self.reasoner.derived
        # Group by relation type
        by_type = {}
        for f in all_facts:
            by_type.setdefault(f.relation.value, []).append(f)

        lines = [f"I know {self.reasoner.total_facts} facts:"]
        for rel, facts in sorted(by_type.items()):
            lines.append(f"\n  {rel} ({len(facts)}):")
            for f in facts[:5]:
                lines.append(f"    {f.subject} -> {f.obj} [{f.confidence:.0%}]")
            if len(facts) > 5:
                lines.append(f"    ... and {len(facts) - 5} more")
        return "\n".join(lines)

    # ---- Persistence ----

    def _save(self) -> None:
        """Save all knowledge to disk."""
        data = {
            "facts": [{"s": f.subject, "r": f.relation.value, "o": f.obj,
                        "c": f.confidence, "src": f.source}
                       for f in self.reasoner.facts],
            "numeric": [{"s": nf.subject, "p": nf.property, "v": nf.value,
                         "u": nf.unit, "c": nf.confidence, "src": nf.source}
                        for nf in self.reasoner.numeric_facts],
            "blocked": list(self.reasoner._blocked) if hasattr(self.reasoner, '_blocked') else [],
            "turns": self.turns,
            "history": self.history[-100:],  # Keep last 100 exchanges
        }
        path = self.data_dir / "brain.json"
        path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load knowledge from disk."""
        path = self.data_dir / "brain.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            rel_map = {r.value: r for r in Rel}
            for f in data.get("facts", []):
                rel = rel_map.get(f["r"])
                if rel:
                    self.reasoner.tell(f["s"], rel, f["o"], f.get("c", 0.8), f.get("src", "loaded"))
            for nf in data.get("numeric", []):
                self.reasoner.tell_numeric(nf["s"], nf["p"], nf["v"], nf["u"],
                                           nf.get("c", 0.8), nf.get("src", "loaded"))
            for blocked in data.get("blocked", []):
                if len(blocked) == 3:
                    rel = rel_map.get(blocked[1])
                    if rel:
                        self.reasoner.deny(blocked[0], rel, blocked[2])
            self.turns = data.get("turns", 0)
            self.history = data.get("history", [])
            # Derive everything from loaded facts
            self.reasoner.forward_chain()
        except Exception:
            pass

    def _seed(self) -> None:
        """Foundational knowledge. Like instincts -- not learned, but needed."""
        r = self.reasoner

        r.tell_many([
            ("dog", Rel.IS_A, "mammal"), ("cat", Rel.IS_A, "mammal"),
            ("horse", Rel.IS_A, "mammal"), ("whale", Rel.IS_A, "mammal"),
            ("bat", Rel.IS_A, "mammal"),
            ("snake", Rel.IS_A, "reptile"), ("lizard", Rel.IS_A, "reptile"),
            ("alligator", Rel.IS_A, "reptile"),
            ("eagle", Rel.IS_A, "bird"), ("penguin", Rel.IS_A, "bird"),
            ("sparrow", Rel.IS_A, "bird"),
            ("salmon", Rel.IS_A, "fish"), ("shark", Rel.IS_A, "fish"),
            ("mammal", Rel.IS_A, "animal"), ("reptile", Rel.IS_A, "animal"),
            ("bird", Rel.IS_A, "animal"), ("fish", Rel.IS_A, "animal"),
            ("insect", Rel.IS_A, "animal"),
            ("animal", Rel.IS_A, "living thing"),
            ("plant", Rel.IS_A, "living thing"),
            ("mammal", Rel.HAS_PROP, "warm-blooded"),
            ("mammal", Rel.HAS_PROP, "has fur"),
            ("reptile", Rel.HAS_PROP, "cold-blooded"),
            ("reptile", Rel.HAS_PROP, "has scales"),
            ("bird", Rel.HAS_PROP, "has feathers"),
            ("bird", Rel.CAN, "fly"),
            ("fish", Rel.HAS_PROP, "has gills"),
            ("fish", Rel.CAN, "swim"),
            ("whale", Rel.CAN, "swim"), ("whale", Rel.LOCATED, "ocean"),
            ("gravity", Rel.CAUSES, "falling"),
            ("heat", Rel.CAUSES, "expansion"),
            ("cold", Rel.CAUSES, "contraction"),
            ("fire", Rel.REQUIRES, "oxygen"),
            ("fire", Rel.REQUIRES, "fuel"),
            ("rain", Rel.CAUSES, "wet ground"),
        ], confidence=0.95, source="boot")

        r.deny("penguin", Rel.CAN, "fly", source="boot")

        r.tell_numeric("whale", "length", 25, "meters", 0.8, "boot")
        r.tell_numeric("whale", "weight", 140000, "kg", 0.8, "boot")
        r.tell_numeric("car", "length", 4.5, "meters", 0.9, "boot")
        r.tell_numeric("car", "weight", 1500, "kg", 0.9, "boot")
        r.tell_numeric("human", "height", 1.7, "meters", 0.8, "boot")
        r.tell_numeric("human", "weight", 70, "kg", 0.8, "boot")

        r.forward_chain()
        self._save()
