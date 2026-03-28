"""
Natural Language Understanding — parse English into structured meaning.

No LLM. No ML model. Algorithmic parsing.

The system needs to understand:
  "gravity pulls objects together" → (gravity, causes, objects moving together)
  "the dog that lives next door is brown" → (dog, is, brown)
  "because heat increases, molecules move faster" → (heat increase, causes, molecule speed increase)
  "Newton discovered gravity in 1687" → (Newton, discovered, gravity) + (gravity, temporal, 1687)

Approach: rule-based constituency parsing + semantic role extraction.
Not as good as a neural parser, but it's OUR understanding, not borrowed.

Pipeline:
  1. Tokenize → split into words + punctuation
  2. POS tag → noun, verb, adjective, etc. (rule-based)
  3. Chunk → noun phrases, verb phrases, prep phrases
  4. Extract triples → (subject, relation, object)
  5. Resolve clauses → handle "which", "that", "because", "when"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class POS(Enum):
    """Part of speech tags."""
    NOUN = "noun"
    VERB = "verb"
    ADJ = "adjective"
    ADV = "adverb"
    DET = "determiner"
    PREP = "preposition"
    CONJ = "conjunction"
    PRON = "pronoun"
    NUM = "number"
    PUNCT = "punctuation"
    REL = "relative"       # which, that, who
    SUB = "subordinator"   # because, when, if, although
    NEG = "negation"


@dataclass
class Token:
    """A word with its part of speech."""
    text: str
    pos: POS
    index: int

    def __repr__(self) -> str:
        return f"{self.text}/{self.pos.value}"


@dataclass
class Triple:
    """A subject-relation-object triple extracted from text."""
    subject: str
    relation: str          # verb or relation type
    object: str
    confidence: float = 0.7
    clause_type: str = "main"  # "main", "relative", "causal", "temporal"
    negated: bool = False
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": round(self.confidence, 3),
            "clause_type": self.clause_type,
            "negated": self.negated,
        }

    def as_belief(self) -> tuple[str, str]:
        """Convert to (subject, predicate) for the belief system."""
        if self.relation in ("is", "are", "was", "were"):
            return (self.subject, self.object)
        return (self.subject, f"{self.relation} {self.object}")

    def as_relation(self) -> tuple[str, str, str] | None:
        """Convert to (source, relation_type, target) if applicable."""
        causal_verbs = {"causes", "cause", "creates", "produces", "leads to",
                        "results in", "generates", "triggers", "induces"}
        require_verbs = {"requires", "require", "needs", "depends on"}
        enable_verbs = {"enables", "enable", "allows", "permits", "lets"}
        contain_verbs = {"contains", "contain", "includes", "has", "have",
                         "consists of", "comprises"}
        use_verbs = {"uses", "use", "employs", "utilizes", "applies"}

        rel_lower = self.relation.lower()
        if rel_lower in causal_verbs:
            return (self.subject, "causes", self.object)
        if rel_lower in require_verbs:
            return (self.subject, "requires", self.object)
        if rel_lower in enable_verbs:
            return (self.subject, "enables", self.object)
        if rel_lower in contain_verbs:
            return (self.object, "part_of", self.subject)  # reverse: X contains Y → Y part_of X
        if rel_lower in use_verbs:
            return (self.subject, "uses", self.object)
        if rel_lower in ("measures", "measure"):
            return (self.subject, "measures", self.object)
        return None


# ── POS Lexicon (rule-based, no ML) ──

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my",
               "your", "his", "her", "its", "our", "their", "some", "any",
               "no", "every", "each", "all", "both", "few", "many", "much",
               "several", "most"}

PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "from", "by", "of",
                "about", "into", "through", "during", "before", "after",
                "above", "below", "between", "under", "over", "near",
                "behind", "across", "along", "around", "against", "within",
                "without", "toward", "towards", "upon", "among", "beyond"}

PRONOUNS = {"i", "me", "you", "he", "him", "she", "her", "it", "we", "us",
            "they", "them", "myself", "yourself", "himself", "herself",
            "itself", "ourselves", "themselves"}

CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "for"}

RELATIVES = {"which", "that", "who", "whom", "whose", "where", "when"}

SUBORDINATORS = {"because", "since", "although", "though", "if", "unless",
                 "while", "whereas", "whenever", "wherever", "until",
                 "so that", "in order to", "even though"}

NEGATIONS = {"not", "never", "no", "neither", "nor", "hardly", "barely",
             "scarcely", "n't", "doesn't", "don't", "didn't", "isn't",
             "aren't", "wasn't", "weren't", "won't", "can't", "cannot"}

# Common verbs (not exhaustive but covers key patterns)
VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did", "doing", "done",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "causes", "cause", "caused", "causing",
    "requires", "require", "required", "requiring",
    "enables", "enable", "enabled", "enabling",
    "uses", "use", "used", "using",
    "contains", "contain", "contained", "containing",
    "includes", "include", "included", "including",
    "creates", "create", "created", "creating",
    "produces", "produce", "produced", "producing",
    "measures", "measure", "measured", "measuring",
    "describes", "describe", "described", "describing",
    "explains", "explain", "explained", "explaining",
    "makes", "make", "made", "making",
    "gives", "give", "gave", "given", "giving",
    "takes", "take", "took", "taken", "taking",
    "becomes", "become", "became", "becoming",
    "shows", "show", "showed", "shown", "showing",
    "leads", "lead", "led", "leading",
    "follows", "follow", "followed", "following",
    "moves", "move", "moved", "moving",
    "changes", "change", "changed", "changing",
    "grows", "grow", "grew", "grown", "growing",
    "increases", "increase", "increased", "increasing",
    "decreases", "decrease", "decreased", "decreasing",
    "pulls", "pull", "pulled", "pulling",
    "pushes", "push", "pushed", "pushing",
    "exists", "exist", "existed", "existing",
    "means", "mean", "meant", "meaning",
    "involves", "involve", "involved", "involving",
    "determines", "determine", "determined", "determining",
    "allows", "allow", "allowed", "allowing",
    "prevents", "prevent", "prevented", "preventing",
    "generates", "generate", "generated", "generating",
    "results", "result", "resulted", "resulting",
    "connects", "connect", "connected", "connecting",
    "relates", "relate", "related", "relating",
    "defines", "define", "defined", "defining",
    "discovered", "discovered", "discovers",
    "invented", "invents",
    "developed", "develops",
    "studied", "studies",
}

ADJECTIVES = {
    "large", "small", "big", "little", "great", "old", "new", "young",
    "good", "bad", "high", "low", "long", "short", "hot", "cold",
    "fast", "slow", "strong", "weak", "heavy", "light", "hard", "soft",
    "important", "fundamental", "basic", "primary", "secondary",
    "positive", "negative", "natural", "physical", "chemical",
    "mathematical", "scientific", "electrical", "magnetic",
    "nuclear", "atomic", "molecular", "biological", "genetic",
    "red", "blue", "green", "yellow", "black", "white", "brown",
    "first", "second", "third", "last", "next", "final",
    "true", "false", "real", "complex", "simple", "pure",
}


class NLU:
    """
    Natural Language Understanding engine.

    Parses English sentences into structured triples.
    No ML. Rule-based POS tagging + chunking + triple extraction.
    """

    def parse(self, text: str) -> list[Triple]:
        """
        Parse text into structured triples.

        Returns list of (subject, relation, object) triples.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            sentences = [text]

        all_triples = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            triples = self._parse_sentence(sentence)
            all_triples.extend(triples)

        return all_triples

    def _parse_sentence(self, sentence: str) -> list[Triple]:
        """Parse a single sentence into triples."""
        # Tokenize and POS tag
        tokens = self._tokenize(sentence)
        tokens = self._pos_tag(tokens)

        triples = []

        # Check for subordinate clauses first (because X, Y)
        sub_triples = self._extract_subordinate(tokens, sentence)
        if sub_triples:
            triples.extend(sub_triples)

        # Check for relative clauses (X which/that Y)
        rel_triples = self._extract_relative(tokens, sentence)
        if rel_triples:
            triples.extend(rel_triples)

        # Extract main clause SVO triples
        main_triples = self._extract_svo(tokens, sentence)
        triples.extend(main_triples)

        return triples

    def _tokenize(self, sentence: str) -> list[Token]:
        """Split into tokens."""
        # Handle contractions
        text = sentence.replace("n't", " not").replace("'s", " is").replace("'re", " are")
        # Split on whitespace and punctuation
        raw_tokens = re.findall(r"[\w]+|[.,;:!?]", text)
        return [Token(text=t, pos=POS.NOUN, index=i) for i, t in enumerate(raw_tokens)]

    def _pos_tag(self, tokens: list[Token]) -> list[Token]:
        """Rule-based POS tagging."""
        for token in tokens:
            word = token.text.lower()

            if word in DETERMINERS:
                token.pos = POS.DET
            elif word in PREPOSITIONS:
                token.pos = POS.PREP
            elif word in PRONOUNS:
                token.pos = POS.PRON
            elif word in CONJUNCTIONS:
                token.pos = POS.CONJ
            elif word in RELATIVES:
                token.pos = POS.REL
            elif word in SUBORDINATORS:
                token.pos = POS.SUB
            elif word in NEGATIONS:
                token.pos = POS.NEG
            elif word in VERBS:
                token.pos = POS.VERB
            elif word in ADJECTIVES:
                token.pos = POS.ADJ
            elif word in {".", ",", ";", ":", "!", "?"}:
                token.pos = POS.PUNCT
            elif re.match(r'^\d+\.?\d*$', word):
                token.pos = POS.NUM
            elif word.endswith("ly"):
                token.pos = POS.ADV
            elif word.endswith(("tion", "ment", "ness", "ity", "ence", "ance", "ism")):
                token.pos = POS.NOUN
            elif word.endswith(("ing", "ed", "es", "s")) and word in VERBS:
                token.pos = POS.VERB
            # Default: NOUN (nouns are most common in factual text)

        return tokens

    def _extract_svo(self, tokens: list[Token], raw: str) -> list[Triple]:
        """Extract Subject-Verb-Object triples from main clause."""
        triples = []

        # Don't extract from questions — questions are queries, not statements
        if raw.strip().endswith("?"):
            return []
        first_word = tokens[0].text.lower() if tokens else ""
        if first_word in ("what", "who", "where", "how", "why", "when", "which",
                          "is", "are", "do", "does", "can", "could", "would", "should"):
            return []

        # Find verb positions
        verb_positions = [i for i, t in enumerate(tokens) if t.pos == POS.VERB]
        if not verb_positions:
            return []

        for verb_idx in verb_positions:
            # Subject = nouns/adjectives before verb
            subject_tokens = []
            for i in range(verb_idx - 1, -1, -1):
                if tokens[i].pos in (POS.NOUN, POS.ADJ, POS.NUM):
                    subject_tokens.insert(0, tokens[i])
                elif tokens[i].pos == POS.DET:
                    continue  # skip determiners
                elif tokens[i].pos in (POS.PREP, POS.CONJ, POS.PUNCT, POS.SUB, POS.REL):
                    break  # stop at clause boundary
                else:
                    break

            # Object = nouns/adjectives after verb
            object_tokens = []
            negated = False
            for i in range(verb_idx + 1, len(tokens)):
                if tokens[i].pos == POS.NEG:
                    negated = True
                    continue
                if tokens[i].pos in (POS.NOUN, POS.ADJ, POS.NUM, POS.ADV):
                    object_tokens.append(tokens[i])
                elif tokens[i].pos == POS.DET:
                    continue
                elif tokens[i].pos == POS.PREP:
                    # Include prepositional phrase: "part of X"
                    prep_obj = []
                    for j in range(i + 1, min(i + 5, len(tokens))):
                        if tokens[j].pos in (POS.NOUN, POS.ADJ):
                            prep_obj.append(tokens[j])
                        elif tokens[j].pos == POS.DET:
                            continue
                        else:
                            break
                    if prep_obj:
                        object_tokens.append(Token(
                            text=tokens[i].text + " " + " ".join(t.text for t in prep_obj),
                            pos=POS.NOUN, index=i))
                    break
                elif tokens[i].pos in (POS.CONJ, POS.PUNCT, POS.REL, POS.SUB):
                    break

            if subject_tokens and object_tokens:
                subject = " ".join(t.text.lower() for t in subject_tokens)
                verb = tokens[verb_idx].text.lower()
                obj = " ".join(t.text.lower() for t in object_tokens)

                triples.append(Triple(
                    subject=subject,
                    relation=verb,
                    object=obj,
                    negated=negated,
                    raw=raw,
                ))

        return triples

    def _extract_subordinate(self, tokens: list[Token], raw: str) -> list[Triple]:
        """Extract causal triples from subordinate clauses (because X, Y)."""
        triples = []
        for i, token in enumerate(tokens):
            if token.pos == POS.SUB and token.text.lower() in ("because", "since"):
                # "because X, Y happens" → X causes Y
                # Find the cause (after "because") and effect (before or after comma)
                cause_tokens = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j].pos in (POS.PUNCT, POS.CONJ):
                        break
                    if tokens[j].pos in (POS.NOUN, POS.ADJ, POS.VERB):
                        cause_tokens.append(tokens[j])

                # Effect is the other clause
                effect_tokens = []
                # Look before the subordinator
                for j in range(i - 1, -1, -1):
                    if tokens[j].pos in (POS.PUNCT, POS.CONJ):
                        break
                    if tokens[j].pos in (POS.NOUN, POS.ADJ, POS.VERB):
                        effect_tokens.insert(0, tokens[j])

                if cause_tokens and effect_tokens:
                    cause = " ".join(t.text.lower() for t in cause_tokens)
                    effect = " ".join(t.text.lower() for t in effect_tokens)
                    triples.append(Triple(
                        subject=cause,
                        relation="causes",
                        object=effect,
                        clause_type="causal",
                        raw=raw,
                    ))

        return triples

    def _extract_relative(self, tokens: list[Token], raw: str) -> list[Triple]:
        """Extract triples from relative clauses (X which/that Y)."""
        triples = []
        for i, token in enumerate(tokens):
            if token.pos == POS.REL and token.text.lower() in ("which", "that", "who"):
                # Antecedent = noun before "which"
                antecedent = ""
                for j in range(i - 1, -1, -1):
                    if tokens[j].pos in (POS.NOUN, POS.ADJ):
                        antecedent = tokens[j].text.lower()
                        break

                # Relative clause = verb + object after "which"
                rel_verb = ""
                rel_obj_tokens = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j].pos == POS.VERB and not rel_verb:
                        rel_verb = tokens[j].text.lower()
                    elif rel_verb and tokens[j].pos in (POS.NOUN, POS.ADJ):
                        rel_obj_tokens.append(tokens[j])
                    elif tokens[j].pos in (POS.PUNCT, POS.CONJ, POS.REL):
                        break

                if antecedent and rel_verb and rel_obj_tokens:
                    rel_obj = " ".join(t.text.lower() for t in rel_obj_tokens)
                    triples.append(Triple(
                        subject=antecedent,
                        relation=rel_verb,
                        object=rel_obj,
                        clause_type="relative",
                        raw=raw,
                    ))

        return triples

    def extract_facts(self, text: str) -> list[tuple[str, str]]:
        """
        Extract (subject, predicate) facts from text.

        Higher quality than regex — understands sentence structure.
        """
        triples = self.parse(text)
        facts = []
        for t in triples:
            subj, pred = t.as_belief()
            if len(subj) > 1 and len(pred) > 3:
                facts.append((subj, pred))
        return facts

    def extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract (source, relation_type, target) relations from text.

        Only returns triples that map to known relation types.
        """
        triples = self.parse(text)
        relations = []
        for t in triples:
            rel = t.as_relation()
            if rel:
                relations.append(rel)
        return relations
