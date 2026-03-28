"""
LLM Translator — the language sense, NOT the brain.

STRICT BOUNDARY:
  The LLM touches INBOUND text only.
  It converts messy English → clean structured triples.
  It NEVER answers questions.
  It NEVER touches reasoning.
  It NEVER touches output.
  It NEVER touches beliefs directly.
  It NEVER touches the knowledge graph directly.

Everything that crosses the boundary gets:
  - Tagged with source="llm_parse"
  - Logged in the audit trail
  - Validated by the brain before storage

The brain can function WITHOUT the LLM. The LLM is a hearing aid.
Turn it off → worse parsing, same brain, same reasoning.

Usage:
    translator = LLMTranslator(api_key="...")
    triples = translator.parse("Because friction generates heat...")
    # → [("friction", "causes", "heat"), ...]
    # Each triple tagged source="llm_parse"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class ParsedTriple:
    """A structured triple parsed by the LLM. Always tagged."""
    subject: str
    relation: str      # "is_a", "causes", "requires", "part_of", "uses", etc.
    object: str
    source: str = "llm_parse"   # ALWAYS tagged
    confidence: float = 0.5     # LLM parsing confidence (lower than human teaching)
    raw_text: str = ""          # the original text that produced this

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class AuditEntry:
    """Record of every LLM interaction for transparency."""
    timestamp: str
    input_text: str
    output_triples: list[dict]
    model_used: str
    tokens_used: int
    duration_ms: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "input": self.input_text[:200],
            "triples_produced": len(self.output_triples),
            "model": self.model_used,
            "tokens": self.tokens_used,
        }


# The prompt that constrains the LLM to ONLY parse, never reason
PARSE_PROMPT = """You are a language parser. Your ONLY job is to extract structured facts from text.

Given a text, extract (subject, relation, object) triples. Nothing else.

Valid relations: is_a, causes, requires, part_of, uses, enables, measures, opposite_of, has_property, located_in, created_by, happened_before, happened_after

Rules:
- Extract ONLY what the text explicitly states
- Do NOT infer or reason about the content
- Do NOT add your own knowledge
- Do NOT answer questions — only parse statements
- Keep subjects and objects short (1-5 words each)
- Output ONLY valid JSON array of triples

Format: [{"s": "subject", "r": "relation", "o": "object"}, ...]

If the text contains no extractable facts, output: []
"""


class LLMTranslator:
    """
    Translates natural language → structured triples.

    The LLM is a MICROPHONE, not a SPEAKER.
    It hears complex English and translates to clean data.
    The brain does ALL the thinking.
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514",
                 enabled: bool = False) -> None:
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.audit_log: list[AuditEntry] = []
        self._call_count = 0
        self._total_tokens = 0

    def parse(self, text: str) -> list[ParsedTriple]:
        """
        Parse text into structured triples.

        If LLM is disabled, falls back to rule-based NLU.
        Everything tagged source="llm_parse".
        """
        if not self.enabled or not self.api_key:
            return self._fallback_parse(text)

        start = time.time()
        triples = self._call_llm(text)
        duration = (time.time() - start) * 1000

        # Audit every call
        self.audit_log.append(AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            input_text=text,
            output_triples=[t.to_dict() for t in triples],
            model_used=self.model,
            tokens_used=0,  # Filled by actual API response
            duration_ms=duration,
        ))

        return triples

    def _call_llm(self, text: str) -> list[ParsedTriple]:
        """Call the LLM API to parse text. Returns tagged triples."""
        try:
            import urllib.request

            body = json.dumps({
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": f"{PARSE_PROMPT}\n\nText to parse:\n{text[:2000]}"}
                ],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                response = json.loads(resp.read().decode("utf-8"))

            # Extract text from response
            content = response.get("content", [{}])
            if content and isinstance(content, list):
                raw = content[0].get("text", "[]")
            else:
                raw = "[]"

            # Track tokens
            usage = response.get("usage", {})
            self._total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            self._call_count += 1

            # Parse JSON triples
            return self._parse_response(raw, text)

        except Exception as e:
            # On any error, fall back to rule-based
            return self._fallback_parse(text)

    def _parse_response(self, raw_json: str, original_text: str) -> list[ParsedTriple]:
        """Parse the LLM's JSON response into tagged triples."""
        try:
            # Find JSON array in response
            start = raw_json.find("[")
            end = raw_json.rfind("]") + 1
            if start < 0 or end <= start:
                return []

            data = json.loads(raw_json[start:end])
            triples = []

            for item in data:
                if isinstance(item, dict) and "s" in item and "r" in item and "o" in item:
                    subj = str(item["s"]).strip().lower()
                    rel = str(item["r"]).strip().lower()
                    obj = str(item["o"]).strip().lower()

                    # Validate: subject and object must be short
                    if len(subj) > 50 or len(obj) > 50:
                        continue
                    # Validate: relation must be a known type
                    valid_rels = {"is_a", "causes", "requires", "part_of", "uses",
                                  "enables", "measures", "opposite_of", "has_property",
                                  "located_in", "created_by", "happened_before",
                                  "happened_after"}
                    if rel not in valid_rels:
                        continue

                    triples.append(ParsedTriple(
                        subject=subj,
                        relation=rel,
                        object=obj,
                        source="llm_parse",
                        confidence=0.4,  # Lower than human teaching (0.5)
                        raw_text=original_text[:200],
                    ))

            return triples

        except (json.JSONDecodeError, KeyError):
            return []

    def _fallback_parse(self, text: str) -> list[ParsedTriple]:
        """Rule-based fallback when LLM is disabled."""
        # Use the existing NLU
        try:
            from klomboagi.reasoning.nlu import NLU
            nlu = NLU()
            triples = nlu.parse(text)
            results = []
            for t in triples:
                rel = t.as_relation()
                if rel:
                    results.append(ParsedTriple(
                        subject=rel[0], relation=rel[1], object=rel[2],
                        source="nlu_fallback", confidence=0.3,
                        raw_text=text[:200],
                    ))
                else:
                    subj, pred = t.as_belief()
                    if subj and pred and len(subj) > 1 and len(pred) > 3:
                        results.append(ParsedTriple(
                            subject=subj, relation="is_a", object=pred,
                            source="nlu_fallback", confidence=0.3,
                            raw_text=text[:200],
                        ))
            return results
        except Exception:
            return []

    # ── Transparency ──

    def audit_report(self) -> str:
        """Full transparency report of all LLM interactions."""
        lines = [
            f"LLM Translator Audit Report",
            f"  Enabled: {self.enabled}",
            f"  Model: {self.model}",
            f"  Total calls: {self._call_count}",
            f"  Total tokens: {self._total_tokens}",
            f"  Audit entries: {len(self.audit_log)}",
        ]
        if self.audit_log:
            lines.append(f"\n  Recent interactions:")
            for entry in self.audit_log[-5:]:
                lines.append(f"    [{entry.timestamp}] {entry.input_text[:60]}...")
                lines.append(f"      → {entry.triples_produced} triples, {entry.tokens_used} tokens")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "calls": self._call_count,
            "tokens": self._total_tokens,
            "audit_entries": len(self.audit_log),
        }
