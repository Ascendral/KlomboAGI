"""
Searcher Sense — the ability to search for information.

Uses multiple knowledge sources to find information:
1. Wikipedia REST API (structured knowledge)
2. Wikidata (structured facts + relationships)
3. DuckDuckGo instant answers (general knowledge)

Returns raw results for the reasoning engine to process.
No API keys required. All free public APIs.
"""

from __future__ import annotations


class Searcher:
    """Searches multiple knowledge sources and returns results."""

    def search(self, query: str) -> str:
        """
        Search for information. Tries multiple sources:
        Wikipedia → Wikidata → DuckDuckGo → variations.
        """
        # Try Wikipedia first — best for concept definitions
        result = self._search_wikipedia(query)
        if result and len(result.strip()) > 50:
            if 'may refer to' not in result.lower():
                return result

        # Try Wikidata — structured facts and relationships
        result = self._search_wikidata(query)
        if result and len(result.strip()) > 50:
            return result

        # Try DuckDuckGo instant answers
        result = self._search_duckduckgo(query)
        if result and len(result.strip()) > 50:
            return result

        # Wikipedia with type hints
        for suffix in [' (programming language)', ' (animal)', ' (software)', ' (concept)', '']:
            result = self._search_wikipedia(query + suffix)
            if result and len(result.strip()) > 50 and 'may refer to' not in result.lower():
                return result

        # DuckDuckGo with 'what is' prefix
        result = self._search_duckduckgo(f'what is {query}')
        if result and len(result.strip()) > 50:
            return result

        # Try Open Library for book/author queries
        result = self._search_open_library(query)
        if result and len(result.strip()) > 50:
            return result

        return f"Could not find information about: {query}"

    def _search_wikipedia(self, query: str) -> str:
        """Search Wikipedia API for a summary."""
        try:
            import urllib.request
            import json

            # Wikipedia REST API — no key needed
            safe_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_query}"

            req = urllib.request.Request(url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system; contact: alex@ascendral.com)"
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            extract = data.get("extract", "")
            title = data.get("title", query)

            if extract:
                return f"[Wikipedia: {title}]\n{extract}"
            return ""
        except Exception:
            return ""

    def _search_wikidata(self, query: str) -> str:
        """Search Wikidata for structured facts about an entity."""
        try:
            import urllib.request
            import urllib.parse
            import json

            # Step 1: Search for entity by name
            safe_query = urllib.parse.quote(query)
            search_url = (
                f"https://www.wikidata.org/w/api.php"
                f"?action=wbsearchentities&search={safe_query}"
                f"&language=en&format=json&limit=1"
            )
            req = urllib.request.Request(search_url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("search", [])
            if not results:
                return ""

            # Filter out scholarly articles and disambiguation pages
            skip_descriptions = {
                "scientific article", "scholarly article", "wikimedia",
                "disambiguation", "family name", "given name",
            }
            entity_id = None
            label = query
            description = ""
            for result in results[:5]:
                desc = result.get("description", "").lower()
                if any(skip in desc for skip in skip_descriptions):
                    continue
                entity_id = result["id"]
                label = result.get("label", query)
                description = result.get("description", "")
                break

            if not entity_id:
                return ""

            # Step 2: Get entity details
            entity_url = (
                f"https://www.wikidata.org/w/api.php"
                f"?action=wbgetentities&ids={entity_id}"
                f"&languages=en&format=json&props=claims|descriptions"
            )
            req2 = urllib.request.Request(entity_url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                entity_data = json.loads(resp2.read().decode("utf-8"))

            entity = entity_data.get("entities", {}).get(entity_id, {})
            claims = entity.get("claims", {})

            # Extract key facts
            facts = [f"[Wikidata: {label}]"]
            if description:
                facts.append(f"Description: {description}")

            # Instance of (P31) — what kind of thing it is
            instance_of = self._wikidata_claim_labels(claims.get("P31", []))
            if instance_of:
                facts.append(f"Type: {', '.join(instance_of[:3])}")

            # Subclass of (P279)
            subclass_of = self._wikidata_claim_labels(claims.get("P279", []))
            if subclass_of:
                facts.append(f"Subclass of: {', '.join(subclass_of[:3])}")

            # Part of (P361)
            part_of = self._wikidata_claim_labels(claims.get("P361", []))
            if part_of:
                facts.append(f"Part of: {', '.join(part_of[:3])}")

            # Has parts (P527)
            has_parts = self._wikidata_claim_labels(claims.get("P527", []))
            if has_parts:
                facts.append(f"Has parts: {', '.join(has_parts[:5])}")

            if len(facts) > 2:  # More than just header + description
                return "\n".join(facts)
            return ""

        except Exception:
            return ""

    def _wikidata_claim_labels(self, claims: list) -> list[str]:
        """Extract labels from Wikidata claim values (best-effort)."""
        labels = []
        for claim in claims[:5]:
            try:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value", {})
                if isinstance(value, dict) and "id" in value:
                    # It's an entity reference — use the ID as placeholder
                    labels.append(value["id"])
                elif isinstance(value, str):
                    labels.append(value)
            except Exception:
                continue
        return labels

    def _search_open_library(self, query: str) -> str:
        """Search Open Library for books/authors."""
        try:
            import urllib.request
            import urllib.parse
            import json

            safe_query = urllib.parse.quote(query)
            url = f"https://openlibrary.org/search.json?q={safe_query}&limit=3"
            req = urllib.request.Request(url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            docs = data.get("docs", [])
            if not docs:
                return ""

            parts = [f"[Open Library: '{query}']"]
            for doc in docs[:3]:
                title = doc.get("title", "")
                authors = doc.get("author_name", [])
                year = doc.get("first_publish_year", "")
                if title:
                    author_str = f" by {', '.join(authors[:2])}" if authors else ""
                    year_str = f" ({year})" if year else ""
                    parts.append(f"  • {title}{author_str}{year_str}")

            return "\n".join(parts) if len(parts) > 1 else ""
        except Exception:
            return ""

    def _search_duckduckgo(self, query: str) -> str:
        """Search DuckDuckGo instant answer API."""
        try:
            import urllib.request
            import urllib.parse
            import json

            safe_query = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1"

            req = urllib.request.Request(url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Try abstract first, then answer
            abstract = data.get("AbstractText", "")
            answer = data.get("Answer", "")
            definition = data.get("Definition", "")

            parts = []
            if abstract:
                parts.append(f"[DuckDuckGo]\n{abstract}")
            if answer:
                parts.append(f"Answer: {answer}")
            if definition:
                parts.append(f"Definition: {definition}")

            return "\n".join(parts) if parts else ""
        except Exception:
            return ""
