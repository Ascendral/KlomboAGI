"""
Searcher Sense — the ability to search for information.

Uses web search to find information about unknown concepts.
Returns raw results for the reasoning engine to process.
"""

from __future__ import annotations


class Searcher:
    """Searches for information and returns results."""

    def search(self, query: str) -> str:
        """
        Search for information. Tries multiple sources and strategies.
        """
        # Try Wikipedia first
        result = self._search_wikipedia(query)
        if result and len(result.strip()) > 50:
            # Check if it's a disambiguation page
            if 'may refer to' not in result.lower():
                return result

        # Try DuckDuckGo
        result = self._search_duckduckgo(query)
        if result and len(result.strip()) > 50:
            return result

        # Try Wikipedia with '(disambiguation)' stripped and more specific queries
        for suffix in [' (programming language)', ' (animal)', ' (software)', ' (concept)', '']:
            result = self._search_wikipedia(query + suffix)
            if result and len(result.strip()) > 50 and 'may refer to' not in result.lower():
                return result

        # Try DuckDuckGo with 'what is' prefix
        result = self._search_duckduckgo(f'what is {query}')
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
