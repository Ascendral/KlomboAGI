"""
Reader Sense — the ability to read and consume information.

Reads files, URLs, and raw text. Extracts facts from what it reads.
Feeds those facts into the reasoning engine.
"""

from __future__ import annotations

import os
from pathlib import Path


class Reader:
    """Reads content from various sources and returns raw text."""

    def read(self, source: str) -> str:
        """
        Read from a source. Auto-detects type:
        - File path → read file
        - URL → fetch URL
        - Raw text → return as-is
        """
        if os.path.exists(source):
            return self.read_file(source)
        if source.startswith(("http://", "https://", "www.")):
            return self.read_url(source)
        return source  # Treat as raw text

    def read_file(self, path: str) -> str:
        """Read a local file."""
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        if p.stat().st_size > 5 * 1024 * 1024:  # 5MB limit
            return f"File too large: {path} ({p.stat().st_size} bytes)"

        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {path}: {e}"

    def read_wikipedia(self, topic: str) -> str:
        """Read a full Wikipedia article's plain text content."""
        try:
            import urllib.request
            import urllib.parse
            import json

            safe = urllib.parse.quote(topic)
            url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&titles={safe}&prop=extracts"
                f"&explaintext=1&format=json&exlimit=1"
            )
            req = urllib.request.Request(url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return ""
                text = page.get("extract", "")
                if text:
                    # Take first 10000 chars to avoid overwhelming
                    return text[:10000]
            return ""
        except Exception as e:
            return f"Error reading Wikipedia: {e}"

    def read_url(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            import urllib.request
            if not url.startswith("http"):
                url = "https://" + url
            req = urllib.request.Request(url, headers={
                "User-Agent": "KlomboAGI/0.1 (learning system)"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            # Strip HTML tags for cleaner text
            return self._strip_html(content)
        except Exception as e:
            return f"Error fetching {url}: {e}"

    def _strip_html(self, html: str) -> str:
        """Basic HTML tag stripping."""
        import re
        # Remove script and style blocks
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Decode entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
        return text
