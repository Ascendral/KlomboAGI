"""
ARC LLM-Guided Solver — use GPT-5.4 to PROPOSE transformation rules.

The LLM sees the training examples and proposes a Python function
that transforms input → output. The function is then verified against
all training examples before being applied to the test.

This is the "library card" approach: the LLM proposes, the system verifies.
The LLM never sees the test output — it only helps discover the rule.

STRICT BOUNDARY:
  - LLM proposes a transformation function (Python code)
  - System executes and verifies against training examples
  - Only verified functions are applied to test input
  - LLM never touches the test output directly
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

Grid = list[list[int]]

PROPOSE_PROMPT = """You are an ARC puzzle solver. Given input/output grid pairs, write a Python function that transforms any input grid to the correct output grid.

Rules:
- Write ONLY the function body, starting with `def transform(grid):`
- grid is a list[list[int]] (2D array of integers 0-9)
- Return a list[list[int]]
- Use ONLY standard Python (no imports except copy, collections.Counter)
- The function must be GENERAL — work for any input following the pattern
- Keep it simple and short

Training examples:
"""


def solve_with_llm(train: list[dict], test_input: Grid,
                   api_key: str = "", model: str = "gpt-5.4",
                   max_attempts: int = 2) -> Grid | None:
    """
    Use LLM to propose a transformation function, verify, apply.

    Returns test output if a valid function is found, else None.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    for attempt in range(max_attempts):
        # Build the prompt with training examples
        prompt = PROPOSE_PROMPT
        for i, ex in enumerate(train):
            prompt += f"\nExample {i + 1}:\n"
            prompt += f"Input:  {json.dumps(ex['input'])}\n"
            prompt += f"Output: {json.dumps(ex['output'])}\n"

        prompt += "\nWrite the transform function:"

        # Call LLM
        code = _call_llm(prompt, api_key, model)
        if not code:
            continue

        # Extract and execute the function
        fn = _extract_function(code)
        if fn is None:
            continue

        # Verify against ALL training examples
        if _verify(fn, train):
            # Apply to test input
            try:
                result = fn(test_input)
                if isinstance(result, list) and result and isinstance(result[0], list):
                    return result
            except Exception:
                continue

    return None


def _call_llm(prompt: str, api_key: str, model: str) -> str | None:
    """Call OpenAI API."""
    try:
        body = json.dumps({
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            response = json.loads(resp.read().decode("utf-8"))

        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None


def _extract_function(code: str) -> callable | None:
    """Extract and compile the transform function from LLM output."""
    # Find the function definition
    # Handle code blocks
    code = code.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # Find def transform
    match = re.search(r'(def transform\(.*?\):.*?)(?=\ndef |\Z)', code, re.DOTALL)
    if not match:
        # Try to wrap bare code in a function
        if "return" in code:
            code = f"def transform(grid):\n" + "\n".join(f"    {line}" for line in code.strip().split("\n"))
            match = re.search(r'(def transform\(.*?\):.*)', code, re.DOTALL)

    if not match:
        return None

    func_code = match.group(1).strip()

    # Safety: only allow safe builtins
    allowed_imports = {"copy", "collections"}
    for line in func_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            module = stripped.split()[1].split(".")[0]
            if module not in allowed_imports:
                return None

    try:
        namespace = {"__builtins__": {
            "range": range, "len": len, "min": min, "max": max,
            "sum": sum, "abs": abs, "zip": zip, "enumerate": enumerate,
            "list": list, "tuple": tuple, "set": set, "dict": dict,
            "int": int, "float": float, "bool": bool, "str": str,
            "sorted": sorted, "reversed": reversed, "any": any, "all": all,
            "map": map, "filter": filter, "isinstance": isinstance,
            "True": True, "False": False, "None": None,
            "print": lambda *a, **k: None,  # silence prints
        }}

        # Allow Counter
        from collections import Counter
        namespace["Counter"] = Counter
        import copy as copy_mod
        namespace["copy"] = copy_mod

        exec(func_code, namespace)
        return namespace.get("transform")
    except Exception:
        return None


def _verify(fn: callable, train: list[dict]) -> bool:
    """Verify the function works for ALL training examples."""
    for ex in train:
        try:
            result = fn(ex["input"])
            if result != ex["output"]:
                return False
        except Exception:
            return False
    return True
