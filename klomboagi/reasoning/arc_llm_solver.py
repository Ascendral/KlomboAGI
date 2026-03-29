"""
ARC LLM-Guided Solver — use GPT to PROPOSE and REFINE transformation rules.

The LLM sees the training examples and proposes a Python function
that transforms input → output. The function is then verified against
all training examples. On failure, the LLM sees EXACTLY which cells
differ (input, expected, actual) and revises the function.

Refinement loop (up to max_attempts):
  1. GPT proposes transform(grid) function
  2. System executes on ALL training examples
  3. On failure: format a diff showing the failing cells
  4. Send diff back to GPT → revised function
  5. Repeat until verified or max_attempts reached
  6. Only apply verified function to test input

STRICT BOUNDARY:
  - LLM proposes Python code
  - System executes and verifies
  - LLM never touches the test output directly
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Callable

Grid = list[list[int]]

PROPOSE_PROMPT = """You are an ARC (Abstraction and Reasoning Corpus) puzzle solver.
Given input/output grid pairs, write a Python function that transforms any input grid to the correct output grid.

Rules:
- Write ONLY the function, starting with `def transform(grid):`
- `grid` is a list[list[int]] (2D array of integers 0-9)
- Return a list[list[int]]
- Use ONLY standard Python (no imports except: copy, collections)
- The function must be GENERAL — it must work for ANY input following the same pattern
- Do NOT hardcode specific grids — derive the rule

Training examples:
{examples}

Study the transformation pattern carefully, then write the transform function:
"""

REFINE_PROMPT = """You are an ARC puzzle solver. Your previous transform function was INCORRECT.

Here is a detailed diff showing where it failed:

{diff}

Your previous function:
```python
{prev_code}
```

Fix the function so it correctly handles ALL training examples.
Write ONLY the corrected `def transform(grid):` function.
Think carefully about the pattern before writing code.
"""


def solve_with_llm(train: list[dict], test_input: Grid,
                   api_key: str = "", model: str = "gpt-4o",
                   max_attempts: int = 5) -> Grid | None:
    """
    Use LLM to propose+refine a transformation function, then apply.

    Returns test output if a valid function is found, else None.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    # Use best available model
    if not model:
        model = "gpt-4o"

    # Format training examples once
    examples_text = _format_examples(train)
    initial_prompt = PROPOSE_PROMPT.format(examples=examples_text)

    prev_code = None
    messages = [{"role": "user", "content": initial_prompt}]

    for attempt in range(max_attempts):
        # Get LLM response
        code = _call_llm(messages, api_key, model)
        if not code:
            break

        # Add assistant response to history
        messages.append({"role": "assistant", "content": code})

        # Extract and compile the function
        fn = _extract_function(code)
        if fn is None:
            # Bad code — ask for syntactically valid function
            messages.append({"role": "user", "content":
                "Your response could not be parsed as a Python function. "
                "Please write ONLY the `def transform(grid):` function with no extra text."})
            continue

        prev_code = _extract_func_code(code)

        # Verify against ALL training examples
        passed, diff_text = _verify_with_diff(fn, train)

        if passed:
            # Apply to test input
            try:
                result = fn(test_input)
                if isinstance(result, list) and result and isinstance(result[0], list):
                    return result
            except Exception as e:
                # Function crashes on test input
                messages.append({"role": "user", "content":
                    f"Your function crashed on the test input with error: {e}\n"
                    "Please fix it to handle edge cases."})
                continue

        else:
            # Build refinement prompt with diff
            refine_msg = REFINE_PROMPT.format(
                diff=diff_text,
                prev_code=prev_code or code
            )
            messages.append({"role": "user", "content": refine_msg})

    return None


def _format_examples(train: list[dict]) -> str:
    """Format training examples as compact text."""
    lines = []
    for i, ex in enumerate(train):
        lines.append(f"Example {i + 1}:")
        lines.append(f"  Input:  {json.dumps(ex['input'])}")
        lines.append(f"  Output: {json.dumps(ex['output'])}")
        # Add visual grid for small grids
        if len(ex["input"]) <= 10 and len(ex["input"][0]) <= 10:
            lines.append("  Input grid:")
            for row in ex["input"]:
                lines.append("    " + " ".join(str(v) for v in row))
            lines.append("  Output grid:")
            for row in ex["output"]:
                lines.append("    " + " ".join(str(v) for v in row))
    return "\n".join(lines)


def _format_diff(fn: Callable, train: list[dict]) -> str:
    """Format a detailed diff of where the function fails."""
    lines = []
    total_wrong = 0

    for i, ex in enumerate(train):
        try:
            result = fn(ex["input"])
        except Exception as e:
            lines.append(f"Example {i + 1}: CRASHED with error: {e}")
            lines.append(f"  Input: {json.dumps(ex['input'])}")
            total_wrong += 1
            continue

        expected = ex["output"]

        if result == expected:
            lines.append(f"Example {i + 1}: ✓ CORRECT")
            continue

        lines.append(f"Example {i + 1}: ✗ WRONG")

        # Check size mismatch first
        if len(result) != len(expected) or (result and expected and len(result[0]) != len(expected[0])):
            lines.append(f"  Size mismatch: got {len(result)}x{len(result[0]) if result else 0}, "
                         f"expected {len(expected)}x{len(expected[0]) if expected else 0}")
            lines.append(f"  Got:      {json.dumps(result)}")
            lines.append(f"  Expected: {json.dumps(expected)}")
            total_wrong += 1
            continue

        # Cell-by-cell diff (show up to 20 differences)
        diffs = []
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if r < len(result) and c < len(result[r]):
                    if result[r][c] != expected[r][c]:
                        diffs.append((r, c, result[r][c], expected[r][c]))

        total_wrong += len(diffs)
        lines.append(f"  {len(diffs)} cells wrong (row, col, got, expected):")

        shown = diffs[:20]
        for r, c, got, exp in shown:
            lines.append(f"    [{r},{c}]: got {got}, expected {exp}")
        if len(diffs) > 20:
            lines.append(f"    ... and {len(diffs) - 20} more")

        # Show input grid for context (small grids only)
        if len(ex["input"]) <= 10 and len(ex["input"][0]) <= 10:
            lines.append("  Input grid:")
            for row in ex["input"]:
                lines.append("    " + " ".join(str(v) for v in row))
            lines.append("  Expected output:")
            for row in expected:
                lines.append("    " + " ".join(str(v) for v in row))
            lines.append("  Your output:")
            for row in result:
                lines.append("    " + " ".join(str(v) for v in row))

    if total_wrong == 0:
        return "All examples passed."

    return f"Total wrong cells/crashes: {total_wrong}\n\n" + "\n".join(lines)


def _call_llm(messages: list[dict], api_key: str, model: str) -> str | None:
    """Call OpenAI chat completions API with message history."""
    for retry in range(3):
        try:
            body = json.dumps({
                "model": model,
                "max_tokens": 2048,
                "temperature": 0.0,
                "messages": messages,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                response = json.loads(resp.read().decode("utf-8"))

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429 and retry < 2:
                # Rate limited — wait and retry
                time.sleep(2 * (retry + 1))
                continue
            return None
        except Exception:
            return None
    return None


def _extract_func_code(code: str) -> str:
    """Extract just the function code as a string (for display in prompts)."""
    code = code.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    match = re.search(r'(def transform\b.*)', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def _extract_function(code: str) -> Callable | None:
    """Extract and compile the transform function from LLM output."""
    code = code.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # Find def transform
    match = re.search(r'(def transform\b.*?)(?=\ndef (?!transform)|\Z)', code, re.DOTALL)
    if not match:
        # Try to find it more broadly
        match = re.search(r'(def transform\b.*)', code, re.DOTALL)

    if not match:
        # Try to wrap bare code in a function
        if "return" in code:
            wrapped = "def transform(grid):\n" + "\n".join(f"    {line}" for line in code.strip().split("\n"))
            match = re.search(r'(def transform\b.*)', wrapped, re.DOTALL)

    if not match:
        return None

    func_code = match.group(1).strip()

    # Safety: only allow safe imports
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
            "frozenset": frozenset,
            "int": int, "float": float, "bool": bool, "str": str,
            "sorted": sorted, "reversed": reversed, "any": any, "all": all,
            "map": map, "filter": filter, "isinstance": isinstance,
            "True": True, "False": False, "None": None,
            "print": lambda *a, **k: None,  # silence prints
            "round": round, "divmod": divmod, "pow": pow,
            "iter": iter, "next": next, "type": type, "id": id,
            "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
            "ValueError": ValueError, "TypeError": TypeError,
            "IndexError": IndexError, "KeyError": KeyError,
            "StopIteration": StopIteration, "Exception": Exception,
            "slice": slice, "object": object, "super": super,
            "staticmethod": staticmethod, "classmethod": classmethod,
            "property": property, "chr": chr, "ord": ord,
            "__import__": lambda name, *a, **k: __import__(name) if name in ("copy", "collections") else None,
        }}

        # Allow Counter and copy
        from collections import Counter, defaultdict, deque
        import copy as copy_mod
        import collections as collections_mod
        namespace["Counter"] = Counter
        namespace["defaultdict"] = defaultdict
        namespace["deque"] = deque
        namespace["copy"] = copy_mod
        namespace["collections"] = collections_mod

        exec(func_code, namespace)
        return namespace.get("transform")
    except Exception:
        return None


def _verify(fn: Callable, train: list[dict]) -> bool:
    """Verify the function works for ALL training examples."""
    for ex in train:
        try:
            result = fn(ex["input"])
            if result != ex["output"]:
                return False
        except Exception:
            return False
    return True


def _verify_with_diff(fn: Callable, train: list[dict]) -> tuple[bool, str]:
    """Verify and return (passed, diff_text)."""
    passed = _verify(fn, train)
    if passed:
        return True, ""
    diff = _format_diff(fn, train)
    return False, diff
