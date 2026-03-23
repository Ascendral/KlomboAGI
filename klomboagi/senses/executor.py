"""
Executor Sense — the ability to write and run code to test understanding.

When the system wants to verify it understands something, it writes
code, runs it, and observes the result. Like a student doing homework
to check they understood the lecture.
"""

from __future__ import annotations

import subprocess
import tempfile
import os


class Executor:
    """Writes and runs code to test understanding."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def execute(self, code_or_query: str) -> str:
        """
        Execute code to test understanding.
        If given a query instead of code, generates a simple test.
        Returns the output or error.
        """
        # If it looks like actual code, run it
        if self._looks_like_code(code_or_query):
            return self.run_python(code_or_query)

        # Otherwise treat as a concept to test
        return self.test_concept(code_or_query)

    def run_python(self, code: str) -> str:
        """Run Python code in a sandboxed subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            path = f.name

        try:
            result = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                return f"[Executed successfully]\n{output}" if output else "[Executed successfully, no output]"
            else:
                return f"[Error (exit code {result.returncode})]\n{error}"

        except subprocess.TimeoutExpired:
            return f"[Timeout after {self.timeout}s]"
        except Exception as e:
            return f"[Execution error: {e}]"
        finally:
            os.unlink(path)

    def test_concept(self, concept: str) -> str:
        """
        Generate and run a simple test for a concept.
        Used when the system wants to verify understanding through code.
        """
        # Simple test: try to use the concept in Python
        test_code = f'''
# Testing understanding of: {concept}
try:
    # Try importing if it's a module
    import importlib
    mod = importlib.import_module("{concept.lower().replace(" ", "_")}")
    print(f"Module found: {{mod.__name__}}")
    if hasattr(mod, "__doc__") and mod.__doc__:
        print(f"Description: {{mod.__doc__[:200]}}")
except ImportError:
    print(f"'{concept}' is not a Python module")
except Exception as e:
    print(f"Error: {{e}}")
'''
        return self.run_python(test_code)

    def _looks_like_code(self, text: str) -> bool:
        """Check if the text looks like code."""
        code_indicators = [
            "import ", "def ", "class ", "print(", "for ", "while ",
            "if ", "return ", "from ", "#", "=", "()", "[]", "{}",
        ]
        return any(ind in text for ind in code_indicators)
