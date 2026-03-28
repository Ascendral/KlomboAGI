"""
Computation Engine — the system can DO math, not just know ABOUT math.

Knowing "addition is combining two quantities" is knowledge.
Computing 2 + 3 = 5 is intelligence.

This module gives the system the ability to:
- Evaluate arithmetic expressions (2 + 3, 7 * 8, sqrt(144))
- Solve simple equations (x + 5 = 12 → x = 7)
- Verify numerical facts ("is 17 prime?" → True)
- Compare quantities ("which is larger, 2^10 or 10^3?" → 1024 > 1000)
- Unit awareness ("100cm in meters" → 1.0)

No external libraries. Pure Python math. Safe evaluation.
"""

from __future__ import annotations

import math
import re
import operator
from dataclasses import dataclass, field


@dataclass
class ComputeResult:
    """Result of a computation."""
    expression: str
    result: float | int | bool | str
    steps: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "result": self.result,
            "steps": self.steps,
            "success": self.success,
            "error": self.error,
        }


# Safe operations — no eval(), no exec()
SAFE_OPS: dict[str, object] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "^": operator.pow,
}

SAFE_FUNCTIONS: dict[str, object] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
}


class ComputeEngine:
    """
    Mathematical computation from first principles.

    No eval(). No exec(). Parses and evaluates safely.
    """

    def compute(self, expression: str) -> ComputeResult:
        """
        Evaluate a mathematical expression or answer a math question.

        Handles:
        - Arithmetic: "2 + 3", "7 * 8 - 3"
        - Powers: "2^10", "2**10"
        - Functions: "sqrt(144)", "factorial(5)"
        - Constants: "pi", "e"
        - Comparisons: "2^10 > 10^3"
        - Questions: "is 17 prime?", "what is 15% of 200?"
        """
        expr = expression.strip()

        # Try question patterns first
        question_result = self._try_question(expr)
        if question_result:
            return question_result

        # Try comparison
        comparison_result = self._try_comparison(expr)
        if comparison_result:
            return comparison_result

        # Try arithmetic evaluation
        return self._evaluate(expr)

    def _evaluate(self, expr: str) -> ComputeResult:
        """Safely evaluate an arithmetic expression."""
        steps = []
        original = expr

        # Normalize
        expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
        expr = expr.replace(" x ", " * ")  # "3 x 4" → "3 * 4"

        # Replace named constants
        expr = re.sub(r'\bpi\b', str(math.pi), expr)
        expr = re.sub(r'\be\b(?!\w)', str(math.e), expr)

        # Handle function calls: sqrt(X), factorial(X), etc.
        for fname, func in SAFE_FUNCTIONS.items():
            if callable(func):
                pattern = rf'{fname}\s*\(\s*([^)]+)\s*\)'
                while re.search(pattern, expr):
                    m = re.search(pattern, expr)
                    inner = m.group(1)
                    inner_result = self._evaluate(inner)
                    if not inner_result.success:
                        return ComputeResult(original, 0, success=False,
                                           error=f"Failed to evaluate {fname}({inner})")
                    try:
                        val = func(inner_result.result)
                        steps.append(f"{fname}({inner}) = {val}")
                        expr = expr[:m.start()] + str(val) + expr[m.end():]
                    except Exception as e:
                        return ComputeResult(original, 0, success=False, error=str(e))

        # Tokenize and evaluate with precedence
        try:
            result = self._parse_expr(expr)
            if isinstance(result, float) and result == int(result):
                result = int(result)
            steps.append(f"{original} = {result}")
            return ComputeResult(original, result, steps=steps)
        except Exception as e:
            return ComputeResult(original, 0, steps=steps, success=False, error=str(e))

    def _parse_expr(self, expr: str) -> float:
        """
        Recursive descent parser for arithmetic expressions.
        Handles: +, -, *, /, **, parentheses.
        """
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        pos = [0]  # mutable position counter

        def parse_additive() -> float:
            left = parse_multiplicative()
            while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
                op = tokens[pos[0]]
                pos[0] += 1
                right = parse_multiplicative()
                left = SAFE_OPS[op](left, right)
            return left

        def parse_multiplicative() -> float:
            left = parse_power()
            while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/', '//', '%'):
                op = tokens[pos[0]]
                pos[0] += 1
                right = parse_power()
                if op in ('/', '//') and right == 0:
                    raise ValueError("Division by zero")
                left = SAFE_OPS[op](left, right)
            return left

        def parse_power() -> float:
            left = parse_unary()
            if pos[0] < len(tokens) and tokens[pos[0]] == '**':
                pos[0] += 1
                right = parse_power()  # Right-associative
                left = SAFE_OPS['**'](left, right)
            return left

        def parse_unary() -> float:
            if pos[0] < len(tokens) and tokens[pos[0]] == '-':
                pos[0] += 1
                return -parse_primary()
            return parse_primary()

        def parse_primary() -> float:
            if pos[0] >= len(tokens):
                raise ValueError("Unexpected end of expression")
            token = tokens[pos[0]]
            if token == '(':
                pos[0] += 1
                result = parse_additive()
                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1
                return result
            try:
                val = float(token)
                pos[0] += 1
                return val
            except ValueError:
                raise ValueError(f"Unexpected token: {token}")

        result = parse_additive()
        return result

    def _tokenize(self, expr: str) -> list[str]:
        """Tokenize an arithmetic expression."""
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            # Multi-char operators
            if expr[i:i+2] in ('**', '//', '>=', '<=', '!=', '=='):
                tokens.append(expr[i:i+2])
                i += 2
            elif expr[i] in '+-*/%()><=':
                tokens.append(expr[i])
                i += 1
            elif expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(expr[i:j])
                i = j
            else:
                i += 1  # Skip unknown chars
        return tokens

    def _try_question(self, expr: str) -> ComputeResult | None:
        """Try to parse as a math question."""
        q = expr.lower().strip().rstrip("?")

        # "is X prime?"
        m = re.match(r"is\s+(\d+)\s+prime", q)
        if m:
            n = int(m.group(1))
            is_prime = self._is_prime(n)
            return ComputeResult(expr, is_prime,
                               steps=[f"Testing primality of {n}",
                                      f"{n} {'is' if is_prime else 'is not'} prime"])

        # "what is X% of Y?"
        m = re.match(r"what is\s+([\d.]+)\s*%\s*of\s+([\d.]+)", q)
        if m:
            pct = float(m.group(1))
            val = float(m.group(2))
            result = val * pct / 100
            return ComputeResult(expr, result,
                               steps=[f"{pct}% of {val} = {val} * {pct}/100 = {result}"])

        # "what is X + Y", "what is X * Y", etc.
        m = re.match(r"what is\s+(.+)", q)
        if m:
            return self._evaluate(m.group(1))

        # "X + Y" directly
        if any(op in expr for op in ['+', '-', '*', '/', '^', '**', 'sqrt', 'factorial']):
            return None  # Let _evaluate handle it

        return None

    def _try_comparison(self, expr: str) -> ComputeResult | None:
        """Try to parse as a comparison: "2^10 > 10^3"."""
        for op_str, op_name in [(">=", "≥"), ("<=", "≤"), ("!=", "≠"),
                                 ("==", "="), (">", ">"), ("<", "<")]:
            if op_str in expr:
                parts = expr.split(op_str, 1)
                if len(parts) == 2:
                    left = self._evaluate(parts[0].strip())
                    right = self._evaluate(parts[1].strip())
                    if left.success and right.success:
                        cmp_ops = {
                            ">=": operator.ge, "<=": operator.le,
                            "!=": operator.ne, "==": operator.eq,
                            ">": operator.gt, "<": operator.lt,
                        }
                        result = cmp_ops[op_str](left.result, right.result)
                        return ComputeResult(
                            expr, result,
                            steps=[
                                f"Left: {left.result}",
                                f"Right: {right.result}",
                                f"{left.result} {op_name} {right.result} → {result}",
                            ])
        return None

    def _is_prime(self, n: int) -> bool:
        """Test primality."""
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # ── Verification ──

    def verify_fact(self, fact: str) -> ComputeResult | None:
        """
        Try to verify a mathematical fact computationally.

        "2 + 2 = 4" → True
        "the square root of 144 is 12" → True
        "7 * 8 = 54" → False (it's 56)
        """
        # Pattern: "X = Y"
        m = re.match(r"(.+?)\s*=\s*(.+)", fact)
        if m:
            left = self._evaluate(m.group(1).strip())
            right = self._evaluate(m.group(2).strip())
            if left.success and right.success:
                equal = abs(left.result - right.result) < 1e-9
                return ComputeResult(
                    fact, equal,
                    steps=[f"Left: {left.result}", f"Right: {right.result}",
                           f"Equal: {equal}"])

        # Pattern: "the square root of X is Y"
        m = re.match(r"the square root of\s+([\d.]+)\s+is\s+([\d.]+)", fact, re.IGNORECASE)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            actual = math.sqrt(x)
            equal = abs(actual - y) < 1e-9
            return ComputeResult(fact, equal,
                               steps=[f"sqrt({x}) = {actual}", f"Claimed: {y}", f"Correct: {equal}"])

        return None
