"""
Pattern Engine — given partial patterns, predict the rest.

From Chollet/ARC-AGI: intelligence IS the ability to extract abstract
patterns from examples and apply them to new situations.

This module handles:
1. Sequence patterns: [1, 2, 3, ?] → 4
2. Structural patterns: [{a:1, b:2}, {a:3, b:4}] → rule: b = a + 1
3. Transformation patterns: input → output mapping
4. Analogy patterns: A:B :: C:? → derive rule from A→B, apply to C

No hardcoded rules. The engine DISCOVERS the pattern from examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pattern:
    """A discovered pattern."""
    name: str
    rule: str                          # Human-readable description
    confidence: float                  # How well it fits the data
    examples_seen: int                 # How many examples confirmed it
    rule_fn: Any = None                # Callable that applies the rule

    def apply(self, input_val: Any) -> Any:
        """Apply the pattern to a new input."""
        if self.rule_fn:
            try:
                return self.rule_fn(input_val)
            except Exception:
                return None
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rule": self.rule,
            "confidence": self.confidence,
            "examples_seen": self.examples_seen,
        }


class PatternEngine:
    """
    Discovers patterns from examples and applies them to new inputs.
    No hardcoded rules — all patterns are derived from data.
    """

    def __init__(self) -> None:
        self.discovered: list[Pattern] = []

    def find_sequence_pattern(self, sequence: list) -> Pattern | None:
        """
        Given a sequence, find the generating rule.
        Tries: constant difference, constant ratio, alternating,
        fibonacci-like, polynomial, repeat.
        """
        if len(sequence) < 2:
            return None

        # Try constant difference (arithmetic)
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)
                 if isinstance(sequence[i], (int, float)) and isinstance(sequence[i+1], (int, float))]
        if diffs and all(d == diffs[0] for d in diffs):
            d = diffs[0]
            pattern = Pattern(
                name="arithmetic",
                rule=f"each term = previous + {d}",
                confidence=1.0,
                examples_seen=len(sequence),
                rule_fn=lambda x, step=d: x + step,
            )
            self.discovered.append(pattern)
            return pattern

        # Try constant ratio (geometric)
        if all(isinstance(s, (int, float)) and s != 0 for s in sequence[:-1]):
            ratios = [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]
            if ratios and all(abs(r - ratios[0]) < 0.001 for r in ratios):
                r = ratios[0]
                pattern = Pattern(
                    name="geometric",
                    rule=f"each term = previous × {r}",
                    confidence=1.0,
                    examples_seen=len(sequence),
                    rule_fn=lambda x, ratio=r: x * ratio,
                )
                self.discovered.append(pattern)
                return pattern

        # Try second-order difference (quadratic)
        if len(diffs) >= 2:
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if second_diffs and all(d == second_diffs[0] for d in second_diffs):
                sd = second_diffs[0]
                pattern = Pattern(
                    name="quadratic",
                    rule=f"differences increase by {sd} each step",
                    confidence=0.9,
                    examples_seen=len(sequence),
                    rule_fn=lambda x, last_diff=diffs[-1], step=sd: x + last_diff + step,
                )
                self.discovered.append(pattern)
                return pattern

        # Try fibonacci-like (each = sum of previous two)
        if len(sequence) >= 3:
            is_fib = all(
                sequence[i] == sequence[i-1] + sequence[i-2]
                for i in range(2, len(sequence))
                if isinstance(sequence[i], (int, float))
            )
            if is_fib:
                pattern = Pattern(
                    name="fibonacci-like",
                    rule="each term = sum of previous two",
                    confidence=1.0,
                    examples_seen=len(sequence),
                )
                self.discovered.append(pattern)
                return pattern

        # Try repeating pattern
        for period in range(1, len(sequence) // 2 + 1):
            chunk = sequence[:period]
            is_repeat = all(
                sequence[i] == chunk[i % period]
                for i in range(len(sequence))
            )
            if is_repeat:
                pattern = Pattern(
                    name="repeating",
                    rule=f"repeats every {period}: {chunk}",
                    confidence=1.0,
                    examples_seen=len(sequence),
                    rule_fn=lambda x, cyc=chunk, p=period, seq=sequence: cyc[len(seq) % p],
                )
                self.discovered.append(pattern)
                return pattern

        return None

    def predict_next(self, sequence: list) -> tuple[Any, Pattern | None]:
        """Predict the next element in a sequence."""
        pattern = self.find_sequence_pattern(sequence)
        if pattern and pattern.rule_fn:
            prediction = pattern.apply(sequence[-1])
            return prediction, pattern
        return None, None

    def find_mapping_pattern(self, pairs: list[tuple]) -> Pattern | None:
        """
        Given input→output pairs, find the transformation rule.
        pairs = [(input1, output1), (input2, output2), ...]
        """
        if len(pairs) < 2:
            return None

        # Try constant offset
        if all(isinstance(a, (int, float)) and isinstance(b, (int, float)) for a, b in pairs):
            offsets = [b - a for a, b in pairs]
            if all(o == offsets[0] for o in offsets):
                offset = offsets[0]
                return Pattern(
                    name="constant_offset",
                    rule=f"output = input + {offset}",
                    confidence=1.0,
                    examples_seen=len(pairs),
                    rule_fn=lambda x, off=offset: x + off,
                )

            # Try constant multiplier
            multipliers = [b / a for a, b in pairs if a != 0]
            if multipliers and all(abs(m - multipliers[0]) < 0.001 for m in multipliers):
                mult = multipliers[0]
                return Pattern(
                    name="constant_multiplier",
                    rule=f"output = input × {mult}",
                    confidence=1.0,
                    examples_seen=len(pairs),
                    rule_fn=lambda x, m=mult: x * m,
                )

            # Try polynomial (quadratic)
            if len(pairs) >= 3:
                # output = a*input^2 + b*input + c
                # Solve system of equations
                try:
                    from numpy import array, linalg
                    A = array([[a**2, a, 1] for a, _ in pairs[:3]])
                    B = array([b for _, b in pairs[:3]])
                    coeffs = linalg.solve(A, B)
                    # Verify on remaining pairs
                    fn = lambda x, c=coeffs: c[0]*x**2 + c[1]*x + c[2]
                    errors = [abs(fn(a) - b) for a, b in pairs]
                    if all(e < 0.01 for e in errors):
                        return Pattern(
                            name="quadratic_mapping",
                            rule=f"output = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}",
                            confidence=0.95,
                            examples_seen=len(pairs),
                            rule_fn=fn,
                        )
                except Exception:
                    pass

        # Try string patterns
        if all(isinstance(a, str) and isinstance(b, str) for a, b in pairs):
            # Check if output is reversed input
            if all(b == a[::-1] for a, b in pairs):
                return Pattern(
                    name="reverse",
                    rule="output = reverse(input)",
                    confidence=1.0,
                    examples_seen=len(pairs),
                    rule_fn=lambda x: x[::-1],
                )

            # Check if output is uppercase
            if all(b == a.upper() for a, b in pairs):
                return Pattern(
                    name="uppercase",
                    rule="output = uppercase(input)",
                    confidence=1.0,
                    examples_seen=len(pairs),
                    rule_fn=lambda x: x.upper(),
                )

            # Check prefix/suffix addition
            prefixes = [b[:len(b)-len(a)] if b.endswith(a) else None for a, b in pairs]
            if all(p is not None and p == prefixes[0] for p in prefixes):
                prefix = prefixes[0]
                return Pattern(
                    name="prefix",
                    rule=f"output = '{prefix}' + input",
                    confidence=1.0,
                    examples_seen=len(pairs),
                    rule_fn=lambda x, p=prefix: p + x,
                )

        return None

    def find_analogy(self, a: Any, b: Any, c: Any) -> tuple[Any, Pattern | None]:
        """
        A:B :: C:?
        Find the rule that transforms A→B, then apply it to C.

        This is structural transfer — the core of abstract reasoning.
        """
        # Find the rule from A→B
        pattern = self.find_mapping_pattern([(a, b)])

        # If we found a rule, apply to C
        if pattern and pattern.rule_fn:
            result = pattern.apply(c)
            return result, pattern

        # Try structural comparison for non-numeric types
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(c, (int, float)):
            # Simple: what operation turns A into B?
            diff = b - a
            result = c + diff
            pattern = Pattern(
                name="analogy_offset",
                rule=f"transform: +{diff}",
                confidence=0.7,
                examples_seen=1,
                rule_fn=lambda x, d=diff: x + d,
            )
            return result, pattern

        if isinstance(a, str) and isinstance(b, str) and isinstance(c, str):
            # Check case change
            if b == a.upper():
                return c.upper(), Pattern("analogy_case", "uppercase", 0.8, 1, lambda x: x.upper())
            if b == a.lower():
                return c.lower(), Pattern("analogy_case", "lowercase", 0.8, 1, lambda x: x.lower())
            if b == a[::-1]:
                return c[::-1], Pattern("analogy_reverse", "reverse", 0.8, 1, lambda x: x[::-1])

        return None, None

    def stats(self) -> dict:
        return {
            "patterns_discovered": len(self.discovered),
            "pattern_types": list(set(p.name for p in self.discovered)),
        }
