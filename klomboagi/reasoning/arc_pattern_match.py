"""
ARC Pattern Matching Solver — learn input→output by finding consistent local patterns.

For same-size tasks where the output is determined by local patterns in the input.
Instead of hand-coding rules, learn the mapping from (local_pattern → output_cell).

Approach:
  1. For each cell that changes between input and output
  2. Extract the NxN neighborhood around it
  3. Build a table: pattern → output value
  4. If the table is consistent across ALL training examples, we have a rule
"""

from __future__ import annotations

from collections import Counter

Grid = list[list[int]]


def learn_pattern_rule(train: list[dict]) -> callable | None:
    """Try to learn a local pattern matching rule."""

    # Only same-size tasks
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    # Try different neighborhood sizes
    for radius in [1, 2, 3]:
        rule = _try_pattern_radius(train, radius)
        if rule:
            return rule

    # Try row-based pattern (whole row determines output row)
    rule = _try_row_pattern(train)
    if rule:
        return rule

    # Try column-based pattern
    rule = _try_col_pattern(train)
    if rule:
        return rule

    return None


def _try_pattern_radius(train, radius):
    """Learn a rule based on the (2r+1)×(2r+1) neighborhood of each cell."""
    rule_table = {}

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])

        for r in range(rows):
            for c in range(cols):
                # Extract neighborhood
                neighborhood = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighborhood.append(inp[nr][nc])
                        else:
                            neighborhood.append(-1)  # border padding

                key = tuple(neighborhood)
                val = out[r][c]

                if key in rule_table:
                    if rule_table[key] != val:
                        return None  # Inconsistent
                else:
                    rule_table[key] = val

    if not rule_table:
        return None

    # Check it's not identity
    size = (2 * radius + 1) ** 2
    center_idx = size // 2
    identity = all(k[center_idx] == v for k, v in rule_table.items())
    if identity:
        return None

    def apply_rule(grid, rt=rule_table, rad=radius):
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                neighborhood = []
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighborhood.append(grid[nr][nc])
                        else:
                            neighborhood.append(-1)
                key = tuple(neighborhood)
                result[r][c] = rt.get(key, grid[r][c])
        return result

    # Final verification
    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_row_pattern(train):
    """Each output row is determined by the content of the input row."""
    rule_table = {}

    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            key = tuple(inp[r])
            val = tuple(out[r])
            if key in rule_table and rule_table[key] != val:
                return None
            rule_table[key] = val

    if not rule_table or all(k == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid, rt=rule_table):
        return [list(rt.get(tuple(row), row)) for row in grid]

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_col_pattern(train):
    """Each output column is determined by the content of the input column."""
    rule_table = {}

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for c in range(cols):
            in_col = tuple(inp[r][c] for r in range(rows))
            out_col = tuple(out[r][c] for r in range(rows))
            if in_col in rule_table and rule_table[in_col] != out_col:
                return None
            rule_table[in_col] = out_col

    if not rule_table or all(k == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid, rt=rule_table):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for c in range(cols):
            in_col = tuple(grid[r][c] for r in range(rows))
            out_col = rt.get(in_col, in_col)
            for r in range(rows):
                result[r][c] = out_col[r]
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule
