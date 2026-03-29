"""
ARC Per-Cell Rule Learner — learn output = f(cell, neighborhood) from examples.

Many ARC tasks transform each cell based on its local context:
  - "If cell is color X and has neighbor Y, change to Z"
  - "If cell has exactly 2 non-bg neighbors, keep it"
  - "Count neighbors of each color, adopt the majority"

This module learns these rules from training examples by extracting
(input_features → output_value) pairs and finding consistent patterns.
"""

from __future__ import annotations

from collections import Counter

Grid = list[list[int]]


def learn_cell_rule(train: list[dict]) -> callable | None:
    """
    Try to learn a per-cell transformation rule from training examples.

    Returns a function (grid) -> grid if a consistent rule is found, else None.
    """
    # All examples must be same-size
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    # Strategy 1: Direct cell mapping (each input color maps to one output color)
    rule = _try_direct_color_map(train)
    if rule:
        return rule

    # Strategy 2: Neighborhood-based rule (cell + 4 neighbors → output)
    rule = _try_neighborhood_4(train)
    if rule:
        return rule

    # Strategy 3: Neighborhood-based rule (cell + 8 neighbors → output)
    rule = _try_neighborhood_8(train)
    if rule:
        return rule

    # Strategy 4: Count-based rule (number of non-bg neighbors determines output)
    rule = _try_count_rule(train)
    if rule:
        return rule

    # Strategy 5: Row/column position rule
    rule = _try_position_rule(train)
    if rule:
        return rule

    return None


def _try_direct_color_map(train: list[dict]) -> callable | None:
    """Check if each input color consistently maps to one output color."""
    mapping = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ic, oc = inp[r][c], out[r][c]
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None  # Inconsistent
                else:
                    mapping[ic] = oc

    if not mapping or all(k == v for k, v in mapping.items()):
        return None  # Identity or empty

    def apply_rule(grid: Grid) -> Grid:
        return [[mapping.get(c, c) for c in row] for row in grid]

    # Verify
    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_neighborhood_4(train: list[dict]) -> callable | None:
    """Learn rule based on cell value + 4-directional neighbors."""
    # Extract features: (cell_color, n_color, s_color, e_color, w_color) → output_color
    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                n = inp[r - 1][c] if r > 0 else -1
                s = inp[r + 1][c] if r < rows - 1 else -1
                w = inp[r][c - 1] if c > 0 else -1
                e = inp[r][c + 1] if c < cols - 1 else -1
                key = (inp[r][c], n, s, w, e)
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None  # Inconsistent
                rule_table[key] = val

    if not rule_table:
        return None

    # Check it's not just identity
    if all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                n = grid[r - 1][c] if r > 0 else -1
                s = grid[r + 1][c] if r < rows - 1 else -1
                w = grid[r][c - 1] if c > 0 else -1
                e = grid[r][c + 1] if c < cols - 1 else -1
                key = (grid[r][c], n, s, w, e)
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_neighborhood_8(train: list[dict]) -> callable | None:
    """Learn rule based on cell + 8 surrounding neighbors (sorted for rotation invariance)."""
    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(inp[nr][nc])
                        else:
                            neighbors.append(-1)
                # Use sorted neighbors for partial rotation invariance
                key = (inp[r][c], tuple(sorted(neighbors)))
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(grid[nr][nc])
                        else:
                            neighbors.append(-1)
                key = (grid[r][c], tuple(sorted(neighbors)))
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_count_rule(train: list[dict]) -> callable | None:
    """Learn rule based on count of non-bg neighbors."""
    # Find bg color
    all_vals = []
    for ex in train:
        for row in ex["input"]:
            all_vals.extend(row)
    bg = Counter(all_vals).most_common(1)[0][0]

    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                n_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and inp[nr][nc] != bg:
                            n_neighbors += 1
                key = (inp[r][c], n_neighbors)
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                n_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
                            n_neighbors += 1
                key = (grid[r][c], n_neighbors)
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_position_rule(train: list[dict]) -> callable | None:
    """Learn rule based on (cell_color, row % N, col % N) for some period N."""
    for period in [2, 3, 4, 5]:
        rule_table = {}
        consistent = True
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    key = (inp[r][c], r % period, c % period)
                    val = out[r][c]
                    if key in rule_table and rule_table[key] != val:
                        consistent = False
                        break
                    rule_table[key] = val
                if not consistent:
                    break
            if not consistent:
                break

        if not consistent or not rule_table:
            continue
        if all(k[0] == v for k, v in rule_table.items()):
            continue

        p = period  # capture for closure

        def apply_rule(grid: Grid, rt=rule_table, per=p) -> Grid:
            rows, cols = len(grid), len(grid[0])
            return [[rt.get((grid[r][c], r % per, c % per), grid[r][c])
                     for c in range(cols)] for r in range(rows)]

        valid = all(apply_rule(ex["input"]) == ex["output"] for ex in train)
        if valid:
            return apply_rule

    return None
