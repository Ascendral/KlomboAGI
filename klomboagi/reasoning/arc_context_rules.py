"""
ARC Context-Based Rules.

Output color depends on compound contextual features:
  - Cell color + position (row mod N, col mod N)
  - Nearest non-bg color (Voronoi)
  - Diagonal patterns
  - Border vs interior of objects
  - Distance from edge
  - Majority neighbor color
  - Fill between same-color flanking cells
"""

from __future__ import annotations
from collections import Counter, deque

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_context_rule(train: list[dict]) -> callable | None:
    """Try context-based rules."""
    for fn in [
        _try_bg_to_nearest_color,
        _try_border_vs_interior,
        _try_fill_between,
        _try_majority_neighbor,
        _try_distance_from_edge,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_bg_to_nearest_color(train):
    """Each bg cell takes color of nearest non-bg cell (Voronoi fill)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Output should have NO bg cells
    for ex in train:
        if any(v == bg for row in ex["output"] for v in row):
            return None

    def nearest_color(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        queue = deque()
        dist = [[float('inf')] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    queue.append((r, c, 0))
                    dist[r][c] = 0

        while queue:
            r, c, d = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and d + 1 < dist[nr][nc]:
                    dist[nr][nc] = d + 1
                    result[nr][nc] = result[r][c]
                    queue.append((nr, nc, d + 1))

        return result

    fn = lambda grid, bg_val=bg: nearest_color(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_border_vs_interior(train):
    """Border cells of objects get one color, interior cells another."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg:
                    continue
                is_border = any(
                    nr < 0 or nr >= rows or nc < 0 or nc >= cols or inp[nr][nc] == bg
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for nr, nc in [(r + dr, c + dc)]
                )
                key = ('border' if is_border else 'interior', inp[r][c])
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[1] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid, bg_val=bg, rt=rule_table):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg_val:
                    continue
                is_border = any(
                    nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc] == bg_val
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for nr, nc in [(r + dr, c + dc)]
                )
                key = ('border' if is_border else 'interior', grid[r][c])
                result[r][c] = rt.get(key, grid[r][c])
        return result

    if all(apply_rule(ex["input"]) == ex["output"] for ex in train):
        return apply_rule

    return None


def _try_fill_between(train):
    """Fill bg gaps between same-color cells (H and V)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def fill_between(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Fill horizontal
        for r in range(rows):
            for c in range(1, cols - 1):
                if grid[r][c] == bg_val:
                    left = right = None
                    for lc in range(c - 1, -1, -1):
                        if grid[r][lc] != bg_val:
                            left = grid[r][lc]
                            break
                    for rc in range(c + 1, cols):
                        if grid[r][rc] != bg_val:
                            right = grid[r][rc]
                            break
                    if left is not None and left == right:
                        result[r][c] = left

        # Fill vertical
        for c in range(cols):
            for r in range(1, rows - 1):
                if grid[r][c] == bg_val:
                    top = bottom = None
                    for tr in range(r - 1, -1, -1):
                        if grid[tr][c] != bg_val:
                            top = grid[tr][c]
                            break
                    for br in range(r + 1, rows):
                        if grid[br][c] != bg_val:
                            bottom = grid[br][c]
                            break
                    if top is not None and top == bottom:
                        result[r][c] = top

        return result

    fn = lambda grid, bg_val=bg: fill_between(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    # Try iterative
    for n in range(2, 6):
        def fill_iter(grid, bg_val=bg, iters=n):
            r = grid
            for _ in range(iters):
                r = fill_between(r, bg_val)
            return r

        if all(fill_iter(ex["input"]) == ex["output"] for ex in train):
            return fill_iter

    return None


def _try_majority_neighbor(train):
    """Each cell becomes the majority color of its 8 neighbors."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    def majority(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
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
                if neighbors:
                    result[r][c] = Counter(neighbors).most_common(1)[0][0]
        return result

    if all(majority(ex["input"]) == ex["output"] for ex in train):
        return majority

    return None


def _try_distance_from_edge(train):
    """Color depends on distance from grid edge."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                dist = min(r, rows - 1 - r, c, cols - 1 - c)
                key = (inp[r][c], dist)
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid, rt=rule_table):
        rows, cols = len(grid), len(grid[0])
        return [[rt.get((grid[r][c], min(r, rows-1-r, c, cols-1-c)), grid[r][c])
                 for c in range(cols)] for r in range(rows)]

    if all(apply_rule(ex["input"]) == ex["output"] for ex in train):
        return apply_rule

    return None
