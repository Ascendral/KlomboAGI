"""
ARC Grid Operations — solve tasks involving divided grids.

Many ARC tasks have grids divided by separator lines into sub-regions.
The output is typically a function of those sub-regions:
  - XOR/AND/OR of two halves
  - The sub-region that's "different"
  - Overlay/combine sub-regions
  - Count objects per sub-region
"""

from __future__ import annotations

from collections import Counter

Grid = list[list[int]]


def _get_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def _find_h_dividers(grid):
    """Find horizontal divider lines (rows of single non-bg color)."""
    bg = _get_bg(grid)
    rows = len(grid)
    dividers = []
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            dividers.append(r)
    return dividers


def _find_v_dividers(grid):
    """Find vertical divider lines (columns of single non-bg color)."""
    bg = _get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    dividers = []
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1 and grid[0][c] != bg:
            dividers.append(c)
    return dividers


def _split_h(grid, dividers):
    """Split grid horizontally at divider lines."""
    parts = []
    prev = 0
    for d in dividers:
        if d > prev:
            parts.append(grid[prev:d])
        prev = d + 1
    if prev < len(grid):
        parts.append(grid[prev:])
    return [p for p in parts if p]


def _split_v(grid, dividers):
    """Split grid vertically at divider columns."""
    parts = []
    prev = 0
    for d in dividers:
        if d > prev:
            parts.append([row[prev:d] for row in grid])
        prev = d + 1
    if prev < len(grid[0]):
        parts.append([row[prev:] for row in grid])
    return [p for p in parts if p]


def learn_grid_rule(train: list[dict]) -> callable | None:
    """Try to learn a rule based on grid structure (dividers)."""

    # Try horizontal split operations
    rule = _try_split_combine(train, "h")
    if rule:
        return rule

    # Try vertical split operations
    rule = _try_split_combine(train, "v")
    if rule:
        return rule

    # Try quadrant operations (split both ways)
    rule = _try_quadrant(train)
    if rule:
        return rule

    return None


def _try_split_combine(train, direction):
    """Split by dividers, try various combinations of the parts."""
    bg = _get_bg(train[0]["input"])

    # Check all examples have dividers
    for ex in train:
        if direction == "h":
            divs = _find_h_dividers(ex["input"])
        else:
            divs = _find_v_dividers(ex["input"])
        if not divs:
            return None

    # Try each combining operation
    operations = [
        ("xor", _xor_parts),
        ("and", _and_parts),
        ("or", _or_parts),
        ("diff", _diff_part),
        ("first", lambda parts, bg: parts[0] if parts else None),
        ("last", lambda parts, bg: parts[-1] if parts else None),
    ]

    for op_name, op_fn in operations:
        def apply_fn(grid, dir_=direction, fn=op_fn, bg_=bg):
            if dir_ == "h":
                divs = _find_h_dividers(grid)
                parts = _split_h(grid, divs) if divs else [grid[:len(grid)//2], grid[len(grid)//2:]]
            else:
                divs = _find_v_dividers(grid)
                parts = _split_v(grid, divs) if divs else [[r[:len(r)//2] for r in grid], [r[len(r)//2:] for r in grid]]
            if len(parts) < 2:
                return None
            return fn(parts, bg_)

        valid = True
        for ex in train:
            result = apply_fn(ex["input"])
            if result is None or result != ex["output"]:
                valid = False
                break
        if valid:
            return apply_fn

    return None


def _xor_parts(parts, bg):
    """Non-bg cells that differ between parts."""
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    if len(a) != len(b) or not a or len(a[0]) != len(b[0]):
        return None
    rows, cols = len(a), len(a[0])
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if a[r][c] != b[r][c]:
                result[r][c] = a[r][c] if a[r][c] != bg else b[r][c]
    return result


def _and_parts(parts, bg):
    """Keep only cells where both parts are non-bg."""
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    if len(a) != len(b) or not a or len(a[0]) != len(b[0]):
        return None
    rows, cols = len(a), len(a[0])
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if a[r][c] != bg and b[r][c] != bg:
                result[r][c] = a[r][c]
    return result


def _or_parts(parts, bg):
    """Keep non-bg from either part (overlay)."""
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    if len(a) != len(b) or not a or len(a[0]) != len(b[0]):
        return None
    rows, cols = len(a), len(a[0])
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if a[r][c] != bg:
                result[r][c] = a[r][c]
            elif b[r][c] != bg:
                result[r][c] = b[r][c]
    return result


def _diff_part(parts, bg):
    """Return the part that's most different from the others (the unique one)."""
    if len(parts) < 2:
        return None
    # Compare each part to every other
    scores = []
    for i, part in enumerate(parts):
        total_diff = 0
        for j, other in enumerate(parts):
            if i == j:
                continue
            if len(part) != len(other) or not part or len(part[0]) != len(other[0]):
                return None
            for r in range(len(part)):
                for c in range(len(part[0])):
                    if part[r][c] != other[r][c]:
                        total_diff += 1
        scores.append((total_diff, i))
    # The part with the most total differences is the "unique" one
    scores.sort(reverse=True)
    return parts[scores[0][1]]


def _try_quadrant(train):
    """Split into 4 quadrants (by h+v dividers or by middle), combine."""
    bg = _get_bg(train[0]["input"])

    for ex in train:
        h_divs = _find_h_dividers(ex["input"])
        v_divs = _find_v_dividers(ex["input"])
        if not h_divs or not v_divs:
            # Try middle split
            rows, cols = len(ex["input"]), len(ex["input"][0])
            if rows % 2 != 0 or cols % 2 != 0:
                return None

    # Get quadrants
    def get_quads(grid):
        h_divs = _find_h_dividers(grid)
        v_divs = _find_v_dividers(grid)
        if h_divs and v_divs:
            top = grid[:h_divs[0]]
            bot = grid[h_divs[-1]+1:]
            tl = [row[:v_divs[0]] for row in top]
            tr = [row[v_divs[-1]+1:] for row in top]
            bl = [row[:v_divs[0]] for row in bot]
            br = [row[v_divs[-1]+1:] for row in bot]
            return [tl, tr, bl, br]
        rows, cols = len(grid), len(grid[0])
        mr, mc = rows // 2, cols // 2
        return [
            [row[:mc] for row in grid[:mr]],
            [row[mc:] for row in grid[:mr]],
            [row[:mc] for row in grid[mr:]],
            [row[mc:] for row in grid[mr:]],
        ]

    # Try XOR of all quadrants
    def xor_quads(grid, bg_=bg):
        quads = get_quads(grid)
        if len(quads) != 4:
            return None
        result = quads[0]
        for q in quads[1:]:
            if len(q) != len(result) or (q and result and len(q[0]) != len(result[0])):
                return None
            result = _xor_parts([result, q], bg_)
            if result is None:
                return None
        return result

    if all(xor_quads(ex["input"]) == ex["output"] for ex in train):
        return xor_quads

    # Try OR of all quadrants
    def or_quads(grid, bg_=bg):
        quads = get_quads(grid)
        if len(quads) != 4:
            return None
        result = quads[0]
        for q in quads[1:]:
            if len(q) != len(result) or (q and result and len(q[0]) != len(result[0])):
                return None
            result = _or_parts([result, q], bg_)
            if result is None:
                return None
        return result

    if all(or_quads(ex["input"]) == ex["output"] for ex in train):
        return or_quads

    return None
