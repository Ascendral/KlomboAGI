"""
ARC Ranking + Ordering Strategies.

Handles tasks where:
  - Objects/columns/rows are recolored based on their rank (size, height, etc.)
  - Colors are assigned by some ordering property
  - Diagonal repeating pattern tiling
  - Stamp/template at marker positions
"""

from __future__ import annotations
from collections import Counter, defaultdict

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_ranking_rule(train: list[dict]) -> callable | None:
    """Try ranking/ordering strategies."""
    for fn in [
        _try_rank_columns_by_height,
        _try_rank_rows_by_width,
        _try_diagonal_tile,
        _try_stamp_pattern_at_markers,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_rank_columns_by_height(train):
    """
    Each column has cells of one color. Recolor based on column height rank.
    Tallest column → color 1, next → 2, etc. (or reverse)
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Check: input has one non-bg color, output has multiple
    in_colors = set()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    in_colors.add(v)
    if len(in_colors) != 1:
        return None

    src_color = list(in_colors)[0]

    # For each column, compute its height (number of non-bg cells)
    def rank_cols(grid, bg_val, src):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Get height of each column
        col_heights = []
        for c in range(cols):
            h = sum(1 for r in range(rows) if grid[r][c] != bg_val)
            if h > 0:
                col_heights.append((c, h))

        if not col_heights:
            return result

        # Sort by height (descending) to assign rank
        col_heights.sort(key=lambda x: -x[1])

        # Assign colors: rank 1 = tallest
        col_to_color = {}
        for rank, (c, h) in enumerate(col_heights):
            col_to_color[c] = rank + 1

        # Apply
        for c, color in col_to_color.items():
            for r in range(rows):
                if grid[r][c] != bg_val:
                    result[r][c] = color

        return result

    fn = lambda grid, bg_val=bg, s=src_color: rank_cols(grid, bg_val, s)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    # Try reverse (shortest = 1)
    def rank_cols_asc(grid, bg_val, src):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        col_heights = []
        for c in range(cols):
            h = sum(1 for r in range(rows) if grid[r][c] != bg_val)
            if h > 0:
                col_heights.append((c, h))

        if not col_heights:
            return result

        col_heights.sort(key=lambda x: x[1])  # ascending
        col_to_color = {}
        for rank, (c, h) in enumerate(col_heights):
            col_to_color[c] = rank + 1

        for c, color in col_to_color.items():
            for r in range(rows):
                if grid[r][c] != bg_val:
                    result[r][c] = color

        return result

    fn2 = lambda grid, bg_val=bg, s=src_color: rank_cols_asc(grid, bg_val, s)
    if all(fn2(ex["input"]) == ex["output"] for ex in train):
        return fn2

    return None


def _try_rank_rows_by_width(train):
    """Same as column ranking but for rows."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    in_colors = set()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    in_colors.add(v)
    if len(in_colors) != 1:
        return None

    def rank_rows_desc(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        row_widths = []
        for r in range(rows):
            w = sum(1 for c in range(cols) if grid[r][c] != bg_val)
            if w > 0:
                row_widths.append((r, w))

        if not row_widths:
            return result

        row_widths.sort(key=lambda x: -x[1])
        row_to_color = {r: rank + 1 for rank, (r, w) in enumerate(row_widths)}

        for r, color in row_to_color.items():
            for c in range(cols):
                if grid[r][c] != bg_val:
                    result[r][c] = color

        return result

    fn = lambda grid, bg_val=bg: rank_rows_desc(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_diagonal_tile(train):
    """
    Extract a repeating color sequence from a diagonal pattern in the input,
    then tile the entire grid with this pattern along diagonals.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def extract_diag_sequence(grid, bg_val):
        """Extract the non-bg color sequence from any diagonal."""
        rows, cols = len(grid), len(grid[0])

        # Collect ALL non-bg cells and check if they form a consistent
        # repeating pattern along (r+c) % N or (r-c) % N
        nonbg = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                 if grid[r][c] != bg_val]
        if len(nonbg) < 2:
            return None

        colors = list(dict.fromkeys(v for _, _, v in nonbg))  # unique, ordered
        if len(colors) < 2:
            return None

        # Try: sequence along (r+c) % N for different N values
        for n in range(2, min(10, rows + cols)):
            # Group by (r+c) % n
            groups = {}
            consistent = True
            for r, c, v in nonbg:
                key = (r + c) % n
                if key in groups:
                    if groups[key] != v:
                        consistent = False
                        break
                else:
                    groups[key] = v
            if consistent and len(groups) >= 2:
                seq = [groups.get(i, None) for i in range(n)]
                if None not in seq:
                    return seq

        # Try: (r-c) % N
        for n in range(2, min(10, rows + cols)):
            groups = {}
            consistent = True
            for r, c, v in nonbg:
                key = (r - c) % n
                if key in groups:
                    if groups[key] != v:
                        consistent = False
                        break
                else:
                    groups[key] = v
            if consistent and len(groups) >= 2:
                seq = [groups.get(i, None) for i in range(n)]
                if None not in seq:
                    return seq

        return None

    # Each example has its own sequence — extract from input, tile to output
    # First, determine which direction and period are consistent

    # Try (r+c) % n
    for n in range(2, 10):
        def tile_sum(grid, bg_val, period=n):
            seq = extract_diag_sequence(grid, bg_val)
            if not seq or len(seq) != period:
                return None
            rows, cols = len(grid), len(grid[0])
            return [[seq[(r + c) % period] for c in range(cols)] for r in range(rows)]

        try:
            results = [tile_sum(ex["input"], bg) for ex in train]
            if all(r is not None for r in results) and all(r == ex["output"] for r, ex in zip(results, train)):
                return lambda grid, bg_val=bg, p=n: tile_sum(grid, bg_val, p)
        except:
            pass

    # Try (r-c) % n
    for n in range(2, 10):
        def tile_diff(grid, bg_val, period=n):
            seq = extract_diag_sequence(grid, bg_val)
            if not seq or len(seq) != period:
                return None
            rows, cols = len(grid), len(grid[0])
            return [[seq[(r - c) % period] for c in range(cols)] for r in range(rows)]

        try:
            results = [tile_diff(ex["input"], bg) for ex in train]
            if all(r is not None for r in results) and all(r == ex["output"] for r, ex in zip(results, train)):
                return lambda grid, bg_val=bg, p=n: tile_diff(grid, bg_val, p)
        except:
            pass

    return None


def _try_stamp_pattern_at_markers(train):
    """
    Each marker color in the input gets a specific small pattern stamped around it.
    E.g., color 1 → '+' of color 7, color 2 → 'X' of color 4.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Find all marker positions and what appears around them in the output
    # Pattern: for each non-bg cell, look at the 3x3 (or 5x5) neighborhood in the output
    # and learn what gets stamped

    # Build: color → delta → stamped_color
    stamp_patterns = defaultdict(dict)  # marker_color → {(dr,dc): stamp_color}

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])

        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg:
                    continue
                marker_color = inp[r][c]

                # Check what new cells appear around this marker in the output
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if inp[nr][nc] == bg and out[nr][nc] != bg:
                                delta = (dr, dc)
                                stamp_color = out[nr][nc]
                                if delta in stamp_patterns[marker_color]:
                                    if stamp_patterns[marker_color][delta] != stamp_color:
                                        return None  # Inconsistent
                                else:
                                    stamp_patterns[marker_color][delta] = stamp_color

    if not stamp_patterns:
        return None
    # Must have at least some stamped cells
    if all(not v for v in stamp_patterns.values()):
        return None

    def apply_stamps(grid, bg_val, patterns):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg_val:
                    continue
                mc = grid[r][c]
                if mc in patterns:
                    for (dr, dc), sc in patterns[mc].items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if result[nr][nc] == bg_val:
                                result[nr][nc] = sc
        return result

    fn = lambda grid, bg_val=bg, pat=dict(stamp_patterns): apply_stamps(grid, bg_val, pat)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None
