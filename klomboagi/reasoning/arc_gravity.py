"""
ARC Gravity + Movement Strategies.

Handles tasks where colored cells "fall" in a direction, or objects
move along paths, or cells are pushed/pulled by rules.
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_gravity_rule(train: list[dict]) -> callable | None:
    """Try all gravity/movement strategies."""
    for fn in [
        _try_gravity_4dir,
        _try_gravity_to_wall,
        _try_gravity_to_object,
        _try_shift_rows_cols,
        _try_compact,
        _try_mirror_to_fill,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_gravity_4dir(train):
    """All non-bg cells fall in one of 4 directions (down, up, left, right)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def gravity_down(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows) if grid[r][c] != bg_val]
            # Pack to bottom
            for i, v in enumerate(reversed(col_vals)):
                result[rows - 1 - i][c] = v
        return result

    def gravity_up(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows) if grid[r][c] != bg_val]
            for i, v in enumerate(col_vals):
                result[i][c] = v
        return result

    def gravity_left(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]
        for r in range(rows):
            row_vals = [grid[r][c] for c in range(cols) if grid[r][c] != bg_val]
            for i, v in enumerate(row_vals):
                result[r][i] = v
        return result

    def gravity_right(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]
        for r in range(rows):
            row_vals = [grid[r][c] for c in range(cols) if grid[r][c] != bg_val]
            for i, v in enumerate(reversed(row_vals)):
                result[r][cols - 1 - i] = v
        return result

    for gfn in [gravity_down, gravity_up, gravity_left, gravity_right]:
        fn = lambda grid, bg_val=bg, g=gfn: g(grid, bg_val)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


def _try_gravity_to_wall(train):
    """Non-bg cells fall until they hit another non-bg cell (not just the edge)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def gravity_down_stack(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        changed = True
        while changed:
            changed = False
            for r in range(rows - 2, -1, -1):
                for c in range(cols):
                    if result[r][c] != bg_val and result[r + 1][c] == bg_val:
                        result[r + 1][c] = result[r][c]
                        result[r][c] = bg_val
                        changed = True
        return result

    def gravity_up_stack(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        changed = True
        while changed:
            changed = False
            for r in range(1, rows):
                for c in range(cols):
                    if result[r][c] != bg_val and result[r - 1][c] == bg_val:
                        result[r - 1][c] = result[r][c]
                        result[r][c] = bg_val
                        changed = True
        return result

    def gravity_left_stack(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        changed = True
        while changed:
            changed = False
            for r in range(rows):
                for c in range(1, cols):
                    if result[r][c] != bg_val and result[r][c - 1] == bg_val:
                        result[r][c - 1] = result[r][c]
                        result[r][c] = bg_val
                        changed = True
        return result

    def gravity_right_stack(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        changed = True
        while changed:
            changed = False
            for r in range(rows):
                for c in range(cols - 2, -1, -1):
                    if result[r][c] != bg_val and result[r][c + 1] == bg_val:
                        result[r][c + 1] = result[r][c]
                        result[r][c] = bg_val
                        changed = True
        return result

    for gfn in [gravity_down_stack, gravity_up_stack, gravity_left_stack, gravity_right_stack]:
        fn = lambda grid, bg_val=bg, g=gfn: g(grid, bg_val)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


def _try_gravity_to_object(train):
    """One color stays fixed (walls), another color slides toward walls."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Find which colors exist
    in_colors = set()
    for ex in train:
        for row in ex["input"]:
            in_colors.update(row)
    in_colors.discard(bg)

    if len(in_colors) < 2:
        return None

    # Try each color as "wall" and others as "movers"
    for wall_color in in_colors:
        mover_colors = in_colors - {wall_color}

        for direction in ['down', 'up', 'left', 'right']:
            def apply(grid, bg_val=bg, wc=wall_color, d=direction):
                rows, cols = len(grid), len(grid[0])
                result = [row[:] for row in grid]
                changed = True
                max_iter = max(rows, cols)
                i = 0
                while changed and i < max_iter:
                    changed = False
                    i += 1
                    if d == 'down':
                        for r in range(rows - 2, -1, -1):
                            for c in range(cols):
                                if result[r][c] not in (bg_val, wc) and result[r + 1][c] == bg_val:
                                    result[r + 1][c] = result[r][c]
                                    result[r][c] = bg_val
                                    changed = True
                    elif d == 'up':
                        for r in range(1, rows):
                            for c in range(cols):
                                if result[r][c] not in (bg_val, wc) and result[r - 1][c] == bg_val:
                                    result[r - 1][c] = result[r][c]
                                    result[r][c] = bg_val
                                    changed = True
                    elif d == 'left':
                        for r in range(rows):
                            for c in range(1, cols):
                                if result[r][c] not in (bg_val, wc) and result[r][c - 1] == bg_val:
                                    result[r][c - 1] = result[r][c]
                                    result[r][c] = bg_val
                                    changed = True
                    elif d == 'right':
                        for r in range(rows):
                            for c in range(cols - 2, -1, -1):
                                if result[r][c] not in (bg_val, wc) and result[r][c + 1] == bg_val:
                                    result[r][c + 1] = result[r][c]
                                    result[r][c] = bg_val
                                    changed = True
                return result

            if all(apply(ex["input"]) == ex["output"] for ex in train):
                return apply

    return None


def _try_shift_rows_cols(train):
    """Each row or column is cyclically shifted by some amount."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    rows = len(train[0]["input"])
    cols = len(train[0]["input"][0])

    # Try row shifts
    for shift in range(1, cols):
        def shift_rows_right(grid, s=shift):
            return [row[-s:] + row[:-s] for row in grid]

        def shift_rows_left(grid, s=shift):
            return [row[s:] + row[:s] for row in grid]

        for fn in [shift_rows_right, shift_rows_left]:
            if all(fn(ex["input"]) == ex["output"] for ex in train):
                return fn

    # Try col shifts
    for shift in range(1, rows):
        def shift_cols_down(grid, s=shift):
            r = len(grid)
            return grid[r - s:] + grid[:r - s]

        def shift_cols_up(grid, s=shift):
            return grid[s:] + grid[:s]

        for fn in [shift_cols_down, shift_cols_up]:
            if all(fn(ex["input"]) == ex["output"] for ex in train):
                return fn

    return None


def _try_compact(train):
    """Remove empty rows/cols or compact non-bg cells."""
    # Check if output is smaller (rows/cols removed)
    for ex in train:
        if len(ex["input"]) == len(ex["output"]) and len(ex["input"][0]) == len(ex["output"][0]):
            continue
        break
    else:
        return None  # All same-size, not compact

    bg = _bg(train)

    # Try removing empty rows
    def remove_empty_rows(grid, bg_val):
        return [row for row in grid if any(v != bg_val for v in row)]

    if all(remove_empty_rows(ex["input"], bg) == ex["output"] for ex in train):
        return lambda grid, bv=bg: remove_empty_rows(grid, bv)

    # Try removing empty cols
    def remove_empty_cols(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        keep = [c for c in range(cols) if any(grid[r][c] != bg_val for r in range(rows))]
        return [[grid[r][c] for c in keep] for r in range(rows)]

    if all(remove_empty_cols(ex["input"], bg) == ex["output"] for ex in train):
        return lambda grid, bv=bg: remove_empty_cols(grid, bv)

    # Try removing both
    def remove_empty_both(grid, bg_val):
        g = remove_empty_rows(grid, bg_val)
        return remove_empty_cols(g, bg_val)

    if all(remove_empty_both(ex["input"], bg) == ex["output"] for ex in train):
        return lambda grid, bv=bg: remove_empty_both(grid, bv)

    return None


def _try_mirror_to_fill(train):
    """Fill empty half with mirror of non-empty half."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def mirror_left_to_right(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                if result[r][c] != bg_val and result[r][mirror_c] == bg_val:
                    result[r][mirror_c] = result[r][c]
                elif result[r][mirror_c] != bg_val and result[r][c] == bg_val:
                    result[r][c] = result[r][mirror_c]
        return result

    def mirror_top_to_bottom(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        mid = rows // 2
        for r in range(mid):
            mirror_r = rows - 1 - r
            for c in range(cols):
                if result[r][c] != bg_val and result[mirror_r][c] == bg_val:
                    result[mirror_r][c] = result[r][c]
                elif result[mirror_r][c] != bg_val and result[r][c] == bg_val:
                    result[r][c] = result[mirror_r][c]
        return result

    def mirror_diag(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        if rows != cols:
            return None
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if result[r][c] != bg_val and result[c][r] == bg_val:
                    result[c][r] = result[r][c]
                elif result[c][r] != bg_val and result[r][c] == bg_val:
                    result[r][c] = result[c][r]
        return result

    for fn in [mirror_left_to_right, mirror_top_to_bottom, mirror_diag]:
        wrapped = lambda grid, bg_val=bg, f=fn: f(grid, bg_val)
        try:
            if all(wrapped(ex["input"]) == ex["output"] for ex in train):
                return wrapped
        except:
            continue

    return None
