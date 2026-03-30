"""
ARC Compositional Strategies.

Handles multi-step transformations by composing primitives:
  - Remove one color, then apply another rule
  - Split grid by color, transform each part
  - Apply rule to each object independently
  - Chain two simple transformations
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_compose_rule(train: list[dict]) -> callable | None:
    """Try compositional strategies."""
    for fn in [
        _try_remove_color_then_apply,
        _try_bool_mask_color,
        _try_input_overlay,
        _try_color_filter_and_crop,
        _try_split_by_divider_and_combine,
        _try_grid_regions_fill,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_remove_color_then_apply(train):
    """
    Remove one specific color (set to bg), then the result matches the output.
    Or: keep only specific colors.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # What non-bg colors exist in input?
    in_colors = set()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    in_colors.add(v)

    # Try removing each color and checking
    for remove_color in in_colors:
        def remove_and_check(grid, bg_val, rc):
            return [[bg_val if v == rc else v for v in row] for row in grid]

        fn = lambda grid, bg_val=bg, rc=remove_color: remove_and_check(grid, bg_val, rc)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


def _try_bool_mask_color(train):
    """
    Input has two layers separated by color. One layer is a "mask",
    the other is the content. Output = content where mask is active.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # What colors exist?
    in_colors = set()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    in_colors.add(v)

    if len(in_colors) < 2:
        return None

    # Try: for each color pair (mask_color, fill_color),
    # where mask cells become fill_color in output
    for mask_color in in_colors:
        for fill_color in in_colors:
            if mask_color == fill_color:
                continue

            def apply_mask(grid, bg_val, mc, fc):
                rows, cols = len(grid), len(grid[0])
                result = [row[:] for row in grid]
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] == mc:
                            result[r][c] = fc
                return result

            fn = lambda grid, bg_val=bg, mc=mask_color, fc=fill_color: apply_mask(grid, bg_val, mc, fc)
            if all(fn(ex["input"]) == ex["output"] for ex in train):
                return fn

    return None


def _try_input_overlay(train):
    """
    Two grids embedded in the input (separated by a divider or by position).
    Output = overlay/combine them.
    """
    bg = _bg(train)

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        otr, otc = len(out), len(out[0])

        # Check for horizontal split (output is half height)
        if rows == 2 * otr + 1 and cols == otc:
            # Find divider row
            for div_r in range(rows):
                if len(set(inp[div_r])) == 1 and inp[div_r][0] != bg:
                    top = [row[:] for row in inp[:div_r]]
                    bot = [row[:] for row in inp[div_r + 1:]]
                    if len(top) == otr and len(bot) == otr:
                        # Try overlay operations
                        break
            else:
                continue
            break
        # Check for vertical split
        elif cols == 2 * otc + 1 and rows == otr:
            break

    return None  # Complex — existing grid_ops handles most of this


def _try_color_filter_and_crop(train):
    """
    Keep only cells of specific colors, then crop to bounding box.
    """
    bg = _bg(train)

    # Check: does the output contain only a subset of input colors?
    in_colors = set()
    out_colors = set()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    in_colors.add(v)
        for row in ex["output"]:
            for v in row:
                if v != bg:
                    out_colors.add(v)

    keep_colors = out_colors - {bg}
    remove_colors = in_colors - out_colors

    if not remove_colors:
        return None

    def filter_and_crop(grid, bg_val, keep):
        rows, cols = len(grid), len(grid[0])

        # Filter: remove colors not in keep set
        filtered = [[v if v in keep or v == bg_val else bg_val for v in row] for row in grid]

        # Crop to bounding box of non-bg cells
        min_r = rows
        max_r = -1
        min_c = cols
        max_c = -1
        for r in range(rows):
            for c in range(cols):
                if filtered[r][c] != bg_val:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        if max_r < 0:
            return filtered

        return [filtered[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]

    fn = lambda grid, bg_val=bg, k=keep_colors: filter_and_crop(grid, bg_val, k)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_split_by_divider_and_combine(train):
    """
    Input has a divider line of a specific color (entire row or column).
    The two halves are combined by AND/XOR/OR of non-bg cells.
    Output uses a new color (learned from examples).
    """
    bg = _bg(train)

    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        or_, oc = len(ex["output"]), len(ex["output"][0])
        if ir != or_ or ic != oc:
            # Check if output is half size (output = result of combining two halves)
            if ir == or_ and ic == oc:
                pass
            else:
                # Could be shrinking — output = combined half
                pass

    def find_divider(grid, bg_val):
        """Find a column or row that is entirely one non-bg color."""
        rows, cols = len(grid), len(grid[0])
        # Column dividers
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1 and vals != {bg_val}:
                return ("col", c, list(vals)[0])
        # Row dividers
        for r in range(rows):
            vals = set(grid[r][c] for c in range(cols))
            if len(vals) == 1 and vals != {bg_val}:
                return ("row", r, list(vals)[0])
        return None

    # Find divider info from first training example
    div0 = find_divider(train[0]["input"], bg)
    if div0 is None:
        return None

    # Check all training examples have same divider type and color
    for ex in train:
        div = find_divider(ex["input"], bg)
        if div is None or div[0] != div0[0] or div[2] != div0[2]:
            return None

    div_type, div_color = div0[0], div0[2]

    # Split input at divider, combine two halves
    def split_and_combine(grid, bg_val, dtype, dcolor, op, out_color):
        rows, cols = len(grid), len(grid[0])
        # Find divider position
        if dtype == "col":
            div_pos = None
            for c in range(cols):
                if all(grid[r][c] == dcolor for r in range(rows)):
                    div_pos = c
                    break
            if div_pos is None:
                return None
            # Left and right (same height, equal or near-equal width)
            left = [[grid[r][c] for c in range(div_pos)] for r in range(rows)]
            right = [[grid[r][c] for c in range(div_pos + 1, cols)] for r in range(rows)]
            # Ensure same width
            lw = len(left[0]) if left and left[0] else 0
            rw = len(right[0]) if right and right[0] else 0
            if lw != rw:
                return None
            h, w = rows, lw
            result = [[bg_val] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    lv = left[r][c] != bg_val
                    rv = right[r][c] != bg_val
                    if op == "and" and lv and rv:
                        result[r][c] = out_color
                    elif op == "or" and (lv or rv):
                        result[r][c] = out_color if lv or rv else bg_val
                    elif op == "xor" and (lv ^ rv):
                        result[r][c] = out_color
                    elif op == "left_only" and lv and not rv:
                        result[r][c] = out_color
                    elif op == "right_only" and rv and not lv:
                        result[r][c] = out_color
            return result
        else:  # row divider
            div_pos = None
            for r in range(rows):
                if all(grid[r][c] == dcolor for c in range(cols)):
                    div_pos = r
                    break
            if div_pos is None:
                return None
            top = grid[:div_pos]
            bot = grid[div_pos + 1:]
            if len(top) != len(bot):
                return None
            h, w = len(top), cols
            result = [[bg_val] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    tv = top[r][c] != bg_val
                    bv = bot[r][c] != bg_val
                    if op == "and" and tv and bv:
                        result[r][c] = out_color
                    elif op == "or" and (tv or bv):
                        result[r][c] = out_color
                    elif op == "xor" and (tv ^ bv):
                        result[r][c] = out_color
                    elif op == "left_only" and tv and not bv:
                        result[r][c] = out_color
                    elif op == "right_only" and bv and not tv:
                        result[r][c] = out_color
            return result

    # Learn output color (non-bg color in outputs)
    out_colors = set()
    for ex in train:
        for row in ex["output"]:
            for v in row:
                if v != bg:
                    out_colors.add(v)

    for out_color in out_colors:
        for op in ["and", "xor", "or", "left_only", "right_only"]:
            results = [split_and_combine(ex["input"], bg, div_type, div_color, op, out_color)
                       for ex in train]
            if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
                return lambda g, b=bg, dt=div_type, dc=div_color, o=op, oc=out_color: \
                    split_and_combine(g, b, dt, dc, o, oc)

    return None


def _try_grid_regions_fill(train):
    """
    Input has separator lines (rows/cols all same color) dividing the grid
    into rectangular regions. Output fills each region with its most common
    non-bg, non-separator color.
    """
    bg = _bg(train)

    # Same size input/output
    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def find_sep_lines(grid, bg_val):
        """Find rows/cols that are entirely one specific non-bg color."""
        rows, cols = len(grid), len(grid[0])
        sep_rows = []
        sep_cols = []
        for r in range(rows):
            vals = set(grid[r])
            if len(vals) == 1 and vals != {bg_val}:
                sep_rows.append((r, list(vals)[0]))
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1 and vals != {bg_val}:
                sep_cols.append((c, list(vals)[0]))
        return sep_rows, sep_cols

    def fill_regions(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        sep_rows, sep_cols = find_sep_lines(grid, bg_val)

        if not sep_rows and not sep_cols:
            return None

        sep_row_indices = {r for r, _ in sep_rows}
        sep_col_indices = {c for c, _ in sep_cols}

        # Build row and col boundaries
        sep_row_list = sorted(sep_row_indices)
        sep_col_list = sorted(sep_col_indices)

        row_bounds = [0] + [r + 1 for r in sep_row_list] + [rows]
        col_bounds = [0] + [c + 1 for c in sep_col_list] + [cols]

        result = [row[:] for row in grid]

        for i in range(len(row_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i + 1]
            for j in range(len(col_bounds) - 1):
                c0, c1 = col_bounds[j], col_bounds[j + 1]

                # Extract region (skip separator rows/cols)
                region_cells = []
                for r in range(r0, r1):
                    if r in sep_row_indices:
                        continue
                    for c in range(c0, c1):
                        if c in sep_col_indices:
                            continue
                        v = grid[r][c]
                        if v != bg_val:
                            region_cells.append(v)

                if not region_cells:
                    continue

                # Fill color = most common non-bg color in region
                fill = Counter(region_cells).most_common(1)[0][0]

                for r in range(r0, r1):
                    if r in sep_row_indices:
                        continue
                    for c in range(c0, c1):
                        if c in sep_col_indices:
                            continue
                        result[r][c] = fill

        return result

    # Check if first example has separator lines
    sr, sc = find_sep_lines(train[0]["input"], bg)
    if not sr and not sc:
        return None

    results = [fill_regions(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: fill_regions(g, b)

    return None
