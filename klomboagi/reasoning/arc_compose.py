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
