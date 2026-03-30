"""
ARC Multi-Object and Composition Strategies.

Handles:
  1. Grid overlay: input contains two halves — combine them
  2. Object sorting: sort objects by size/color/position → rearrange
  3. Object template: use one colored object as a template, stamp elsewhere
  4. Object movement to target: objects slide toward a target color
  5. Isolate object by uniqueness: unique shape/position/color wins
  6. Frame/border drawing around objects
  7. Object deduplication: keep one copy of repeated objects
  8. Input grid = multiple choices + key → select matching
  9. Pixelwise XOR/AND/OR of two halves
  10. Object scaling by color/size rule
  11. Color spread from seed: flood fill colors through background
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Callable

Grid = list[list[int]]


def _bg(train) -> int:
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def _find_objects(grid: Grid, bg: int) -> list[dict]:
    """BFS connected components of non-bg cells."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            cells = []
            queue = [(sr, sc)]
            while queue:
                r, c = queue.pop(0)
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    continue
                if visited[r][c] or grid[r][c] == bg:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((r + dr, c + dc))

            if cells:
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                sub = [[bg] * w for _ in range(h)]
                for r, c in cells:
                    sub[r - min_r][c - min_c] = grid[r][c]
                color_count = Counter(grid[r][c] for r, c in cells)
                objects.append({
                    "cells": cells,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "h": h, "w": w,
                    "size": len(cells),
                    "sub": sub,
                    "primary": color_count.most_common(1)[0][0],
                    "colors": set(color_count.keys()),
                    "color_count": color_count,
                    "center": ((min_r + max_r) / 2, (min_c + max_c) / 2),
                })
    return objects


def learn_multiobj_rule(train: list[dict]) -> Callable | None:
    """Try all multi-object strategies."""
    for fn in [
        _try_overlay_halves,
        _try_overlay_halves_xor,
        _try_sort_objects_by_position,
        _try_color_from_neighbor_object,
        _try_draw_bounding_box,
        _try_remove_noise_keep_rectangle,
        _try_select_grid_matching_key,
        _try_object_gravity_stack,
        _try_fill_object_interior,
        _try_unique_object_wins,
        _try_smallest_object_color_fill,
        _try_connect_same_color_objects,
        _try_object_count_to_size,
        _try_inside_outside_recolor,
        _try_color_objects_by_row_position,
        _try_color_objects_by_col_position,
        _try_cross_divider_quadrants,
        _try_cross_quadrant_select,
        _try_corner_frame_extract,
        _try_most_common_object,
        _try_color_specific_neighbors,
        _try_row_col_paint,
        _try_nearest_border_recolor,
        _try_fill_interior_from_external,
        _try_project_dots_onto_block,
        _try_dot_to_block_line,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


# ─── Grid overlay (combine two halves) ──────────────────────────────────────

def _try_overlay_halves(train):
    """
    Input contains two halves (left/right or top/bottom).
    Output = overlay: where both have non-bg, take one; else take non-bg value.
    """
    # Check same-size input/output
    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    bg = _bg(train)

    def split_and_overlay(grid, bg_val, direction, conflict):
        rows, cols = len(grid), len(grid[0])
        if direction == "h":
            if cols % 2 != 0:
                return None
            half = cols // 2
            a = [[grid[r][c] for c in range(half)] for r in range(rows)]
            b = [[grid[r][c] for c in range(half, cols)] for r in range(rows)]
        else:
            if rows % 2 != 0:
                return None
            half = rows // 2
            a = [[grid[r][c] for c in range(cols)] for r in range(half)]
            b = [[grid[r][c] for c in range(cols)] for r in range(half, rows)]

        h2, w2 = len(a), len(a[0])
        result = [[bg_val] * w2 for _ in range(h2)]
        for r in range(h2):
            for c in range(w2):
                av, bv = a[r][c], b[r][c]
                if av == bg_val and bv == bg_val:
                    result[r][c] = bg_val
                elif av != bg_val and bv == bg_val:
                    result[r][c] = av
                elif av == bg_val and bv != bg_val:
                    result[r][c] = bv
                else:
                    result[r][c] = conflict(av, bv)
        return result

    for direction in ["h", "v"]:
        for conflict in [lambda a, b: a, lambda a, b: b, lambda a, b: bg]:
            def make_fn(d=direction, c=conflict, bg_=bg):
                def fn(grid):
                    r = split_and_overlay(grid, bg_, d, c)
                    return r
                return fn
            f = make_fn()
            try:
                results = [f(ex["input"]) for ex in train]
                if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
                    return f
            except Exception:
                continue

    return None


def _try_overlay_halves_xor(train):
    """Output = XOR of two halves: non-bg iff exactly one half has non-bg."""
    bg = _bg(train)

    # Check output is smaller than input (half the size)
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    or_, oc = len(ex0["output"]), len(ex0["output"][0])

    if not (ir == or_ * 2 or ic == oc * 2):
        return None

    def do_xor(grid, bg_val, direction):
        rows, cols = len(grid), len(grid[0])
        if direction == "h":
            if cols % 2 != 0:
                return None
            half = cols // 2
            a = [[grid[r][c] for c in range(half)] for r in range(rows)]
            b = [[grid[r][c] for c in range(half, cols)] for r in range(rows)]
        else:
            if rows % 2 != 0:
                return None
            half = rows // 2
            a = grid[:half]
            b = grid[half:]

        h2, w2 = len(a), len(a[0])
        result = [[bg_val] * w2 for _ in range(h2)]
        for r in range(h2):
            for c in range(w2):
                av = a[r][c] != bg_val
                bv = b[r][c] != bg_val
                if av ^ bv:
                    result[r][c] = a[r][c] if av else b[r][c]
        return result

    for direction in ["h", "v"]:
        def make_fn(d=direction, bg_=bg):
            return lambda grid: do_xor(grid, bg_, d)
        f = make_fn()
        try:
            results = [f(ex["input"]) for ex in train]
            if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
                return f
        except Exception:
            continue

    return None


# ─── Sort objects by position ────────────────────────────────────────────────

def _try_sort_objects_by_position(train):
    """
    Objects in input are rearranged in output sorted by size (desc/asc).
    Or they're stacked to one side.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # Strategy: objects sorted by height (y position) descending → placed left-to-right
    def sort_objects_vertical(grid, bg_val):
        """Sort column objects by their height, place from left."""
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        if not objects:
            return grid

        # Sort by size descending
        objects.sort(key=lambda o: -o["size"])

        result = [[bg_val] * cols for _ in range(rows)]
        # Place objects left-aligned, maintaining row positions
        col_offset = 0
        for obj in objects:
            h, w = obj["h"], obj["w"]
            if col_offset + w > cols:
                break
            min_r = obj["bbox"][0]
            for r in range(h):
                for c in range(w):
                    result[min_r + r][col_offset + c] = obj["sub"][r][c]
            col_offset += w + 1  # 1 gap between objects
        return result

    # This is too complex to learn generically — skip the generic approach
    # and just check if sorting by specific keys works
    return None


# ─── Color from neighbor object ──────────────────────────────────────────────

def _try_color_from_neighbor_object(train):
    """
    Input has objects of color A next to an object of color B.
    Output: A objects are recolored to B.
    Learn: each input color maps to its nearest non-bg neighbor's color.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # Check if it's a simple global color mapping
    # Learn color_in → color_out from all training examples
    mapping = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv == bg and ov == bg:
                    continue
                if iv != bg and ov != bg and iv != ov:
                    if iv in mapping and mapping[iv] != ov:
                        return None
                    mapping[iv] = ov

    if not mapping:
        return None

    def apply(grid, m=mapping, bg_=bg):
        return [[m.get(v, v) for v in row] for row in grid]

    # Already covered by cell_rules — only proceed if output adds new cells
    # Check if ANY bg cell becomes non-bg
    for ex in train:
        for r in range(len(ex["input"])):
            for c in range(len(ex["input"][0])):
                if ex["input"][r][c] == bg and ex["output"][r][c] != bg:
                    return None  # This is fill, not recolor

    if all(apply(ex["input"]) == ex["output"] for ex in train):
        return apply
    return None


# ─── Draw bounding box ───────────────────────────────────────────────────────

def _try_draw_bounding_box(train):
    """
    Output draws the bounding box (rectangle outline) around each object.
    Various color rules: same color, specific color, outline color.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # Strategy A: Each object gets its bounding box outlined in same color
    def draw_bbox_same_color(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        result = [row[:] for row in grid]
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            color = obj["primary"]
            for c in range(c0, c1 + 1):
                result[r0][c] = color
                result[r1][c] = color
            for r in range(r0, r1 + 1):
                result[r][c0] = color
                result[r][c1] = color
        return result

    if all(draw_bbox_same_color(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: draw_bbox_same_color(g, b)

    # Strategy B: Draw bbox, clear interior (just the outline)
    def draw_bbox_outline_only(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        result = [[bg_val] * cols for _ in range(rows)]
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            color = obj["primary"]
            for c in range(c0, c1 + 1):
                result[r0][c] = color
                result[r1][c] = color
            for r in range(r0, r1 + 1):
                result[r][c0] = color
                result[r][c1] = color
        return result

    if all(draw_bbox_outline_only(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: draw_bbox_outline_only(g, b)

    # Strategy C: Learn bbox color from output
    # Find a single output color used for all bboxes
    all_out_colors = set()
    for ex in train:
        for row in ex["output"]:
            for v in row:
                if v != bg:
                    all_out_colors.add(v)

    for bbox_color in all_out_colors:
        def draw_bbox_color(grid, bg_val, bc=bbox_color):
            rows, cols = len(grid), len(grid[0])
            objects = _find_objects(grid, bg_val)
            result = [[bg_val] * cols for _ in range(rows)]
            for obj in objects:
                r0, c0, r1, c1 = obj["bbox"]
                for c in range(c0, c1 + 1):
                    result[r0][c] = bc
                    result[r1][c] = bc
                for r in range(r0, r1 + 1):
                    result[r][c0] = bc
                    result[r][c1] = bc
            return result

        if all(draw_bbox_color(ex["input"], bg) == ex["output"] for ex in train):
            return lambda g, b=bg, bc=bbox_color: draw_bbox_color(g, b, bc)

    return None


# ─── Remove noise, keep rectangle ────────────────────────────────────────────

def _try_remove_noise_keep_rectangle(train):
    """
    Input has a mostly-clean rectangular object with some noise.
    Output = the clean rectangle.
    Detect by: most cells of non-bg are in a rectangular arrangement.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def reconstruct_rectangle(grid, bg_val):
        """Find the dominant rectangular object and clean it."""
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        if not objects:
            return grid

        # For each object, check if it's "rectangle-like": fills most of its bbox
        best = None
        best_score = 0
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            bbox_size = (r1 - r0 + 1) * (c1 - c0 + 1)
            if bbox_size == 0:
                continue
            rect_score = obj["size"] / bbox_size
            if rect_score > best_score:
                best_score = rect_score
                best = obj

        if best is None or best_score < 0.7:
            return None

        # Fill the bbox of the best object
        r0, c0, r1, c1 = best["bbox"]
        result = [[bg_val] * cols for _ in range(rows)]
        color = best["primary"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                result[r][c] = color
        return result

    results = [reconstruct_rectangle(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: reconstruct_rectangle(g, b)

    return None


# ─── Select grid matching key ─────────────────────────────────────────────────

def _try_select_grid_matching_key(train):
    """
    Input has a small 'key' object and multiple larger 'choice' grids.
    Output is the choice grid that has the same pattern as the key.
    """
    # This is complex; check basic structure first
    bg = _bg(train)

    # Check: output is smaller than input
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    or_, oc = len(ex0["output"]), len(ex0["output"][0])

    if or_ >= ir or oc >= ic:
        return None

    # Try: input is divided into a grid of sub-grids separated by bg rows/cols
    def find_separators(grid, bg_val):
        """Find rows/cols that are entirely background."""
        rows, cols = len(grid), len(grid[0])
        sep_rows = [r for r in range(rows) if all(grid[r][c] == bg_val for c in range(cols))]
        sep_cols = [c for c in range(cols) if all(grid[r][c] == bg_val for r in range(rows))]
        return sep_rows, sep_cols

    def split_by_separators(grid, sep_rows, sep_cols, bg_val):
        """Split grid into sub-grids at separator positions."""
        rows, cols = len(grid), len(grid[0])
        row_bounds = [0] + [r + 1 for r in sep_rows] + [rows]
        col_bounds = [0] + [c + 1 for c in sep_cols] + [cols]

        sub_grids = []
        for i in range(len(row_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i + 1]
            for j in range(len(col_bounds) - 1):
                c0, c1 = col_bounds[j], col_bounds[j + 1]
                sub = [[grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]
                # Skip empty sub-grids
                if any(v != bg_val for row in sub for v in row):
                    sub_grids.append({"grid": sub, "pos": (r0, c0)})
        return sub_grids

    # Check if input can be split into sub-grids, one of which matches output
    for ex in train[:1]:
        sep_rows, sep_cols = find_separators(ex["input"], bg)
        if not sep_rows and not sep_cols:
            return None

        sub_grids = split_by_separators(ex["input"], sep_rows, sep_cols, bg)
        if len(sub_grids) < 2:
            return None

        # Does the output match one of the sub-grids?
        matching = [sg for sg in sub_grids if sg["grid"] == ex["output"]]
        if not matching:
            return None

    # Pattern found in first example — build a selector function
    def select_matching(grid, target, bg_val):
        """Return the sub-grid that matches the target pattern."""
        sep_rows, sep_cols = find_separators(grid, bg_val)
        sub_grids = split_by_separators(grid, sep_rows, sep_cols, bg_val)
        for sg in sub_grids:
            if sg["grid"] == target:
                return sg["grid"]
        return None

    # The target changes each example — need a different approach
    # This is hard to generalize without knowing what "key" is
    return None


# ─── Object gravity with stacking ────────────────────────────────────────────

def _try_object_gravity_stack(train):
    """
    Objects in input fall in some direction and stack on each other.
    More sophisticated than simple per-cell gravity.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def gravity_objects_down(grid, bg_val):
        """Each connected object falls straight down."""
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        result = [[bg_val] * cols for _ in range(rows)]

        # Sort objects by bottom edge (process bottom-most first to handle stacking)
        objects.sort(key=lambda o: -o["bbox"][2])

        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            # Find how far this object can fall
            # Cells of this object: cells list
            # Find lowest position for each column of the object
            max_drop = rows - 1 - r1

            for drop in range(max_drop, -1, -1):
                can_place = True
                for r, c in obj["cells"]:
                    nr = r + drop
                    if nr >= rows or result[nr][c] != bg_val:
                        can_place = False
                        break
                if can_place:
                    for r, c in obj["cells"]:
                        result[r + drop][c] = grid[r][c]
                    break
            else:
                # Can't drop at all — place in original position
                for r, c in obj["cells"]:
                    result[r][c] = grid[r][c]

        return result

    if all(gravity_objects_down(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: gravity_objects_down(g, b)

    # Try gravity up
    def gravity_objects_up(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        result = [[bg_val] * cols for _ in range(rows)]
        objects.sort(key=lambda o: o["bbox"][0])

        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            max_rise = r0

            for rise in range(max_rise, -1, -1):
                can_place = True
                for r, c in obj["cells"]:
                    nr = r - rise
                    if nr < 0 or result[nr][c] != bg_val:
                        can_place = False
                        break
                if can_place:
                    for r, c in obj["cells"]:
                        result[r - rise][c] = grid[r][c]
                    break
            else:
                for r, c in obj["cells"]:
                    result[r][c] = grid[r][c]
        return result

    if all(gravity_objects_up(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: gravity_objects_up(g, b)

    return None


# ─── Fill object interior ─────────────────────────────────────────────────────

def _try_fill_object_interior(train):
    """
    Objects are outlines; fill their interior with a specific color.
    The fill color may be: same as outline, a specific fixed color, or
    a color found inside the outline in the input.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def flood_fill_from_outside(grid, bg_val):
        """Flood fill from all border cells; everything not reached is 'inside'."""
        rows, cols = len(grid), len(grid[0])
        outside = [[False] * cols for _ in range(rows)]
        queue = []
        for r in range(rows):
            for c in [0, cols - 1]:
                if grid[r][c] == bg_val and not outside[r][c]:
                    outside[r][c] = True
                    queue.append((r, c))
        for c in range(cols):
            for r in [0, rows - 1]:
                if grid[r][c] == bg_val and not outside[r][c]:
                    outside[r][c] = True
                    queue.append((r, c))

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not outside[nr][nc] and grid[nr][nc] == bg_val:
                        outside[nr][nc] = True
                        queue.append((nr, nc))

        return outside

    # Check what happens: do bg cells inside outlines get a specific color?
    # Learn: for each bg cell that becomes non-bg, what color does it get?
    interior_colors = set()
    for ex in train:
        outside = flood_fill_from_outside(ex["input"], bg)
        rows, cols = len(ex["input"]), len(ex["input"][0])
        for r in range(rows):
            for c in range(cols):
                if ex["input"][r][c] == bg and not outside[r][c]:
                    # Interior bg cell
                    expected_out = ex["output"][r][c]
                    if expected_out != bg:
                        interior_colors.add(expected_out)

    if not interior_colors:
        return None

    # Try: fill interior with each candidate color
    for fill_color in interior_colors:
        def fill_interior(grid, bg_val, fc=fill_color):
            rows, cols = len(grid), len(grid[0])
            outside = flood_fill_from_outside(grid, bg_val)
            result = [row[:] for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == bg_val and not outside[r][c]:
                        result[r][c] = fc
            return result

        if all(fill_interior(ex["input"], bg) == ex["output"] for ex in train):
            return lambda g, b=bg, fc=fill_color: fill_interior(g, b, fc)

    # Try: fill interior with the color of the surrounding outline
    def fill_interior_with_outline_color(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        outside = flood_fill_from_outside(grid, bg_val)
        result = [row[:] for row in grid]

        # For each interior region, find surrounding outline color
        visited = [[False] * cols for _ in range(rows)]
        for sr in range(rows):
            for sc in range(cols):
                if visited[sr][sc] or grid[sr][sc] != bg_val or outside[sr][sc]:
                    continue
                # BFS interior region
                region = []
                adj_colors = Counter()
                queue = [(sr, sc)]
                while queue:
                    r, c = queue.pop(0)
                    if visited[r][c]:
                        continue
                    visited[r][c] = True
                    region.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr][nc]:
                                if grid[nr][nc] != bg_val:
                                    adj_colors[grid[nr][nc]] += 1
                                else:
                                    queue.append((nr, nc))

                if adj_colors:
                    fill = adj_colors.most_common(1)[0][0]
                    for r, c in region:
                        result[r][c] = fill
        return result

    if all(fill_interior_with_outline_color(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: fill_interior_with_outline_color(g, b)

    return None


# ─── Unique object wins ───────────────────────────────────────────────────────

def _try_unique_object_wins(train):
    """
    Multiple similar objects + one unique one. Output = the unique one.
    'Unique' can mean: different color, different shape, different size.
    """
    bg = _bg(train)
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    or_, oc = len(ex0["output"]), len(ex0["output"][0])

    # Try different uniqueness criteria
    def extract_unique_by_color(grid, bg_val):
        objects = _find_objects(grid, bg_val)
        if len(objects) < 2:
            return None
        color_counts = Counter(o["primary"] for o in objects)
        unique = [o for o in objects if color_counts[o["primary"]] == 1]
        if len(unique) == 1:
            return unique[0]["sub"]
        return None

    results = [extract_unique_by_color(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: extract_unique_by_color(g, b)

    def extract_unique_by_shape(grid, bg_val):
        objects = _find_objects(grid, bg_val)
        if len(objects) < 2:
            return None
        # Normalize shapes (remove color, just structural)
        def normalize(obj):
            return tuple(tuple(1 if v != bg_val else 0 for v in row) for row in obj["sub"])
        shape_counts = Counter(normalize(o) for o in objects)
        unique = [o for o in objects if shape_counts[normalize(o)] == 1]
        if len(unique) == 1:
            return unique[0]["sub"]
        return None

    results = [extract_unique_by_shape(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: extract_unique_by_shape(g, b)

    return None


# ─── Smallest object color → fill ────────────────────────────────────────────

def _try_smallest_object_color_fill(train):
    """
    Smallest object in input = fill color.
    Largest object gets filled with that color.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def apply(grid, bg_val):
        objects = _find_objects(grid, bg_val)
        if len(objects) < 2:
            return None
        smallest = min(objects, key=lambda o: o["size"])
        largest = max(objects, key=lambda o: o["size"])
        if smallest is largest:
            return None
        fill_color = smallest["primary"]
        result = [row[:] for row in grid]
        for r, c in largest["cells"]:
            result[r][c] = fill_color
        return result

    results = [apply(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: apply(g, b)

    return None


# ─── Connect same-color objects ───────────────────────────────────────────────

def _try_connect_same_color_objects(train):
    """
    Same-color objects in input get connected with a line (H or V).
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def connect_objects(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        result = [row[:] for row in grid]

        # Group by color
        by_color = defaultdict(list)
        for obj in objects:
            by_color[obj["primary"]].append(obj)

        for color, objs in by_color.items():
            if len(objs) < 2:
                continue
            # Try to connect each pair with H or V line
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i], objs[j]
                    ar0, ac0, ar1, ac1 = a["bbox"]
                    br0, bc0, br1, bc1 = b["bbox"]

                    # Horizontal connection: same row range overlap
                    row_overlap = max(ar0, br0), min(ar1, br1)
                    if row_overlap[0] <= row_overlap[1]:
                        r_mid = (row_overlap[0] + row_overlap[1]) // 2
                        c_start = min(ac1, bc1)
                        c_end = max(ac0, bc0)
                        if c_start < c_end:
                            for c in range(c_start + 1, c_end):
                                if result[r_mid][c] == bg_val:
                                    result[r_mid][c] = color

                    # Vertical connection: same col range overlap
                    col_overlap = max(ac0, bc0), min(ac1, bc1)
                    if col_overlap[0] <= col_overlap[1]:
                        c_mid = (col_overlap[0] + col_overlap[1]) // 2
                        r_start = min(ar1, br1)
                        r_end = max(ar0, br0)
                        if r_start < r_end:
                            for r in range(r_start + 1, r_end):
                                if result[r][c_mid] == bg_val:
                                    result[r][c_mid] = color

        return result

    if all(connect_objects(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: connect_objects(g, b)

    return None


# ─── Object count to size ─────────────────────────────────────────────────────

def _try_object_count_to_size(train):
    """
    Count objects of each color in input.
    Output: a colored bar/shape of size = count.
    """
    bg = _bg(train)
    # Too complex to learn generically here — handled by arc_advanced
    return None


# ─── Inside/outside recolor ───────────────────────────────────────────────────

def _try_inside_outside_recolor(train):
    """
    Cells inside a closed shape get one color, cells outside get another.
    The shape is defined by a specific 'wall' color.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # Find the 'wall' color (appears in both input and output unchanged)
    # And find inside/outside colors in output

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])

        # Find cells that are bg in input but non-bg in output
        newly_colored = {}
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg and out[r][c] != bg:
                    newly_colored[(r, c)] = out[r][c]

        if not newly_colored:
            continue

        # Multiple colors being filled in
        fill_colors = set(newly_colored.values())
        if len(fill_colors) > 2:
            return None

    return None


# ─── Color objects by row position ───────────────────────────────────────────

def _try_color_objects_by_row_position(train):
    """
    Objects in different rows get different colors based on their row index.
    E.g., top row objects → color 1, middle → 2, bottom → 3.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # Learn: which output color corresponds to which row band
    # Check if objects are recolored based on their row position
    row_to_color = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        objects = _find_objects(inp, bg)
        for obj in objects:
            r_center = int(obj["center"][0])
            in_color = obj["primary"]

            # Find the color of this object's cells in output
            out_colors = Counter(out[r][c] for r, c in obj["cells"] if out[r][c] != bg)
            if not out_colors:
                continue
            out_color = out_colors.most_common(1)[0][0]

            if r_center in row_to_color:
                if row_to_color[r_center] != out_color:
                    return None
            else:
                row_to_color[r_center] = out_color

    if not row_to_color:
        return None

    # Try applying this mapping
    def apply(grid, bg_val, r2c=row_to_color):
        objects = _find_objects(grid, bg_val)
        result = [row[:] for row in grid]
        for obj in objects:
            r_center = int(obj["center"][0])
            # Find nearest known row
            nearest = min(r2c.keys(), key=lambda k: abs(k - r_center))
            new_color = r2c[nearest]
            for r, c in obj["cells"]:
                result[r][c] = new_color
        return result

    if all(apply(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: apply(g, b)

    return None


def _try_corner_frame_extract(train):
    """
    Input has 4 corner markers (same color) at the corners of a rectangle.
    Those 4 points define a frame. The interior contains cells of another color.
    Output = interior content, with the interior cells recolored to the corner marker color.
    """
    bg = _bg(train)

    def find_4_corners(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        by_color = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg_val:
                    if v not in by_color:
                        by_color[v] = []
                    by_color[v].append((r, c))

        for color, cells in by_color.items():
            if len(cells) != 4:
                continue
            rs = sorted(set(r for r, c in cells))
            cs_vals = sorted(set(c for r, c in cells))
            if len(rs) != 2 or len(cs_vals) != 2:
                continue
            r0, r1 = rs[0], rs[1]
            c0, c1 = cs_vals[0], cs_vals[1]
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if corners == set(cells):
                return color, r0, r1, c0, c1
        return None

    def apply_frame_extract(grid, bg_val):
        result = find_4_corners(grid, bg_val)
        if result is None:
            return None
        frame_color, r0, r1, c0, c1 = result
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        h = r1 - r0 - 1
        w = c1 - c0 - 1
        output = [[bg_val] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                v = grid[r0 + 1 + r][c0 + 1 + c]
                if v != bg_val and v != frame_color:
                    output[r][c] = frame_color
        return output

    results = [apply_frame_extract(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: apply_frame_extract(g, b)

    return None


def _try_most_common_object(train):
    """
    Input contains multiple copies of the same pattern (possibly non-connected),
    plus different noise patterns. Output = the pattern that repeats most.
    """
    bg = _bg(train)

    def normalize_cells(cells, grid):
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        return tuple(
            tuple(grid[min_r + dr][min_c + dc] for dc in range(w))
            for dr in range(h)
        )

    def nearest_neighbor_cluster(cells, k):
        """Greedily partition cells into groups of size k using nearest-neighbor."""
        remaining = list(cells)
        groups = []
        while len(remaining) >= k:
            group = [remaining.pop(0)]
            while len(group) < k:
                gr = sum(r for r, c in group) / len(group)
                gc = sum(c for r, c in group) / len(group)
                best_dist = float('inf')
                best_idx = -1
                for i, (r, c) in enumerate(remaining):
                    d = (r - gr) ** 2 + (c - gc) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                if best_idx < 0:
                    break
                group.append(remaining.pop(best_idx))
            if len(group) == k:
                groups.append(group)
        return groups

    def get_candidates(grid, bg_val):
        """Return list of (pattern, num_copies) tuples."""
        rows, cols = len(grid), len(grid[0])
        # (pattern_tuple, num_copies) → track best per pattern
        pattern_to_copies = {}

        # Strategy 1: connected components
        visited = [[False] * cols for _ in range(rows)]
        components = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val and not visited[r][c]:
                    cells = []
                    queue = [(r, c)]
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.pop()
                        cells.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg_val:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                    components.append(cells)

        if components:
            normed = [normalize_cells(c, grid) for c in components]
            shape_counts = Counter(normed)
            for shape, count in shape_counts.items():
                if count >= 2:
                    pattern_to_copies[shape] = max(pattern_to_copies.get(shape, 0), count)

        # Strategy 2: spatial clustering by color
        by_color = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg_val:
                    if v not in by_color:
                        by_color[v] = []
                    by_color[v].append((r, c))

        for color, cells in by_color.items():
            n = len(cells)
            if n < 4:
                continue
            for k in range(2, n):
                if n % k != 0:
                    continue
                num_groups = n // k
                if num_groups < 2:
                    continue
                sorted_cells = sorted(cells)
                groups = nearest_neighbor_cluster(sorted_cells, k)
                if len(groups) != num_groups:
                    continue
                normed = [normalize_cells(g, grid) for g in groups]
                if len(set(normed)) == 1:
                    pattern_to_copies[normed[0]] = max(pattern_to_copies.get(normed[0], 0), num_groups)

        # Return list of (pattern_list, copies) sorted by:
        # 1. Has internal bg cells (non-trivial patterns first)
        # 2. Num copies (more = more likely signal)
        # 3. Area (larger = more informative)
        results = []
        for shape, copies in pattern_to_copies.items():
            pattern = [list(row) for row in shape]
            area = len(pattern) * len(pattern[0]) if pattern else 0
            has_bg = any(v == bg_val for row in pattern for v in row)
            results.append((pattern, copies, area, has_bg))
        # Sort: has_bg desc, copies desc, area desc
        results.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
        return results

    # Find which candidate pattern is consistent across all training examples
    # Build candidates from first training example, then verify on rest
    if not train:
        return None

    candidates0 = get_candidates(train[0]["input"], bg)
    if not candidates0:
        return None

    # Check all training examples have the correct output in candidates
    # (candidates are (pattern, copies, area, has_bg) tuples)
    for ex in train:
        ex_cands = get_candidates(ex["input"], bg)
        patterns_only = [c[0] for c in ex_cands]
        if ex["output"] not in patterns_only:
            return None

    # Lambda: return best candidate (most copies, then largest area)
    def apply_most_common(grid, bg_val, fn=get_candidates):
        cands = fn(grid, bg_val)
        if not cands:
            return None
        return cands[0][0]  # already sorted by (copies desc, area desc)

    fn = lambda g, b=bg: apply_most_common(g, b)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_cross_divider_quadrants(train):
    """
    Input has a cross divider (same color in one full row + one full col).
    Creates 4 quadrants, each with one object.
    Output = 2×2 arrangement of the 4 objects (side-by-side).
    """
    bg = _bg(train)

    def find_cross_divider(grid, bg_val):
        """Find (div_row, div_col, div_color) where one row and one col are entirely that color."""
        rows, cols = len(grid), len(grid[0])
        div_rows = {}
        div_cols = {}
        for r in range(rows):
            vals = set(grid[r])
            if len(vals) == 1 and vals != {bg_val}:
                div_rows[r] = list(vals)[0]
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1 and vals != {bg_val}:
                div_cols[c] = list(vals)[0]
        # Find matching row+col with same color
        for r, rv in div_rows.items():
            for c, cv in div_cols.items():
                if rv == cv:
                    return r, c, rv
        return None

    def extract_object_from_region(grid, r0, r1, c0, c1, bg_val, div_color):
        """Extract the single non-bg, non-divider object in the region as its bounding box."""
        cells = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                v = grid[r][c]
                if v != bg_val and v != div_color:
                    cells.append((r, c))
        if not cells:
            return None
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        sub = [[bg_val] * w for _ in range(h)]
        for r, c in cells:
            sub[r - min_r][c - min_c] = grid[r][c]
        return sub

    def apply_cross_quadrants(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        div = find_cross_divider(grid, bg_val)
        if div is None:
            return None
        div_r, div_c, div_color = div

        # Extract 4 quadrant objects
        tl = extract_object_from_region(grid, 0, div_r, 0, div_c, bg_val, div_color)
        tr = extract_object_from_region(grid, 0, div_r, div_c + 1, cols, bg_val, div_color)
        bl = extract_object_from_region(grid, div_r + 1, rows, 0, div_c, bg_val, div_color)
        br = extract_object_from_region(grid, div_r + 1, rows, div_c + 1, cols, bg_val, div_color)

        if tl is None or tr is None or bl is None or br is None:
            return None

        # All objects must have same dimensions (or we pad to match)
        # Find unified h and w for each half
        top_h = max(len(tl), len(tr))
        bot_h = max(len(bl), len(br))
        left_w = max(len(tl[0]) if tl else 0, len(bl[0]) if bl else 0)
        right_w = max(len(tr[0]) if tr else 0, len(br[0]) if br else 0)

        def pad(obj, h, w):
            result = [[bg_val] * w for _ in range(h)]
            for r in range(len(obj)):
                for c in range(len(obj[r])):
                    result[r][c] = obj[r][c]
            return result

        tl = pad(tl, top_h, left_w)
        tr = pad(tr, top_h, right_w)
        bl = pad(bl, bot_h, left_w)
        br = pad(br, bot_h, right_w)

        # Build output
        out_h = top_h + bot_h
        out_w = left_w + right_w
        result = [[bg_val] * out_w for _ in range(out_h)]
        for r in range(top_h):
            for c in range(left_w):
                result[r][c] = tl[r][c]
            for c in range(right_w):
                result[r][left_w + c] = tr[r][c]
        for r in range(bot_h):
            for c in range(left_w):
                result[top_h + r][c] = bl[r][c]
            for c in range(right_w):
                result[top_h + r][left_w + c] = br[r][c]
        return result

    # Verify all training examples
    results = [apply_cross_quadrants(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: apply_cross_quadrants(g, b)

    return None


def _try_color_objects_by_col_position(train):
    """Same as above but for column position."""
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    col_to_color = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        objects = _find_objects(inp, bg)
        for obj in objects:
            c_center = int(obj["center"][1])
            out_colors = Counter(out[r][c] for r, c in obj["cells"] if out[r][c] != bg)
            if not out_colors:
                continue
            out_color = out_colors.most_common(1)[0][0]

            if c_center in col_to_color:
                if col_to_color[c_center] != out_color:
                    return None
            else:
                col_to_color[c_center] = out_color

    if not col_to_color:
        return None

    def apply(grid, bg_val, c2c=col_to_color):
        objects = _find_objects(grid, bg_val)
        result = [row[:] for row in grid]
        for obj in objects:
            c_center = int(obj["center"][1])
            nearest = min(c2c.keys(), key=lambda k: abs(k - c_center))
            new_color = c2c[nearest]
            for r, c in obj["cells"]:
                result[r][c] = new_color
        return result

    if all(apply(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: apply(g, b)

    return None


def _try_cross_quadrant_select(train):
    """
    Input has a cross divider (one full row + one full col of same color).
    This creates 4 quadrants, each mostly filled with one dominant color.
    One quadrant has a cell that differs from the dominant color.
    Output = that unique quadrant.
    """
    def find_cross_and_unique_quadrant(grid):
        rows, cols = len(grid), len(grid[0])

        # Find rows that are entirely one color
        row_dividers = []
        for r in range(rows):
            vals = set(grid[r])
            if len(vals) == 1:
                row_dividers.append((r, list(vals)[0]))

        # Find cols that are entirely one color
        col_dividers = []
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1:
                col_dividers.append((c, list(vals)[0]))

        if not row_dividers or not col_dividers:
            return None

        # Find matching color between row+col dividers
        row_div_colors = {color for _, color in row_dividers}
        col_div_colors = {color for _, color in col_dividers}
        shared = row_div_colors & col_div_colors
        if not shared:
            return None

        div_color = next(iter(shared))
        div_row = next(r for r, c in row_dividers if c == div_color)
        div_col = next(c for c, color in col_dividers if color == div_color)

        # Extract the 4 quadrants
        q_tl = [grid[r][:div_col] for r in range(div_row)]
        q_tr = [grid[r][div_col+1:] for r in range(div_row)]
        q_bl = [grid[r][:div_col] for r in range(div_row+1, rows)]
        q_br = [grid[r][div_col+1:] for r in range(div_row+1, rows)]

        quadrants = [q for q in [q_tl, q_tr, q_bl, q_br] if q and q[0]]

        # For each quadrant, find dominant color + check for non-dominant cell
        unique_qs = []
        for q in quadrants:
            all_vals = [v for row in q for v in row]
            if not all_vals:
                continue
            dominant = Counter(all_vals).most_common(1)[0][0]
            # Has a cell different from both dominant and divider color?
            non_dominant = [v for v in all_vals if v != dominant and v != div_color]
            if non_dominant:
                unique_qs.append(q)

        if len(unique_qs) != 1:
            return None

        return unique_qs[0]

    results = [find_cross_and_unique_quadrant(ex["input"]) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return find_cross_and_unique_quadrant

    return None


def _try_color_specific_neighbors(train):
    """
    Each non-bg color in input has a specific neighbor pattern added around it.
    Example: color 1 → add color 7 in NSEW positions
             color 2 → add color 4 in diagonal positions
    Learns strictly: an offset (dr, dc) is a rule for color C only if EVERY
    occurrence of C in training has that neighbor position painted the same way.
    """
    bg = _bg(train)

    # Only applies to same-size transforms
    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    # For each (src_color, dr, dc), collect ALL observed output values
    # Key: (src_color, dr, dc) → list of output values at (r+dr, c+dc)
    # Only when output[r+dr][c+dc] was bg in input
    candidate_map = defaultdict(list)

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                src = inp[r][c]
                if src == bg:
                    continue
                # Only consider immediate neighbors (NSEW + diagonals, distance 1)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if inp[nr][nc] == bg:
                        # This position was bg in input — record what output has
                        out_v = out[nr][nc]
                        candidate_map[(src, dr, dc)].append(out_v)

    # Build strict neighbor map: only include (src, dr, dc) → paint_color if:
    # 1. ALL occurrences have the SAME non-bg output value
    # 2. At least one occurrence was actually painted (non-bg)
    neighbor_map = {}
    for (src, dr, dc), vals in candidate_map.items():
        unique_vals = set(vals)
        if len(unique_vals) == 1:
            v = list(unique_vals)[0]
            if v != bg:
                neighbor_map[(src, dr, dc)] = v

    if not neighbor_map:
        return None

    def apply_neighbors(grid, bg_val, nm=neighbor_map):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                src = grid[r][c]
                if src == bg_val:
                    continue
                for (s, dr, dc), paint_color in nm.items():
                    if s != src:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if grid[nr][nc] == bg_val:
                        result[nr][nc] = paint_color
        return result

    if all(apply_neighbors(ex["input"], bg) == ex["output"] for ex in train):
        return lambda g, b=bg: apply_neighbors(g, b)

    return None


def _try_row_col_paint(train):
    """
    Non-bg objects in the input define which rows/columns get fully painted.
    - If object width > height: paint ALL rows the object occupies with that color
    - If object height > width: paint ALL columns the object occupies with that color
    Row paints override column paints.

    Handles task 9344f635 and similar.
    """
    # Must be same-size input/output
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def apply_row_col_paint(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]

        # Find all non-bg objects
        objects = _find_objects(grid, bg_val)

        # Separate into row-painters and col-painters
        col_paints = {}   # col_idx → color
        row_paints = {}   # row_idx → color

        for obj in objects:
            color = obj["primary"]
            cells = obj["cells"]
            obj_rows = set(r for r, c in cells)
            obj_cols = set(c for r, c in cells)
            height = len(obj_rows)
            width = len(obj_cols)

            if width > height:
                # Row painter: paint each row the object occupies
                for r in obj_rows:
                    row_paints[r] = color
            elif height > width:
                # Column painter: paint each col the object occupies
                for c in obj_cols:
                    col_paints[c] = color
            # else: square or single cell — ambiguous, skip

        # Start with bg, apply col paints first
        for c, color in col_paints.items():
            for r in range(rows):
                result[r][c] = color

        # Then apply row paints (override cols)
        for r, color in row_paints.items():
            for c in range(cols):
                result[r][c] = color

        return result

    if all(apply_row_col_paint(ex["input"]) == ex["output"] for ex in train):
        return apply_row_col_paint

    return None


def _try_nearest_border_recolor(train):
    """
    Uniform rows or columns define borders with specific colors.
    Non-bg, non-border cells get recolored with the color of their nearest border.

    Handles 2204b7a8 and similar tasks.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def apply_nearest_border(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])

        # Find uniform rows (all same non-bg color)
        horiz_borders = {}
        for r in range(rows):
            vals = set(grid[r])
            if len(vals) == 1 and list(vals)[0] != bg_val:
                horiz_borders[r] = list(vals)[0]

        # Find uniform cols
        vert_borders = {}
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1 and list(vals)[0] != bg_val:
                vert_borders[c] = list(vals)[0]

        if not horiz_borders and not vert_borders:
            return grid  # No borders found

        border_colors = set(horiz_borders.values()) | set(vert_borders.values())
        result = [row[:] for row in grid]

        for r in range(rows):
            if r in horiz_borders:
                continue
            for c in range(cols):
                if c in vert_borders:
                    continue
                if grid[r][c] == bg_val or grid[r][c] in border_colors:
                    continue
                # Find nearest border
                min_dist = float('inf')
                nearest_color = None
                for br, bcolor in horiz_borders.items():
                    d = abs(r - br)
                    if d < min_dist:
                        min_dist = d
                        nearest_color = bcolor
                for bc, bcolor in vert_borders.items():
                    d = abs(c - bc)
                    if d < min_dist:
                        min_dist = d
                        nearest_color = bcolor
                if nearest_color is not None:
                    result[r][c] = nearest_color

        return result

    fn = lambda grid, b=bg: apply_nearest_border(grid, b)

    # Must not be identity
    if all(fn(ex["input"]) == ex["input"] for ex in train):
        return None

    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_fill_interior_from_external(train):
    """
    Fill a rectangle's interior with the color of a nearby small object,
    then remove the small object.

    Handles task 465b7d93.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def flood_outside(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        outside = [[False] * cols for _ in range(rows)]
        queue = []
        for r in range(rows):
            for c in [0, cols - 1]:
                if grid[r][c] == bg_val and not outside[r][c]:
                    outside[r][c] = True
                    queue.append((r, c))
        for c in range(cols):
            for r in [0, rows - 1]:
                if grid[r][c] == bg_val and not outside[r][c]:
                    outside[r][c] = True
                    queue.append((r, c))
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not outside[nr][nc] and grid[nr][nc] == bg_val:
                        outside[nr][nc] = True
                        queue.append((nr, nc))
        return outside

    def apply_fill_remove(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])
        outside = flood_outside(grid, bg_val)

        # Find interior bg cells
        interior = [(r, c) for r in range(rows) for c in range(cols)
                     if grid[r][c] == bg_val and not outside[r][c]]
        if not interior:
            return None

        # Find the outline color (adjacent to interior)
        outline_colors = set()
        for r, c in interior:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg_val:
                    outline_colors.add(grid[nr][nc])

        # Find external non-bg, non-outline cells (the small donor object)
        ext_colors = set()
        ext_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val and grid[r][c] not in outline_colors:
                    if outside[r][c] or (r, c) not in set(interior):
                        ext_colors.add(grid[r][c])
                        ext_cells.append((r, c))

        if len(ext_colors) != 1 or not ext_cells:
            return None
        fill_color = ext_colors.pop()

        result = [row[:] for row in grid]
        for r, c in interior:
            result[r][c] = fill_color
        for r, c in ext_cells:
            result[r][c] = bg_val
        return result

    results = [apply_fill_remove(ex["input"]) for ex in train]
    if any(r is None for r in results):
        return None
    if all(r == ex["input"] for r, ex in zip(results, train)):
        return None
    if all(r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: apply_fill_remove(g, b)
    return None


def _try_dot_to_block_line(train):
    """
    Isolated dots draw lines to the nearest edge of a rectangular block.

    A dot whose column falls within the block's col range draws a vertical
    line to the block. A dot whose row falls within the block's row range
    draws a horizontal line. Non-aligned dots stay unchanged.

    Handles task 2c608aff.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    def _grid_bg(grid):
        """Background = most common color in this grid."""
        flat = [c for row in grid for c in row]
        return Counter(flat).most_common(1)[0][0] if flat else 0

    def _find_rect_and_dots(grid, bg_val):
        """Find the rectangle (largest single-color blob) and dot cells."""
        rows, cols = len(grid), len(grid[0])
        color_cells = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg_val:
                    color_cells.setdefault(v, []).append((r, c))

        if len(color_cells) < 2:
            return None, None, None, None

        # Rectangle = largest group that forms a solid rectangle
        rect_color = None
        rect_bbox = None
        for color, cells in sorted(color_cells.items(), key=lambda x: -len(x[1])):
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            expected = (rmax - rmin + 1) * (cmax - cmin + 1)
            if len(cells) == expected and expected >= 4:
                rect_color = color
                rect_bbox = (rmin, rmax, cmin, cmax)
                break

        if rect_color is None:
            return None, None, None, None

        # Dots = all other non-bg cells
        dot_color = None
        dots = []
        for color, cells in color_cells.items():
            if color == rect_color:
                continue
            for r, c in cells:
                dots.append((r, c))
                if dot_color is None:
                    dot_color = color
                elif color != dot_color:
                    return None, None, None, None  # Multiple dot colors

        return rect_bbox, rect_color, dot_color, dots

    def apply_dot_line(grid):
        rows, cols = len(grid), len(grid[0])
        bg_val = _grid_bg(grid)
        info = _find_rect_and_dots(grid, bg_val)
        if info[0] is None:
            return grid
        rect_bbox, rect_color, dot_color, dots = info
        rmin, rmax, cmin, cmax = rect_bbox

        result = [row[:] for row in grid]
        for dr, dc in dots:
            in_col_range = cmin <= dc <= cmax
            in_row_range = rmin <= dr <= rmax

            if in_col_range and not in_row_range:
                # Draw vertical line from dot to nearest horizontal edge
                if dr < rmin:
                    for r in range(dr, rmin):
                        result[r][dc] = dot_color
                else:
                    for r in range(rmax + 1, dr + 1):
                        result[r][dc] = dot_color
            elif in_row_range and not in_col_range:
                # Draw horizontal line from dot to nearest vertical edge
                if dc < cmin:
                    for c in range(dc, cmin):
                        result[dr][c] = dot_color
                else:
                    for c in range(cmax + 1, dc + 1):
                        result[dr][c] = dot_color
            # Non-aligned or both-aligned → leave as-is

        return result

    if all(apply_dot_line(ex["input"]) == ex["input"] for ex in train):
        return None
    results = [apply_dot_line(ex["input"]) for ex in train]
    if any(r is None for r in results):
        return None
    if all(r == ex["output"] for r, ex in zip(results, train)):
        return apply_dot_line
    return None


def _try_project_dots_onto_block(train):
    """
    Isolated dots project their color onto the nearest edge cell of a
    rectangular block. Only the block's edge cell changes; the line
    between dot and block stays background.

    Handles task 1f642eb9.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    def _grid_bg_local(grid):
        flat = [c for row in grid for c in row]
        return Counter(flat).most_common(1)[0][0] if flat else 0

    def _find_rect_and_dots_local(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        color_cells = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg_val:
                    color_cells.setdefault(v, []).append((r, c))

        if len(color_cells) < 2:
            return None, None, None, None

        rect_color = None
        rect_bbox = None
        for color, cells in sorted(color_cells.items(), key=lambda x: -len(x[1])):
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            expected = (rmax - rmin + 1) * (cmax - cmin + 1)
            if len(cells) == expected and expected >= 4:
                rect_color = color
                rect_bbox = (rmin, rmax, cmin, cmax)
                break

        if rect_color is None:
            return None, None, None, None

        dots = []
        for color, cells in color_cells.items():
            if color == rect_color:
                continue
            for r, c in cells:
                dots.append((r, c, color))

        return rect_bbox, rect_color, dots, bg_val

    def apply_project(grid):
        rows, cols = len(grid), len(grid[0])
        bg_val = _grid_bg_local(grid)
        info = _find_rect_and_dots_local(grid, bg_val)
        if info[0] is None:
            return grid
        rect_bbox, rect_color, dots, _ = info
        rmin, rmax, cmin, cmax = rect_bbox

        result = [row[:] for row in grid]
        for dr, dc, color in dots:
            in_col_range = cmin <= dc <= cmax
            in_row_range = rmin <= dr <= rmax

            if in_col_range and not in_row_range:
                # Project vertically to nearest horizontal edge
                if dr < rmin:
                    result[rmin][dc] = color
                else:
                    result[rmax][dc] = color
            elif in_row_range and not in_col_range:
                # Project horizontally to nearest vertical edge
                if dc < cmin:
                    result[dr][cmin] = color
                else:
                    result[dr][cmax] = color

        return result

    if all(apply_project(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_project(ex["input"]) == ex["output"] for ex in train):
        return apply_project
    return None
