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
