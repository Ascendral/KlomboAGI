"""
ARC Region-Based Strategies.

Handles tasks involving:
  - Enclosed region filling (bg cells inside colored borders → fill)
  - Region coloring by property (size, neighbor count, shape)
  - Connected component operations (keep largest, filter by size, etc.)
"""

from __future__ import annotations
from collections import Counter, defaultdict

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_region_rule(train: list[dict]) -> callable | None:
    """Try all region-based strategies."""
    for fn in [
        _try_fill_enclosed_by_color,
        _try_fill_by_region_size,
        _try_recolor_by_neighbor_count,
        _try_keep_largest_object,
        _try_keep_smallest_object,
        _try_remove_small_objects,
        _try_mark_enclosed_with_dot_color,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _find_objects(grid, bg_val):
    """Find connected components of non-bg cells."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] == bg_val:
                continue
            obj = []
            queue = [(r, c)]
            while queue:
                cr, cc = queue.pop(0)
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr][cc] or grid[cr][cc] == bg_val:
                    continue
                visited[cr][cc] = True
                obj.append((cr, cc, grid[cr][cc]))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cr + dr, cc + dc))
            if obj:
                objects.append(obj)
    return objects


def _find_bg_regions(grid, bg_val):
    """Find connected regions of bg cells, categorize as touching border or enclosed."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    regions = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] != bg_val:
                continue
            region = []
            queue = [(r, c)]
            touches_border = False
            adj_colors = Counter()

            while queue:
                cr, cc = queue.pop(0)
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    touches_border = True
                    continue
                if visited[cr][cc]:
                    continue
                if grid[cr][cc] != bg_val:
                    adj_colors[grid[cr][cc]] += 1
                    continue
                visited[cr][cc] = True
                region.append((cr, cc))
                if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                    touches_border = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cr + dr, cc + dc))

            if region:
                regions.append({
                    "cells": region,
                    "touches_border": touches_border,
                    "adj_colors": adj_colors,
                    "size": len(region),
                })
    return regions


def _try_fill_enclosed_by_color(train):
    """Fill enclosed bg regions with a color based on the enclosing border color."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Strategy: fill enclosed bg regions with the most common adjacent color
    def fill_enclosed(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        regions = _find_bg_regions(grid, bg_val)

        for region in regions:
            if not region["touches_border"] and region["adj_colors"]:
                fill_color = region["adj_colors"].most_common(1)[0][0]
                for r, c in region["cells"]:
                    result[r][c] = fill_color
        return result

    fn = lambda grid, bg_val=bg: fill_enclosed(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_fill_by_region_size(train):
    """Fill enclosed regions with different colors based on region size."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Learn: size → fill_color mapping from training
    size_to_color = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        regions = _find_bg_regions(inp, bg)
        for region in regions:
            if not region["touches_border"]:
                # What color was this region filled with?
                r, c = region["cells"][0]
                fill = out[r][c]
                if fill != bg:
                    sz = region["size"]
                    if sz in size_to_color and size_to_color[sz] != fill:
                        return None  # Inconsistent
                    size_to_color[sz] = fill

    if not size_to_color:
        return None

    def fill_by_size(grid, bg_val, stc):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        regions = _find_bg_regions(grid, bg_val)
        for region in regions:
            if not region["touches_border"]:
                sz = region["size"]
                if sz in stc:
                    for r, c in region["cells"]:
                        result[r][c] = stc[sz]
        return result

    fn = lambda grid, bg_val=bg, stc=size_to_color: fill_by_size(grid, bg_val, stc)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_recolor_by_neighbor_count(train):
    """Recolor non-bg cells based on how many non-bg neighbors they have."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Learn: neighbor_count → output_color
    count_to_color = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg:
                    continue
                nc = sum(
                    1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= r + dr < rows and 0 <= c + dc < cols
                    and inp[r + dr][c + dc] != bg
                )
                oc = out[r][c]
                if nc in count_to_color and count_to_color[nc] != oc:
                    return None
                count_to_color[nc] = oc

    if not count_to_color:
        return None
    # Must be non-trivial (not all same)
    if len(set(count_to_color.values())) <= 1:
        return None

    def recolor(grid, bg_val, ctc):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg_val:
                    continue
                nc = sum(
                    1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= r + dr < rows and 0 <= c + dc < cols
                    and grid[r + dr][c + dc] != bg_val
                )
                if nc in ctc:
                    result[r][c] = ctc[nc]
        return result

    fn = lambda grid, bg_val=bg, ctc=count_to_color: recolor(grid, bg_val, ctc)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_keep_largest_object(train):
    """Output = only the largest connected component from input."""
    bg = _bg(train)

    def keep_largest(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        if not objects:
            return grid

        largest = max(objects, key=len)
        largest_set = {(r, c) for r, c, v in largest}

        result = [[bg_val] * cols for _ in range(rows)]
        for r, c, v in largest:
            result[r][c] = v
        return result

    fn = lambda grid, bg_val=bg: keep_largest(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    # Also try: crop to bounding box of largest
    def keep_largest_cropped(grid, bg_val):
        objects = _find_objects(grid, bg_val)
        if not objects:
            return grid
        largest = max(objects, key=len)
        min_r = min(r for r, c, v in largest)
        max_r = max(r for r, c, v in largest)
        min_c = min(c for r, c, v in largest)
        max_c = max(c for r, c, v in largest)

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        result = [[bg_val] * w for _ in range(h)]
        for r, c, v in largest:
            result[r - min_r][c - min_c] = v
        return result

    fn2 = lambda grid, bg_val=bg: keep_largest_cropped(grid, bg_val)
    if all(fn2(ex["input"]) == ex["output"] for ex in train):
        return fn2

    return None


def _try_keep_smallest_object(train):
    """Output = only the smallest connected component."""
    bg = _bg(train)

    def keep_smallest(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        objects = _find_objects(grid, bg_val)
        if not objects:
            return grid

        smallest = min(objects, key=len)
        min_r = min(r for r, c, v in smallest)
        max_r = max(r for r, c, v in smallest)
        min_c = min(c for r, c, v in smallest)
        max_c = max(c for r, c, v in smallest)

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        result = [[bg_val] * w for _ in range(h)]
        for r, c, v in smallest:
            result[r - min_r][c - min_c] = v
        return result

    fn = lambda grid, bg_val=bg: keep_smallest(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_remove_small_objects(train):
    """Remove objects smaller than some threshold."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Find threshold: what sizes are kept vs removed?
    for threshold in range(1, 10):
        def remove_small(grid, bg_val, thresh):
            rows, cols = len(grid), len(grid[0])
            result = [[bg_val] * cols for _ in range(rows)]
            objects = _find_objects(grid, bg_val)
            for obj in objects:
                if len(obj) > thresh:
                    for r, c, v in obj:
                        result[r][c] = v
            return result

        fn = lambda grid, bg_val=bg, t=threshold: remove_small(grid, bg_val, t)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


def _try_mark_enclosed_with_dot_color(train):
    """
    Each enclosed region has a marker (single non-bg, non-wall cell).
    Fill the enclosed region with the marker's color.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Find the "wall" color (most common non-bg)
    non_bg = Counter()
    for ex in train:
        for row in ex["input"]:
            for v in row:
                if v != bg:
                    non_bg[v] += 1

    if not non_bg:
        return None

    wall_color = non_bg.most_common(1)[0][0]

    def fill_with_marker(grid, bg_val, wall):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Find enclosed bg+marker regions
        visited = [[False] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or grid[r][c] == wall:
                    continue

                # BFS to find the region
                region_cells = []
                marker_color = None
                queue = [(r, c)]
                touches_border = False

                while queue:
                    cr, cc = queue.pop(0)
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        touches_border = True
                        continue
                    if visited[cr][cc] or grid[cr][cc] == wall:
                        continue
                    visited[cr][cc] = True
                    region_cells.append((cr, cc))
                    if grid[cr][cc] != bg_val and grid[cr][cc] != wall:
                        marker_color = grid[cr][cc]
                    if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                        touches_border = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        queue.append((cr + dr, cc + dc))

                if not touches_border and marker_color is not None:
                    for rr, cc in region_cells:
                        if grid[rr][cc] == bg_val:
                            result[rr][cc] = marker_color

        return result

    fn = lambda grid, bg_val=bg, w=wall_color: fill_with_marker(grid, bg_val, w)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None
