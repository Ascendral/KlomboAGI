"""
ARC Legend/Key Strategies.

Handles tasks where a small section of the input encodes the transformation:
  - Color mapping legend (2x2 or NxN key in corner)
  - Template that defines the transformation pattern
  - Lookup table embedded in the grid
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_legend_rule(train: list[dict]) -> callable | None:
    """Try legend/key-based strategies."""
    for fn in [
        _try_corner_color_legend,
        _try_recolor_by_object_property,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_corner_color_legend(train):
    """
    A small NxN section in a corner of the grid defines a color rotation.
    Each color in the legend maps to another color.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # Try extracting a legend from each corner
    for legend_size in [2, 3]:
        for corner in ['tl', 'tr', 'bl', 'br']:
            # Extract legend colors and build mapping
            mapping = {}
            consistent = True

            for ex in train:
                inp = ex["input"]
                rows, cols = len(inp), len(inp[0])

                if corner == 'tl':
                    legend = [inp[r][:legend_size] for r in range(legend_size)]
                    rest_inp = inp  # We'll check the rest of the grid
                elif corner == 'tr':
                    legend = [inp[r][cols-legend_size:] for r in range(legend_size)]
                    rest_inp = inp
                elif corner == 'bl':
                    legend = [inp[rows-legend_size+r][:legend_size] for r in range(legend_size)]
                    rest_inp = inp
                elif corner == 'br':
                    legend = [inp[rows-legend_size+r][cols-legend_size:] for r in range(legend_size)]
                    rest_inp = inp

                # The legend should have unique non-bg colors
                legend_colors = set()
                for row in legend:
                    for v in row:
                        if v != bg:
                            legend_colors.add(v)

                if len(legend_colors) < 2:
                    consistent = False
                    break

                # Check if the same colors appear in the rest of the grid
                # and get mapped to different colors in the output
                out = ex["output"]
                for r in range(rows):
                    for c in range(cols):
                        # Skip legend area
                        if corner == 'tl' and r < legend_size and c < legend_size:
                            continue
                        elif corner == 'tr' and r < legend_size and c >= cols - legend_size:
                            continue
                        elif corner == 'bl' and r >= rows - legend_size and c < legend_size:
                            continue
                        elif corner == 'br' and r >= rows - legend_size and c >= cols - legend_size:
                            continue

                        if inp[r][c] != bg and out[r][c] != inp[r][c]:
                            ic = inp[r][c]
                            oc = out[r][c]
                            if ic in mapping:
                                if mapping[ic] != oc:
                                    consistent = False
                                    break
                            else:
                                mapping[ic] = oc
                    if not consistent:
                        break
                if not consistent:
                    break

            if not consistent or not mapping:
                continue

            # Verify: the legend itself defines the mapping
            # The legend positions should show each color and its target
            # Build: color at position (r, c) in legend → color at same position in output legend
            legend_mapping = {}
            for ex in train:
                inp, out = ex["input"], ex["output"]
                rows, cols = len(inp), len(inp[0])
                for r in range(legend_size):
                    for c in range(legend_size):
                        if corner == 'tl':
                            lr, lc = r, c
                        elif corner == 'tr':
                            lr, lc = r, cols - legend_size + c
                        elif corner == 'bl':
                            lr, lc = rows - legend_size + r, c
                        elif corner == 'br':
                            lr, lc = rows - legend_size + r, cols - legend_size + c

                        iv = inp[lr][lc]
                        ov = out[lr][lc]
                        if iv != bg:
                            if iv in legend_mapping:
                                if legend_mapping[iv] != ov:
                                    consistent = False
                                    break
                            else:
                                legend_mapping[iv] = ov
                    if not consistent:
                        break
                if not consistent:
                    break

            if not consistent:
                continue

            # Merge legend_mapping with grid mapping
            full_mapping = {**legend_mapping, **mapping}

            def apply_mapping(grid, fm=full_mapping, bg_val=bg):
                return [[fm.get(v, v) for v in row] for row in grid]

            if all(apply_mapping(ex["input"]) == ex["output"] for ex in train):
                return apply_mapping

    return None


def _try_recolor_by_object_property(train):
    """
    Each connected object is recolored based on its size, number of unique
    colors, or some other scalar property.
    """
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def find_objects(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
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
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        queue.append((cr+dr, cc+dc))
                if obj:
                    objects.append(obj)
        return objects

    # Learn: object_size → output_color
    size_to_color = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        objects = find_objects(inp, bg)
        for obj in objects:
            sz = len(obj)
            # What color does this object become in the output?
            r0, c0, _ = obj[0]
            out_color = out[r0][c0]
            if sz in size_to_color:
                if size_to_color[sz] != out_color:
                    size_to_color = None
                    break
            else:
                size_to_color[sz] = out_color
        if size_to_color is None:
            break

    if size_to_color and len(set(size_to_color.values())) > 1:
        def recolor_by_size(grid, bg_val=bg, stc=size_to_color):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            objects = find_objects(grid, bg_val)
            for obj in objects:
                sz = len(obj)
                if sz in stc:
                    for r, c, v in obj:
                        result[r][c] = stc[sz]
            return result

        if all(recolor_by_size(ex["input"]) == ex["output"] for ex in train):
            return recolor_by_size

    return None
