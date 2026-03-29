"""
ARC DSL v2 — Composable Grid Primitives + Program Synthesis.

Instead of 106 hand-coded strategies, define ~30 atomic operations
and a search engine that composes them into programs.

A program is a sequence: [op1, op2, op3, ...]
Each op transforms a grid → grid.
The synthesizer finds the sequence that maps training inputs to outputs.

Primitives fall into categories:
  DETECT  — find objects, colors, regions, patterns
  FILTER  — select specific objects/cells by property
  TRANSFORM — move, rotate, reflect, recolor, fill
  COMBINE — overlay, merge, tile

The key insight from ARC research: programs of 2-5 composed primitives
solve the vast majority of tasks. We don't need infinite depth.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Callable

Grid = list[list[int]]


# ═══════════════════════════════════════════════════════════════════
# ATOMIC PRIMITIVES — each takes a grid, returns a grid
# ═══════════════════════════════════════════════════════════════════

def get_bg(grid: Grid) -> int:
    """Most common color = background."""
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def grid_copy(grid: Grid) -> Grid:
    return [row[:] for row in grid]


# ── Color Operations ──

def recolor(grid: Grid, from_c: int, to_c: int) -> Grid:
    """Replace all cells of from_c with to_c."""
    return [[to_c if c == from_c else c for c in row] for row in grid]


def swap_colors(grid: Grid, c1: int, c2: int) -> Grid:
    """Swap two colors."""
    return [[c2 if c == c1 else c1 if c == c2 else c for c in row] for row in grid]


def map_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    """Apply a color mapping."""
    return [[mapping.get(c, c) for c in row] for row in grid]


def most_common_color(grid: Grid, exclude_bg: bool = True) -> int:
    """Most common non-bg color."""
    bg = get_bg(grid) if exclude_bg else -1
    flat = [c for row in grid for c in row if c != bg]
    return Counter(flat).most_common(1)[0][0] if flat else 0


# ── Spatial Transforms ──

def rotate_90(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows - 1 - c][r] for c in range(rows)] for r in range(cols)]


def rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    return [[grid[c][cols - 1 - r] for c in range(rows)] for r in range(cols)]


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(rows)] for c in range(cols)]


# ── Region Operations ──

def flood_fill_enclosed(grid: Grid) -> Grid:
    """Fill enclosed bg regions with the enclosing color."""
    rows, cols = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = grid_copy(grid)
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] != bg:
                continue
            region = []
            queue = [(r, c)]
            touches_border = False
            adj_colors = Counter()

            while queue:
                cr, cc = queue.pop(0)
                if visited[cr][cc]:
                    continue
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    touches_border = True
                    continue
                if grid[cr][cc] != bg:
                    adj_colors[grid[cr][cc]] += 1
                    continue
                visited[cr][cc] = True
                region.append((cr, cc))
                if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                    touches_border = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        queue.append((nr, nc))

            if not touches_border and adj_colors:
                fill = adj_colors.most_common(1)[0][0]
                for rr, cc in region:
                    result[rr][cc] = fill
    return result


def gravity_down(grid: Grid) -> Grid:
    """Drop all non-bg cells down (gravity)."""
    rows, cols = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg] * cols for _ in range(rows)]
    for c in range(cols):
        non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
        for i, val in enumerate(non_bg):
            result[rows - len(non_bg) + i][c] = val
    return result


def gravity_up(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg] * cols for _ in range(rows)]
    for c in range(cols):
        non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
        for i, val in enumerate(non_bg):
            result[i][c] = val
    return result


def gravity_left(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
        for i, val in enumerate(non_bg):
            result[r][i] = val
    return result


def gravity_right(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
        for i, val in enumerate(non_bg):
            result[r][cols - len(non_bg) + i] = val
    return result


# ── Extraction ──

def crop_to_content(grid: Grid) -> Grid:
    """Crop to bounding box of non-bg content."""
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    min_r = min_c = float('inf')
    max_r = max_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return grid
    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


def extract_top_half(grid: Grid) -> Grid:
    return grid[:len(grid) // 2]


def extract_bottom_half(grid: Grid) -> Grid:
    return grid[len(grid) // 2:]


def extract_left_half(grid: Grid) -> Grid:
    mid = len(grid[0]) // 2
    return [row[:mid] for row in grid]


def extract_right_half(grid: Grid) -> Grid:
    mid = len(grid[0]) // 2
    return [row[mid:] for row in grid]


# ── Tiling ──

def tile_2x2(grid: Grid) -> Grid:
    """Tile the grid into a 2x2 arrangement."""
    rows = len(grid)
    cols = len(grid[0])
    result = []
    for r in range(rows * 2):
        row = []
        for c in range(cols * 2):
            row.append(grid[r % rows][c % cols])
        result.append(row)
    return result


def tile_3x3(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    result = []
    for r in range(rows * 3):
        row = []
        for c in range(cols * 3):
            row.append(grid[r % rows][c % cols])
        result.append(row)
    return result


# ── Overlay ──

def overlay(base: Grid, layer: Grid, bg: int = 0) -> Grid:
    """Overlay layer on base — non-bg cells from layer replace base cells."""
    result = grid_copy(base)
    for r in range(min(len(base), len(layer))):
        for c in range(min(len(base[0]), len(layer[0]))):
            if layer[r][c] != bg:
                result[r][c] = layer[r][c]
    return result


def xor_grids(a: Grid, b: Grid) -> Grid:
    """XOR two grids — cells that differ get marked."""
    rows = min(len(a), len(b))
    cols = min(len(a[0]), len(b[0]))
    return [[a[r][c] if a[r][c] == b[r][c] else max(a[r][c], b[r][c])
             for c in range(cols)] for r in range(rows)]


# ── Object-Level Operations ──

def find_objects(grid: Grid) -> list[dict]:
    """Find connected non-bg regions. Returns list of {cells, color, bbox}."""
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] == bg:
                continue
            # BFS to find connected region of same color
            color = grid[r][c]
            cells = []
            queue = [(r, c)]
            while queue:
                cr, cc = queue.pop(0)
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr][cc] or grid[cr][cc] != color:
                    continue
                visited[cr][cc] = True
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cr + dr, cc + dc))

            if cells:
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                objects.append({
                    "cells": cells,
                    "color": color,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "size": len(cells),
                })
    return objects


def extract_largest_object(grid: Grid) -> Grid:
    """Extract the largest non-bg object as a cropped grid."""
    objects = find_objects(grid)
    if not objects:
        return grid
    largest = max(objects, key=lambda o: o["size"])
    r1, c1, r2, c2 = largest["bbox"]
    bg = get_bg(grid)
    result = [[bg] * (c2 - c1 + 1) for _ in range(r2 - r1 + 1)]
    for r, c in largest["cells"]:
        result[r - r1][c - c1] = grid[r][c]
    return result


def extract_smallest_object(grid: Grid) -> Grid:
    """Extract the smallest non-bg object as a cropped grid."""
    objects = find_objects(grid)
    if not objects:
        return grid
    smallest = min(objects, key=lambda o: o["size"])
    r1, c1, r2, c2 = smallest["bbox"]
    bg = get_bg(grid)
    result = [[bg] * (c2 - c1 + 1) for _ in range(r2 - r1 + 1)]
    for r, c in smallest["cells"]:
        result[r - r1][c - c1] = grid[r][c]
    return result


def keep_most_common_color_objects(grid: Grid) -> Grid:
    """Keep only objects of the most common non-bg color, clear rest."""
    bg = get_bg(grid)
    objects = find_objects(grid)
    if not objects:
        return grid
    color_counts = Counter(o["color"] for o in objects)
    keep_color = color_counts.most_common(1)[0][0]
    result = [[bg] * len(grid[0]) for _ in range(len(grid))]
    for obj in objects:
        if obj["color"] == keep_color:
            for r, c in obj["cells"]:
                result[r][c] = grid[r][c]
    return result


def remove_smallest_objects(grid: Grid) -> Grid:
    """Remove objects smaller than the median size."""
    bg = get_bg(grid)
    objects = find_objects(grid)
    if len(objects) < 2:
        return grid
    sizes = sorted(o["size"] for o in objects)
    median = sizes[len(sizes) // 2]
    result = grid_copy(grid)
    for obj in objects:
        if obj["size"] < median:
            for r, c in obj["cells"]:
                result[r][c] = bg
    return result


def extract_unique_subgrid(grid: Grid) -> Grid:
    """
    If the grid has a repeating pattern with one unique section, extract it.
    Common ARC pattern: grid is tiled but one tile is different — extract that tile.
    """
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])

    # Try different sub-grid sizes
    for bh in range(2, rows // 2 + 1):
        if rows % bh != 0:
            continue
        for bw in range(2, cols // 2 + 1):
            if cols % bw != 0:
                continue
            # Extract all blocks
            blocks = []
            for br in range(0, rows, bh):
                for bc in range(0, cols, bw):
                    block = tuple(
                        tuple(grid[r][c] for c in range(bc, bc + bw))
                        for r in range(br, br + bh)
                    )
                    blocks.append((br, bc, block))

            # Find the unique block (appears only once)
            block_counts = Counter(b[2] for b in blocks)
            unique = [b for b in blocks if block_counts[b[2]] == 1]
            if len(unique) == 1:
                br, bc, block = unique[0]
                return [list(row) for row in block]

    return grid  # No unique sub-grid found


def extract_non_bg_bbox(grid: Grid) -> Grid:
    """Extract the bounding box of ALL non-bg content."""
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    min_r = min_c = float('inf')
    max_r = max_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return grid
    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


def apply_majority_neighbor(grid: Grid) -> Grid:
    """
    For each bg cell, if majority of 4-neighbors are same color, adopt that color.
    Single pass — good for "spreading" or "smoothing" patterns.
    """
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = grid_copy(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                continue
            neighbors = Counter()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
                    neighbors[grid[nr][nc]] += 1
            if neighbors:
                top_color, top_count = neighbors.most_common(1)[0]
                if top_count >= 2:
                    result[r][c] = top_color
    return result


def outline_all_objects(grid: Grid) -> Grid:
    """
    For each non-bg cell that borders a bg cell, keep it.
    Interior cells (all 4 neighbors are non-bg) become bg.
    Produces outlines of objects.
    """
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg:
                continue
            # Check if this cell borders bg
            is_border = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc] == bg:
                    is_border = True
                    break
            if is_border:
                result[r][c] = grid[r][c]
    return result


def fill_bg_between_colors(grid: Grid) -> Grid:
    """
    On each row and column, fill bg cells between two same-colored markers.
    Common pattern: two dots of color X → fill the line between them with X.
    """
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = grid_copy(grid)

    # Fill rows
    for r in range(rows):
        colors_in_row = {}
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in colors_in_row:
                    colors_in_row[color] = [c]
                else:
                    colors_in_row[color].append(c)
        for color, positions in colors_in_row.items():
            if len(positions) >= 2:
                for c in range(min(positions), max(positions) + 1):
                    if result[r][c] == bg:
                        result[r][c] = color

    # Fill columns
    for c in range(cols):
        colors_in_col = {}
        for r in range(rows):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in colors_in_col:
                    colors_in_col[color] = [r]
                else:
                    colors_in_col[color].append(r)
        for color, positions in colors_in_col.items():
            if len(positions) >= 2:
                for r in range(min(positions), max(positions) + 1):
                    if result[r][c] == bg:
                        result[r][c] = color

    return result


def split_by_horizontal_divider(grid: Grid) -> list[Grid]:
    """Split grid at horizontal divider lines (rows of single color)."""
    bg = get_bg(grid)
    rows = len(grid)
    dividers = []
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            dividers.append(r)
    if not dividers:
        return [grid]
    # Split between dividers
    parts = []
    prev = 0
    for d in dividers:
        if d > prev:
            parts.append(grid[prev:d])
        prev = d + 1
    if prev < rows:
        parts.append(grid[prev:])
    return [p for p in parts if p]


def split_by_vertical_divider(grid: Grid) -> list[Grid]:
    """Split grid at vertical divider lines (columns of single color)."""
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    dividers = []
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1 and grid[0][c] != bg:
            dividers.append(c)
    if not dividers:
        return [grid]
    parts = []
    prev = 0
    for d in dividers:
        if d > prev:
            parts.append([row[prev:d] for row in grid])
        prev = d + 1
    if prev < cols:
        parts.append([row[prev:] for row in grid])
    return [p for p in parts if p]


def xor_split_halves_h(grid: Grid) -> Grid:
    """Split horizontally, XOR the two halves. Highlights differences."""
    halves = split_by_horizontal_divider(grid)
    if len(halves) < 2:
        # Try simple split at middle
        mid = len(grid) // 2
        halves = [grid[:mid], grid[mid:]]
    if len(halves) >= 2 and len(halves[0]) == len(halves[1]):
        return xor_grids(halves[0], halves[1])
    return grid


def xor_split_halves_v(grid: Grid) -> Grid:
    """Split vertically, XOR the two halves."""
    halves = split_by_vertical_divider(grid)
    if len(halves) < 2:
        mid = len(grid[0]) // 2
        halves = [[row[:mid] for row in grid], [row[mid:] for row in grid]]
    if len(halves) >= 2:
        h0, h1 = halves[0], halves[1]
        if len(h0) == len(h1) and len(h0[0]) == len(h1[0]):
            return xor_grids(h0, h1)
    return grid


def and_split_halves_h(grid: Grid) -> Grid:
    """Split horizontally, AND (keep cells where both halves agree)."""
    mid = len(grid) // 2
    top, bot = grid[:mid], grid[mid:]
    if len(top) != len(bot):
        return grid
    bg = get_bg(grid)
    rows, cols = len(top), len(top[0])
    result = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(min(len(top[0]), len(bot[0]))):
            if top[r][c] == bot[r][c] and top[r][c] != bg:
                result[r][c] = top[r][c]
    return result


def and_split_halves_v(grid: Grid) -> Grid:
    """Split vertically, AND (keep cells where both halves agree)."""
    cols = len(grid[0])
    mid = cols // 2
    left = [row[:mid] for row in grid]
    right = [row[mid:] for row in grid]
    if len(left[0]) != len(right[0]):
        return grid
    bg = get_bg(grid)
    rows = len(grid)
    result = [[bg] * mid for _ in range(rows)]
    for r in range(rows):
        for c in range(mid):
            if left[r][c] == right[r][c] and left[r][c] != bg:
                result[r][c] = left[r][c]
    return result


def or_split_halves_h(grid: Grid) -> Grid:
    """Split horizontally, OR (keep non-bg from either half)."""
    mid = len(grid) // 2
    top, bot = grid[:mid], grid[mid:]
    if len(top) != len(bot):
        return grid
    bg = get_bg(grid)
    return overlay(top, bot, bg)


def or_split_halves_v(grid: Grid) -> Grid:
    """Split vertically, OR (keep non-bg from either half)."""
    cols = len(grid[0])
    mid = cols // 2
    left = [row[:mid] for row in grid]
    right = [row[mid:] for row in grid]
    if len(left[0]) != len(right[0]):
        return grid
    bg = get_bg(grid)
    return overlay(left, right, bg)


def sort_objects_by_size(grid: Grid) -> Grid:
    """Rearrange objects left-to-right by size (smallest first)."""
    bg = get_bg(grid)
    objects = find_objects(grid)
    if len(objects) < 2:
        return grid
    objects.sort(key=lambda o: o["size"])
    # Place objects side by side
    result_rows = max(o["bbox"][2] - o["bbox"][0] + 1 for o in objects)
    result = []
    col_offset = 0
    for obj in objects:
        r1, c1, r2, c2 = obj["bbox"]
        w = c2 - c1 + 1
        h = r2 - r1 + 1
        for r, c in obj["cells"]:
            while len(result) <= r - r1:
                result.append([bg] * (col_offset + w + 1))
            while len(result[r - r1]) <= col_offset + c - c1:
                result[r - r1].append(bg)
            result[r - r1][col_offset + c - c1] = grid[r][c]
        col_offset += w + 1
    # Pad to rectangular
    max_cols = max(len(row) for row in result) if result else 0
    for row in result:
        while len(row) < max_cols:
            row.append(bg)
    return result if result else grid


# ═══════════════════════════════════════════════════════════════════
# THE DSL — named operations with metadata
# ═══════════════════════════════════════════════════════════════════

class Op:
    """A named grid operation."""
    def __init__(self, name: str, fn: Callable, preserves_size: bool = True,
                 category: str = "transform"):
        self.name = name
        self.fn = fn
        self.preserves_size = preserves_size
        self.category = category

    def apply(self, grid: Grid) -> Grid | None:
        try:
            return self.fn(grid)
        except Exception:
            return None

    def __repr__(self):
        return self.name


# All available operations
ALL_OPS = [
    # Spatial
    Op("rotate_90", rotate_90, True, "spatial"),
    Op("rotate_180", rotate_180, True, "spatial"),
    Op("rotate_270", rotate_270, True, "spatial"),
    Op("flip_h", flip_h, True, "spatial"),
    Op("flip_v", flip_v, True, "spatial"),
    Op("transpose", transpose, False, "spatial"),

    # Gravity
    Op("gravity_down", gravity_down, True, "gravity"),
    Op("gravity_up", gravity_up, True, "gravity"),
    Op("gravity_left", gravity_left, True, "gravity"),
    Op("gravity_right", gravity_right, True, "gravity"),

    # Region
    Op("flood_fill", flood_fill_enclosed, True, "region"),

    # Extraction
    Op("crop", crop_to_content, False, "extract"),
    Op("top_half", extract_top_half, False, "extract"),
    Op("bottom_half", extract_bottom_half, False, "extract"),
    Op("left_half", extract_left_half, False, "extract"),
    Op("right_half", extract_right_half, False, "extract"),

    # Tiling
    Op("tile_2x2", tile_2x2, False, "tile"),
    Op("tile_3x3", tile_3x3, False, "tile"),

    # Object-level
    Op("largest_object", extract_largest_object, False, "object"),
    Op("smallest_object", extract_smallest_object, False, "object"),
    Op("keep_common_color", keep_most_common_color_objects, True, "object"),
    Op("remove_small", remove_smallest_objects, True, "object"),

    # Sub-grid extraction
    Op("unique_subgrid", extract_unique_subgrid, False, "extract"),
    Op("non_bg_bbox", extract_non_bg_bbox, False, "extract"),

    # Per-cell neighborhood rules
    Op("majority_neighbor", apply_majority_neighbor, True, "cell_rule"),
    Op("outline_objects", outline_all_objects, True, "cell_rule"),
    Op("fill_bg_between", fill_bg_between_colors, True, "cell_rule"),

    # Grid splitting + combining halves
    Op("xor_halves_h", xor_split_halves_h, False, "split"),
    Op("xor_halves_v", xor_split_halves_v, False, "split"),
    Op("and_halves_h", and_split_halves_h, False, "split"),
    Op("and_halves_v", and_split_halves_v, False, "split"),
    Op("or_halves_h", or_split_halves_h, False, "split"),
    Op("or_halves_v", or_split_halves_v, False, "split"),
]


# ═══════════════════════════════════════════════════════════════════
# PROGRAM SYNTHESIZER — search over compositions
# ═══════════════════════════════════════════════════════════════════

class Program:
    """A sequence of operations."""
    def __init__(self, ops: list[Op]):
        self.ops = ops

    def execute(self, grid: Grid) -> Grid | None:
        result = grid_copy(grid)
        for op in self.ops:
            result = op.apply(result)
            if result is None:
                return None
        return result

    def __repr__(self):
        return " → ".join(op.name for op in self.ops)


def synthesize(train: list[dict], test_input: Grid,
               max_depth: int = 3, timeout_ms: int = 5000) -> Grid | None:
    """
    Find a program (sequence of ops) that maps all training inputs to outputs.

    Uses iterative deepening: try depth 1, then 2, then 3.
    Cross-validates against ALL training examples.
    Returns the test output if a valid program is found.
    """
    import time
    start = time.time()

    # Determine if output size differs from input
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )

    # Filter ops by size compatibility
    if same_size:
        ops = [op for op in ALL_OPS if op.preserves_size]
    else:
        ops = ALL_OPS

    # Also generate color-specific ops from the training data
    color_ops = _generate_color_ops(train)
    ops = ops + color_ops

    for depth in range(1, max_depth + 1):
        if (time.time() - start) * 1000 > timeout_ms:
            break
        result = _search_depth(train, test_input, ops, depth, start, timeout_ms)
        if result is not None:
            return result

    return None


def _generate_color_ops(train: list[dict]) -> list[Op]:
    """Generate color-swap operations specific to this task."""
    in_colors = set()
    out_colors = set()
    for ex in train:
        for row in ex["input"]:
            in_colors.update(row)
        for row in ex["output"]:
            out_colors.update(row)

    ops = []
    all_colors = in_colors | out_colors
    for c1 in all_colors:
        for c2 in all_colors:
            if c1 < c2:  # avoid duplicates
                ops.append(Op(
                    f"swap_{c1}_{c2}",
                    lambda g, a=c1, b=c2: swap_colors(g, a, b),
                    True, "color"
                ))
    # Color mappings: find consistent per-cell color change
    if len(train) >= 1:
        ex = train[0]
        inp, out = ex["input"], ex["output"]
        if len(inp) == len(out) and len(inp[0]) == len(out[0]):
            mapping = {}
            consistent = True
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    ic, oc = inp[r][c], out[r][c]
                    if ic != oc:
                        if ic in mapping and mapping[ic] != oc:
                            consistent = False
                            break
                        mapping[ic] = oc
                if not consistent:
                    break
            if consistent and mapping:
                ops.append(Op(
                    f"colormap_{mapping}",
                    lambda g, m=mapping: map_colors(g, m),
                    True, "color"
                ))

    return ops


def _search_depth(train: list[dict], test_input: Grid,
                  ops: list[Op], depth: int,
                  start_time: float, timeout_ms: int) -> Grid | None:
    """Search all programs of exactly `depth` operations."""
    import time

    if depth == 1:
        for op in ops:
            if (time.time() - start_time) * 1000 > timeout_ms:
                return None
            # Check if this single op works for ALL training examples
            valid = True
            for ex in train:
                result = op.apply(ex["input"])
                if result != ex["output"]:
                    valid = False
                    break
            if valid:
                return op.apply(test_input)
        return None

    if depth == 2:
        for op1 in ops:
            if (time.time() - start_time) * 1000 > timeout_ms:
                return None
            for op2 in ops:
                if op1.name == op2.name:
                    continue  # skip identity compositions
                valid = True
                for ex in train:
                    r1 = op1.apply(ex["input"])
                    if r1 is None:
                        valid = False
                        break
                    r2 = op2.apply(r1)
                    if r2 != ex["output"]:
                        valid = False
                        break
                if valid:
                    r1 = op1.apply(test_input)
                    return op2.apply(r1) if r1 else None
        return None

    if depth == 3:
        for op1 in ops:
            if (time.time() - start_time) * 1000 > timeout_ms:
                return None
            # Pre-compute op1 results for all examples
            r1s = []
            skip = False
            for ex in train:
                r1 = op1.apply(ex["input"])
                if r1 is None:
                    skip = True
                    break
                r1s.append(r1)
            if skip:
                continue

            for op2 in ops:
                if op1.name == op2.name:
                    continue
                r2s = []
                skip2 = False
                for r1 in r1s:
                    r2 = op2.apply(r1)
                    if r2 is None:
                        skip2 = True
                        break
                    r2s.append(r2)
                if skip2:
                    continue

                for op3 in ops:
                    if (time.time() - start_time) * 1000 > timeout_ms:
                        return None
                    if op2.name == op3.name:
                        continue
                    valid = True
                    for i, r2 in enumerate(r2s):
                        r3 = op3.apply(r2)
                        if r3 != train[i]["output"]:
                            valid = False
                            break
                    if valid:
                        r1 = op1.apply(test_input)
                        r2 = op2.apply(r1) if r1 else None
                        return op3.apply(r2) if r2 else None
        return None

    return None
