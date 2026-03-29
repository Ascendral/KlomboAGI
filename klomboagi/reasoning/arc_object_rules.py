"""
ARC Object-Level Rule Learner — learn transformations per-object.

Many ARC tasks work at the object level:
  - "Move each object to the nearest corner"
  - "Recolor each object based on its size"
  - "Keep only the largest object"
  - "Sort objects by position"
  - "Replace each object with a scaled version"

This module detects objects, extracts features, and learns per-object rules.
"""

from __future__ import annotations

import numpy as np
from collections import Counter

Grid = list[list[int]]


def _get_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def _find_objects(grid, bg):
    """Find connected non-bg regions using BFS."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] == bg:
                continue
            cells = []
            color = grid[r][c]
            queue = [(r, c)]
            while queue:
                cr, cc = queue.pop(0)
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr][cc] or grid[cr][cc] == bg:
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

                # Extract the object's sub-grid
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                sub_grid = [[bg] * w for _ in range(h)]
                for cr, cc in cells:
                    sub_grid[cr - min_r][cc - min_c] = grid[cr][cc]

                objects.append({
                    "cells": cells,
                    "colors": set(grid[cr][cc] for cr, cc in cells),
                    "primary_color": Counter(grid[cr][cc] for cr, cc in cells).most_common(1)[0][0],
                    "bbox": (min_r, min_c, max_r, max_c),
                    "size": len(cells),
                    "h": h, "w": w,
                    "sub_grid": sub_grid,
                    "center": ((min_r + max_r) / 2, (min_c + max_c) / 2),
                })
    return objects


def learn_object_rule(train: list[dict]) -> callable | None:
    """Try to learn an object-level transformation rule."""

    # Strategy 1: Output = largest object extracted
    rule = _try_extract_by_size(train, "largest")
    if rule:
        return rule

    # Strategy 2: Output = smallest object extracted
    rule = _try_extract_by_size(train, "smallest")
    if rule:
        return rule

    # Strategy 3: Output = object with unique color
    rule = _try_extract_unique_color_object(train)
    if rule:
        return rule

    # Strategy 4: Output = object with most colors
    rule = _try_extract_most_colors_object(train)
    if rule:
        return rule

    # Strategy 5: Recolor objects based on size
    rule = _try_recolor_by_size(train)
    if rule:
        return rule

    # Strategy 6: Keep only objects matching a size threshold
    rule = _try_filter_by_size(train)
    if rule:
        return rule

    return None


def _try_extract_by_size(train, which="largest"):
    """Output = extract the largest/smallest object."""
    bg = _get_bg(train[0]["input"])

    def apply_fn(grid, which_=which, bg_=bg):
        objects = _find_objects(grid, bg_)
        if not objects:
            return grid
        if which_ == "largest":
            obj = max(objects, key=lambda o: o["size"])
        else:
            obj = min(objects, key=lambda o: o["size"])
        return obj["sub_grid"]

    for ex in train:
        result = apply_fn(ex["input"])
        if result != ex["output"]:
            return None
    return apply_fn


def _try_extract_unique_color_object(train):
    """Output = the object whose color appears only once."""
    bg = _get_bg(train[0]["input"])

    def apply_fn(grid, bg_=bg):
        objects = _find_objects(grid, bg_)
        if not objects:
            return grid
        color_counts = Counter(o["primary_color"] for o in objects)
        unique = [o for o in objects if color_counts[o["primary_color"]] == 1]
        if len(unique) == 1:
            return unique[0]["sub_grid"]
        return grid

    for ex in train:
        if apply_fn(ex["input"]) != ex["output"]:
            return None
    return apply_fn


def _try_extract_most_colors_object(train):
    """Output = the object with the most distinct colors."""
    bg = _get_bg(train[0]["input"])

    def apply_fn(grid, bg_=bg):
        objects = _find_objects(grid, bg_)
        if not objects:
            return grid
        # Find object with most colors (connected region with multiple colors)
        # Re-detect with multi-color connectivity
        multi = max(objects, key=lambda o: len(o["colors"]))
        return multi["sub_grid"]

    for ex in train:
        if apply_fn(ex["input"]) != ex["output"]:
            return None
    return apply_fn


def _try_recolor_by_size(train):
    """Recolor each object based on its size ranking."""
    bg = _get_bg(train[0]["input"])

    # Check if objects get recolored consistently based on size order
    color_mappings = []
    for ex in train:
        in_objects = _find_objects(ex["input"], bg)
        out_objects = _find_objects(ex["output"], bg)
        if len(in_objects) != len(out_objects):
            return None
        # Sort both by position (top-left)
        in_objects.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
        out_objects.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
        mapping = {}
        for io, oo in zip(in_objects, out_objects):
            if io["primary_color"] != oo["primary_color"]:
                mapping[io["primary_color"]] = oo["primary_color"]
        color_mappings.append(mapping)

    # Check consistency
    if not color_mappings or not color_mappings[0]:
        return None
    ref = color_mappings[0]
    for m in color_mappings[1:]:
        if m != ref:
            return None

    def apply_fn(grid, mapping=ref, bg_=bg):
        return [[mapping.get(c, c) for c in row] for row in grid]

    for ex in train:
        if apply_fn(ex["input"]) != ex["output"]:
            return None
    return apply_fn


def _try_filter_by_size(train):
    """Keep only objects above/below a certain size, clear the rest."""
    bg = _get_bg(train[0]["input"])

    # Find what sizes are kept vs removed across examples
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    # Try: keep objects >= threshold
    for threshold in range(1, 20):
        def apply_fn(grid, t=threshold, bg_=bg):
            objects = _find_objects(grid, bg_)
            result = [[bg_] * len(grid[0]) for _ in range(len(grid))]
            for obj in objects:
                if obj["size"] >= t:
                    for r, c in obj["cells"]:
                        result[r][c] = grid[r][c]
            return result

        if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
            return apply_fn

    # Try: keep objects <= threshold
    for threshold in range(1, 20):
        def apply_fn(grid, t=threshold, bg_=bg):
            objects = _find_objects(grid, bg_)
            result = [[bg_] * len(grid[0]) for _ in range(len(grid))]
            for obj in objects:
                if obj["size"] <= t:
                    for r, c in obj["cells"]:
                        result[r][c] = grid[r][c]
            return result

        if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
            return apply_fn

    return None
