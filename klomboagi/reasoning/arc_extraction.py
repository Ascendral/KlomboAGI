"""
ARC Extraction Solver — for tasks where output is a sub-region of input.

110 unsolved extraction tasks. Common patterns:
  - Output = the unique/different sub-grid among repeating tiles
  - Output = count of objects encoded as grid size or color
  - Output = the smallest/largest distinct region
  - Output = sub-grid at a specific position (marked by a special color)
  - Output = the region inside a border/frame
"""

from __future__ import annotations

from collections import Counter

Grid = list[list[int]]


def _get_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_extraction_rule(train: list[dict]) -> callable | None:
    """Try to learn an extraction rule from training examples."""

    # Only for tasks where output is smaller
    for ex in train:
        if len(ex["output"]) >= len(ex["input"]) and len(ex["output"][0]) >= len(ex["input"][0]):
            return None

    # Strategy 1: Output appears literally as a sub-grid in input
    rule = _try_literal_subgrid(train)
    if rule:
        return rule

    # Strategy 2: Output = unique tile in a repeating grid
    rule = _try_unique_tile(train)
    if rule:
        return rule

    # Strategy 3: Output = region inside a rectangular border
    rule = _try_extract_bordered_region(train)
    if rule:
        return rule

    # Strategy 4: Output = sub-grid at position of a marker color
    rule = _try_extract_at_marker(train)
    if rule:
        return rule

    # Strategy 5: Output size encodes a count
    rule = _try_count_as_size(train)
    if rule:
        return rule

    # Strategy 6: Output = the non-bg content cropped tight
    rule = _try_tight_crop(train)
    if rule:
        return rule

    return None


def _try_literal_subgrid(train):
    """Output appears exactly as a sub-grid in the input."""
    def find_subgrid(grid, target):
        rows, cols = len(grid), len(grid[0])
        tr, tc = len(target), len(target[0])
        for r in range(rows - tr + 1):
            for c in range(cols - tc + 1):
                match = True
                for dr in range(tr):
                    for dc in range(tc):
                        if grid[r + dr][c + dc] != target[dr][dc]:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    return (r, c)
        return None

    # Check if output is always found at a consistent relative position
    positions = []
    for ex in train:
        pos = find_subgrid(ex["input"], ex["output"])
        if pos is None:
            return None
        positions.append(pos)

    # Check if position is consistent (same absolute or relative position)
    if len(set(positions)) == 1:
        # Same absolute position in every example
        r, c = positions[0]
        or_, oc = len(train[0]["output"]), len(train[0]["output"][0])

        def apply_fn(grid, r_=r, c_=c, h=or_, w=oc):
            return [grid[r_ + dr][c_:c_ + w] for dr in range(h)]
        if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
            return apply_fn

    return None


def _try_unique_tile(train):
    """Grid is tiled, output = the unique tile."""
    for ex in train:
        inp = ex["input"]
        out = ex["output"]
        oh, ow = len(out), len(out[0])
        ih, iw = len(inp), len(inp[0])

        if ih % oh != 0 or iw % ow != 0:
            continue

        # Extract all tiles of output size
        tiles = []
        for r in range(0, ih, oh):
            for c in range(0, iw, ow):
                tile = [inp[r + dr][c:c + ow] for dr in range(oh)]
                tiles.append(tuple(tuple(row) for row in tile))

        tile_counts = Counter(tiles)
        unique = [t for t, count in tile_counts.items() if count == 1]
        if len(unique) == 1:
            candidate = [list(row) for row in unique[0]]
            if candidate == out:
                # This works for this example, check all
                break
        else:
            continue
    else:
        return None

    def apply_fn(grid, h=oh, w=ow):
        rows, cols = len(grid), len(grid[0])
        if rows % h != 0 or cols % w != 0:
            return grid
        tiles = []
        for r in range(0, rows, h):
            for c in range(0, cols, w):
                tile = tuple(tuple(grid[r + dr][c:c + w]) for dr in range(h))
                tiles.append(tile)
        counts = Counter(tiles)
        unique = [t for t, count in counts.items() if count == 1]
        if len(unique) == 1:
            return [list(row) for row in unique[0]]
        # If no unique tile, return the most different one
        if tiles:
            # Find tile most different from others
            best_diff = -1
            best_tile = tiles[0]
            for t in tiles:
                diff = sum(1 for other in tiles if other != t)
                if diff > best_diff:
                    best_diff = diff
                    best_tile = t
            return [list(row) for row in best_tile]
        return grid

    if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
        return apply_fn
    return None


def _try_extract_bordered_region(train):
    """Output = content inside a rectangular border of non-bg color."""
    bg = _get_bg(train[0]["input"])

    def extract_bordered(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        # Find rectangular borders
        for r1 in range(rows - 2):
            for c1 in range(cols - 2):
                if grid[r1][c1] == bg_val:
                    continue
                border_color = grid[r1][c1]
                # Check if this starts a rectangle
                for r2 in range(r1 + 2, rows):
                    for c2 in range(c1 + 2, cols):
                        # Check top edge
                        if not all(grid[r1][c] == border_color for c in range(c1, c2 + 1)):
                            continue
                        # Check bottom edge
                        if not all(grid[r2][c] == border_color for c in range(c1, c2 + 1)):
                            continue
                        # Check left edge
                        if not all(grid[r][c1] == border_color for r in range(r1, r2 + 1)):
                            continue
                        # Check right edge
                        if not all(grid[r][c2] == border_color for r in range(r1, r2 + 1)):
                            continue
                        # Found a bordered region — extract interior
                        interior = [grid[r][c1 + 1:c2] for r in range(r1 + 1, r2)]
                        if interior and interior[0]:
                            return interior
        return None

    results = [extract_bordered(ex["input"], bg) for ex in train]
    if all(r == ex["output"] for r, ex in zip(results, train) if r is not None):
        if all(r is not None for r in results):
            def apply_fn(grid, bg_=bg):
                return extract_bordered(grid, bg_)
            return apply_fn
    return None


def _try_extract_at_marker(train):
    """A special marker color indicates where to extract from."""
    bg = _get_bg(train[0]["input"])
    oh, ow = len(train[0]["output"]), len(train[0]["output"][0])

    # Find colors that appear exactly once per example (potential markers)
    marker_candidates = set(range(10))
    for ex in train:
        flat = [c for row in ex["input"] for c in row]
        counts = Counter(flat)
        # Marker = color appearing exactly 1 time
        once = {c for c, n in counts.items() if n == 1 and c != bg}
        marker_candidates &= once
        if not marker_candidates:
            return None

    for marker in marker_candidates:
        def apply_fn(grid, m=marker, h=oh, w=ow, bg_=bg):
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == m:
                        # Extract h×w region centered on or starting at marker
                        for dr in range(-h, 1):
                            for dc in range(-w, 1):
                                sr, sc = r + dr, c + dc
                                if sr >= 0 and sc >= 0 and sr + h <= rows and sc + w <= cols:
                                    region = [grid[sr + rr][sc:sc + w] for rr in range(h)]
                                    return region
            return grid

        if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
            return apply_fn

    return None


def _try_count_as_size(train):
    """Output dimensions encode a count from the input."""
    bg = _get_bg(train[0]["input"])

    # Check if output is always a solid color rectangle
    for ex in train:
        out = ex["output"]
        vals = set(c for row in out for c in row)
        if len(vals) > 2:  # Allow bg + one color
            return None

    # Check if output height or width correlates with object count
    def count_objects(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        count = 0
        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or grid[r][c] == bg_val:
                    continue
                count += 1
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr][cc] or grid[cr][cc] == bg_val:
                        continue
                    visited[cr][cc] = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        queue.append((cr + dr, cc + dc))
        return count

    # Check: output width == object count?
    matches_w = all(
        len(ex["output"][0]) == count_objects(ex["input"], bg)
        for ex in train
    )
    if matches_w:
        oh = len(train[0]["output"])
        out_color = [c for row in train[0]["output"] for c in row if c != bg]
        oc = out_color[0] if out_color else 1

        def apply_fn(grid, bg_=bg, h=oh, color=oc):
            n = count_objects(grid, bg_)
            return [[color] * n for _ in range(h)]
        if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
            return apply_fn

    return None


def _try_tight_crop(train):
    """Output = all non-bg content cropped to bounding box."""
    bg = _get_bg(train[0]["input"])

    def crop(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        min_r = min_c = float('inf')
        max_r = max_c = -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        if max_r < 0:
            return grid
        return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]

    if all(crop(ex["input"], bg) == ex["output"] for ex in train):
        def apply_fn(grid, bg_=bg):
            return crop(grid, bg_)
        return apply_fn
    return None
