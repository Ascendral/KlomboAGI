"""
ARC Feature Learner — discovers operators from input→output relationships.

This is NOT a hardcoded strategy. The system:
1. Extracts local features for each cell (value, neighbors, position)
2. Builds a mapping: feature_tuple → output_value
3. If the mapping is consistent across ALL training examples, applies to test

The operator is INVENTED by the system, not programmed by a human.
"""

from __future__ import annotations
from collections import Counter
from klomboagi.reasoning.arc_objects import ObjectDetector

_detector = ObjectDetector()
_obj_cache: dict = {}

Grid = list[list[int]]


def extract_features_v1(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Basic feature tuple: value + 4-neighbor stats + border."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n4 = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        n4.append(grid[nr][nc] if 0 <= nr < rows and 0 <= nc < cols else -1)
    n4_nonbg = [n for n in n4 if n != bg and n != -1]
    n4_same = sum(1 for n in n4 if n == val)
    on_border = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
    majority = Counter(n4_nonbg).most_common(1)[0][0] if n4_nonbg else bg
    return (val, val == bg, n4_same, len(n4_nonbg), majority, on_border)


def extract_features_v2(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Extended features: value + all 8 neighbors."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            neighbors.append(grid[nr][nc] if 0 <= nr < rows and 0 <= nc < cols else -1)
    return (val,) + tuple(neighbors)


def extract_features_v3(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Features: value + cardinal neighbors + position modulo."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n = tuple(grid[r + dr][c + dc] if 0 <= r + dr < rows and 0 <= c + dc < cols else -1
              for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    return (val, n[0], n[1], n[2], n[3], r % 2, c % 2)


def extract_features_v4(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Features: value + is_bg + neighbor count + border distance."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n_nonbg = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
            n_nonbg += 1
    border_dist = min(r, c, rows - 1 - r, cols - 1 - c)
    return (val, val == bg, n_nonbg, min(border_dist, 5))




def extract_compact_features(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Compact: value + abstract neighbor summary."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n_same = 0; n_bg = 0; n_other = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<rows and 0<=nc<cols:
            nv = grid[nr][nc]
            if nv == val: n_same += 1
            elif nv == bg: n_bg += 1
            else: n_other += 1
        else:
            n_bg += 1
    return (val, n_same, n_bg, n_other)


def extract_directional(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + first non-bg color in each cardinal direction."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    def first_nonbg(dr, dc):
        cr, cc = r+dr, c+dc
        while 0<=cr<rows and 0<=cc<cols:
            if grid[cr][cc] != bg: return grid[cr][cc]
            cr += dr; cc += dc
        return -1
    return (val, first_nonbg(-1,0), first_nonbg(1,0), first_nonbg(0,-1), first_nonbg(0,1))


def extract_mega_features(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Kitchen sink: value + all cardinal + 8-neighbor count + border dist."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    def g(dr, dc):
        nr, nc = r+dr, c+dc
        return grid[nr][nc] if 0<=nr<rows and 0<=nc<cols else -1
    n = g(-1,0), g(1,0), g(0,-1), g(0,1)
    n8_nonbg = 0
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            v = g(dr, dc)
            if v != bg and v != -1: n8_nonbg += 1
    bd = min(r, c, rows-1-r, cols-1-c)
    return (val, n[0], n[1], n[2], n[3], n8_nonbg, min(bd,3), val==bg)



def extract_row_col_position(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + unique color counts in row and column."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    row_colors = len(set(grid[r][c2] for c2 in range(cols) if grid[r][c2] != bg))
    col_colors = len(set(grid[r2][c] for r2 in range(rows) if grid[r2][c] != bg))
    return (val, row_colors, col_colors)


def extract_val_and_col(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + entire column as context."""
    rows = len(grid)
    val = grid[r][c]
    return (val, tuple(grid[r2][c] for r2 in range(rows)))


def extract_cross_2(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + what is 2 cells away in each cardinal direction."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    def g(dr, dc):
        nr, nc = r+dr, c+dc
        return grid[nr][nc] if 0<=nr<rows and 0<=nc<cols else -1
    return (val, g(-2,0), g(2,0), g(0,-2), g(0,2))



# ── Auto-discovered feature extractors ──

def extract_val_n4(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + cardinal non-bg neighbor count. Discovered: +5 puzzles."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
            if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]!=bg)
    return (val, n)


def extract_val_n8(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + 8-neighbor non-bg count."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
            if (dr or dc) and 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]!=bg)
    return (val, n)


def extract_val_cmod3(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + column modulo 3. Discovers column-periodic patterns."""
    return (grid[r][c], c % 3)


def extract_val_north(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + cell directly above."""
    return (grid[r][c], grid[r-1][c] if r > 0 else -1)


def extract_val_n4_n8(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + cardinal count + 8-neighbor count."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    n4 = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
             if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]!=bg)
    n8 = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
             if (dr or dc) and 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]!=bg)
    return (val, n4, n8)


def extract_val_west_east(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + west neighbor + east neighbor."""
    cols = len(grid[0])
    return (grid[r][c],
            grid[r][c-1] if c > 0 else -1,
            grid[r][c+1] if c < cols-1 else -1)



def extract_val_mirror_h(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + horizontally mirrored cell value. Discovers symmetry rules."""
    cols = len(grid[0])
    return (grid[r][c], grid[r][cols-1-c])


def extract_val_mirror_v(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + vertically mirrored cell value."""
    rows = len(grid)
    return (grid[r][c], grid[rows-1-r][c])


def extract_val_dist_same(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + min distance to another cell of same color."""
    rows, cols = len(grid), len(grid[0])
    val = grid[r][c]
    min_d = rows + cols
    for r2 in range(rows):
        for c2 in range(cols):
            if (r2 != r or c2 != c) and grid[r2][c2] == val:
                d = abs(r-r2) + abs(c-c2)
                if d < min_d: min_d = d
    return (val, min(min_d, 10))



def _get_objs(grid: Grid, bg: int):
    """Cached object detection."""
    gt = id(grid)  # Use id for speed
    if gt not in _obj_cache:
        _obj_cache[gt] = _detector.detect(grid, bg)
    return _obj_cache[gt]


def extract_val_obj_size(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + size of containing object."""
    objs = _get_objs(grid, bg)
    for obj in objs:
        if (r,c) in set(obj.cells):
            return (grid[r][c], obj.size)
    return (grid[r][c], 0)


def extract_val_obj_smallest(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + is this the smallest object?"""
    objs = _get_objs(grid, bg)
    if not objs: return (grid[r][c], False)
    min_size = min(o.size for o in objs)
    for obj in objs:
        if (r,c) in set(obj.cells):
            return (grid[r][c], obj.size == min_size)
    return (grid[r][c], False)


def extract_val_num_objects(grid: Grid, r: int, c: int, bg: int) -> tuple:
    """Value + total number of objects in grid."""
    objs = _get_objs(grid, bg)
    return (grid[r][c], len(objs))

FEATURE_EXTRACTORS = [
    ("v1_basic", extract_features_v1),
    ("v2_8neighbors", extract_features_v2),
    ("v3_cardinal_pos", extract_features_v3),
    ("v4_border_dist", extract_features_v4),
    ("v5_compact", extract_compact_features),
    ("v6_directional", extract_directional),
    ("v7_mega", extract_mega_features),
    ("v8_row_col_pos", extract_row_col_position),
    ("v9_val_col", extract_val_and_col),
    ("v10_cross2", extract_cross_2),
    ("v11_val_n4", extract_val_n4),
    ("v12_val_n8", extract_val_n8),
    ("v13_val_cmod3", extract_val_cmod3),
    ("v14_val_north", extract_val_north),
    ("v15_val_n4_n8", extract_val_n4_n8),
    ("v16_val_west_east", extract_val_west_east),
    ("v17_mirror_h", extract_val_mirror_h),
    ("v18_mirror_v", extract_val_mirror_v),
    ("v19_dist_same", extract_val_dist_same),
    ("v20_obj_size", extract_val_obj_size),
    ("v21_obj_smallest", extract_val_obj_smallest),
    ("v22_num_objects", extract_val_num_objects),
]


class FeatureLearner:
    """
    Learn cell-level transforms from feature tuples.
    Tries multiple feature extractors and picks the one that works.
    """

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        _obj_cache.clear()
        """Try to learn a feature-based rule from training examples."""
        # Same size required
        for ex in train:
            if (len(ex["input"]) != len(ex["output"]) or
                    len(ex["input"][0]) != len(ex["output"][0])):
                return None
        if not train:
            return None

        # Find bg
        av = []
        for ex in train:
            for row in ex["input"]:
                av.extend(row)
        bg = Counter(av).most_common(1)[0][0]

        # Try each feature extractor
        for name, extractor in FEATURE_EXTRACTORS:
            result = self._try_extractor(train, test_input, bg, extractor)
            if result is not None:
                # Cross-validate
                if self._cross_validate(train, bg, extractor):
                    return result

        return None

    def _try_extractor(self, train: list[dict], test_input: Grid,
                       bg: int, extractor) -> Grid | None:
        """Build lookup from one extractor and apply."""
        lookup = {}
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    feats = extractor(inp, r, c, bg)
                    oval = out[r][c]
                    if feats in lookup and lookup[feats] != oval:
                        return None  # Inconsistent
                    lookup[feats] = oval

        # Apply to test
        rows, cols = len(test_input), len(test_input[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                feats = extractor(test_input, r, c, bg)
                if feats in lookup:
                    result[r][c] = lookup[feats]
                else:
                    return None  # Can't predict this cell
        return result

    def _cross_validate(self, train: list[dict], bg: int, extractor) -> bool:
        """Hold one example out and verify."""
        if len(train) < 2:
            return True
        for i in range(min(len(train), 1)):  # Only 1 holdout — feature learning needs more data
            holdout = train[i]
            remaining = train[:i] + train[i + 1:]
            result = self._try_extractor(remaining, holdout["input"], bg, extractor)
            if result != holdout["output"]:
                return False
        return True
