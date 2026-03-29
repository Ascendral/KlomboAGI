"""
ARC Tiling + Scaling Strategies.

Handles tasks where:
  - Input is scaled up (each cell → NxN block)
  - Input is tiled (repeated N times)
  - Output is a mosaic of the input
  - Pixel upscaling (each cell value determines a tile)
  - Self-similar tiling (input pattern repeated where cells are non-bg)
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_tiling_rule(train: list[dict]) -> callable | None:
    """Try all tiling/scaling strategies."""
    for fn in [
        _try_upscale,
        _try_self_tile,
        _try_tile_hv,
        _try_pixel_to_block,
        _try_downsample,
        _try_unique_tile_extract,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


def _try_upscale(train):
    """Output = input with each cell expanded to NxN block of same color."""
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if ir == 0 or ic == 0 or otr % ir != 0 or otc % ic != 0:
            return None
        sr, sc = otr // ir, otc // ic
        if sr != sc:
            continue  # Non-uniform scaling, try next
        break
    else:
        return None

    # Verify scale factor consistent across examples
    ex0 = train[0]
    ir0, ic0 = len(ex0["input"]), len(ex0["input"][0])
    otr0, otc0 = len(ex0["output"]), len(ex0["output"][0])
    if ir0 == 0 or ic0 == 0:
        return None
    sr = otr0 // ir0
    sc = otc0 // ic0

    if sr < 2 or sc < 2:
        return None

    def upscale(grid, scale_r=sr, scale_c=sc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for r in range(rows):
            for _ in range(scale_r):
                row = []
                for c in range(cols):
                    row.extend([grid[r][c]] * scale_c)
                result.append(row)
        return result

    if all(upscale(ex["input"]) == ex["output"] for ex in train):
        return upscale

    return None


def _try_self_tile(train):
    """
    Output = input tiled onto itself: where input cell is non-bg,
    place a copy of input; where bg, place bg block.
    """
    bg = _bg(train)
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    otr, otc = len(ex0["output"]), len(ex0["output"][0])

    if otr != ir * ir or otc != ic * ic:
        return None

    def self_tile(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        out_rows = rows * rows
        out_cols = cols * cols
        result = [[bg_val] * out_cols for _ in range(out_rows)]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    # Place a copy of grid at (r*rows, c*cols)
                    for dr in range(rows):
                        for dc in range(cols):
                            result[r * rows + dr][c * cols + dc] = grid[dr][dc]
        return result

    fn = lambda grid, bg_val=bg: self_tile(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None


def _try_tile_hv(train):
    """Output = input tiled horizontally or vertically (2x1, 1x2, 2x2, 3x3)."""
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    otr, otc = len(ex0["output"]), len(ex0["output"][0])

    if ir == 0 or ic == 0:
        return None
    if otr % ir != 0 or otc % ic != 0:
        return None

    tr = otr // ir
    tc = otc // ic

    if tr < 1 or tc < 1 or (tr == 1 and tc == 1):
        return None

    def tile(grid, tile_r=tr, tile_c=tc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for _ in range(tile_r):
            for row in grid:
                result.append(row * tile_c)
        return result

    if all(tile(ex["input"]) == ex["output"] for ex in train):
        return tile

    # Try with alternating flips
    def tile_with_hflip(grid, tile_r=tr, tile_c=tc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for tri in range(tile_r):
            src = grid if tri % 2 == 0 else grid[::-1]
            for row in src:
                new_row = []
                for tci in range(tile_c):
                    if tci % 2 == 0:
                        new_row.extend(row)
                    else:
                        new_row.extend(row[::-1])
                result.append(new_row)
        return result

    if all(tile_with_hflip(ex["input"]) == ex["output"] for ex in train):
        return tile_with_hflip

    return None


def _try_pixel_to_block(train):
    """
    Each input cell value maps to a specific NxN tile/block pattern.
    Output = grid where each cell is replaced by its block.
    """
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    otr, otc = len(ex0["output"]), len(ex0["output"][0])

    if ir == 0 or ic == 0:
        return None
    if otr % ir != 0 or otc % ic != 0:
        return None

    bh = otr // ir
    bw = otc // ic

    if bh < 2 or bw < 2:
        return None

    # Learn color → block mapping from first example
    color_to_block = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                color = inp[r][c]
                block = tuple(
                    tuple(out[r * bh + dr][c * bw + dc] for dc in range(bw))
                    for dr in range(bh)
                )
                if color in color_to_block:
                    if color_to_block[color] != block:
                        return None  # Inconsistent
                else:
                    color_to_block[color] = block

    if not color_to_block:
        return None

    def apply_blocks(grid, ctb=color_to_block, block_h=bh, block_w=bw):
        rows, cols = len(grid), len(grid[0])
        result = []
        for r in range(rows):
            for dr in range(block_h):
                row = []
                for c in range(cols):
                    block = ctb.get(grid[r][c])
                    if block is None:
                        row.extend([0] * block_w)
                    else:
                        row.extend(block[dr])
                result.append(row)
        return result

    if all(apply_blocks(ex["input"]) == ex["output"] for ex in train):
        return apply_blocks

    return None


def _try_downsample(train):
    """Output = input downsampled by taking every Nth pixel or block majority."""
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if otr == 0 or otc == 0:
            return None
        if ir % otr != 0 or ic % otc != 0:
            return None
        sr, sc = ir // otr, ic // otc
        if sr < 2 or sc < 2:
            return None

    sr = len(train[0]["input"]) // len(train[0]["output"])
    sc = len(train[0]["input"][0]) // len(train[0]["output"][0])

    # Try: take top-left of each block
    def downsample_tl(grid, s_r=sr, s_c=sc):
        rows, cols = len(grid), len(grid[0])
        return [[grid[r * s_r][c * s_c]
                 for c in range(cols // s_c)]
                for r in range(rows // s_r)]

    if all(downsample_tl(ex["input"]) == ex["output"] for ex in train):
        return downsample_tl

    # Try: majority color in each block
    def downsample_majority(grid, s_r=sr, s_c=sc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for r in range(rows // s_r):
            row = []
            for c in range(cols // s_c):
                block_vals = []
                for dr in range(s_r):
                    for dc in range(s_c):
                        block_vals.append(grid[r * s_r + dr][c * s_c + dc])
                row.append(Counter(block_vals).most_common(1)[0][0])
            result.append(row)
        return result

    if all(downsample_majority(ex["input"]) == ex["output"] for ex in train):
        return downsample_majority

    return None


def _try_unique_tile_extract(train):
    """Input is a tiled grid. Output = the one tile that differs from the rest."""
    bg = _bg(train)

    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])

        if otr >= ir or otc >= ic:
            return None
        if ir % otr != 0 or ic % otc != 0:
            return None

    otr = len(train[0]["output"])
    otc = len(train[0]["output"][0])

    def extract_unique_tile(grid, th=otr, tw=otc):
        rows, cols = len(grid), len(grid[0])
        tiles = []
        for r in range(0, rows, th):
            for c in range(0, cols, tw):
                tile = tuple(
                    tuple(grid[r + dr][c + dc] for dc in range(tw))
                    for dr in range(th)
                )
                tiles.append(tile)

        if len(tiles) < 2:
            return None

        # Find the unique one
        tile_counts = Counter(tiles)
        if len(tile_counts) == 2:
            for tile, count in tile_counts.items():
                if count == 1:
                    return [list(row) for row in tile]

        return None

    fn = lambda grid, th=otr, tw=otc: extract_unique_tile(grid, th, tw)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn

    return None
