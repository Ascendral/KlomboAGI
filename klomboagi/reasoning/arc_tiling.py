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
        _try_self_tile_inverted,
        _try_tile_hv,
        _try_rotation_tile,
        _try_tile_mark_diagonal,
        _try_pixel_to_block,
        _try_downsample,
        _try_unique_tile_extract,
        _try_symmetric_3x3_tile,
        _try_scale_by_color_count,
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


def _try_self_tile_inverted(train):
    """
    For each non-bg cell at (r,c), stamp the INVERTED input at output block (r*rows, c*cols).
    Inversion: bg → input_color, non-bg → bg.
    """
    bg = _bg(train)
    ex0 = train[0]
    ir, ic = len(ex0["input"]), len(ex0["input"][0])
    otr, otc = len(ex0["output"]), len(ex0["output"][0])

    if otr != ir * ir or otc != ic * ic:
        return None

    def self_tile_inv(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * (cols * cols) for _ in range(rows * rows)]

        # Build inverted pattern: bg→primary_color, non-bg→bg
        non_bg = [v for row in grid for v in row if v != bg_val]
        if not non_bg:
            return result
        primary = Counter(non_bg).most_common(1)[0][0]
        inverted = [[primary if grid[r][c] == bg_val else bg_val
                     for c in range(cols)] for r in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    # Place inverted at (r*rows, c*cols)
                    for dr in range(rows):
                        for dc in range(cols):
                            result[r * rows + dr][c * cols + dc] = inverted[dr][dc]
        return result

    fn = lambda grid, bg_val=bg: self_tile_inv(grid, bg_val)
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

    # Alternating H-flip per tile-row (even rows: original, odd rows: hflipped)
    # Columns tiled without flip
    def tile_row_hflip(grid, tile_r=tr, tile_c=tc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for tri in range(tile_r):
            for row in grid:
                if tri % 2 == 0:
                    result.append(row * tile_c)
                else:
                    result.append(row[::-1] * tile_c)
        return result

    if all(tile_row_hflip(ex["input"]) == ex["output"] for ex in train):
        return tile_row_hflip

    # Alternating V-flip per tile-col (even cols: original, odd cols: vflipped)
    def tile_col_vflip(grid, tile_r=tr, tile_c=tc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for _ in range(tile_r):
            for row in grid:
                result.append(row * tile_c)
        # Now flip alternating column blocks
        h = len(result)
        w = len(result[0]) if result else 0
        for tci in range(1, tile_c, 2):
            c0 = tci * cols
            for r in range(rows):
                for c in range(cols):
                    src_r = rows - 1 - r
                    for tri in range(tile_r):
                        result[tri * rows + r][c0 + c] = grid[src_r][c]
        return result

    if all(tile_col_vflip(ex["input"]) == ex["output"] for ex in train):
        return tile_col_vflip

    # Full alternating: even tile rows use original, odd use both-flipped
    def tile_full_alt(grid, tile_r=tr, tile_c=tc):
        rows, cols = len(grid), len(grid[0])
        result = []
        for tri in range(tile_r):
            src_rows = grid if tri % 2 == 0 else [row[::-1] for row in grid[::-1]]
            for row in src_rows:
                result.append(row * tile_c)
        return result

    if all(tile_full_alt(ex["input"]) == ex["output"] for ex in train):
        return tile_full_alt

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


def _try_symmetric_3x3_tile(train):
    """
    Output = input tiled 3x3 with symmetry:
      rot180  vflip  rot180
      hflip   orig   hflip
      rot180  vflip  rot180
    """
    # All outputs must be 3x input
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if otr != 3 * ir or otc != 3 * ic:
            return None

    def rot180(grid):
        return [row[::-1] for row in grid[::-1]]

    def vflip(grid):
        return grid[::-1]

    def hflip(grid):
        return [row[::-1] for row in grid]

    def make_symmetric_tile(grid):
        r180 = rot180(grid)
        vf = vflip(grid)
        hf = hflip(grid)
        rows, cols = len(grid), len(grid[0])
        result = []
        for tile_row in range(3):
            for r in range(rows):
                row = []
                for tile_col in range(3):
                    if tile_row == 0 and tile_col == 0:
                        src = r180
                    elif tile_row == 0 and tile_col == 1:
                        src = vf
                    elif tile_row == 0 and tile_col == 2:
                        src = r180
                    elif tile_row == 1 and tile_col == 0:
                        src = hf
                    elif tile_row == 1 and tile_col == 1:
                        src = grid
                    elif tile_row == 1 and tile_col == 2:
                        src = hf
                    elif tile_row == 2 and tile_col == 0:
                        src = r180
                    elif tile_row == 2 and tile_col == 1:
                        src = vf
                    else:
                        src = r180
                    row.extend(src[r])
                result.append(row)
        return result

    if all(make_symmetric_tile(ex["input"]) == ex["output"] for ex in train):
        return make_symmetric_tile

    # Try other 3x3 symmetry arrangements
    def make_mirror_tile(grid):
        """
        orig   hflip  orig
        vflip  rot180 vflip
        orig   hflip  orig
        """
        r180 = rot180(grid)
        vf = vflip(grid)
        hf = hflip(grid)
        rows, cols = len(grid), len(grid[0])
        result = []
        for tile_row in range(3):
            for r in range(rows):
                row = []
                for tile_col in range(3):
                    if tile_row == 0 and tile_col == 0:
                        src = grid
                    elif tile_row == 0 and tile_col == 1:
                        src = hf
                    elif tile_row == 0 and tile_col == 2:
                        src = grid
                    elif tile_row == 1 and tile_col == 0:
                        src = vf
                    elif tile_row == 1 and tile_col == 1:
                        src = r180
                    elif tile_row == 1 and tile_col == 2:
                        src = vf
                    elif tile_row == 2 and tile_col == 0:
                        src = grid
                    elif tile_row == 2 and tile_col == 1:
                        src = hf
                    else:
                        src = grid
                    row.extend(src[r])
                result.append(row)
        return result

    if all(make_mirror_tile(ex["input"]) == ex["output"] for ex in train):
        return make_mirror_tile

    return None


def _try_scale_by_color_count(train):
    """
    Output = input upscaled by N where N = number of distinct non-bg colors.
    """
    bg = _bg(train)

    scales = []
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if ir == 0 or ic == 0:
            return None
        if otr % ir != 0 or otc % ic != 0:
            return None
        sr, sc = otr // ir, otc // ic
        if sr != sc:
            return None
        distinct = len(set(v for row in ex["input"] for v in row if v != bg))
        if distinct != sr:
            return None
        scales.append(sr)

    if not scales or len(set(scales)) == 1 and scales[0] < 2:
        return None

    def upscale_by_distinct(grid, bg_val=bg):
        distinct = len(set(v for row in grid for v in row if v != bg_val))
        if distinct < 2:
            return None
        n = distinct
        rows, cols = len(grid), len(grid[0])
        result = []
        for r in range(rows):
            for _ in range(n):
                row = []
                for c in range(cols):
                    row.extend([grid[r][c]] * n)
                result.append(row)
        return result

    if all(upscale_by_distinct(ex["input"]) == ex["output"] for ex in train):
        return upscale_by_distinct

    return None


def _try_rotation_tile(train):
    """
    2×2 tiling using rotations:
      orig    | rot90CW
      ---------+---------
      rot90CCW | rot180

    All examples must double in both dimensions.
    """
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if otr != 2 * ir or otc != 2 * ic:
            return None

    def rot90cw(grid):
        # rows×cols → cols×rows
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows - 1 - c][r] for c in range(rows)] for r in range(cols)]

    def rot90ccw(grid):
        # rows×cols → cols×rows
        rows, cols = len(grid), len(grid[0])
        return [[grid[c][cols - 1 - r] for c in range(rows)] for r in range(cols)]

    def rot180(grid):
        return [row[::-1] for row in grid[::-1]]

    # 90° rotations only work cleanly for tiling if grid is square
    # (non-square grids produce different-height tiles that can't be side-by-side)
    ex0_ir, ex0_ic = len(train[0]["input"]), len(train[0]["input"][0])
    if ex0_ir == ex0_ic:
        def make_rotation_tile(grid):
            cw = rot90cw(grid)
            ccw = rot90ccw(grid)
            r180 = rot180(grid)
            rows = len(grid)
            result = []
            for r in range(rows):
                result.append(grid[r] + cw[r])
            for r in range(rows):
                result.append(ccw[r] + r180[r])
            return result

        if all(make_rotation_tile(ex["input"]) == ex["output"] for ex in train):
            return make_rotation_tile

    # Try other rotation arrangements (square only)
    if ex0_ir == ex0_ic:
        def make_rotation_tile_v2(grid):
            """rot90CW | orig / rot180 | rot90CCW"""
            cw = rot90cw(grid); ccw = rot90ccw(grid); r180 = rot180(grid)
            rows = len(grid); result = []
            for r in range(rows): result.append(cw[r] + grid[r])
            for r in range(rows): result.append(r180[r] + ccw[r])
            return result

        if all(make_rotation_tile_v2(ex["input"]) == ex["output"] for ex in train):
            return make_rotation_tile_v2

        def make_rotation_tile_v3(grid):
            """rot90CCW | rot180 / orig | rot90CW"""
            cw = rot90cw(grid); ccw = rot90ccw(grid); r180 = rot180(grid)
            rows = len(grid); result = []
            for r in range(rows): result.append(ccw[r] + r180[r])
            for r in range(rows): result.append(grid[r] + cw[r])
            return result

        if all(make_rotation_tile_v3(ex["input"]) == ex["output"] for ex in train):
            return make_rotation_tile_v3

        def make_rotation_tile_v4(grid):
            """rot180 | rot90CCW / rot90CW | orig"""
            cw = rot90cw(grid); ccw = rot90ccw(grid); r180 = rot180(grid)
            rows = len(grid); result = []
            for r in range(rows): result.append(r180[r] + ccw[r])
            for r in range(rows): result.append(cw[r] + grid[r])
            return result

        if all(make_rotation_tile_v4(ex["input"]) == ex["output"] for ex in train):
            return make_rotation_tile_v4

    # Try reflection-based 2×2 tiling: rot180|vflip / hflip|orig
    def rot180_local(grid):
        return [row[::-1] for row in grid[::-1]]

    def vflip_local(grid):
        return grid[::-1]

    def hflip_local(grid):
        return [row[::-1] for row in grid]

    def make_reflect_tile(grid):
        """rot180 | vflip / hflip | orig"""
        r180 = rot180_local(grid); vf = vflip_local(grid); hf = hflip_local(grid)
        rows = len(grid)
        result = []
        for r in range(rows):
            result.append(r180[r] + vf[r])
        for r in range(rows):
            result.append(hf[r] + grid[r])
        return result

    if all(make_reflect_tile(ex["input"]) == ex["output"] for ex in train):
        return make_reflect_tile

    def make_reflect_tile_v2(grid):
        """vflip | rot180 / orig | hflip"""
        r180 = rot180_local(grid); vf = vflip_local(grid); hf = hflip_local(grid)
        rows = len(grid)
        result = []
        for r in range(rows):
            result.append(vf[r] + r180[r])
        for r in range(rows):
            result.append(grid[r] + hf[r])
        return result

    if all(make_reflect_tile_v2(ex["input"]) == ex["output"] for ex in train):
        return make_reflect_tile_v2

    def make_reflect_tile_v3(grid):
        """hflip | orig / rot180 | vflip"""
        r180 = rot180_local(grid); vf = vflip_local(grid); hf = hflip_local(grid)
        rows = len(grid)
        result = []
        for r in range(rows):
            result.append(hf[r] + grid[r])
        for r in range(rows):
            result.append(r180[r] + vf[r])
        return result

    if all(make_reflect_tile_v3(ex["input"]) == ex["output"] for ex in train):
        return make_reflect_tile_v3

    def make_reflect_tile_v4(grid):
        """orig | hflip / vflip | rot180"""
        r180 = rot180_local(grid); vf = vflip_local(grid); hf = hflip_local(grid)
        rows = len(grid)
        result = []
        for r in range(rows):
            result.append(grid[r] + hf[r])
        for r in range(rows):
            result.append(vf[r] + r180[r])
        return result

    if all(make_reflect_tile_v4(ex["input"]) == ex["output"] for ex in train):
        return make_reflect_tile_v4

    return None


def _try_tile_mark_diagonal(train):
    """
    Output = input tiled 2×2, then bg cells with a diagonal non-bg neighbor are marked 8.

    Handles tasks where tiling + diagonal marking gives the output.
    """
    bg = _bg(train)

    # All outputs must be exactly 2× input size
    for ex in train:
        ir, ic = len(ex["input"]), len(ex["input"][0])
        otr, otc = len(ex["output"]), len(ex["output"][0])
        if otr != 2 * ir or otc != 2 * ic:
            return None

    def apply_tile_mark(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])

        # Tile 2×2
        tiled = []
        for _ in range(2):
            for row in grid:
                tiled.append(row + row)

        # Mark diagonal neighbors
        tr, tc = len(tiled), len(tiled[0])
        result = [row[:] for row in tiled]
        for r in range(tr):
            for c in range(tc):
                if tiled[r][c] == bg_val:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < tr and 0 <= nc < tc and tiled[nr][nc] != bg_val:
                            result[r][c] = 8
                            break
        return result

    if all(apply_tile_mark(ex["input"]) == ex["output"] for ex in train):
        return apply_tile_mark

    return None
