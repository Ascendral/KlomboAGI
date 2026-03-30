"""
ARC Advanced Strategies — handles patterns not covered by existing solvers.

Strategy families:
  1. Symmetry: make grid symmetric (H, V, diagonal, rotational)
  2. Color propagation: colors spread/wave through background
  3. Object transformation: rotate/flip individual objects in place
  4. Path drawing: connect points with lines/paths
  5. Counting encoders: count something → encode as shape/color/size
  6. Frame/border operations
  7. Sorting (rows/columns by value or object properties)
"""

from __future__ import annotations
from collections import Counter, defaultdict

Grid = list[list[int]]


def _bg(train):
    flat = [c for ex in train for row in ex["input"] for c in row]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def learn_advanced_rule(train: list[dict]) -> callable | None:
    """Try all advanced strategies, return first that works."""
    for fn in [
        _try_symmetry,
        _try_color_propagate,
        _try_object_transform,
        _try_draw_lines,
        _try_sort_rows,
        _try_sort_cols,
        _try_frame_extract,
        _try_frame_fill,
        _try_diff_of_examples,
        _try_checkerboard,
        _try_keep_only_color,
        _try_majority_color_grid,
        _try_output_is_single_color,
        _try_output_is_row_count,
        _try_output_encodes_object_count,
        _try_tile_denoise_majority,
        _try_expand_cross_shape,
        _try_diagonal_gravity_stack,
    ]:
        try:
            rule = fn(train)
            if rule is not None:
                return rule
        except Exception:
            continue
    return None


# ─── Symmetry ────────────────────────────────────────────────────────────────

def _try_symmetry(train):
    """Output = input made symmetric (H, V, diagonal, or rotational)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    def h_sym(grid):
        """Mirror left half over right (or right over left — whichever matches more)."""
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols // 2):
                mirror = cols - 1 - c
                result[r][mirror] = grid[r][c]
        return result

    def h_sym_r(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols // 2):
                mirror = cols - 1 - c
                result[r][c] = grid[r][mirror]
        return result

    def v_sym(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows // 2):
            mirror = rows - 1 - r
            result[mirror] = list(grid[r])
        return result

    def v_sym_r(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows // 2):
            mirror = rows - 1 - r
            result[r] = list(grid[mirror])
        return result

    def rot180(grid):
        return [row[::-1] for row in grid[::-1]]

    for fn in [h_sym, h_sym_r, v_sym, v_sym_r, rot180]:
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


# ─── Color Propagation ───────────────────────────────────────────────────────

def _try_color_propagate(train):
    """Colors spread outward from source cells through background, N steps."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def propagate_n(grid, bg_val, n):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for _ in range(n):
            new = [row[:] for row in result]
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] == bg_val:
                        # Take color from any non-bg neighbor
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and result[nr][nc] != bg_val:
                                new[r][c] = result[nr][nc]
                                break
            result = new
        return result

    # Try 1-5 propagation steps
    for n in range(1, 16):
        fn = lambda grid, bg_val=bg, steps=n: propagate_n(grid, bg_val, steps)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    # Try diagonal propagation
    def propagate_n_diag(grid, bg_val, n):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for _ in range(n):
            new = [row[:] for row in result]
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] == bg_val:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and result[nr][nc] != bg_val:
                                new[r][c] = result[nr][nc]
                                break
            result = new
        return result

    for n in range(1, 16):
        fn = lambda grid, bg_val=bg, steps=n: propagate_n_diag(grid, bg_val, steps)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


# ─── Object Transformation ───────────────────────────────────────────────────

def _try_object_transform(train):
    """Each connected object is transformed (rotated 90/180/270, flipped H/V)."""
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
                if not visited[r][c] and grid[r][c] != bg_val:
                    obj = []
                    queue = [(r, c)]
                    while queue:
                        cr, cc = queue.pop(0)
                        if cr<0 or cr>=rows or cc<0 or cc>=cols:
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

    def apply_transform(grid, bg_val, transform_fn):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val]*cols for _ in range(rows)]
        objects = find_objects(grid, bg_val)
        for obj in objects:
            # Get bounding box
            min_r = min(r for r,c,v in obj)
            max_r = max(r for r,c,v in obj)
            min_c = min(c for r,c,v in obj)
            max_c = max(c for r,c,v in obj)
            h = max_r - min_r + 1
            w = max_c - min_c + 1

            # Build local grid
            local = [[bg_val]*w for _ in range(h)]
            for r, c, v in obj:
                local[r - min_r][c - min_c] = v

            # Apply transform
            new_local = transform_fn(local)

            # Place back at same top-left (may change size for rotations)
            new_h = len(new_local)
            new_w = len(new_local[0]) if new_local else 0
            for r in range(new_h):
                for c in range(new_w):
                    nr, nc = min_r + r, min_c + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if new_local[r][c] != bg_val:
                            result[nr][nc] = new_local[r][c]
        return result

    def flip_h(grid): return [row[::-1] for row in grid]
    def flip_v(grid): return grid[::-1]
    def rot90(grid):
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows-1-c][r] for c in range(rows)] for r in range(cols)]
    def rot180(grid): return [row[::-1] for row in grid[::-1]]
    def rot270(grid):
        rows, cols = len(grid), len(grid[0])
        return [[grid[c][cols-1-r] for c in range(rows)] for r in range(cols)]

    for tfn in [flip_h, flip_v, rot90, rot180, rot270]:
        fn = lambda grid, bg_val=bg, t=tfn: apply_transform(grid, bg_val, t)
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


# ─── Path Drawing ────────────────────────────────────────────────────────────

def _try_draw_lines(train):
    """Connect pairs of same-color markers with straight lines (H, V, or diagonal)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def draw_lines(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        # Find all non-bg cells, group by color
        by_color = defaultdict(list)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    by_color[grid[r][c]].append((r, c))

        for color, points in by_color.items():
            if len(points) == 2:
                (r1, c1), (r2, c2) = points
                if r1 == r2:
                    for c in range(min(c1,c2), max(c1,c2)+1):
                        result[r1][c] = color
                elif c1 == c2:
                    for r in range(min(r1,r2), max(r1,r2)+1):
                        result[r][c1] = color
                elif abs(r2-r1) == abs(c2-c1):
                    # Diagonal
                    dr = 1 if r2 > r1 else -1
                    dc = 1 if c2 > c1 else -1
                    r, c = r1, c1
                    while (r, c) != (r2 + dr, c2 + dc):
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = color
                        r += dr
                        c += dc
        return result

    def draw_lines_extend(grid, bg_val):
        """Extend lines from each point to the edge."""
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    color = grid[r][c]
                    # Extend in all 4 directions
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        while 0 <= nr < rows and 0 <= nc < cols:
                            if result[nr][nc] == bg_val:
                                result[nr][nc] = color
                            else:
                                break
                            nr += dr
                            nc += dc
        return result

    for fn in [draw_lines, draw_lines_extend]:
        wrapped = lambda grid, bg_val=bg, f=fn: f(grid, bg_val)
        if all(wrapped(ex["input"]) == ex["output"] for ex in train):
            return wrapped

    return None


# ─── Sort Rows/Cols ──────────────────────────────────────────────────────────

def _try_sort_rows(train):
    """Output = input with rows sorted by some criterion."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def count_nonbg(row, bg_val):
        return sum(1 for v in row if v != bg_val)

    def sort_rows_by_count_asc(grid, bg_val):
        return sorted(grid, key=lambda row: count_nonbg(row, bg_val))

    def sort_rows_by_count_desc(grid, bg_val):
        return sorted(grid, key=lambda row: count_nonbg(row, bg_val), reverse=True)

    def sort_rows_lex(grid, bg_val):
        return sorted(grid)

    for fn in [sort_rows_by_count_asc, sort_rows_by_count_desc, sort_rows_lex]:
        wrapped = lambda grid, bg_val=bg, f=fn: f(grid, bg_val)
        if all(wrapped(ex["input"]) == ex["output"] for ex in train):
            return wrapped

    return None


def _try_sort_cols(train):
    """Output = input with columns sorted."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def transpose(grid):
        return [list(row) for row in zip(*grid)]

    def sort_cols_by_count_asc(grid, bg_val):
        t = transpose(grid)
        t.sort(key=lambda col: sum(1 for v in col if v != bg_val))
        return transpose(t)

    def sort_cols_by_count_desc(grid, bg_val):
        t = transpose(grid)
        t.sort(key=lambda col: sum(1 for v in col if v != bg_val), reverse=True)
        return transpose(t)

    for fn in [sort_cols_by_count_asc, sort_cols_by_count_desc]:
        wrapped = lambda grid, bg_val=bg, f=fn: f(grid, bg_val)
        if all(wrapped(ex["input"]) == ex["output"] for ex in train):
            return wrapped

    return None


# ─── Frame Extract ───────────────────────────────────────────────────────────

def _try_frame_extract(train):
    """Output = just the border/frame of the input."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    def extract_frame(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val]*cols for _ in range(rows)]
        for c in range(cols):
            result[0][c] = grid[0][c]
            result[rows-1][c] = grid[rows-1][c]
        for r in range(rows):
            result[r][0] = grid[r][0]
            result[r][cols-1] = grid[r][cols-1]
        return result

    fn = lambda grid, bg_val=bg: extract_frame(grid, bg_val)
    if all(fn(ex["input"]) == ex["output"] for ex in train):
        return fn
    return None


def _try_frame_fill(train):
    """Output = input with the border set to a specific color."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    # Detect what color the border becomes
    for color in range(10):
        def fill_frame(grid, fill_color=color):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            for c in range(cols):
                result[0][c] = fill_color
                result[rows-1][c] = fill_color
            for r in range(rows):
                result[r][0] = fill_color
                result[r][cols-1] = fill_color
            return result

        if all(fill_frame(ex["input"]) == ex["output"] for ex in train):
            return fill_frame

    return None


# ─── Diff of Examples ────────────────────────────────────────────────────────

def _try_diff_of_examples(train):
    """Two examples differ by their output — output = XOR/diff of two grids in input."""
    # This handles tasks where input contains two grids separated by a divider
    bg = _bg(train)

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])

        # Check for vertical divider
        for dc in range(1, cols-1):
            col = [inp[r][dc] for r in range(rows)]
            if len(set(col)) == 1 and col[0] != bg:
                left = [inp[r][:dc] for r in range(rows)]
                right = [inp[r][dc+1:] for r in range(rows)]
                if len(left[0]) == len(right[0]):
                    # Try various combinations
                    def xor_combine(a, b, bg_val):
                        h, w = len(a), len(a[0])
                        result = []
                        for r in range(h):
                            row = []
                            for c in range(w):
                                av, bv = a[r][c], b[r][c]
                                if av != bg_val and bv == bg_val: row.append(av)
                                elif av == bg_val and bv != bg_val: row.append(bv)
                                else: row.append(bg_val)
                            result.append(row)
                        return result

                    if xor_combine(left, right, bg) == out:
                        break
        else:
            continue
        break

    return None  # Complex — skip for now


# ─── Checkerboard ────────────────────────────────────────────────────────────

def _try_checkerboard(train):
    """Output = input with checkerboard pattern applied."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    # Detect two alternating colors
    for ex in train:
        out = ex["output"]
        vals = set(v for row in out for v in row)
        if len(vals) != 2:
            return None

    c0 = train[0]["output"][0][0]
    c1 = train[0]["output"][0][1] if len(train[0]["output"][0]) > 1 else None
    if c1 is None or c0 == c1:
        return None

    def checkerboard(grid, color0=c0, color1=c1):
        rows, cols = len(grid), len(grid[0])
        return [[color0 if (r+c)%2==0 else color1 for c in range(cols)] for r in range(rows)]

    def checkerboard_r(grid, color0=c0, color1=c1):
        rows, cols = len(grid), len(grid[0])
        return [[color1 if (r+c)%2==0 else color0 for c in range(cols)] for r in range(rows)]

    for fn in [checkerboard, checkerboard_r]:
        if all(fn(ex["input"]) == ex["output"] for ex in train):
            return fn

    return None


# ─── Keep Only Color ─────────────────────────────────────────────────────────

def _try_keep_only_color(train):
    """Output = input with only one specific color kept (others → bg)."""
    same_size = all(
        len(ex["input"]) == len(ex["output"]) and
        len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_size:
        return None

    bg = _bg(train)

    # What colors appear in outputs?
    out_colors = set()
    for ex in train:
        for row in ex["output"]:
            for v in row:
                out_colors.add(v)
    out_colors.discard(bg)

    if len(out_colors) != 1:
        return None

    keep_color = list(out_colors)[0]

    def keep_only(grid, color=keep_color, bg_val=bg):
        return [[v if v == color else bg_val for v in row] for row in grid]

    if all(keep_only(ex["input"]) == ex["output"] for ex in train):
        return keep_only

    return None


# ─── Majority Color ──────────────────────────────────────────────────────────

def _try_majority_color_grid(train):
    """Output = solid grid of the most common color (excluding bg)."""
    bg = _bg(train)

    for ex in train:
        out = ex["output"]
        vals = set(v for row in out for v in row)
        if len(vals) != 1:
            return None

    def majority_color_grid(grid, bg_val=bg):
        flat = [v for row in grid for v in row if v != bg_val]
        if not flat:
            return grid
        mc = Counter(flat).most_common(1)[0][0]
        rows, cols = len(grid), len(grid[0])
        return [[mc]*cols for _ in range(rows)]

    if all(majority_color_grid(ex["input"]) == ex["output"] for ex in train):
        return majority_color_grid

    return None


# ─── Single Color Output ─────────────────────────────────────────────────────

def _try_output_is_single_color(train):
    """Output is 1x1 or NxN of a single color derived from input."""
    for ex in train:
        out = ex["output"]
        vals = set(v for row in out for v in row)
        if len(vals) != 1:
            return None

    # What determines the color?
    bg = _bg(train)
    oh = len(train[0]["output"])
    ow = len(train[0]["output"][0])

    if not all(len(ex["output"]) == oh and len(ex["output"][0]) == ow for ex in train):
        return None

    def unique_color(grid, bg_val=bg):
        """Find the color that appears exactly once."""
        flat = [v for row in grid for v in row if v != bg_val]
        counts = Counter(flat)
        unique = [k for k, n in counts.items() if n == 1]
        if len(unique) == 1:
            return unique[0]
        return None

    def rarest_color(grid, bg_val=bg):
        flat = [v for row in grid for v in row if v != bg_val]
        if not flat:
            return None
        return Counter(flat).most_common()[-1][0]

    for color_fn in [unique_color, rarest_color]:
        results = [color_fn(ex["input"]) for ex in train]
        if all(r is not None for r in results):
            expected = [ex["output"][0][0] for ex in train]
            if results == expected:
                def apply_fn(grid, cfn=color_fn, h=oh, w=ow):
                    c = cfn(grid)
                    if c is None:
                        return [[0]*w for _ in range(h)]
                    return [[c]*w for _ in range(h)]
                if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
                    return apply_fn

    return None


# ─── Row Count Encoder ───────────────────────────────────────────────────────

def _try_output_is_row_count(train):
    """Output height/width encodes count of rows with non-bg content."""
    bg = _bg(train)

    for ex in train:
        if len(ex["output"]) > 10 or len(ex["output"][0]) > 10:
            return None

    def count_nonempty_rows(grid, bg_val):
        return sum(1 for row in grid if any(v != bg_val for v in row))

    def count_nonempty_cols(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        return sum(1 for c in range(cols) if any(grid[r][c] != bg_val for r in range(rows)))

    oh = len(train[0]["output"])
    ow = len(train[0]["output"][0])

    # Check if output height = count of non-empty rows
    if all(count_nonempty_rows(ex["input"], bg) == len(ex["output"]) for ex in train):
        # What fills the output?
        out_vals = list(set(v for ex in train for row in ex["output"] for v in row if v != bg))
        if len(out_vals) == 1:
            oc = out_vals[0]
            def apply_fn(grid, bg_val=bg, color=oc, w=ow):
                h = count_nonempty_rows(grid, bg_val)
                return [[color]*w for _ in range(h)]
            if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
                return apply_fn

    return None


# ─── Object Count Encoder ────────────────────────────────────────────────────

def _try_output_encodes_object_count(train):
    """Output is a grid whose size/color encodes the number of objects in input."""
    bg = _bg(train)

    def count_objects(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
        count = 0
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] != bg_val:
                    count += 1
                    queue = [(r, c)]
                    while queue:
                        cr, cc = queue.pop(0)
                        if cr<0 or cr>=rows or cc<0 or cc>=cols:
                            continue
                        if visited[cr][cc] or grid[cr][cc] == bg_val:
                            continue
                        visited[cr][cc] = True
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            queue.append((cr+dr, cc+dc))
        return count

    # Check if output is NxN or Nx1 or 1xN solid grid
    for ex in train:
        out = ex["output"]
        vals = set(v for row in out for v in row) - {bg}
        if len(vals) > 2:
            return None

    # Check: output 1xN where N = object count
    counts = [count_objects(ex["input"], bg) for ex in train]

    # Nx1 grid
    if all(len(ex["output"]) == counts[i] and len(ex["output"][0]) == 1 for i, ex in enumerate(train)):
        out_colors = [ex["output"][0][0] for ex in train]
        if len(set(out_colors)) == 1:
            oc = out_colors[0]
            def apply_fn(grid, bg_val=bg, color=oc):
                n = count_objects(grid, bg_val)
                return [[color]] * n
            if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
                return apply_fn

    # 1xN grid
    if all(len(ex["output"]) == 1 and len(ex["output"][0]) == counts[i] for i, ex in enumerate(train)):
        out_colors = [ex["output"][0][0] for ex in train]
        if len(set(out_colors)) == 1:
            oc = out_colors[0]
            def apply_fn(grid, bg_val=bg, color=oc):
                n = count_objects(grid, bg_val)
                return [[color] * n]
            if all(apply_fn(ex["input"]) == ex["output"] for ex in train):
                return apply_fn

    return None


# ─── Tile denoise by majority vote ───────────────────────────────────────────

def _try_tile_denoise_majority(train):
    """
    Input has a repeating pattern (tiles) with noise scattered.
    Separator lines (single-color rows/cols) divide tiles.
    Output = same grid with all tiles replaced by the majority-vote canonical tile.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def find_sep_lines(grid):
        rows, cols = len(grid), len(grid[0])
        sep_rows, sep_cols = [], []
        for r in range(rows):
            vals = set(grid[r])
            if len(vals) == 1:
                sep_rows.append(r)
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1:
                sep_cols.append(c)
        return sep_rows, sep_cols

    def denoise_tiles(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        sep_rows, sep_cols = find_sep_lines(grid)

        if not sep_rows and not sep_cols:
            return None

        sep_row_set = set(sep_rows)
        sep_col_set = set(sep_cols)

        row_bounds = [0] + [r + 1 for r in sorted(sep_rows)] + [rows]
        col_bounds = [0] + [c + 1 for c in sorted(sep_cols)] + [cols]

        # Extract all tile sub-grids (non-separator regions with actual content)
        tiles = []
        tile_positions = []
        for i in range(len(row_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i + 1]
            if r1 - r0 == 0:
                continue
            for j in range(len(col_bounds) - 1):
                c0, c1 = col_bounds[j], col_bounds[j + 1]
                if c1 - c0 == 0:
                    continue
                tile = [[grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]
                tiles.append(tile)
                tile_positions.append((r0, r1, c0, c1))

        if len(tiles) < 2:
            return None

        # Check all tiles same dimensions
        h0, w0 = len(tiles[0]), len(tiles[0][0]) if tiles[0] else 0
        if not all(len(t) == h0 and (len(t[0]) if t else 0) == w0 for t in tiles):
            return None

        # Majority vote per position
        canonical = []
        for r in range(h0):
            row = []
            for c in range(w0):
                vals = [tiles[ti][r][c] for ti in range(len(tiles))]
                row.append(Counter(vals).most_common(1)[0][0])
            canonical.append(row)

        # Must differ from at least one tile (otherwise nothing to denoise)
        if all(t == canonical for t in tiles):
            return None

        # Rebuild output
        result = [row[:] for row in grid]
        for (r0, r1, c0, c1) in tile_positions:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    result[r][c] = canonical[r - r0][c - c0]

        return result

    # Check first example has separator lines in input
    in_sr, in_sc = find_sep_lines(train[0]["input"])
    out_sr, out_sc = find_sep_lines(train[0]["output"])

    # Need separators in at least 2 directions for tile grid
    # If input has incomplete separators, fall through to output-based approach
    input_has_full_seps = bool(in_sr) and bool(in_sc)

    if not in_sr and not in_sc:
        if not out_sr and not out_sc:
            return None

        # Build denoise_tiles using separator structure from OUTPUT
        def denoise_tiles_output_sep(grid, bg_val, sep_rows_out, sep_cols_out):
            rows, cols = len(grid), len(grid[0])
            row_bounds = [0] + [r + 1 for r in sorted(sep_rows_out)] + [rows]
            col_bounds = [0] + [c + 1 for c in sorted(sep_cols_out)] + [cols]

            all_locs = []
            for i in range(len(row_bounds) - 1):
                r0, r1 = row_bounds[i], row_bounds[i + 1]
                if r1 - r0 == 0:
                    continue
                for j in range(len(col_bounds) - 1):
                    c0, c1 = col_bounds[j], col_bounds[j + 1]
                    if c1 - c0 == 0:
                        continue
                    all_locs.append((r0, r1, c0, c1))

            # Filter: only use tiles of the MOST COMMON size
            size_count = Counter((r1-r0, c1-c0) for r0,r1,c0,c1 in all_locs)
            if not size_count:
                return None
            dominant_h, dominant_w = size_count.most_common(1)[0][0]
            tiles_loc = [(r0,r1,c0,c1) for r0,r1,c0,c1 in all_locs
                        if (r1-r0) == dominant_h and (c1-c0) == dominant_w]

            if len(tiles_loc) < 2:
                return None

            tiles = [[[grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]
                     for r0, r1, c0, c1 in tiles_loc]

            h0, w0 = dominant_h, dominant_w
            canonical = []
            for r in range(h0):
                row = []
                for c in range(w0):
                    vals = [tiles[ti][r][c] for ti in range(len(tiles))]
                    row.append(Counter(vals).most_common(1)[0][0])
                canonical.append(row)

            if all(t == canonical for t in tiles):
                return None

            result = [row[:] for row in grid]
            # Fill separator lines with bg
            for r in sep_rows_out:
                for c in range(cols):
                    result[r][c] = bg_val
            for c in sep_cols_out:
                for r in range(rows):
                    result[r][c] = bg_val

            # Fill all dominant-size tile locations with canonical
            for (r0, r1, c0, c1) in tiles_loc:
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        result[r][c] = canonical[r - r0][c - c0]

            # Fill other (non-dominant) tile locations with bg
            for (r0, r1, c0, c1) in all_locs:
                if (r1-r0) != dominant_h or (c1-c0) != dominant_w:
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            result[r][c] = bg_val

            return result

        out_sr2, out_sc2 = find_sep_lines(train[0]["output"])
        results = [denoise_tiles_output_sep(ex["input"], bg, out_sr2, out_sc2) for ex in train]
        if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
            return lambda g, b=bg, sr=out_sr2, sc=out_sc2: \
                denoise_tiles_output_sep(g, b, sr, sc)
        return None

    # If input has only partial separators, try using per-example output sep structure
    if not input_has_full_seps and out_sr and out_sc:
        def per_example_denoise(grid, bg_val):
            # Derive separator structure from this grid's own approximate sep lines
            # Use output sep structure from the first training example as template
            r_srs, r_scs = find_sep_lines(grid)
            # If partial, infer from spacing
            if r_srs and not r_scs:
                # Find tile width from column periodicity
                pass
            if r_scs and not r_srs:
                pass
            # Fall back: use output sep from whichever training example has same structure
            return None

        # Per-example approach: for each training example, get its own output sep
        def denoise_per_example(ex_input, ex_output, bg_val):
            e_out_sr, e_out_sc = find_sep_lines(ex_output)
            if not e_out_sr and not e_out_sc:
                return None
            return denoise_tiles_output_sep(ex_input, bg_val, e_out_sr, e_out_sc)

        # Check if per-example approach works on training
        train_results = [denoise_per_example(ex["input"], ex["output"], bg) for ex in train]
        if all(r is not None and r == ex["output"] for r, ex in zip(train_results, train)):
            # For test: use partially-visible separators in test input to find full structure
            # Strategy: find the most "separator-like" rows/cols
            def denoise_from_partial_seps(grid, bg_val):
                rows, cols = len(grid), len(grid[0])
                # Find rows with max bg fraction
                row_bg = [sum(1 for v in grid[r] if v == bg_val) / cols for r in range(rows)]
                col_bg = [sum(1 for r in range(rows) if grid[r][c] == bg_val) / rows
                          for c in range(cols)]

                # Strict sep lines first
                sep_rows = [r for r in range(rows) if row_bg[r] == 1.0]
                sep_cols = [c for c in range(cols) if col_bg[c] == 1.0]

                if sep_rows and sep_cols:
                    return denoise_tiles_output_sep(grid, bg_val, sep_rows, sep_cols)

                # Infer from detected separator + periodicity
                if sep_rows and not sep_cols:
                    # Find tile width by checking the most bg-heavy cols
                    bg_sorted_cols = sorted(range(cols), key=lambda c: -col_bg[c])
                    candidates = sorted(bg_sorted_cols[:max(1, cols//5)])
                    if candidates:
                        return denoise_tiles_output_sep(grid, bg_val, sep_rows, candidates)

                if sep_cols and not sep_rows:
                    bg_sorted_rows = sorted(range(rows), key=lambda r: -row_bg[r])
                    candidates = sorted(bg_sorted_rows[:max(1, rows//5)])
                    if candidates:
                        return denoise_tiles_output_sep(grid, bg_val, candidates, sep_cols)

                return None

            return lambda g, b=bg: denoise_from_partial_seps(g, b)

    results = [denoise_tiles(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: denoise_tiles(g, b)

    return None


# ─── Expand cross shape ───────────────────────────────────────────────────────

def _try_expand_cross_shape(train):
    """
    Input has small cross shapes (center_color + 4 arm_color neighbors).
    Output expands each cross: arms extend 2 steps, center color appears
    at diagonal positions (distance 1 and 2 from center).
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def find_crosses(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        crosses = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                v = grid[r][c]
                if v == bg_val:
                    continue
                arm_color = None
                arms = []
                ok = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    nv = grid[nr][nc]
                    if nv == bg_val or nv == v:
                        ok = False
                        break
                    if arm_color is None:
                        arm_color = nv
                    elif nv != arm_color:
                        ok = False
                        break
                    arms.append((nr, nc))
                if ok and len(arms) == 4:
                    crosses.append((r, c, v, arm_color))
        return crosses

    def expand_cross(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        crosses = find_crosses(grid, bg_val)
        if not crosses:
            return None
        result = [row[:] for row in grid]
        for cr, cc, center_color, arm_color in crosses:
            # Extend arms (distance 2)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + 2 * dr, cc + 2 * dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = arm_color
            # Center color at distance-1 diagonals
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = center_color
            # Center color at distance-2 diagonals
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = center_color
        return result

    results = [expand_cross(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: expand_cross(g, b)

    # Variant: only distance-1 diagonals (no distance-2)
    def expand_cross_v2(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        crosses = find_crosses(grid, bg_val)
        if not crosses:
            return None
        result = [row[:] for row in grid]
        for cr, cc, center_color, arm_color in crosses:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + 2 * dr, cc + 2 * dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = arm_color
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = center_color
        return result

    results = [expand_cross_v2(ex["input"], bg) for ex in train]
    if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
        return lambda g, b=bg: expand_cross_v2(g, b)

    return None


# ─── Diagonal gravity stacking ────────────────────────────────────────────────

def _try_diagonal_gravity_stack(train):
    """
    Rectangular objects form a diagonal staircase stack.
    Objects are sorted by column (left-to-right), then stacked from a corner:
    each object's top-left = previous object's bottom-right.
    Later objects overwrite the shared corner pixel.
    """
    bg = _bg(train)

    if not all(len(ex["input"]) == len(ex["output"]) and
               len(ex["input"][0]) == len(ex["output"][0]) for ex in train):
        return None

    def find_rect_objects(grid, bg_val):
        """Find solid-color rectangular objects."""
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        objects = []
        for sr in range(rows):
            for sc in range(cols):
                if visited[sr][sc] or grid[sr][sc] == bg_val:
                    continue
                cells = []
                color = grid[sr][sc]
                queue = [(sr, sc)]
                while queue:
                    r, c = queue.pop(0)
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    if visited[r][c] or grid[r][c] != color:
                        continue
                    visited[r][c] = True
                    cells.append((r, c))
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        queue.append((r+dr2, c+dc2))
                if cells:
                    min_r = min(r for r, c in cells)
                    max_r = max(r for r, c in cells)
                    min_c = min(c for r, c in cells)
                    max_c = max(c for r, c in cells)
                    h, w = max_r - min_r + 1, max_c - min_c + 1
                    if len(cells) == h * w:
                        objects.append({
                            "r0": min_r, "c0": min_c, "h": h, "w": w,
                            "color": color,
                        })
        return objects

    def staircase_stack(grid, bg_val, sort_by="col_asc", corner="top_left"):
        rows, cols = len(grid), len(grid[0])
        objects = find_rect_objects(grid, bg_val)
        if not objects:
            return grid

        # Sort objects by position
        if sort_by == "col_asc":
            objects.sort(key=lambda o: o["c0"])
        elif sort_by == "col_desc":
            objects.sort(key=lambda o: -o["c0"])
        elif sort_by == "row_asc":
            objects.sort(key=lambda o: o["r0"])
        elif sort_by == "row_desc":
            objects.sort(key=lambda o: -o["r0"])
        elif sort_by == "size_desc":
            objects.sort(key=lambda o: -(o["h"] * o["w"]))

        result = [[bg_val] * cols for _ in range(rows)]

        # Stack from corner: each object's top-left = previous object's bottom-right
        if corner == "top_left":
            cur_r, cur_c = 0, 0
            for obj in objects:
                h, w = obj["h"], obj["w"]
                for dr in range(h):
                    for dc in range(w):
                        r, c = cur_r + dr, cur_c + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = obj["color"]
                cur_r += h - 1
                cur_c += w - 1
        elif corner == "top_right":
            cur_r, cur_c = 0, cols - 1
            for obj in objects:
                h, w = obj["h"], obj["w"]
                c_start = cur_c - w + 1
                for dr in range(h):
                    for dc in range(w):
                        r, c = cur_r + dr, c_start + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = obj["color"]
                cur_r += h - 1
                cur_c = c_start + w - 1 - (w - 1)

        return result

    for sort_by in ["col_asc", "col_desc", "row_asc", "row_desc", "size_desc"]:
        for corner in ["top_left", "top_right"]:
            results = [staircase_stack(ex["input"], bg, sort_by, corner) for ex in train]
            if all(r is not None and r == ex["output"] for r, ex in zip(results, train)):
                return lambda g, b=bg, sb=sort_by, co=corner: staircase_stack(g, b, sb, co)

    return None
