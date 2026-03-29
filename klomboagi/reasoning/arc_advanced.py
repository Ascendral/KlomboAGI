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
