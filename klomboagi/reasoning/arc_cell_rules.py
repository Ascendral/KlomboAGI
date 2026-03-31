"""
ARC Per-Cell Rule Learner — learn output = f(cell, neighborhood) from examples.

Many ARC tasks transform each cell based on its local context:
  - "If cell is color X and has neighbor Y, change to Z"
  - "If cell has exactly 2 non-bg neighbors, keep it"
  - "Count neighbors of each color, adopt the majority"

This module learns these rules from training examples by extracting
(input_features → output_value) pairs and finding consistent patterns.
"""

from __future__ import annotations

from collections import Counter

Grid = list[list[int]]


def learn_cell_rule(train: list[dict]) -> callable | None:
    """
    Try to learn a per-cell transformation rule from training examples.

    Returns a function (grid) -> grid if a consistent rule is found, else None.
    """
    # All examples must be same-size
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    # Strategy 1: Direct cell mapping (each input color maps to one output color)
    rule = _try_direct_color_map(train)
    if rule:
        return rule

    # Strategy 2: Neighborhood-based rule (cell + 4 neighbors → output)
    rule = _try_neighborhood_4(train)
    if rule:
        return rule

    # Strategy 3: Neighborhood-based rule (cell + 8 neighbors → output)
    rule = _try_neighborhood_8(train)
    if rule:
        return rule

    # Strategy 4: Count-based rule (number of non-bg neighbors determines output)
    rule = _try_count_rule(train)
    if rule:
        return rule

    # Strategy 5: Row/column position rule
    rule = _try_position_rule(train)
    if rule:
        return rule

    return None


def _try_direct_color_map(train: list[dict]) -> callable | None:
    """Check if each input color consistently maps to one output color."""
    mapping = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ic, oc = inp[r][c], out[r][c]
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None  # Inconsistent
                else:
                    mapping[ic] = oc

    if not mapping or all(k == v for k, v in mapping.items()):
        return None  # Identity or empty

    def apply_rule(grid: Grid) -> Grid:
        return [[mapping.get(c, c) for c in row] for row in grid]

    # Verify
    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_neighborhood_4(train: list[dict]) -> callable | None:
    """Learn rule based on cell value + 4-directional neighbors."""
    # Extract features: (cell_color, n_color, s_color, e_color, w_color) → output_color
    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                n = inp[r - 1][c] if r > 0 else -1
                s = inp[r + 1][c] if r < rows - 1 else -1
                w = inp[r][c - 1] if c > 0 else -1
                e = inp[r][c + 1] if c < cols - 1 else -1
                key = (inp[r][c], n, s, w, e)
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None  # Inconsistent
                rule_table[key] = val

    if not rule_table:
        return None

    # Check it's not just identity
    if all(k[0] == v for k, v in rule_table.items()):
        return None

    # Anti-overfitting check: if bg cells (value=0) map to different values via
    # unique neighborhood signatures (one-to-one memorization), the rule will
    # fail on test inputs with different contexts. Reject if too many unique
    # bg-to-nonbg mappings compared to distinct target values.
    bg_nonbg_entries = [(k, v) for k, v in rule_table.items() if k[0] == 0 and v != 0]
    if bg_nonbg_entries:
        unique_targets = len(set(v for _, v in bg_nonbg_entries))
        if len(bg_nonbg_entries) > unique_targets * 3:
            # Too many unique neighborhood signatures for few target values → overfitting
            return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                n = grid[r - 1][c] if r > 0 else -1
                s = grid[r + 1][c] if r < rows - 1 else -1
                w = grid[r][c - 1] if c > 0 else -1
                e = grid[r][c + 1] if c < cols - 1 else -1
                key = (grid[r][c], n, s, w, e)
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_neighborhood_8(train: list[dict]) -> callable | None:
    """Learn rule based on cell + 8 surrounding neighbors (sorted for rotation invariance)."""
    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(inp[nr][nc])
                        else:
                            neighbors.append(-1)
                # Use sorted neighbors for partial rotation invariance
                key = (inp[r][c], tuple(sorted(neighbors)))
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(grid[nr][nc])
                        else:
                            neighbors.append(-1)
                key = (grid[r][c], tuple(sorted(neighbors)))
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_count_rule(train: list[dict]) -> callable | None:
    """Learn rule based on count of non-bg neighbors."""
    # Find bg color
    all_vals = []
    for ex in train:
        for row in ex["input"]:
            all_vals.extend(row)
    bg = Counter(all_vals).most_common(1)[0][0]

    rule_table = {}
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                n_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and inp[nr][nc] != bg:
                            n_neighbors += 1
                key = (inp[r][c], n_neighbors)
                val = out[r][c]
                if key in rule_table and rule_table[key] != val:
                    return None
                rule_table[key] = val

    if not rule_table or all(k[0] == v for k, v in rule_table.items()):
        return None

    def apply_rule(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                n_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
                            n_neighbors += 1
                key = (grid[r][c], n_neighbors)
                result[r][c] = rule_table.get(key, grid[r][c])
        return result

    for ex in train:
        if apply_rule(ex["input"]) != ex["output"]:
            return None
    return apply_rule


def _try_position_rule(train: list[dict]) -> callable | None:
    """Learn rule based on (cell_color, row % N, col % N) for some period N."""
    for period in [2, 3, 4, 5]:
        rule_table = {}
        consistent = True
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    key = (inp[r][c], r % period, c % period)
                    val = out[r][c]
                    if key in rule_table and rule_table[key] != val:
                        consistent = False
                        break
                    rule_table[key] = val
                if not consistent:
                    break
            if not consistent:
                break

        if not consistent or not rule_table:
            continue
        if all(k[0] == v for k, v in rule_table.items()):
            continue

        p = period  # capture for closure

        def apply_rule(grid: Grid, rt=rule_table, per=p) -> Grid:
            rows, cols = len(grid), len(grid[0])
            return [[rt.get((grid[r][c], r % per, c % per), grid[r][c])
                     for c in range(cols)] for r in range(rows)]

        valid = all(apply_rule(ex["input"]) == ex["output"] for ex in train)
        if valid:
            return apply_rule

    return None


def learn_span_fill_rule(train: list[dict]) -> callable | None:
    """
    Try fill-row-span: in each row, fill between leftmost and rightmost cell of each color.
    """
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    from collections import defaultdict

    def fill_row_span(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            color_cols: dict[int, list] = defaultdict(list)
            for c in range(cols):
                if grid[r][c] != 0:
                    color_cols[grid[r][c]].append(c)
            for color, col_list in color_cols.items():
                if len(col_list) >= 2:
                    for c in range(min(col_list), max(col_list) + 1):
                        result[r][c] = color
        return result

    # Must not be identity
    if all(fill_row_span(ex["input"]) == ex["input"] for ex in train):
        return None

    if all(fill_row_span(ex["input"]) == ex["output"] for ex in train):
        return fill_row_span

    # Try column span instead: in each col, fill between topmost and bottommost cell
    def fill_col_span(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for c in range(cols):
            color_rows: dict[int, list] = defaultdict(list)
            for r in range(rows):
                if grid[r][c] != 0:
                    color_rows[grid[r][c]].append(r)
            for color, row_list in color_rows.items():
                if len(row_list) >= 2:
                    for r in range(min(row_list), max(row_list) + 1):
                        result[r][c] = color
        return result

    if all(fill_col_span(ex["input"]) == ex["input"] for ex in train):
        return None

    if all(fill_col_span(ex["input"]) == ex["output"] for ex in train):
        return fill_col_span

    return None


# ─── Color Key Swap ──────────────────────────────────────────────────────────

def learn_color_key_swap(train: list[dict]) -> callable | None:
    """
    Top-left 2×2 defines two color-swap pairs.

    Key layout:
        [a, b]   →  swap a↔b
        [c, d]   →  swap c↔d

    All non-zero, non-key cells in the grid are recolored according to
    these pairs.  The key itself is left untouched.
    Handles task 0becf7df.
    """
    # Same-size, at least 3×3
    for ex in train:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        if len(inp) < 3 or len(inp[0]) < 3:
            return None

    def apply_key_swap(grid):
        a, b = grid[0][0], grid[0][1]
        c, d = grid[1][0], grid[1][1]
        # All four key cells must be non-zero and distinct pairs
        if 0 in (a, b, c, d):
            return None
        swap = {a: b, b: a, c: d, d: c}
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            for cc in range(cols):
                if r < 2 and cc < 2:
                    continue  # preserve key
                v = grid[r][cc]
                if v in swap:
                    result[r][cc] = swap[v]
        return result

    results = [apply_key_swap(ex["input"]) for ex in train]
    if any(r is None for r in results):
        return None
    # Must actually change something
    if all(r == ex["input"] for r, ex in zip(results, train)):
        return None
    if all(r == ex["output"] for r, ex in zip(results, train)):
        return apply_key_swap
    return None


# ─── Template Row Stamp ──────────────────────────────────────────────────────

def learn_template_row_stamp(train: list[dict]) -> callable | None:
    """
    Row 0 is a template pattern.  One column (the "marker column") has
    isolated non-bg cells at certain rows — those rows get the template
    pattern stamped with a learned stamp color.

    Handles task 2281f1f4.
    """
    # Same-size
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0  # typical ARC background

    # ── Identify marker column: column where non-bg cells appear only in
    #    scattered single rows (not row 0) ──
    def _find_marker_col(grid, bg_val):
        rows, cols = len(grid), len(grid[0])
        for c in range(cols - 1, -1, -1):  # try rightmost first
            markers = [r for r in range(1, rows) if grid[r][c] != bg_val]
            if markers and grid[0][c] == bg_val:
                return c, markers
        return None, []

    # ── Identify template row: row 0 ──
    # ── Find stamp color from training output ──
    stamp_colors = set()
    for ex in train:
        inp, out = ex["input"], ex["output"]
        mc, markers = _find_marker_col(inp, bg)
        if mc is None:
            return None
        template_positions = [c for c in range(len(inp[0]))
                              if inp[0][c] != bg and c != mc]
        if not template_positions:
            return None
        for mr in markers:
            for c in template_positions:
                if out[mr][c] != bg:
                    stamp_colors.add(out[mr][c])
    if len(stamp_colors) != 1:
        return None
    stamp_color = stamp_colors.pop()

    def apply_stamp(grid, bg_val=bg, sc=stamp_color):
        rows, cols = len(grid), len(grid[0])
        mc, markers = _find_marker_col(grid, bg_val)
        if mc is None:
            return grid
        template_positions = [c for c in range(cols)
                              if grid[0][c] != bg_val and c != mc]
        result = [row[:] for row in grid]
        for mr in markers:
            for c in template_positions:
                result[mr][c] = sc
        return result

    if all(apply_stamp(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_stamp(ex["input"]) == ex["output"] for ex in train):
        return apply_stamp
    return None


# ─── Connect Dot Pairs ──────────────────────────────────────────────────────

def learn_connect_dot_pairs(train: list[dict]) -> callable | None:
    """
    Pairs of same-color dots sharing a row or column are connected
    by a line of a learned fill color between them.

    Handles task 253bf280.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    # Learn dot color and fill color from training
    dot_colors = set()
    fill_colors = set()
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != bg:
                    dot_colors.add(inp[r][c])
                if out[r][c] != bg and out[r][c] != inp[r][c]:
                    fill_colors.add(out[r][c])

    if len(dot_colors) != 1 or len(fill_colors) > 1:
        return None
    dot_color = dot_colors.pop()
    fill_color = fill_colors.pop() if fill_colors else None
    if fill_color is None:
        return None  # No change detected

    def apply_connect(grid, bg_val=bg, dc=dot_color, fc=fill_color):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Find all dot positions
        dots = [(r, c) for r in range(rows) for c in range(cols)
                if grid[r][c] == dc]

        # Group by row
        from collections import defaultdict
        by_row = defaultdict(list)
        by_col = defaultdict(list)
        for r, c in dots:
            by_row[r].append(c)
            by_col[c].append(r)

        # Connect pairs sharing a row
        for r, col_list in by_row.items():
            if len(col_list) == 2:
                c1, c2 = sorted(col_list)
                for c in range(c1 + 1, c2):
                    result[r][c] = fc

        # Connect pairs sharing a column
        for c, row_list in by_col.items():
            if len(row_list) == 2:
                r1, r2 = sorted(row_list)
                for r in range(r1 + 1, r2):
                    result[r][c] = fc

        return result

    if all(apply_connect(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_connect(ex["input"]) == ex["output"] for ex in train):
        return apply_connect
    return None


# ─── Grid Gap Fill ───────────────────────────────────────────────────────────

def learn_grid_gap_fill(train: list[dict]) -> callable | None:
    """
    Fill gaps in a regular block grid with 2 (inner) or 1 (edge/trailing).

    Input: regular grid of NxN blocks of color C with zero-gaps.
    Output: gaps between blocks → 2, gaps at edges → 1, outside → 0.

    Rules based on V-position × H-position:
      gap_row × in_group  → 2     gap_row × between → 2     gap_row × outer → 1
      blk_row × in_group  → keep  blk_row × between → 2     blk_row × outer → 0
      trail   × in_group  → 0     trail   × between → 1     trail   × outer → 0

    Handles task 137f0df0.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    def _group_runs(indices):
        """Group sorted indices into consecutive runs → list of sets."""
        if not indices:
            return []
        groups, cur = [], {indices[0]}
        for i in indices[1:]:
            if i == max(cur) + 1:
                cur.add(i)
            else:
                groups.append(cur)
                cur = {i}
        groups.append(cur)
        return groups

    def apply_gap_fill(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])

        # Find block color (non-bg color present in many cells)
        from collections import Counter
        flat = [v for row in grid for v in row if v != bg_val]
        if not flat:
            return grid
        block_color = Counter(flat).most_common(1)[0][0]

        # Find block rows and cols
        blk_rows = sorted({r for r in range(rows)
                           for c in range(cols) if grid[r][c] == block_color})
        blk_cols = sorted({c for r in range(rows)
                           for c in range(cols) if grid[r][c] == block_color})
        if not blk_rows or not blk_cols:
            return grid

        row_groups = _group_runs(blk_rows)
        col_groups = _group_runs(blk_cols)

        # Classify each column
        all_blk_cols = set().union(*col_groups)
        between_cols = set()
        for i in range(len(col_groups) - 1):
            lo = max(col_groups[i]) + 1
            hi = min(col_groups[i + 1])
            between_cols.update(range(lo, hi))
        outer_cols = set(range(cols)) - all_blk_cols - between_cols

        # Classify each row
        all_blk_rows = set().union(*row_groups)
        gap_rows = set()
        for i in range(len(row_groups) - 1):
            lo = max(row_groups[i]) + 1
            hi = min(row_groups[i + 1])
            gap_rows.update(range(lo, hi))
        rmin, rmax = min(all_blk_rows), max(all_blk_rows)
        trail_rows = set(range(rmax + 1, rows)) | set(range(0, rmin))

        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    continue  # Keep blocks as-is

                if r in gap_rows:
                    if c in between_cols or c in all_blk_cols:
                        result[r][c] = 2
                    elif c in outer_cols:
                        result[r][c] = 1
                elif r in all_blk_rows:
                    if c in between_cols:
                        result[r][c] = 2
                    # outer → 0 (already bg)
                elif r in trail_rows:
                    if c in between_cols:
                        result[r][c] = 1
                    # in_group or outer → 0 (already bg)

        return result

    if all(apply_gap_fill(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_gap_fill(ex["input"]) == ex["output"] for ex in train):
        return apply_gap_fill
    return None


# ─── Single-Cell Row/Col Paint ───────────────────────────────────────────────

def learn_single_cell_paint(train: list[dict]) -> callable | None:
    """
    Each isolated non-bg cell paints its entire row or entire column.

    Direction (row vs col) is learned per-color from training.
    Handles task 178fcbfb.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    # ── Learn direction per color from training (two-pass) ──
    color_dir: dict[int, str] = {}  # color → "row" or "col"

    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        # Pass 1: find row-painters (entire output row = color)
        row_paint_rows: set[int] = set()
        for r in range(rows):
            for c in range(cols):
                v = inp[r][c]
                if v == bg:
                    continue
                if all(out[r][cc] == v for cc in range(cols)):
                    if v in color_dir and color_dir[v] != "row":
                        return None
                    color_dir[v] = "row"
                    row_paint_rows.add(r)

        # Pass 2: find col-painters (column = color except at row-paint rows)
        for r in range(rows):
            for c in range(cols):
                v = inp[r][c]
                if v == bg or r in row_paint_rows:
                    continue
                # Check if column c has v at every non-row-paint row
                is_col = all(
                    out[rr][c] == v
                    for rr in range(rows) if rr not in row_paint_rows
                )
                if is_col:
                    if v in color_dir and color_dir[v] != "col":
                        return None
                    color_dir[v] = "col"
                else:
                    return None

    if not color_dir:
        return None

    def apply_paint(grid, bg_val=bg, cdir=color_dir):
        rows, cols = len(grid), len(grid[0])
        # Collect cells with directions
        row_paints: dict[int, int] = {}   # row → color
        col_paints: dict[int, int] = {}   # col → color
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == bg_val or v not in cdir:
                    continue
                if cdir[v] == "row":
                    row_paints[r] = v
                else:
                    col_paints[c] = v

        result = [[bg_val] * cols for _ in range(rows)]
        # Col paints first (rows override)
        for c, color in col_paints.items():
            for r in range(rows):
                result[r][c] = color
        for r, color in row_paints.items():
            for c in range(cols):
                result[r][c] = color
        return result

    if all(apply_paint(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_paint(ex["input"]) == ex["output"] for ex in train):
        return apply_paint
    return None


# ─── Cross from Dots ─────────────────────────────────────────────────────────

def learn_cross_from_dots(train: list[dict]) -> callable | None:
    """
    Each dot paints BOTH its full row AND full column with its color.
    Where two different-color crosses overlap → a learned intersection color.

    Handles task 23581191.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    # Learn intersection color from training
    int_colors = set()
    for ex in train:
        inp, out = ex["input"], ex["output"]
        rows, cols = len(inp), len(inp[0])
        for r in range(rows):
            for c in range(cols):
                if out[r][c] != bg and out[r][c] != inp[r][c]:
                    # Check if this is a new color not in input
                    if out[r][c] not in {v for row in inp for v in row if v != bg}:
                        int_colors.add(out[r][c])
    if len(int_colors) != 1:
        return None
    int_color = int_colors.pop()

    def apply_cross(grid, bg_val=bg, ic=int_color):
        rows, cols = len(grid), len(grid[0])
        # Find dots
        dots = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    dots.append((r, c, grid[r][c]))

        if len(dots) < 2:
            return grid

        # Paint row and col for each dot
        result = [[bg_val] * cols for _ in range(rows)]
        # Track which colors are assigned per cell
        cell_colors: dict = {}  # (r,c) -> set of colors

        for dr, dc, color in dots:
            for c in range(cols):
                cell_colors.setdefault((dr, c), set()).add(color)
            for r in range(rows):
                cell_colors.setdefault((r, dc), set()).add(color)

        for (r, c), colors in cell_colors.items():
            if len(colors) == 1:
                result[r][c] = list(colors)[0]
            else:
                result[r][c] = ic  # intersection

        return result

    if all(apply_cross(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_cross(ex["input"]) == ex["output"] for ex in train):
        return apply_cross
    return None


# ─── Diamond Expansion from Dots ─────────────────────────────────────────────

def learn_diamond_expand(train: list[dict]) -> callable | None:
    """
    Each dot in the first row expands into a diamond/V pattern downward.
    Even rows: center column. Odd rows: left and right columns.

    Handles task 3ac3eb23.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    # All non-bg cells must be in row 0
    for ex in train:
        inp = ex["input"]
        for r in range(1, len(inp)):
            if any(v != bg for v in inp[r]):
                return None

    def apply_diamond(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])
        result = [[bg_val] * cols for _ in range(rows)]

        # Find dots in row 0
        dots = [(c, grid[0][c]) for c in range(cols) if grid[0][c] != bg_val]

        for dot_c, color in dots:
            for r in range(rows):
                if r % 2 == 0:
                    if 0 <= dot_c < cols:
                        result[r][dot_c] = color
                else:
                    if 0 <= dot_c - 1 < cols:
                        result[r][dot_c - 1] = color
                    if 0 <= dot_c + 1 < cols:
                        result[r][dot_c + 1] = color

        return result

    if all(apply_diamond(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_diamond(ex["input"]) == ex["output"] for ex in train):
        return apply_diamond
    return None


# ─── Arrow Ray Shoot ─────────────────────────────────────────────────────────

def learn_arrow_ray(train: list[dict]) -> callable | None:
    """
    A triangle/arrow of one color has a differently-colored dot at its base.
    The dot shoots a ray in the direction the arrow points, to the grid edge.

    The arrow points AWAY from its widest row/col toward its narrowest.
    The ray extends from the tip (narrow end) outward.

    Handles task 25d487eb.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0

    def apply_arrow_ray(grid, bg_val=bg):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Find non-bg cells
        from collections import Counter
        color_cells = {}
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg_val:
                    color_cells.setdefault(grid[r][c], []).append((r, c))

        if len(color_cells) < 2:
            return result

        # Arrow color = most cells, dot color = fewest
        sorted_colors = sorted(color_cells.items(), key=lambda x: -len(x[1]))
        arrow_color = sorted_colors[0][0]
        dot_color = sorted_colors[1][0]
        dot_cells = color_cells[dot_color]
        arrow_cells = color_cells[arrow_color]

        if len(dot_cells) != 1:
            return result
        dot_r, dot_c = dot_cells[0]

        # Find arrow bounding box
        a_rows = [r for r, c in arrow_cells]
        a_cols = [c for r, c in arrow_cells]
        a_rmin, a_rmax = min(a_rows), max(a_rows)
        a_cmin, a_cmax = min(a_cols), max(a_cols)

        # Determine arrow direction by finding which end is narrow
        # Count cells per row and per col
        row_counts = Counter(r for r, c in arrow_cells)
        col_counts = Counter(c for r, c in arrow_cells)

        # Arrow points toward the narrow end
        # Check vertical direction (top vs bottom narrower)
        top_width = row_counts.get(a_rmin, 0)
        bot_width = row_counts.get(a_rmax, 0)
        left_height = col_counts.get(a_cmin, 0)
        right_height = col_counts.get(a_cmax, 0)

        # Determine direction: the tip (narrow end) is where the ray goes
        dr, dc = 0, 0
        if top_width < bot_width:
            dr, dc = -1, 0  # points up
        elif bot_width < top_width:
            dr, dc = 1, 0   # points down
        elif left_height < right_height:
            dr, dc = 0, -1  # points left
        elif right_height < left_height:
            dr, dc = 0, 1   # points right
        else:
            return result

        # Shoot ray from the tip in the arrow direction
        # Find the tip position (narrowest end, same row/col as dot)
        if dr != 0:  # vertical arrow
            tip_r = a_rmin if dr == -1 else a_rmax
            r, c = tip_r + dr, dot_c
        else:  # horizontal arrow
            tip_c = a_cmin if dc == -1 else a_cmax
            r, c = dot_r, tip_c + dc

        while 0 <= r < rows and 0 <= c < cols:
            if result[r][c] == bg_val:
                result[r][c] = dot_color
            r += dr
            c += dc

        return result

    if all(apply_arrow_ray(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_arrow_ray(ex["input"]) == ex["output"] for ex in train):
        return apply_arrow_ray
    return None


# ─── L-Shape Concavity Fill ──────────────────────────────────────────────────

def learn_lshape_concavity(train: list[dict]) -> callable | None:
    """
    Each 3-cell L-shaped connected component gets a marker at the cell
    that would complete it into a 2×2 square (the concavity).

    Handles task 3aa6fb7a.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0
    from collections import Counter

    # Learn shape color and fill color
    shape_colors = set()
    fill_colors = set()
    for ex in train:
        inp, out = ex["input"], ex["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != bg:
                    shape_colors.add(inp[r][c])
                if out[r][c] != bg and out[r][c] != inp[r][c]:
                    fill_colors.add(out[r][c])

    if len(shape_colors) != 1 or len(fill_colors) != 1:
        return None
    shape_color = shape_colors.pop()
    fill_color = fill_colors.pop()

    def apply_concavity(grid, bg_val=bg, sc=shape_color, fc=fill_color):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]

        # Find all 2×2 windows with exactly 3 shape-color cells
        for r in range(rows - 1):
            for c in range(cols - 1):
                block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
                if block.count(sc) == 3 and block.count(bg_val) == 1:
                    # Find the empty cell and fill it
                    for dr, dc in [(0,0), (0,1), (1,0), (1,1)]:
                        if grid[r+dr][c+dc] == bg_val:
                            result[r+dr][c+dc] = fc
                            break

        return result

    if all(apply_concavity(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(apply_concavity(ex["input"]) == ex["output"] for ex in train):
        return apply_concavity
    return None


# ─── Conditional Span Fill ───────────────────────────────────────────────────

def learn_conditional_span_fill(train: list[dict]) -> callable | None:
    """
    Like span_fill but only fills between two same-color cells if ALL
    cells in the gap are background. Won't fill if there are other
    non-bg cells in between.

    Handles task 5ad8a7c0.
    """
    for ex in train:
        if (len(ex["input"]) != len(ex["output"]) or
                len(ex["input"][0]) != len(ex["output"][0])):
            return None

    bg = 0
    from collections import defaultdict

    def fill_row_conditional(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(rows):
            color_cols = defaultdict(list)
            for c in range(cols):
                if grid[r][c] != bg:
                    color_cols[grid[r][c]].append(c)
            for color, col_list in color_cols.items():
                if len(col_list) >= 2:
                    lo, hi = min(col_list), max(col_list)
                    # Only fill if all cells between are bg
                    if all(grid[r][c] == bg for c in range(lo + 1, hi)):
                        for c in range(lo, hi + 1):
                            result[r][c] = color
        return result

    if all(fill_row_conditional(ex["input"]) == ex["input"] for ex in train):
        pass  # Try col version
    elif all(fill_row_conditional(ex["input"]) == ex["output"] for ex in train):
        return fill_row_conditional

    def fill_col_conditional(grid):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for c in range(cols):
            color_rows = defaultdict(list)
            for r in range(rows):
                if grid[r][c] != bg:
                    color_rows[grid[r][c]].append(r)
            for color, row_list in color_rows.items():
                if len(row_list) >= 2:
                    lo, hi = min(row_list), max(row_list)
                    if all(grid[r][c] == bg for r in range(lo + 1, hi)):
                        for r in range(lo, hi + 1):
                            result[r][c] = color
        return result

    if all(fill_col_conditional(ex["input"]) == ex["input"] for ex in train):
        return None
    if all(fill_col_conditional(ex["input"]) == ex["output"] for ex in train):
        return fill_col_conditional

    return None
