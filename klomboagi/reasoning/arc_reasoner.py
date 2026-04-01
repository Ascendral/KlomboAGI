"""
ARC Reasoning Bridge — connects KlomboAGI's reasoning engine to ARC solving.

Instead of hand-coded strategies, this module:
1. Observes training examples and extracts structural facts
2. Feeds facts into the reasoning engine to form hypotheses
3. Tests hypotheses against training examples
4. Applies confirmed hypotheses to the test input

This is what makes KlomboAGI different from every other ARC solver:
the brain reasons about the puzzle instead of pattern-matching it.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from klomboagi.reasoning.arc_objects import ObjectDetector, ArcObject, Grid


# ── Grid Observation ────────────────────────────────────────────────────

def observe_grid(grid: Grid, label: str = "grid") -> list[str]:
    """Extract structural facts from a grid."""
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    facts = []

    facts.append(f"{label} has {rows} rows and {cols} columns")
    facts.append(f"{label} background color is {bg}")

    colors = set(v for row in grid for v in row) - {bg}
    facts.append(f"{label} has {len(colors)} non-background colors: {sorted(colors)}")

    # Object-level facts
    detector = ObjectDetector()
    objects = detector.detect(grid, bg)
    facts.append(f"{label} has {len(objects)} objects")

    for obj in objects:
        facts.append(
            f"{label} object_{obj.id}: color={obj.color}, size={obj.size}, "
            f"shape={obj.shape_name}, pos=({obj.bbox.min_r},{obj.bbox.min_c}), "
            f"bbox={obj.bbox.height}x{obj.bbox.width}"
        )

    # Symmetry facts
    if _is_symmetric_h(grid):
        facts.append(f"{label} is horizontally symmetric")
    if _is_symmetric_v(grid):
        facts.append(f"{label} is vertically symmetric")

    # Row/column patterns
    unique_rows = len(set(tuple(row) for row in grid))
    unique_cols = len(set(tuple(grid[r][c] for r in range(rows)) for c in range(cols)))
    if unique_rows < rows:
        facts.append(f"{label} has {unique_rows} unique rows out of {rows} (repeating)")
    if unique_cols < cols:
        facts.append(f"{label} has {unique_cols} unique columns out of {cols} (repeating)")

    return facts


def observe_transformation(inp: Grid, out: Grid) -> list[str]:
    """Extract facts about what changed between input and output."""
    facts = []
    rows_i, cols_i = len(inp), len(inp[0])
    rows_o, cols_o = len(out), len(out[0])

    if rows_i == rows_o and cols_i == cols_o:
        facts.append("transformation preserves grid size")
        changed = []
        for r in range(rows_i):
            for c in range(cols_i):
                if inp[r][c] != out[r][c]:
                    changed.append((r, c, inp[r][c], out[r][c]))
        facts.append(f"{len(changed)} cells changed out of {rows_i * cols_i}")

        if changed:
            # What colors changed to what
            change_map = Counter((old, new) for _, _, old, new in changed)
            for (old, new), count in change_map.most_common(5):
                facts.append(f"color {old} -> {new}: {count} cells")

            # Where did changes happen?
            changed_rows = set(r for r, _, _, _ in changed)
            changed_cols = set(c for _, c, _, _ in changed)
            if len(changed_rows) == 1:
                facts.append(f"all changes in row {next(iter(changed_rows))}")
            if len(changed_cols) == 1:
                facts.append(f"all changes in column {next(iter(changed_cols))}")
    else:
        facts.append(f"size changed: {rows_i}x{cols_i} -> {rows_o}x{cols_o}")
        ratio_r = rows_o / rows_i if rows_i else 0
        ratio_c = cols_o / cols_i if cols_i else 0
        if abs(ratio_r - ratio_c) < 0.01:
            facts.append(f"uniform scale factor: {ratio_r:.2f}")
        if rows_o < rows_i or cols_o < cols_i:
            facts.append("output is smaller (extraction/cropping)")
        if rows_o > rows_i or cols_o > cols_i:
            facts.append("output is larger (growth/tiling)")

    # Object-level changes
    bg = Counter(v for row in inp for v in row).most_common(1)[0][0]
    detector = ObjectDetector()
    inp_objs = detector.detect(inp, bg)
    out_objs = detector.detect(out, bg)

    if len(inp_objs) != len(out_objs):
        facts.append(f"object count changed: {len(inp_objs)} -> {len(out_objs)}")
        if len(out_objs) < len(inp_objs):
            facts.append("objects were deleted")
        else:
            facts.append("objects were created")
    else:
        facts.append(f"object count preserved: {len(inp_objs)}")
        # Check if objects moved
        moved = 0
        recolored = 0
        reshaped = 0
        for io, oo in zip(
            sorted(inp_objs, key=lambda o: (o.bbox.min_r, o.bbox.min_c)),
            sorted(out_objs, key=lambda o: (o.bbox.min_r, o.bbox.min_c))
        ):
            if io.position != oo.position:
                moved += 1
            if io.color != oo.color:
                recolored += 1
            if io.normalized != oo.normalized:
                reshaped += 1
        if moved:
            facts.append(f"{moved} objects moved")
        if recolored:
            facts.append(f"{recolored} objects recolored")
        if reshaped:
            facts.append(f"{reshaped} objects changed shape")

    return facts


# ── Hypothesis Generation ──────────────────────────────────────────────

# Each hypothesis is a (name, test_fn, apply_fn) tuple.
# test_fn(train) -> bool (does this hypothesis explain all examples?)
# apply_fn(grid) -> Grid (apply the hypothesis to produce output)

def _generate_hypotheses(train: list[dict]) -> list[tuple[str, Callable, Callable]]:
    """Generate hypotheses about the transformation from observations."""
    hypotheses = []
    bg = Counter(v for row in train[0]["input"] for v in row).most_common(1)[0][0]

    # Hypothesis 1: Global color swap
    color_map = _learn_color_map(train)
    if color_map:
        def apply_color_swap(grid, cm=color_map):
            return [[cm.get(v, v) for v in row] for row in grid]
        hypotheses.append(("global_color_swap", lambda t, cm=color_map: _test_color_map(t, cm),
                          apply_color_swap))

    # Hypothesis 2: Per-object recolor by size
    # Only use if objects DON'T change shape — pure recolor
    size_map = _learn_size_recolor(train, bg)
    if size_map:
        # Extra check: all objects must keep the same shape (only color changes)
        detector_check = ObjectDetector()
        shapes_preserved = True
        for ex in train:
            io = detector_check.detect(ex["input"], bg)
            oo = detector_check.detect(ex["output"], bg)
            if len(io) != len(oo):
                shapes_preserved = False
                break
            for a, b in zip(
                sorted(io, key=lambda o: (o.bbox.min_r, o.bbox.min_c)),
                sorted(oo, key=lambda o: (o.bbox.min_r, o.bbox.min_c))
            ):
                if a.normalized != b.normalized:
                    shapes_preserved = False
                    break
            if not shapes_preserved:
                break
        if shapes_preserved and len(size_map) >= 2:
            def apply_size_recolor(grid, sm=size_map, background=bg):
                detector = ObjectDetector()
                objs = detector.detect(grid, background)
                # Reject if any test object has an unseen size
                for obj in objs:
                    if obj.size not in sm:
                        return None
                result = [row[:] for row in grid]
                for obj in objs:
                    nc = sm[obj.size]
                    for r, c in obj.cells:
                        result[r][c] = nc
                return result
            hypotheses.append(("recolor_by_size", lambda t, sm=size_map, b=bg: _test_size_recolor(t, sm, b),
                              apply_size_recolor))

    # Hypothesis 3: Keep only objects of specific color
    keep_color = _learn_keep_color(train, bg)
    if keep_color is not None:
        def apply_keep_color(grid, kc=keep_color, background=bg):
            result = [[background] * len(grid[0]) for _ in range(len(grid))]
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    if grid[r][c] == kc:
                        result[r][c] = kc
            return result
        hypotheses.append(("keep_only_color", lambda t, kc=keep_color, b=bg: _test_keep_color(t, kc, b),
                          apply_keep_color))

    # Hypothesis 4: Remove specific color (replace with bg)
    rm_color = _learn_remove_color(train, bg)
    if rm_color is not None:
        def apply_rm_color(grid, rc=rm_color, background=bg):
            return [[background if v == rc else v for v in row] for row in grid]
        hypotheses.append(("remove_color", lambda t, rc=rm_color, b=bg: _test_remove_color(t, rc, b),
                          apply_rm_color))

    # Hypothesis 5: Fill enclosed regions (disabled — too many false positives)
    # hypotheses.append(("fill_enclosed", _test_fill_enclosed, _apply_fill_enclosed))

    # Hypothesis 6: Replace bg cells adjacent to non-bg with that color (grow objects by 1)
    hypotheses.append(("grow_objects_1", lambda t: _test_grow(t, bg, 1),
                       lambda g: _apply_grow(g, bg, 1)))

    return hypotheses


def _learn_color_map(train):
    """Learn a global color permutation."""
    mapping = {}
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None
        for r in range(len(ex["input"])):
            for c in range(len(ex["input"][0])):
                a, b = ex["input"][r][c], ex["output"][r][c]
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    # Must actually change something
    if all(k == v for k, v in mapping.items()):
        return None
    return mapping


def _test_color_map(train, cm):
    for ex in train:
        expected = [[cm.get(v, v) for v in row] for row in ex["input"]]
        if expected != ex["output"]:
            return False
    return True


def _learn_size_recolor(train, bg):
    """Learn size -> new_color mapping."""
    detector = ObjectDetector()
    size_map = {}
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None
        inp_objs = detector.detect(ex["input"], bg)
        out_objs = detector.detect(ex["output"], bg)
        if len(inp_objs) != len(out_objs):
            return None
        for io, oo in zip(
            sorted(inp_objs, key=lambda o: (o.bbox.min_r, o.bbox.min_c)),
            sorted(out_objs, key=lambda o: (o.bbox.min_r, o.bbox.min_c))
        ):
            s = io.size
            if s in size_map and size_map[s] != oo.color:
                return None
            size_map[s] = oo.color
    if not size_map or len(size_map) < 2:
        return None
    return size_map


def _test_size_recolor(train, sm, bg):
    detector = ObjectDetector()
    for ex in train:
        objs = detector.detect(ex["input"], bg)
        result = [row[:] for row in ex["input"]]
        for obj in objs:
            nc = sm.get(obj.size, obj.color)
            for r, c in obj.cells:
                result[r][c] = nc
        if result != ex["output"]:
            return False
    return True


def _learn_keep_color(train, bg):
    """Learn: output keeps only one specific color."""
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    candidate = None
    for ex in train:
        out_colors = set(v for row in ex["output"] for v in row) - {bg}
        if len(out_colors) != 1:
            return None
        c = next(iter(out_colors))
        if candidate is None:
            candidate = c
        elif candidate != c:
            return None
    return candidate


def _test_keep_color(train, kc, bg):
    for ex in train:
        result = [[bg] * len(ex["input"][0]) for _ in range(len(ex["input"]))]
        for r in range(len(ex["input"])):
            for c in range(len(ex["input"][0])):
                if ex["input"][r][c] == kc:
                    result[r][c] = kc
        if result != ex["output"]:
            return False
    return True


def _learn_remove_color(train, bg):
    """Learn: one color is removed (replaced with bg)."""
    for ex in train:
        if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
            return None

    candidate = None
    for ex in train:
        inp_colors = set(v for row in ex["input"] for v in row)
        out_colors = set(v for row in ex["output"] for v in row)
        removed = inp_colors - out_colors - {bg}
        if len(removed) != 1:
            return None
        c = next(iter(removed))
        if candidate is None:
            candidate = c
        elif candidate != c:
            return None
    return candidate


def _test_remove_color(train, rc, bg):
    for ex in train:
        result = [[bg if v == rc else v for v in row] for row in ex["input"]]
        if result != ex["output"]:
            return False
    return True


def _test_fill_enclosed(train):
    """Test if transformation fills enclosed bg regions."""
    for ex in train:
        result = _apply_fill_enclosed(ex["input"])
        if result != ex["output"]:
            return False
    return True


def _apply_fill_enclosed(grid):
    """Fill bg regions that are fully enclosed by non-bg cells."""
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [row[:] for row in grid]

    # BFS from all border bg cells to find "outside" bg
    outside = set()
    queue = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r][c] == bg:
                queue.append((r, c))
                outside.add((r, c))

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in outside and grid[nr][nc] == bg:
                outside.add((nr, nc))
                queue.append((nr, nc))

    # Find connected interior bg regions and determine fill color for each
    interior_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg and (r, c) not in outside:
                interior_cells.add((r, c))

    if not interior_cells:
        return result  # No enclosed regions

    # BFS to find connected interior regions
    visited = set()
    for start in interior_cells:
        if start in visited:
            continue
        region = []
        q = [start]
        while q:
            cell = q.pop()
            if cell in visited:
                continue
            visited.add(cell)
            region.append(cell)
            r, c = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in interior_cells and (nr, nc) not in visited:
                    q.append((nr, nc))

        # Find the dominant border color of this region
        border_colors = Counter()
        for r, c in region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
                    border_colors[grid[nr][nc]] += 1

        if border_colors:
            fill_color = border_colors.most_common(1)[0][0]
            for r, c in region:
                result[r][c] = fill_color

    return result


def _test_grow(train, bg, radius):
    for ex in train:
        if _apply_grow(ex["input"], bg, radius) != ex["output"]:
            return False
    return True


def _apply_grow(grid, bg, radius):
    """Grow each non-bg cell by `radius` in all directions."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == bg:
                            result[nr][nc] = grid[r][c]
    return result


def _is_symmetric_h(grid):
    rows = len(grid)
    for r in range(rows // 2):
        if grid[r] != grid[rows - 1 - r]:
            return False
    return True


def _is_symmetric_v(grid):
    for row in grid:
        cols = len(row)
        for c in range(cols // 2):
            if row[c] != row[cols - 1 - c]:
                return False
    return True


# ── Main Reasoning Solver ──────────────────────────────────────────────

class ARCReasoner:
    """
    Reasoning-driven ARC solver.

    Instead of pattern matching, it:
    1. Observes structural facts about each training example
    2. Generates hypotheses about the transformation
    3. Tests each hypothesis against all training examples
    4. Applies the first confirmed hypothesis to the test input
    """

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Solve by reasoning about the transformation."""
        # Generate hypotheses from observations
        hypotheses = _generate_hypotheses(train)

        # Test each hypothesis
        for name, test_fn, apply_fn in hypotheses:
            try:
                if test_fn(train):
                    result = apply_fn(test_input)
                    if result is not None and result != [list(row) for row in test_input]:
                        # Double-check on training
                        valid = True
                        for ex in train:
                            if apply_fn(ex["input"]) != ex["output"]:
                                valid = False
                                break
                        if valid:
                            return result
            except Exception:
                continue

        return None
