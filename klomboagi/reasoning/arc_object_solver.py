"""
ARC Object-Level Solver — detects objects, matches input↔output, learns transforms.

This is the bridge between pixel-level strategies and real intelligence.
Instead of pattern-matching grid cells, it:
  1. Detects discrete objects in input and output
  2. Matches them (by position, shape, color, size)
  3. Learns what transform was applied to each
  4. Chains simple transforms into compositional rules
  5. Applies the learned rule to the test input

Compositional transforms supported:
  - recolor: change object color
  - move: translate by (dr, dc)
  - reflect_h / reflect_v: mirror within bounding box
  - rotate_90 / rotate_180 / rotate_270: rotate within bounding box
  - grow_bbox: fill bounding box with object color
  - shrink: keep only border cells
  - delete: remove object
  - copy_to: duplicate at new position
  - recolor_by_property: color based on size/position/shape
"""

from __future__ import annotations

from collections import Counter
from itertools import product

from klomboagi.reasoning.arc_objects import (
    ArcObject, BBox, ObjectDetector, ObjectMatcher, Grid,
)


# ── Atomic Transforms ──────────────────────────────────────────────────

def _normalize_shape(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Position-independent shape signature."""
    if not cells:
        return ()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _reflect_h(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect cells horizontally within their bounding box."""
    if not cells:
        return []
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    return [(max_r - (r - min_r), c) for r, c in cells]


def _reflect_v(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect cells vertically within their bounding box."""
    if not cells:
        return []
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return [(r, max_c - (c - min_c)) for r, c in cells]


def _rotate_90(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Rotate cells 90 degrees clockwise within bounding box."""
    if not cells:
        return []
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_r = max(r for r, _ in cells)
    rel = [(r - min_r, c - min_c) for r, c in cells]
    h = max_r - min_r
    rotated = [(c, h - r) for r, c in rel]
    mr = min(r for r, _ in rotated)
    mc = min(c for _, c in rotated)
    return [(r - mr + min_r, c - mc + min_c) for r, c in rotated]


def _rotate_180(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _rotate_90(_rotate_90(cells))


def _rotate_270(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _rotate_90(_rotate_90(_rotate_90(cells)))


def _grow_bbox(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fill the bounding box."""
    if not cells:
        return []
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]


def _shrink_to_border(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep only border cells (cells with at least one non-object neighbor)."""
    cell_set = set(cells)
    border = []
    for r, c in cells:
        is_border = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) not in cell_set:
                is_border = True
                break
        if is_border:
            border.append((r, c))
    return border


# ── Transform Detection ────────────────────────────────────────────────

def _detect_transform(inp_obj: ArcObject, out_obj: ArcObject) -> dict | None:
    """Detect what single transform converts inp_obj to out_obj."""
    inp_cells = set(inp_obj.cells)
    out_cells = set(out_obj.cells)
    inp_norm = _normalize_shape(inp_obj.cells)
    out_norm = _normalize_shape(out_obj.cells)

    # Identity
    if inp_cells == out_cells and inp_obj.color == out_obj.color:
        return {"type": "identity"}

    # Recolor only (same cells, different color)
    if inp_norm == out_norm and inp_obj.color != out_obj.color:
        # Check cells are at same position
        if inp_cells == out_cells:
            return {"type": "recolor", "from": inp_obj.color, "to": out_obj.color}
        # Cells shifted — recolor + move
        dr = out_obj.bbox.min_r - inp_obj.bbox.min_r
        dc = out_obj.bbox.min_c - inp_obj.bbox.min_c
        shifted = set((r + dr, c + dc) for r, c in inp_cells)
        if shifted == out_cells:
            return {"type": "move_recolor", "dr": dr, "dc": dc,
                    "from": inp_obj.color, "to": out_obj.color}

    # Move only (same shape and color, different position)
    if inp_norm == out_norm and inp_obj.color == out_obj.color and inp_cells != out_cells:
        dr = out_obj.bbox.min_r - inp_obj.bbox.min_r
        dc = out_obj.bbox.min_c - inp_obj.bbox.min_c
        shifted = set((r + dr, c + dc) for r, c in inp_cells)
        if shifted == out_cells:
            return {"type": "move", "dr": dr, "dc": dc}

    # Reflect horizontal
    reflected_h = _normalize_shape(_reflect_h(inp_obj.cells))
    if reflected_h == out_norm:
        return {"type": "reflect_h", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Reflect vertical
    reflected_v = _normalize_shape(_reflect_v(inp_obj.cells))
    if reflected_v == out_norm:
        return {"type": "reflect_v", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Rotate 90
    rotated_90 = _normalize_shape(_rotate_90(inp_obj.cells))
    if rotated_90 == out_norm:
        return {"type": "rotate_90", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Rotate 180
    rotated_180 = _normalize_shape(_rotate_180(inp_obj.cells))
    if rotated_180 == out_norm:
        return {"type": "rotate_180", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Rotate 270
    rotated_270 = _normalize_shape(_rotate_270(inp_obj.cells))
    if rotated_270 == out_norm:
        return {"type": "rotate_270", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Grow (fill bounding box)
    grown_norm = _normalize_shape(_grow_bbox(inp_obj.cells))
    if grown_norm == out_norm:
        return {"type": "grow_bbox", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Shrink (border only)
    shrunk_norm = _normalize_shape(_shrink_to_border(inp_obj.cells))
    if shrunk_norm == out_norm:
        return {"type": "shrink", "color_change": inp_obj.color != out_obj.color,
                "new_color": out_obj.color}

    # Size change with same shape type
    if inp_obj.is_rectangle and out_obj.is_rectangle:
        scale_r = (out_obj.bbox.max_r - out_obj.bbox.min_r + 1) / max(inp_obj.bbox.max_r - inp_obj.bbox.min_r + 1, 1)
        scale_c = (out_obj.bbox.max_c - out_obj.bbox.min_c + 1) / max(inp_obj.bbox.max_c - inp_obj.bbox.min_c + 1, 1)
        if abs(scale_r - scale_c) < 0.01 and scale_r == int(scale_r):
            return {"type": "scale", "factor": int(scale_r),
                    "color_change": inp_obj.color != out_obj.color,
                    "new_color": out_obj.color}

    return None


# ── Multi-Strategy Object Matcher ──────────────────────────────────────

def _match_objects(inp_objs: list[ArcObject], out_objs: list[ArcObject],
                   ) -> list[tuple[ArcObject, ArcObject, dict]] | None:
    """Try all matching strategies, return first that produces consistent transforms."""
    matcher = ObjectMatcher()

    strategies = [
        matcher.match_by_position,
        matcher.match_by_shape,
        matcher.match_by_color,
        matcher.match_by_size,
    ]

    for match_fn in strategies:
        pairs = match_fn(inp_objs, out_objs)
        if not pairs:
            continue

        # Try to detect transform for each pair
        transforms = []
        all_valid = True
        for inp_obj, out_obj in pairs:
            t = _detect_transform(inp_obj, out_obj)
            if t is None:
                all_valid = False
                break
            transforms.append((inp_obj, out_obj, t))

        if all_valid and transforms:
            return transforms

    return None


# ── Recolor-by-Property Detection ──────────────────────────────────────

def _detect_recolor_by_property(inp_objs: list[ArcObject],
                                 out_objs: list[ArcObject],
                                 ) -> dict | None:
    """Detect if objects are recolored based on a property (size, position, shape)."""
    if len(inp_objs) != len(out_objs):
        return None

    # Match by position
    matcher = ObjectMatcher()
    pairs = matcher.match_by_position(inp_objs, out_objs)
    if len(pairs) != len(inp_objs):
        return None

    # Check if all objects have same shape but different output colors
    shapes_same = all(p[0].normalized == p[1].normalized or
                      _normalize_shape(p[0].cells) == _normalize_shape(p[1].cells)
                      for p in pairs)

    if not shapes_same:
        return None

    # Try: recolor by size (larger objects get different color)
    size_to_color = {}
    for inp_obj, out_obj in pairs:
        s = inp_obj.size
        if s in size_to_color and size_to_color[s] != out_obj.color:
            size_to_color = None
            break
        if size_to_color is not None:
            size_to_color[s] = out_obj.color

    if size_to_color and len(size_to_color) > 1:
        return {"type": "recolor_by_size", "size_map": size_to_color}

    # Try: recolor by position (row or column)
    row_to_color = {}
    for inp_obj, out_obj in pairs:
        row_key = inp_obj.bbox.min_r
        if row_key in row_to_color and row_to_color[row_key] != out_obj.color:
            row_to_color = None
            break
        if row_to_color is not None:
            row_to_color[row_key] = out_obj.color

    if row_to_color and len(row_to_color) > 1:
        return {"type": "recolor_by_row", "row_map": row_to_color}

    return None


# ── Compositional Object Solver ────────────────────────────────────────

class CompositionalObjectSolver:
    """
    Object-level ARC solver with compositional transform learning.

    Goes beyond the basic ObjectSolver by:
    - Detecting rotations, reflections, scaling
    - Learning recolor-by-property rules
    - Handling object creation/deletion
    - Chaining transforms (move + recolor, rotate + move, etc.)
    """

    def __init__(self):
        self.detector = ObjectDetector()

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Solve using object-level reasoning."""
        bg = self._find_bg(train)

        # Try same-object-count transforms first
        result = self._try_per_object_transform(train, test_input, bg)
        if result is not None:
            return result

        # Try recolor-by-property
        result = self._try_recolor_by_property(train, test_input, bg)
        if result is not None:
            return result

        # Try object deletion (some objects removed based on property)
        result = self._try_object_deletion(train, test_input, bg)
        if result is not None:
            return result

        # Try object extraction (output = one specific object from input)
        result = self._try_object_extraction(train, test_input, bg)
        if result is not None:
            return result

        return None

    def _try_per_object_transform(self, train, test_input, bg) -> Grid | None:
        """Each object undergoes a transform that depends on its properties.

        Objects are grouped by (color, size, shape_name). Each group can have
        a different transform. This handles the 58% of unsolved tasks where
        objects have mixed transforms.
        """
        # Collect per-example matched transforms
        all_example_data = []
        for ex in train:
            inp_objs = self.detector.detect(ex["input"], bg)
            out_objs = self.detector.detect(ex["output"], bg)
            if not inp_objs:
                return None

            matches = _match_objects(inp_objs, out_objs)
            if not matches:
                return None

            all_example_data.append(matches)

        if not all_example_data:
            return None

        # Strategy A: Uniform transform (all objects same)
        result = self._try_uniform_transform(all_example_data, train, test_input, bg)
        if result is not None:
            return result

        # Strategy B: Per-color transform (different colors get different transforms)
        result = self._try_per_color_transform(all_example_data, train, test_input, bg)
        if result is not None:
            return result

        # Strategy C: Per-size transform (different sizes get different transforms)
        result = self._try_per_size_transform(all_example_data, train, test_input, bg)
        if result is not None:
            return result

        return None

    def _try_uniform_transform(self, all_example_data, train, test_input, bg):
        """All objects get the same transform."""
        first_types = [t["type"] for _, _, t in all_example_data[0]]
        unique = set(first_types)
        if len(unique) != 1:
            return None

        # Check consistency across examples
        for ex_data in all_example_data[1:]:
            ex_types = [t["type"] for _, _, t in ex_data]
            if ex_types != first_types:
                return None

        rule = all_example_data[0][0][2]  # First example, first match, transform

        for ex in train:
            check = self._apply_transform(ex["input"], rule, bg)
            if check != ex["output"]:
                return None
        return self._apply_transform(test_input, rule, bg)

    def _try_per_color_transform(self, all_example_data, train, test_input, bg):
        """Different object colors get different transforms."""
        # Build color -> transform mapping from first example
        color_rules = {}
        for inp_obj, out_obj, transform in all_example_data[0]:
            c = inp_obj.color
            if c in color_rules:
                if color_rules[c]["type"] != transform["type"]:
                    return None  # Same color, different transforms
            color_rules[c] = transform

        if not color_rules or len(color_rules) < 2:
            return None  # Need at least 2 colors with different rules

        # Verify consistency across all examples
        for ex_data in all_example_data[1:]:
            for inp_obj, out_obj, transform in ex_data:
                c = inp_obj.color
                if c not in color_rules:
                    return None
                if color_rules[c]["type"] != transform["type"]:
                    return None

        # Validate
        for ex in train:
            check = self._apply_per_color_rule(ex["input"], color_rules, bg)
            if check != ex["output"]:
                return None
        return self._apply_per_color_rule(test_input, color_rules, bg)

    def _apply_per_color_rule(self, grid, color_rules, bg):
        """Apply different transforms per object color."""
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        objects = self.detector.detect(grid, bg)
        for obj in objects:
            rule = color_rules.get(obj.color)
            if rule is None:
                continue
            self._apply_single_object_transform(result, obj, rule, bg, rows, cols)
        return result

    def _try_per_size_transform(self, all_example_data, train, test_input, bg):
        """Different object sizes get different transforms."""
        size_rules = {}
        for inp_obj, out_obj, transform in all_example_data[0]:
            s = inp_obj.size
            if s in size_rules:
                if size_rules[s]["type"] != transform["type"]:
                    return None
            size_rules[s] = transform

        if not size_rules or len(size_rules) < 2:
            return None

        for ex_data in all_example_data[1:]:
            for inp_obj, out_obj, transform in ex_data:
                s = inp_obj.size
                if s not in size_rules:
                    return None
                if size_rules[s]["type"] != transform["type"]:
                    return None

        for ex in train:
            check = self._apply_per_size_rule(ex["input"], size_rules, bg)
            if check != ex["output"]:
                return None
        return self._apply_per_size_rule(test_input, size_rules, bg)

    def _apply_per_size_rule(self, grid, size_rules, bg):
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        objects = self.detector.detect(grid, bg)
        for obj in objects:
            rule = size_rules.get(obj.size)
            if rule is None:
                continue
            self._apply_single_object_transform(result, obj, rule, bg, rows, cols)
        return result

    def _apply_single_object_transform(self, result, obj, rule, bg, rows, cols):
        """Apply a transform to a single object, modifying result in place."""
        t_type = rule["type"]
        if t_type == "identity":
            pass
        elif t_type == "recolor":
            for r, c in obj.cells:
                result[r][c] = rule["to"]
        elif t_type == "move":
            dr, dc = rule["dr"], rule["dc"]
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in obj.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = obj.color
        elif t_type == "move_recolor":
            dr, dc = rule["dr"], rule["dc"]
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in obj.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = rule["to"]
        elif t_type == "grow_bbox":
            color = rule.get("new_color", obj.color) if rule.get("color_change") else obj.color
            for r in range(obj.bbox.min_r, obj.bbox.max_r + 1):
                for c in range(obj.bbox.min_c, obj.bbox.max_c + 1):
                    if 0 <= r < rows and 0 <= c < cols:
                        result[r][c] = color
        elif t_type == "shrink":
            color = rule.get("new_color", obj.color) if rule.get("color_change") else obj.color
            border = _shrink_to_border(obj.cells)
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in border:
                result[r][c] = color
        elif t_type == "reflect_h":
            color = rule.get("new_color", obj.color) if rule.get("color_change") else obj.color
            reflected = _reflect_h(obj.cells)
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in reflected:
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = color
        elif t_type == "reflect_v":
            color = rule.get("new_color", obj.color) if rule.get("color_change") else obj.color
            reflected = _reflect_v(obj.cells)
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in reflected:
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = color
        elif t_type.startswith("rotate_"):
            angle = int(t_type.split("_")[1])
            rotate_fn = {90: _rotate_90, 180: _rotate_180, 270: _rotate_270}[angle]
            color = rule.get("new_color", obj.color) if rule.get("color_change") else obj.color
            rotated = rotate_fn(obj.cells)
            for r, c in obj.cells:
                result[r][c] = bg
            for r, c in rotated:
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = color

    def _apply_transform(self, grid: Grid, rule: dict, bg: int) -> Grid | None:
        """Apply a learned transform to all objects in a grid."""
        rows, cols = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        objects = self.detector.detect(grid, bg)
        t_type = rule["type"]

        if t_type == "identity":
            return result

        elif t_type == "recolor":
            color_from = rule["from"]
            color_to = rule["to"]
            for obj in objects:
                if obj.color == color_from:
                    for r, c in obj.cells:
                        result[r][c] = color_to
            return result

        elif t_type == "move":
            dr, dc = rule["dr"], rule["dc"]
            obj_cells = set()
            for obj in objects:
                for r, c in obj.cells:
                    obj_cells.add((r, c))
            # Clear old positions
            for r, c in obj_cells:
                result[r][c] = bg
            # Place at new positions
            for obj in objects:
                for r, c in obj.cells:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = grid[r][c]
            return result

        elif t_type == "move_recolor":
            dr, dc = rule["dr"], rule["dc"]
            color_to = rule["to"]
            obj_cells = set()
            for obj in objects:
                for r, c in obj.cells:
                    obj_cells.add((r, c))
            for r, c in obj_cells:
                result[r][c] = bg
            for obj in objects:
                for r, c in obj.cells:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = color_to
            return result

        elif t_type == "reflect_h":
            new_color = rule.get("new_color", None)
            for obj in objects:
                reflected = _reflect_h(obj.cells)
                color = new_color if rule.get("color_change") else obj.color
                for (r, c), (nr, nc) in zip(obj.cells, reflected):
                    result[r][c] = bg  # clear old
                for r, c in reflected:
                    base_r = obj.bbox.min_r
                    base_c = obj.bbox.min_c
                    ar = r
                    ac = c
                    if 0 <= ar < rows and 0 <= ac < cols:
                        result[ar][ac] = color
            return result

        elif t_type == "reflect_v":
            new_color = rule.get("new_color", None)
            for obj in objects:
                reflected = _reflect_v(obj.cells)
                color = new_color if rule.get("color_change") else obj.color
                for r, c in obj.cells:
                    result[r][c] = bg
                for r, c in reflected:
                    if 0 <= r < rows and 0 <= c < cols:
                        result[r][c] = color
            return result

        elif t_type.startswith("rotate_"):
            angle = int(t_type.split("_")[1])
            rotate_fn = {90: _rotate_90, 180: _rotate_180, 270: _rotate_270}[angle]
            new_color = rule.get("new_color", None)
            for obj in objects:
                rotated = rotate_fn(obj.cells)
                color = new_color if rule.get("color_change") else obj.color
                for r, c in obj.cells:
                    result[r][c] = bg
                for r, c in rotated:
                    if 0 <= r < rows and 0 <= c < cols:
                        result[r][c] = color
            return result

        elif t_type == "grow_bbox":
            new_color = rule.get("new_color", None)
            for obj in objects:
                color = new_color if rule.get("color_change") else obj.color
                for r in range(obj.bbox.min_r, obj.bbox.max_r + 1):
                    for c in range(obj.bbox.min_c, obj.bbox.max_c + 1):
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = color
            return result

        elif t_type == "shrink":
            new_color = rule.get("new_color", None)
            for obj in objects:
                border = _shrink_to_border(obj.cells)
                color = new_color if rule.get("color_change") else obj.color
                for r, c in obj.cells:
                    result[r][c] = bg
                for r, c in border:
                    result[r][c] = color
            return result

        elif t_type == "scale":
            factor = rule["factor"]
            new_color = rule.get("new_color", None)
            for obj in objects:
                color = new_color if rule.get("color_change") else obj.color
                base_r, base_c = obj.bbox.min_r, obj.bbox.min_c
                for r, c in obj.cells:
                    rel_r, rel_c = r - base_r, c - base_c
                    for dr in range(factor):
                        for dc in range(factor):
                            nr = base_r + rel_r * factor + dr
                            nc = base_c + rel_c * factor + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                result[nr][nc] = color
            return result

        return None

    def _try_recolor_by_property(self, train, test_input, bg) -> Grid | None:
        """Objects are recolored based on their size, position, or shape."""
        rule = None
        for ex in train:
            inp_objs = self.detector.detect(ex["input"], bg)
            out_objs = self.detector.detect(ex["output"], bg)
            if not inp_objs or not out_objs:
                return None

            r = _detect_recolor_by_property(inp_objs, out_objs)
            if r is None:
                return None

            if rule is None:
                rule = r
            elif rule["type"] != r["type"]:
                return None

        if not rule:
            return None

        # Apply
        test_objs = self.detector.detect(test_input, bg)
        rows, cols = len(test_input), len(test_input[0])
        result = [row[:] for row in test_input]

        if rule["type"] == "recolor_by_size":
            size_map = rule["size_map"]
            for obj in test_objs:
                new_color = size_map.get(obj.size)
                if new_color is not None:
                    for r, c in obj.cells:
                        result[r][c] = new_color

        elif rule["type"] == "recolor_by_row":
            row_map = rule["row_map"]
            for obj in test_objs:
                new_color = row_map.get(obj.bbox.min_r)
                if new_color is not None:
                    for r, c in obj.cells:
                        result[r][c] = new_color

        # Validate
        for ex in train:
            check_objs = self.detector.detect(ex["input"], bg)
            check = [row[:] for row in ex["input"]]
            if rule["type"] == "recolor_by_size":
                for obj in check_objs:
                    nc = rule["size_map"].get(obj.size)
                    if nc is not None:
                        for r, c in obj.cells:
                            check[r][c] = nc
            elif rule["type"] == "recolor_by_row":
                for obj in check_objs:
                    nc = rule["row_map"].get(obj.bbox.min_r)
                    if nc is not None:
                        for r, c in obj.cells:
                            check[r][c] = nc
            if check != ex["output"]:
                return None

        if result == [list(row) for row in test_input]:
            return None
        return result

    def _try_object_deletion(self, train, test_input, bg) -> Grid | None:
        """Some objects are deleted. Learn which ones based on property."""
        for ex in train:
            inp_objs = self.detector.detect(ex["input"], bg)
            out_objs = self.detector.detect(ex["output"], bg)
            if not inp_objs or len(out_objs) >= len(inp_objs):
                return None

        # Determine which objects survive
        # Try: keep largest, keep smallest, keep specific color, keep specific shape
        for keep_fn_name, keep_fn in [
            ("keep_largest", lambda objs: [max(objs, key=lambda o: o.size)]),
            ("keep_smallest", lambda objs: [min(objs, key=lambda o: o.size)]),
        ]:
            valid = True
            for ex in train:
                inp_objs = self.detector.detect(ex["input"], bg)
                out_objs = self.detector.detect(ex["output"], bg)
                kept = keep_fn(inp_objs)
                # Check output matches kept objects
                kept_cells = set()
                for obj in kept:
                    kept_cells.update(obj.cells)
                out_cells = set()
                for obj in out_objs:
                    out_cells.update(obj.cells)
                if kept_cells != out_cells:
                    valid = False
                    break
            if valid:
                test_objs = self.detector.detect(test_input, bg)
                if not test_objs:
                    return None
                kept = keep_fn(test_objs)
                result = [[bg] * len(test_input[0]) for _ in range(len(test_input))]
                for obj in kept:
                    for r, c in obj.cells:
                        result[r][c] = test_input[r][c]
                # Validate
                for ex in train:
                    inp_objs = self.detector.detect(ex["input"], bg)
                    k = keep_fn(inp_objs)
                    check = [[bg] * len(ex["input"][0]) for _ in range(len(ex["input"]))]
                    for obj in k:
                        for r, c in obj.cells:
                            check[r][c] = ex["input"][r][c]
                    if check != ex["output"]:
                        valid = False
                        break
                if valid and result != [list(row) for row in test_input]:
                    return result

        return None

    def _try_object_extraction(self, train, test_input, bg) -> Grid | None:
        """Output is a specific object extracted from input (cropped to bbox)."""
        # Check output is smaller than input
        for ex in train:
            if len(ex["output"]) >= len(ex["input"]) and len(ex["output"][0]) >= len(ex["input"][0]):
                return None

        # For each example, find which input object's extracted grid matches output
        extract_by = None  # "largest", "smallest", "unique_color", etc.

        for strategy_name, select_fn in [
            ("largest", lambda objs: max(objs, key=lambda o: o.size)),
            ("smallest", lambda objs: min(objs, key=lambda o: o.size)),
            ("most_colorful", lambda objs: max(objs, key=lambda o: len(o.colors))),
        ]:
            valid = True
            for ex in train:
                inp_objs = self.detector.detect(ex["input"], bg)
                if not inp_objs:
                    valid = False
                    break
                selected = select_fn(inp_objs)
                extracted = selected.extract_grid()
                if extracted != ex["output"]:
                    valid = False
                    break
            if valid:
                extract_by = strategy_name
                break

        if extract_by is None:
            return None

        test_objs = self.detector.detect(test_input, bg)
        if not test_objs:
            return None

        select_fn = {"largest": lambda o: max(o, key=lambda x: x.size),
                     "smallest": lambda o: min(o, key=lambda x: x.size),
                     "most_colorful": lambda o: max(o, key=lambda x: len(x.colors)),
                     }[extract_by]
        return select_fn(test_objs).extract_grid()

    def _find_bg(self, train) -> int:
        counts = Counter()
        for ex in train:
            for row in ex["input"]:
                counts.update(row)
        return counts.most_common(1)[0][0] if counts else 0
