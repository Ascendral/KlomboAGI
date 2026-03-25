"""
ARC Object Detection — see shapes, not pixels.

This is the missing piece. Current solver treats grids as flat arrays.
Real ARC puzzles are about OBJECTS — shapes that move, transform, copy,
and interact with each other.

An object is:
- A connected component of non-background cells
- Has properties: color, size, shape, bounding box, position
- Has relationships to other objects: above, below, inside, touching, same-shape

The object system:
1. DETECT — find all objects in a grid
2. DESCRIBE — extract properties of each object
3. MATCH — find corresponding objects between input and output
4. LEARN — discover what transform was applied per-object
5. APPLY — apply learned transform to test input objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

Grid = list[list[int]]


@dataclass
class BBox:
    """Bounding box."""
    min_r: int
    max_r: int
    min_c: int
    max_c: int

    @property
    def height(self) -> int: return self.max_r - self.min_r + 1

    @property
    def width(self) -> int: return self.max_c - self.min_c + 1

    @property
    def area(self) -> int: return self.height * self.width

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_r + self.max_r) / 2, (self.min_c + self.max_c) / 2)

    def overlaps(self, other: BBox) -> bool:
        return not (self.max_r < other.min_r or self.min_r > other.max_r or
                    self.max_c < other.min_c or self.min_c > other.max_c)

    def contains(self, other: BBox) -> bool:
        return (self.min_r <= other.min_r and self.max_r >= other.max_r and
                self.min_c <= other.min_c and self.max_c >= other.max_c)

    def to_dict(self) -> dict:
        return {"min_r": self.min_r, "max_r": self.max_r,
                "min_c": self.min_c, "max_c": self.max_c}


@dataclass
class ArcObject:
    """A discrete object detected in an ARC grid."""
    id: int
    cells: list[tuple[int, int]]         # (row, col) positions
    color: int                            # Primary color
    colors: dict[int, int] = field(default_factory=dict)  # color → count
    bbox: BBox = field(default=None)
    
    # Shape properties (computed)
    is_rectangle: bool = False
    is_square: bool = False
    is_line_h: bool = False
    is_line_v: bool = False
    is_dot: bool = False
    is_l_shape: bool = False
    
    # Normalized shape (position-independent)
    normalized: tuple = field(default_factory=tuple)
    
    def __post_init__(self):
        if self.cells:
            rs = [r for r, c in self.cells]
            cs = [c for r, c in self.cells]
            self.bbox = BBox(min(rs), max(rs), min(cs), max(cs))
            self._compute_shape()
            self._normalize()

    def _compute_shape(self):
        """Determine what shape this object is."""
        h, w = self.bbox.height, self.bbox.width
        n = len(self.cells)
        
        self.is_dot = (n == 1)
        self.is_line_h = (h == 1 and w > 1)
        self.is_line_v = (w == 1 and h > 1)
        self.is_rectangle = (n == h * w and h > 1 and w > 1)
        self.is_square = (self.is_rectangle and h == w)
        
        # L-shape: 3 cells in an L pattern
        if n == 3 and not self.is_line_h and not self.is_line_v:
            self.is_l_shape = True

    def _normalize(self):
        """Create position-independent representation of shape."""
        if not self.cells:
            self.normalized = ()
            return
        min_r = self.bbox.min_r
        min_c = self.bbox.min_c
        # Relative positions sorted
        self.normalized = tuple(sorted((r - min_r, c - min_c) for r, c in self.cells))

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def position(self) -> tuple[int, int]:
        """Top-left corner."""
        return (self.bbox.min_r, self.bbox.min_c)

    @property
    def shape_name(self) -> str:
        if self.is_dot: return "dot"
        if self.is_line_h: return "hline"
        if self.is_line_v: return "vline"
        if self.is_square: return "square"
        if self.is_rectangle: return "rectangle"
        if self.is_l_shape: return "l_shape"
        return f"shape_{self.size}"

    def same_shape(self, other: ArcObject) -> bool:
        """Same shape regardless of position and color."""
        return self.normalized == other.normalized

    def extract_grid(self) -> Grid:
        """Extract this object as a minimal grid."""
        h, w = self.bbox.height, self.bbox.width
        grid = [[0] * w for _ in range(h)]
        for r, c in self.cells:
            grid[r - self.bbox.min_r][c - self.bbox.min_c] = self.color
        return grid

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "color": self.color,
            "size": self.size,
            "shape": self.shape_name,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "position": self.position,
            "is_rectangle": self.is_rectangle,
            "is_square": self.is_square,
        }


class ObjectDetector:
    """Detect discrete objects in an ARC grid."""

    def __init__(self):
        self._next_id = 0

    def detect(self, grid: Grid, bg: int | None = None,
               connectivity: int = 4) -> list[ArcObject]:
        """
        Find all objects in a grid.
        
        connectivity: 4 (cardinal only) or 8 (including diagonals)
        """
        if bg is None:
            bg = self._find_bg(grid)

        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        objects = []

        deltas_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        deltas_8 = deltas_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        deltas = deltas_8 if connectivity == 8 else deltas_4

        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or grid[r][c] == bg:
                    continue

                # BFS to find connected component
                color = grid[r][c]
                cells = []
                colors = Counter()
                queue = [(r, c)]

                while queue:
                    cr, cc = queue.pop(0)
                    if visited[cr][cc]:
                        continue
                    if grid[cr][cc] == bg:
                        continue
                    # Same-color connectivity
                    if grid[cr][cc] != color:
                        continue

                    visited[cr][cc] = True
                    cells.append((cr, cc))
                    colors[grid[cr][cc]] += 1

                    for dr, dc in deltas:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                            queue.append((nr, nc))

                if cells:
                    obj = ArcObject(
                        id=self._next_id,
                        cells=cells,
                        color=color,
                        colors=dict(colors),
                    )
                    objects.append(obj)
                    self._next_id += 1

        return objects

    def detect_multicolor(self, grid: Grid, bg: int | None = None) -> list[ArcObject]:
        """Detect objects that may contain multiple colors (connected non-bg regions)."""
        if bg is None:
            bg = self._find_bg(grid)

        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        objects = []

        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or grid[r][c] == bg:
                    continue

                cells = []
                colors = Counter()
                queue = [(r, c)]

                while queue:
                    cr, cc = queue.pop(0)
                    if visited[cr][cc] or grid[cr][cc] == bg:
                        continue

                    visited[cr][cc] = True
                    cells.append((cr, cc))
                    colors[grid[cr][cc]] += 1

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                            queue.append((nr, nc))

                if cells:
                    primary = colors.most_common(1)[0][0]
                    obj = ArcObject(
                        id=self._next_id,
                        cells=cells,
                        color=primary,
                        colors=dict(colors),
                    )
                    objects.append(obj)
                    self._next_id += 1

        return objects

    def _find_bg(self, grid: Grid) -> int:
        """Find background color (most common)."""
        counts = Counter()
        for row in grid:
            counts.update(row)
        return counts.most_common(1)[0][0] if counts else 0


class ObjectMatcher:
    """Match objects between input and output grids."""

    def match_by_position(self, input_objs: list[ArcObject],
                          output_objs: list[ArcObject]) -> list[tuple[ArcObject, ArcObject]]:
        """Match objects that are at the same position."""
        matches = []
        used_out = set()
        for io in input_objs:
            best = None
            best_dist = float('inf')
            for oo in output_objs:
                if oo.id in used_out:
                    continue
                # Manhattan distance between centers
                ic = io.bbox.center
                oc = oo.bbox.center
                dist = abs(ic[0] - oc[0]) + abs(ic[1] - oc[1])
                if dist < best_dist:
                    best_dist = dist
                    best = oo
            if best and best_dist < max(len(input_objs), len(output_objs)) * 2:
                matches.append((io, best))
                used_out.add(best.id)
        return matches

    def match_by_shape(self, input_objs: list[ArcObject],
                       output_objs: list[ArcObject]) -> list[tuple[ArcObject, ArcObject]]:
        """Match objects with the same normalized shape."""
        matches = []
        used_out = set()
        for io in input_objs:
            for oo in output_objs:
                if oo.id in used_out:
                    continue
                if io.same_shape(oo):
                    matches.append((io, oo))
                    used_out.add(oo.id)
                    break
        return matches

    def match_by_color(self, input_objs: list[ArcObject],
                       output_objs: list[ArcObject]) -> list[tuple[ArcObject, ArcObject]]:
        """Match objects with the same color."""
        matches = []
        used_out = set()
        for io in input_objs:
            for oo in output_objs:
                if oo.id in used_out:
                    continue
                if io.color == oo.color:
                    matches.append((io, oo))
                    used_out.add(oo.id)
                    break
        return matches

    def match_by_size(self, input_objs: list[ArcObject],
                      output_objs: list[ArcObject]) -> list[tuple[ArcObject, ArcObject]]:
        """Match objects with the same size (cell count)."""
        matches = []
        used_out = set()
        for io in sorted(input_objs, key=lambda o: o.size, reverse=True):
            for oo in sorted(output_objs, key=lambda o: o.size, reverse=True):
                if oo.id in used_out:
                    continue
                if io.size == oo.size:
                    matches.append((io, oo))
                    used_out.add(oo.id)
                    break
        return matches


class ObjectTransformLearner:
    """Learn what transforms were applied to matched objects."""

    def learn(self, matches: list[tuple[ArcObject, ArcObject]]) -> list[dict]:
        """For each matched pair, determine what changed."""
        transforms = []
        for inp_obj, out_obj in matches:
            t = {
                "color_changed": inp_obj.color != out_obj.color,
                "old_color": inp_obj.color,
                "new_color": out_obj.color,
                "moved": inp_obj.position != out_obj.position,
                "delta_r": out_obj.bbox.min_r - inp_obj.bbox.min_r,
                "delta_c": out_obj.bbox.min_c - inp_obj.bbox.min_c,
                "size_changed": inp_obj.size != out_obj.size,
                "old_size": inp_obj.size,
                "new_size": out_obj.size,
                "shape_changed": not inp_obj.same_shape(out_obj),
                "old_shape": inp_obj.shape_name,
                "new_shape": out_obj.shape_name,
            }

            # Detect specific transforms
            if inp_obj.same_shape(out_obj) and not t["moved"]:
                if t["color_changed"]:
                    t["transform"] = "recolor"
                else:
                    t["transform"] = "identity"
            elif inp_obj.same_shape(out_obj) and t["moved"]:
                t["transform"] = "move"
            elif t["size_changed"] and out_obj.size > inp_obj.size:
                t["transform"] = "grow"
            elif t["size_changed"] and out_obj.size < inp_obj.size:
                t["transform"] = "shrink"
            else:
                t["transform"] = "complex"

            transforms.append(t)
        return transforms

    def find_consistent_rule(self, all_transforms: list[list[dict]]) -> dict | None:
        """
        Find a rule that's consistent across all training examples.
        
        all_transforms: list of transform lists (one per training example)
        """
        if not all_transforms:
            return None

        # Check if all examples have the same transform type for corresponding objects
        first = all_transforms[0]

        for t_list in all_transforms[1:]:
            if len(t_list) != len(first):
                return None
            for t1, t2 in zip(first, t_list):
                if t1["transform"] != t2["transform"]:
                    return None

        # All consistent — extract the rule
        if not first:
            return None

        rule = {
            "transform": first[0]["transform"],
            "consistent": True,
            "num_objects": len(first),
        }

        # Check if movement is consistent
        if rule["transform"] == "move":
            deltas = set()
            for t_list in all_transforms:
                for t in t_list:
                    deltas.add((t["delta_r"], t["delta_c"]))
            if len(deltas) == 1:
                rule["delta"] = list(deltas)[0]
            else:
                rule["delta"] = "varies"

        # Check if recolor is consistent
        if rule["transform"] == "recolor":
            color_maps = set()
            for t_list in all_transforms:
                for t in t_list:
                    color_maps.add((t["old_color"], t["new_color"]))
            rule["color_map"] = dict(color_maps)

        return rule


class ObjectSolver:
    """
    Solve ARC puzzles using object-level reasoning.
    
    1. Detect objects in input and output
    2. Match them
    3. Learn per-object transforms
    4. Apply to test input
    """

    def __init__(self):
        self.detector = ObjectDetector()
        self.matcher = ObjectMatcher()
        self.learner = ObjectTransformLearner()

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Attempt to solve using object-level transforms."""
        bg = self._find_bg(train)

        # Detect objects in all examples
        all_transforms = []
        for ex in train:
            inp_objs = self.detector.detect(ex["input"], bg)
            out_objs = self.detector.detect(ex["output"], bg)

            if not inp_objs and not out_objs:
                return None

            # Try different matching strategies
            matches = None
            for match_fn in [self.matcher.match_by_position,
                            self.matcher.match_by_shape,
                            self.matcher.match_by_color,
                            self.matcher.match_by_size]:
                m = match_fn(inp_objs, out_objs)
                if len(m) == max(len(inp_objs), len(out_objs)):
                    matches = m
                    break

            if not matches:
                # Try: all input objects map to one output, or objects appear/disappear
                continue

            transforms = self.learner.learn(matches)
            all_transforms.append(transforms)

        if not all_transforms:
            return None

        # Find consistent rule
        rule = self.learner.find_consistent_rule(all_transforms)
        if not rule or not rule.get("consistent"):
            return None

        # Apply rule to test input
        test_objs = self.detector.detect(test_input, bg)
        if not test_objs:
            return None

        return self._apply_rule(test_input, test_objs, rule, bg)

    def _apply_rule(self, grid: Grid, objects: list[ArcObject],
                    rule: dict, bg: int) -> Grid | None:
        """Apply a learned object transform rule to a grid."""
        rows, cols = len(grid), len(grid[0])
        result = [[bg] * cols for _ in range(rows)]

        transform = rule["transform"]

        if transform == "identity":
            return [row[:] for row in grid]

        elif transform == "recolor":
            color_map = rule.get("color_map", {})
            result = [row[:] for row in grid]
            for obj in objects:
                new_color = color_map.get(obj.color, obj.color)
                for r, c in obj.cells:
                    result[r][c] = new_color
            return result

        elif transform == "move":
            delta = rule.get("delta")
            if delta == "varies" or not delta:
                return None
            dr, dc = delta
            result = [[bg] * cols for _ in range(rows)]
            # Copy non-object cells
            obj_cells = set()
            for obj in objects:
                for r, c in obj.cells:
                    obj_cells.add((r, c))
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in obj_cells:
                        result[r][c] = grid[r][c]
            # Move objects
            for obj in objects:
                for r, c in obj.cells:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = grid[r][c]
            return result

        elif transform == "grow":
            # Fill each object's bounding box
            result = [row[:] for row in grid]
            for obj in objects:
                for r in range(obj.bbox.min_r, obj.bbox.max_r + 1):
                    for c in range(obj.bbox.min_c, obj.bbox.max_c + 1):
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = obj.color
            return result

        return None

    def _find_bg(self, train: list[dict]) -> int:
        counts = Counter()
        for ex in train:
            for row in ex["input"]:
                counts.update(row)
        return counts.most_common(1)[0][0] if counts else 0


# ── Relationship Analysis ──

class RelationshipAnalyzer:
    """Analyze spatial relationships between objects."""

    def analyze(self, objects: list[ArcObject]) -> list[dict]:
        """Find all pairwise relationships."""
        relationships = []
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i >= j:
                    continue
                rel = self._classify(a, b)
                if rel:
                    relationships.append({
                        "obj_a": a.id,
                        "obj_b": b.id,
                        "relation": rel,
                    })
        return relationships

    def _classify(self, a: ArcObject, b: ArcObject) -> str | None:
        """Classify the relationship between two objects."""
        if a.bbox.contains(b.bbox):
            return "a_contains_b"
        if b.bbox.contains(a.bbox):
            return "b_contains_a"

        # Above/below
        if a.bbox.max_r < b.bbox.min_r:
            return "a_above_b"
        if b.bbox.max_r < a.bbox.min_r:
            return "b_above_a"

        # Left/right
        if a.bbox.max_c < b.bbox.min_c:
            return "a_left_of_b"
        if b.bbox.max_c < a.bbox.min_c:
            return "b_left_of_a"

        # Touching
        a_cells = set(a.cells)
        for r, c in b.cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in a_cells:
                    return "touching"

        if a.bbox.overlaps(b.bbox):
            return "overlapping"

        return "separate"

    def find_pattern(self, train_relationships: list[list[dict]]) -> dict | None:
        """Find consistent relationship patterns across training examples."""
        if not train_relationships:
            return None

        # Check if number of relationships is consistent
        counts = [len(rels) for rels in train_relationships]
        if len(set(counts)) > 1:
            return None

        # Check if relationship types match
        for i in range(len(train_relationships[0])):
            types = set()
            for rels in train_relationships:
                if i < len(rels):
                    types.add(rels[i]["relation"])
            if len(types) > 1:
                return None

        return {
            "consistent": True,
            "num_relationships": counts[0] if counts else 0,
            "pattern": [r["relation"] for r in train_relationships[0]] if train_relationships else [],
        }
