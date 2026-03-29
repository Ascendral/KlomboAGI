"""
ARC Task Analyzer — understand WHAT a task is asking before trying to solve it.

Instead of blindly trying 106 strategies, first analyze the training examples
to categorize the task and narrow the search space.

Categories:
  EXTRACT    — output is a sub-region of input
  RECOLOR    — same layout, different colors
  TRANSFORM  — spatial transform (flip, rotate, tile)
  FILL       — fill regions based on rules
  COMPOSE    — combine/overlay sub-grids
  CONSTRUCT  — build output from detected objects
  PER_CELL   — each output cell determined by local neighborhood rule
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass, field


Grid = list[list[int]]


@dataclass
class TaskAnalysis:
    """Analysis of what an ARC task is asking."""
    category: str = "unknown"
    confidence: float = 0.0
    properties: dict = field(default_factory=dict)

    # Size relationship
    same_size: bool = False
    output_smaller: bool = False
    output_larger: bool = False
    size_ratio: tuple[float, float] = (1.0, 1.0)

    # Color analysis
    same_palette: bool = False
    new_colors_introduced: bool = False
    colors_removed: bool = False
    bg_color: int = 0
    n_input_colors: int = 0
    n_output_colors: int = 0

    # Change analysis (for same-size)
    pct_changed: float = 0.0
    n_changed: int = 0

    # Structure
    has_divider_h: bool = False
    has_divider_v: bool = False
    has_grid_structure: bool = False
    n_objects_in: int = 0
    n_objects_out: int = 0

    # Consistency across training examples
    consistent_transform: bool = False

    def summary(self) -> str:
        lines = [f"Category: {self.category} ({self.confidence:.0%})"]
        if self.same_size:
            lines.append(f"  Same size, {self.pct_changed:.0%} changed ({self.n_changed} cells)")
        elif self.output_smaller:
            lines.append(f"  Shrinks {self.size_ratio}")
        else:
            lines.append(f"  Grows {self.size_ratio}")
        lines.append(f"  Colors: {self.n_input_colors} in → {self.n_output_colors} out")
        if self.has_divider_h:
            lines.append("  Has horizontal divider")
        if self.has_divider_v:
            lines.append("  Has vertical divider")
        lines.append(f"  Objects: {self.n_objects_in} in → {self.n_objects_out} out")
        return "\n".join(lines)


def analyze_task(train: list[dict]) -> TaskAnalysis:
    """Analyze training examples to understand the task type."""
    a = TaskAnalysis()

    if not train:
        return a

    # Use first example for detailed analysis
    inp = np.array(train[0]["input"])
    out = np.array(train[0]["output"])
    ir, ic = inp.shape
    or_, oc = out.shape

    # Size analysis
    a.same_size = ir == or_ and ic == oc
    a.output_smaller = or_ < ir or oc < ic
    a.output_larger = or_ > ir or oc > ic
    a.size_ratio = (or_ / ir if ir else 1, oc / ic if ic else 1)

    # Color analysis
    in_flat = inp.flatten().tolist()
    out_flat = out.flatten().tolist()
    in_colors = set(in_flat)
    out_colors = set(out_flat)
    a.bg_color = Counter(in_flat).most_common(1)[0][0]
    a.n_input_colors = len(in_colors)
    a.n_output_colors = len(out_colors)
    a.same_palette = in_colors == out_colors
    a.new_colors_introduced = bool(out_colors - in_colors)
    a.colors_removed = bool(in_colors - out_colors)

    # Change analysis
    if a.same_size:
        a.n_changed = int(np.sum(inp != out))
        a.pct_changed = a.n_changed / inp.size

    # Divider detection
    for r in range(ir):
        vals = set(inp[r].tolist())
        if len(vals) == 1 and inp[r][0] != a.bg_color:
            a.has_divider_h = True
            break
    for c in range(ic):
        vals = set(inp[r][c] for r in range(ir))
        if len(vals) == 1 and inp[0][c] != a.bg_color:
            a.has_divider_v = True
            break

    # Object counting
    a.n_objects_in = _count_objects(inp.tolist(), a.bg_color)
    a.n_objects_out = _count_objects(out.tolist(), a.bg_color)

    # Grid structure detection
    a.has_grid_structure = a.has_divider_h or a.has_divider_v

    # Categorize
    _categorize(a, train)

    return a


def _count_objects(grid: Grid, bg: int) -> int:
    """Count connected non-bg regions."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or grid[r][c] == bg:
                continue
            count += 1
            queue = [(r, c)]
            while queue:
                cr, cc = queue.pop(0)
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr][cc] or grid[cr][cc] == bg:
                    continue
                visited[cr][cc] = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cr + dr, cc + dc))
    return count


def _categorize(a: TaskAnalysis, train: list[dict]) -> None:
    """Determine the task category from analysis."""

    # EXTRACT: output is smaller and appears as sub-region of input
    if a.output_smaller and not a.new_colors_introduced:
        out = train[0]["output"]
        inp = train[0]["input"]
        # Check if output appears as a sub-grid in input
        or_, oc = len(out), len(out[0])
        ir, ic = len(inp), len(inp[0])
        for r in range(ir - or_ + 1):
            for c in range(ic - oc + 1):
                match = all(
                    inp[r + dr][c + dc] == out[dr][dc]
                    for dr in range(or_) for dc in range(oc)
                )
                if match:
                    a.category = "extract"
                    a.confidence = 0.9
                    return

        a.category = "extract"
        a.confidence = 0.6
        return

    # TRANSFORM: same size, no new colors, moderate-heavy change
    if a.same_size and a.same_palette and a.pct_changed > 0.3:
        a.category = "transform"
        a.confidence = 0.6
        return

    # RECOLOR: same size, few cells change
    if a.same_size and a.pct_changed < 0.15:
        if a.new_colors_introduced:
            a.category = "fill"
            a.confidence = 0.6
        else:
            a.category = "recolor"
            a.confidence = 0.6
        return

    # FILL: same size, sparse change with new colors
    if a.same_size and a.new_colors_introduced:
        a.category = "fill"
        a.confidence = 0.5
        return

    # COMPOSE: has divider structure
    if a.has_grid_structure:
        a.category = "compose"
        a.confidence = 0.5
        return

    # GROW: output larger
    if a.output_larger:
        r_ratio, c_ratio = a.size_ratio
        if r_ratio == int(r_ratio) and c_ratio == int(c_ratio):
            a.category = "tile"
            a.confidence = 0.7
        else:
            a.category = "construct"
            a.confidence = 0.4
        return

    # PER_CELL: same size, moderate change, same palette
    if a.same_size and a.same_palette:
        a.category = "per_cell"
        a.confidence = 0.5
        return

    a.category = "unknown"
    a.confidence = 0.1


def get_targeted_ops(analysis: TaskAnalysis) -> list[str]:
    """Return the op categories most likely to solve this task type."""
    category_ops = {
        "extract": ["extract", "object", "spatial"],
        "recolor": ["color", "cell_rule"],
        "transform": ["spatial", "gravity"],
        "fill": ["region", "cell_rule", "color"],
        "compose": ["split", "spatial"],
        "tile": ["tile", "spatial"],
        "construct": ["object", "extract", "spatial"],
        "per_cell": ["cell_rule", "color", "region"],
        "unknown": None,  # try everything
    }
    return category_ops.get(analysis.category)
