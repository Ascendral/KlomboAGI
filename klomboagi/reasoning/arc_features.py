"""
ARC Task Feature Extraction — extract numerical features from ARC tasks
for strategy routing classification.

Features capture grid properties, color distributions, structural patterns,
and spatial relationships that predict which strategy family will solve the task.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

Grid = list[list[int]]


def extract_features(train: list[dict]) -> dict[str, float]:
    """
    Extract ~30 numerical features from an ARC task's training examples.

    Args:
        train: list of {"input": Grid, "output": Grid} dicts

    Returns:
        dict mapping feature names to float values
    """
    f: dict[str, float] = {}

    # Use first example as representative (features should be consistent)
    inp0 = train[0]["input"]
    out0 = train[0]["output"]
    ir, ic = len(inp0), len(inp0[0])
    otr, otc = len(out0), len(out0[0])

    # ── Grid size features ──
    f["input_rows"] = ir
    f["input_cols"] = ic
    f["input_cells"] = ir * ic
    f["output_rows"] = otr
    f["output_cols"] = otc
    f["output_cells"] = otr * otc
    f["same_size"] = float(ir == otr and ic == otc)
    f["shrinks"] = float(otr * otc < ir * ic)
    f["grows"] = float(otr * otc > ir * ic)
    f["size_ratio"] = (otr * otc) / max(ir * ic, 1)

    # ── Color features ──
    in_flat = [v for row in inp0 for v in row]
    out_flat = [int(v) for row in out0 for v in row]
    in_colors = set(in_flat)
    out_colors = set(out_flat)

    bg = Counter(in_flat).most_common(1)[0][0] if in_flat else 0
    f["bg_color"] = bg
    f["bg_percentage"] = in_flat.count(bg) / max(len(in_flat), 1)
    f["n_input_colors"] = len(in_colors)
    f["n_output_colors"] = len(out_colors)
    f["n_new_colors"] = len(out_colors - in_colors)
    f["n_removed_colors"] = len(in_colors - out_colors)
    f["same_colors"] = float(in_colors == out_colors)

    # Non-bg color distribution
    non_bg_counts = Counter(v for v in in_flat if v != bg)
    if non_bg_counts:
        vals = list(non_bg_counts.values())
        f["max_nonbg_count"] = max(vals)
        f["min_nonbg_count"] = min(vals)
        f["nonbg_color_balance"] = min(vals) / max(max(vals), 1)
    else:
        f["max_nonbg_count"] = 0
        f["min_nonbg_count"] = 0
        f["nonbg_color_balance"] = 0

    # ── Change features (same-size only) ──
    if ir == otr and ic == otc:
        n_changed = sum(1 for r in range(ir) for c in range(ic)
                        if inp0[r][c] != int(out0[r][c]))
        f["diff_percentage"] = n_changed / max(ir * ic, 1)
        f["n_cells_changed"] = n_changed
    else:
        f["diff_percentage"] = 1.0
        f["n_cells_changed"] = ir * ic  # everything changes for diff-size

    # ── Structural features ──
    f["has_grid_dividers"] = float(_has_grid_dividers(inp0, bg))
    f["n_connected_components"] = _count_components(inp0, bg)
    comp_sizes = _component_sizes(inp0, bg)
    f["max_component_size"] = max(comp_sizes) if comp_sizes else 0
    f["min_component_size"] = min(comp_sizes) if comp_sizes else 0
    f["has_border_frame"] = float(_has_border_frame(inp0, bg))
    f["has_isolated_dots"] = float(1 in comp_sizes)
    f["n_isolated_dots"] = comp_sizes.count(1)
    f["has_rectangular_objects"] = float(_has_rectangular_objects(inp0, bg))

    # ── Symmetry features ──
    f["input_h_symmetric"] = float(_is_h_symmetric(inp0))
    f["input_v_symmetric"] = float(_is_v_symmetric(inp0))

    # ── Multi-example consistency ──
    f["n_train_examples"] = len(train)
    if len(train) >= 2:
        # Check if all examples have same size ratio
        ratios = set()
        for ex in train:
            eir, eic = len(ex["input"]), len(ex["input"][0])
            eor, eoc = len(ex["output"]), len(ex["output"][0])
            ratios.add((eor / max(eir, 1), eoc / max(eic, 1)))
        f["consistent_size_ratio"] = float(len(ratios) == 1)
    else:
        f["consistent_size_ratio"] = 1.0

    return f


def extract_feature_vector(train: list[dict]) -> list[float]:
    """Extract features as an ordered list (for sklearn)."""
    features = extract_features(train)
    return [features[k] for k in sorted(features.keys())]


def feature_names() -> list[str]:
    """Return sorted feature names (matches extract_feature_vector order)."""
    # Use a dummy task to get the keys
    dummy = [{"input": [[0]], "output": [[0]]}]
    return sorted(extract_features(dummy).keys())


# ── Helper functions ──

def _has_grid_dividers(grid: Grid, bg: int) -> bool:
    """Check if the grid has uniform rows or cols of a single non-bg color."""
    rows, cols = len(grid), len(grid[0])

    # Check for uniform rows
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and list(vals)[0] != bg:
            return True

    # Check for uniform cols
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1 and list(vals)[0] != bg:
            return True

    return False


def _count_components(grid: Grid, bg: int) -> int:
    """Count connected components of non-bg cells (4-connected)."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            # BFS
            count += 1
            queue = [(sr, sc)]
            while queue:
                r, c = queue.pop(0)
                if visited[r][c]:
                    continue
                visited[r][c] = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr][nc] and grid[nr][nc] != bg):
                        queue.append((nr, nc))
    return count


def _component_sizes(grid: Grid, bg: int) -> list[int]:
    """Return sizes of all connected components."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    sizes = []

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            size = 0
            queue = [(sr, sc)]
            while queue:
                r, c = queue.pop(0)
                if visited[r][c]:
                    continue
                visited[r][c] = True
                size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr][nc] and grid[nr][nc] != bg):
                        queue.append((nr, nc))
            sizes.append(size)
    return sizes


def _has_border_frame(grid: Grid, bg: int) -> bool:
    """Check if the grid has a non-bg border frame."""
    rows, cols = len(grid), len(grid[0])
    if rows < 3 or cols < 3:
        return False

    # Check all border cells are the same non-bg color
    border_vals = set()
    for c in range(cols):
        border_vals.add(grid[0][c])
        border_vals.add(grid[rows - 1][c])
    for r in range(rows):
        border_vals.add(grid[r][0])
        border_vals.add(grid[r][cols - 1])

    border_vals.discard(bg)
    return len(border_vals) == 1 and len(border_vals) > 0


def _has_rectangular_objects(grid: Grid, bg: int) -> bool:
    """Check if any connected component forms a perfect rectangle."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            cells = []
            queue = [(sr, sc)]
            while queue:
                r, c = queue.pop(0)
                if visited[r][c]:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr][nc] and grid[nr][nc] != bg):
                        queue.append((nr, nc))

            if len(cells) >= 4:
                rs = [r for r, c in cells]
                cs = [c for r, c in cells]
                expected = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
                if len(cells) == expected:
                    return True
    return False


def _is_h_symmetric(grid: Grid) -> bool:
    """Check if grid is horizontally symmetric."""
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols // 2):
            if grid[r][c] != grid[r][cols - 1 - c]:
                return False
    return True


def _is_v_symmetric(grid: Grid) -> bool:
    """Check if grid is vertically symmetric."""
    rows = len(grid)
    for r in range(rows // 2):
        if grid[r] != grid[rows - 1 - r]:
            return False
    return True
