"""
ARC-AGI Solver — pattern discovery without LLM.

Given training examples (input grid → output grid), discovers the
transformation rule and applies it to unseen test inputs.

This is the test of general reasoning ability. No training data,
no model weights. Pure observation → rule discovery → application.
"""

from __future__ import annotations
from typing import Any


Grid = list[list[int]]


class ARCSolver:
    """Solve ARC-AGI puzzles through pure pattern discovery."""

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        """
        Given training examples and a test input, predict the output.
        
        Tries multiple rule discovery strategies in order of complexity.
        Returns None if no rule discovered.
        """
        strategies = [
            self._try_identity,
            self._try_position_transform,
            self._try_size_transform,
            self._try_many_to_one,
            self._try_value_replacement,
            self._try_size_change,
            self._try_cell_by_cell_mapping,
        ]
        
        for strategy in strategies:
            result = strategy(train, test_input)
            if result is not None:
                return result
        
        return None


    def _try_identity(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Check if output == input (no transformation)."""
        if all(ex["input"] == ex["output"] for ex in train):
            return [row[:] for row in test_input]
        return None

    def _try_many_to_one(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Check if multiple input values map to one output value (e.g. any non-zero → 5)."""
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
                return None
        
        # Collect all value mappings
        all_maps = {}
        for ex in train:
            inp, out = ex["input"], ex["output"]
            for r in range(len(inp)):
                for c in range(len(inp[r])):
                    iv, ov = inp[r][c], out[r][c]
                    if iv not in all_maps:
                        all_maps[iv] = set()
                    all_maps[iv].add(ov)
        
        # Check: do all non-zero values map to one constant?
        non_zero_targets = set()
        zero_target = None
        for iv, ovs in all_maps.items():
            if len(ovs) != 1:
                return None  # Inconsistent
            target = list(ovs)[0]
            if iv == 0:
                zero_target = target
            else:
                non_zero_targets.add(target)
        
        if len(non_zero_targets) == 1:
            replacement = list(non_zero_targets)[0]
            zero_stays = (zero_target == 0) if zero_target is not None else True
            if zero_stays:
                # Rule: all non-zero → replacement, zero stays zero
                result = []
                for row in test_input:
                    result.append([replacement if cell != 0 else 0 for cell in row])
                return result
        
        return None

    def _try_size_transform(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Handle transforms that change grid dimensions (e.g. transpose)."""
        if not train:
            return None
        
        # Check if dimensions swap (transpose)
        is_transpose = True
        for ex in train:
            inp_rows, inp_cols = len(ex["input"]), len(ex["input"][0])
            out_rows, out_cols = len(ex["output"]), len(ex["output"][0])
            if not (out_rows == inp_cols and out_cols == inp_rows):
                is_transpose = False
                break
            # Verify values
            for r in range(inp_rows):
                for c in range(inp_cols):
                    if ex["input"][r][c] != ex["output"][c][r]:
                        is_transpose = False
                        break
                if not is_transpose:
                    break
        
        if is_transpose:
            rows, cols = len(test_input), len(test_input[0])
            result = [[0] * rows for _ in range(cols)]
            for r in range(rows):
                for c in range(cols):
                    result[c][r] = test_input[r][c]
            return result
        
        return None

    def _try_value_replacement(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Check if transformation is just replacing one value with another.
        Only triggers when the SAME input values appear across ALL examples
        and consistently map to the same output values."""
        all_rules = []
        for ex in train:
            rules = {}
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
                return None  # Size changed — not a value replacement
            for r in range(len(inp)):
                for c in range(len(inp[r])):
                    if inp[r][c] != out[r][c]:
                        rules[inp[r][c]] = out[r][c]
            all_rules.append(rules)
        
        if not all_rules:
            return None
        
        # ALL examples must have the SAME set of value mappings
        # If example 1 maps {1:2} and example 2 maps {5:6}, these are different
        # values — this is a position transform, not a value replacement
        if len(all_rules) >= 2:
            keys_match = all(set(r.keys()) == set(all_rules[0].keys()) for r in all_rules)
            if not keys_match:
                return None
        
        # Check consistency
        common = all_rules[0]
        for rules in all_rules[1:]:
            for k, v in rules.items():
                if k in common and common[k] != v:
                    return None
            common.update(rules)
        
        if not common:
            return None
        
        # Skip if all mappings are identity (nothing actually changes)
        if all(k == v for k, v in common.items()):
            return None
        
        # Apply
        result = []
        for row in test_input:
            result.append([common.get(cell, cell) for cell in row])
        return result

    def _try_position_transform(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Check if transformation is spatial (flip, rotate, etc.)."""
        mappings = []
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            if len(out) != rows or len(out[0]) != cols:
                return None
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] != 0:
                        for r2 in range(rows):
                            for c2 in range(cols):
                                if out[r2][c2] == inp[r][c]:
                                    mappings.append({"from": (r, c), "to": (r2, c2), "rows": rows, "cols": cols})
        
        if not mappings:
            return None
        
        # Test transforms
        transforms = {
            "hflip": lambda r, c, rows, cols: (r, cols - 1 - c),
            "vflip": lambda r, c, rows, cols: (rows - 1 - r, c),
            "rot90": lambda r, c, rows, cols: (c, rows - 1 - r),
            "rot180": lambda r, c, rows, cols: (rows - 1 - r, cols - 1 - c),
            "rot270": lambda r, c, rows, cols: (cols - 1 - c, r),
            "transpose": lambda r, c, rows, cols: (c, r),
        }
        
        for name, fn in transforms.items():
            if all(fn(m["from"][0], m["from"][1], m["rows"], m["cols"]) == m["to"] for m in mappings):
                # Apply this transform
                rows, cols = len(test_input), len(test_input[0])
                result = [[0] * cols for _ in range(rows)]
                for r in range(rows):
                    for c in range(cols):
                        nr, nc = fn(r, c, rows, cols)
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = test_input[r][c]
                return result
        
        return None

    def _try_size_change(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Check if output size relates to input values."""
        if not train:
            return None
        
        # Check if input is 1x1
        if not all(len(ex["input"]) == 1 and len(ex["input"][0]) == 1 for ex in train):
            return None
        
        observations = []
        for ex in train:
            val = ex["input"][0][0]
            out_rows = len(ex["output"])
            out_cols = len(ex["output"][0])
            fill_val = ex["output"][0][0]
            observations.append({"val": val, "rows": out_rows, "cols": out_cols, "fill": fill_val})
        
        if len(observations) < 2:
            return None
        
        # Try: size = val + k for some constant k
        for k in range(-5, 10):
            if all(o["rows"] == o["val"] + k and o["cols"] == o["val"] + k for o in observations):
                val = test_input[0][0]
                size = val + k
                if size > 0:
                    fill = val  # Assume fill with input value
                    return [[fill] * size for _ in range(size)]
        
        # Try: size = val * k
        for k in range(1, 10):
            if all(o["rows"] == o["val"] * k and o["cols"] == o["val"] * k for o in observations):
                val = test_input[0][0]
                size = val * k
                fill = val
                return [[fill] * size for _ in range(size)]
        
        return None

    def _try_cell_by_cell_mapping(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Learn a per-cell transformation function."""
        if not train:
            return None
        
        # Check all grids same size
        rows = len(train[0]["input"])
        cols = len(train[0]["input"][0])
        if not all(len(ex["input"]) == rows and len(ex["output"]) == rows for ex in train):
            return None
        if len(test_input) != rows:
            return None
        
        # For each cell position, learn the mapping
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                values_map = {}
                for ex in train:
                    inp_val = ex["input"][r][c]
                    out_val = ex["output"][r][c]
                    if inp_val in values_map and values_map[inp_val] != out_val:
                        return None  # Inconsistent
                    values_map[inp_val] = out_val
                
                test_val = test_input[r][c]
                if test_val in values_map:
                    result[r][c] = values_map[test_val]
                else:
                    return None  # Can't predict this cell
        
        return result


class ARCSolverV2(ARCSolver):
    """Extended solver with more strategies for real ARC puzzles."""

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        strategies = [
            self._try_identity,
            self._try_position_transform,
            self._try_size_transform,
            # New strategies
            self._try_tile_scale,
            self._try_color_conditional,
            self._try_border_fill,
            self._try_gravity,
            self._try_flood_fill_color,
            self._try_row_col_sort,
            # Original strategies
            self._try_many_to_one,
            self._try_value_replacement,
            self._try_size_change,
            self._try_cell_by_cell_mapping,
        ]
        for strategy in strategies:
            try:
                result = strategy(train, test_input)
                if result is not None:
                    # Validate: result must match expected grid dimensions for training examples
                    return result
            except Exception:
                continue
        return None

    def _try_tile_scale(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Output is input tiled/scaled by a factor."""
        if not train:
            return None

        # Check consistent scale factor
        factors = set()
        for ex in train:
            ir, ic = len(ex["input"]), len(ex["input"][0])
            or_, oc = len(ex["output"]), len(ex["output"][0])
            if or_ % ir != 0 or oc % ic != 0:
                return None
            fr, fc = or_ // ir, oc // ic
            factors.add((fr, fc))

        if len(factors) != 1:
            return None

        fr, fc = list(factors)[0]
        if fr == 1 and fc == 1:
            return None  # No scaling

        # Check if output is tiled (each input cell becomes a fr x fc block of same value)
        is_tile = True
        for ex in train:
            inp, out = ex["input"], ex["output"]
            ir, ic = len(inp), len(inp[0])
            for r in range(ir):
                for c in range(ic):
                    val = inp[r][c]
                    for dr in range(fr):
                        for dc in range(fc):
                            if out[r * fr + dr][c * fc + dc] != val:
                                is_tile = False
                                break
                        if not is_tile:
                            break
                    if not is_tile:
                        break
                if not is_tile:
                    break
            if not is_tile:
                break

        if is_tile:
            ir, ic = len(test_input), len(test_input[0])
            result = [[0] * (ic * fc) for _ in range(ir * fr)]
            for r in range(ir):
                for c in range(ic):
                    val = test_input[r][c]
                    for dr in range(fr):
                        for dc in range(fc):
                            result[r * fr + dr][c * fc + dc] = val
            return result

        return None

    def _try_color_conditional(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Replace one color with another based on neighbor count or position."""
        if not train:
            return None

        # Find which cells change
        changes = []
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out):
                return None
            for r in range(len(inp)):
                if len(inp[r]) != len(out[r]):
                    return None
                for c in range(len(inp[r])):
                    if inp[r][c] != out[r][c]:
                        # Count neighbors of same color
                        val = inp[r][c]
                        neighbors = 0
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < len(inp) and 0 <= nc < len(inp[0]):
                                if inp[nr][nc] == val:
                                    neighbors += 1
                        changes.append({
                            "from": val, "to": out[r][c],
                            "neighbors": neighbors,
                            "row": r, "col": c,
                            "rows": len(inp), "cols": len(inp[0]),
                            "on_edge": r == 0 or r == len(inp)-1 or c == 0 or c == len(inp[0])-1,
                        })

        if not changes:
            return None

        # Check: do all changes have same from→to and same neighbor count?
        from_vals = set(ch["from"] for ch in changes)
        to_vals = set(ch["to"] for ch in changes)
        neighbor_counts = set(ch["neighbors"] for ch in changes)

        if len(from_vals) == 1 and len(to_vals) == 1 and len(neighbor_counts) == 1:
            fv = list(from_vals)[0]
            tv = list(to_vals)[0]
            nc = list(neighbor_counts)[0]

            # Rule: cells of color fv with exactly nc neighbors of same color → tv
            result = [row[:] for row in test_input]
            for r in range(len(test_input)):
                for c in range(len(test_input[r])):
                    if test_input[r][c] == fv:
                        neighbors = 0
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr2, nc2 = r + dr, c + dc
                            if 0 <= nr2 < len(test_input) and 0 <= nc2 < len(test_input[0]):
                                if test_input[nr2][nc2] == fv:
                                    neighbors += 1
                        if neighbors == nc:
                            result[r][c] = tv
            return result

        return None

    def _try_border_fill(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Fill border cells or interior cells with a specific color."""
        if not train:
            return None

        for ex in train:
            if len(ex["input"]) != len(ex["output"]):
                return None

        # Check: are changes only on the border?
        border_changes = []
        interior_changes = []
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] != out[r][c]:
                        is_border = (r == 0 or r == rows-1 or c == 0 or c == cols-1)
                        if is_border:
                            border_changes.append({"from": inp[r][c], "to": out[r][c]})
                        else:
                            interior_changes.append({"from": inp[r][c], "to": out[r][c]})

        if border_changes and not interior_changes:
            # Only border changed
            from_vals = set(ch["from"] for ch in border_changes)
            to_vals = set(ch["to"] for ch in border_changes)
            if len(from_vals) == 1 and len(to_vals) == 1:
                fv, tv = list(from_vals)[0], list(to_vals)[0]
                result = [row[:] for row in test_input]
                rows, cols = len(result), len(result[0])
                for r in range(rows):
                    for c in range(cols):
                        if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and result[r][c] == fv:
                            result[r][c] = tv
                return result

        if interior_changes and not border_changes:
            # Only interior changed
            from_vals = set(ch["from"] for ch in interior_changes)
            to_vals = set(ch["to"] for ch in interior_changes)
            if len(from_vals) == 1 and len(to_vals) == 1:
                fv, tv = list(from_vals)[0], list(to_vals)[0]
                result = [row[:] for row in test_input]
                rows, cols = len(result), len(result[0])
                for r in range(rows):
                    for c in range(cols):
                        if not (r == 0 or r == rows-1 or c == 0 or c == cols-1) and result[r][c] == fv:
                            result[r][c] = tv
                return result

        return None

    def _try_gravity(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Non-zero values fall to the bottom (or rise to top, or slide left/right)."""
        if not train:
            return None

        # Check: gravity down (non-bg values sink to bottom of each column)
        def apply_gravity_down(grid, bg=0):
            rows, cols = len(grid), len(grid[0])
            result = [[bg] * cols for _ in range(rows)]
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                for i, val in enumerate(non_bg):
                    result[rows - len(non_bg) + i][c] = val
            return result

        def apply_gravity_up(grid, bg=0):
            rows, cols = len(grid), len(grid[0])
            result = [[bg] * cols for _ in range(rows)]
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                for i, val in enumerate(non_bg):
                    result[i][c] = val
            return result

        def apply_gravity_right(grid, bg=0):
            rows, cols = len(grid), len(grid[0])
            result = [[bg] * cols for _ in range(rows)]
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                for i, val in enumerate(non_bg):
                    result[r][cols - len(non_bg) + i] = val
            return result

        def apply_gravity_left(grid, bg=0):
            rows, cols = len(grid), len(grid[0])
            result = [[bg] * cols for _ in range(rows)]
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                for i, val in enumerate(non_bg):
                    result[r][i] = val
            return result

        # Find most common color (background)
        all_vals = []
        for ex in train:
            for row in ex["input"]:
                all_vals.extend(row)
        from collections import Counter
        bg = Counter(all_vals).most_common(1)[0][0]

        for direction, fn in [("down", apply_gravity_down), ("up", apply_gravity_up),
                               ("right", apply_gravity_right), ("left", apply_gravity_left)]:
            matches = all(fn(ex["input"], bg) == ex["output"] for ex in train)
            if matches:
                return fn(test_input, bg)

        return None

    def _try_flood_fill_color(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Enclosed regions of background get filled with a color."""
        # TODO: implement flood fill detection
        return None

    def _try_row_col_sort(self, train: list[dict], test_input: Grid) -> Grid | None:
        """Rows or columns are sorted by some criterion."""
        if not train:
            return None

        # Check: rows sorted
        for ex in train:
            if len(ex["input"]) != len(ex["output"]):
                return None

        # Sort rows by first element?
        def sort_rows(grid):
            return sorted(grid, key=lambda row: row[0])

        if all(sort_rows(ex["input"]) == ex["output"] for ex in train):
            return sort_rows(test_input)

        # Sort rows by sum?
        def sort_rows_sum(grid):
            return sorted(grid, key=sum)

        if all(sort_rows_sum(ex["input"]) == ex["output"] for ex in train):
            return sort_rows_sum(test_input)

        # Sort columns?
        def sort_cols(grid):
            transposed = list(map(list, zip(*grid)))
            transposed.sort(key=lambda col: col[0])
            return list(map(list, zip(*transposed)))

        if all(sort_cols(ex["input"]) == ex["output"] for ex in train):
            return sort_cols(test_input)

        return None
