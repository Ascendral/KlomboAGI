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
