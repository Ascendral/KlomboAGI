"""
Neighborhood Rule Learner — learns pixel transformation rules from
the 8-neighbor context of each cell.

For each cell, computes (center_color, neighbor_color_counts) and
maps to output color. This catches cellular automata-like rules,
Game of Life patterns, and local color transformations.
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


class NeighborhoodSolver:

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        for fn in [self._sorted_rule, self._count_rule]:
            try:
                result = fn(train, test_input)
                if result is not None:
                    return result
            except:
                continue
        return None

    def _sorted_rule(self, train, test_input):
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        lookup = {}
        for ex in train:
            inp, out = ex["input"], ex["output"]
            R, C = len(inp), len(inp[0])
            for r in range(R):
                for c in range(C):
                    n = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            n.append(inp[nr][nc] if 0<=nr<R and 0<=nc<C else -1)
                    key = (inp[r][c], tuple(sorted(n)))
                    val = out[r][c]
                    if key in lookup and lookup[key] != val:
                        return None
                    lookup[key] = val
        
        R, C = len(test_input), len(test_input[0])
        result = [row[:] for row in test_input]
        for r in range(R):
            for c in range(C):
                n = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        n.append(test_input[nr][nc] if 0<=nr<R and 0<=nc<C else -1)
                key = (test_input[r][c], tuple(sorted(n)))
                if key in lookup:
                    result[r][c] = lookup[key]
        return result

    def _count_rule(self, train, test_input):
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        lookup = {}
        for ex in train:
            inp, out = ex["input"], ex["output"]
            R, C = len(inp), len(inp[0])
            for r in range(R):
                for c in range(C):
                    n = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if 0<=nr<R and 0<=nc<C:
                                n.append(inp[nr][nc])
                    counts = tuple(sorted(Counter(n).items()))
                    key = (inp[r][c], counts)
                    val = out[r][c]
                    if key in lookup and lookup[key] != val:
                        return None
                    lookup[key] = val
        
        R, C = len(test_input), len(test_input[0])
        result = [row[:] for row in test_input]
        for r in range(R):
            for c in range(C):
                n = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C:
                            n.append(test_input[nr][nc])
                counts = tuple(sorted(Counter(n).items()))
                key = (test_input[r][c], counts)
                if key in lookup:
                    result[r][c] = lookup[key]
        return result
