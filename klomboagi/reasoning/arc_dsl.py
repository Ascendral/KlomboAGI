"""
DSL-based ARC solver. Defines grid primitives and searches
over compositions of them to find programs that solve puzzles.
"""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def select_color(grid, color):
    return [[grid[r][c] == color for c in range(len(grid[0]))] for r in range(len(grid))]

def select_not(grid, color):
    return [[grid[r][c] != color for c in range(len(grid[0]))] for r in range(len(grid))]

def crop_to_mask(grid, mask):
    R, C = len(grid), len(grid[0])
    mr, xr, mc, xc = R, -1, C, -1
    for r in range(R):
        for c in range(C):
            if mask[r][c]:
                mr, xr = min(mr, r), max(xr, r)
                mc, xc = min(mc, c), max(xc, c)
    if xr == -1: return grid
    return [grid[r][mc:xc+1] for r in range(mr, xr+1)]

def expand_mask(mask, n=1):
    R, C = len(mask), len(mask[0])
    result = [row[:] for row in mask]
    for _ in range(n):
        new = [row[:] for row in result]
        for r in range(R):
            for c in range(C):
                if result[r][c]:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C: new[nr][nc] = True
        result = new
    return result

def keep_only(grid, color, bg=0):
    return [[grid[r][c] if grid[r][c] == color else bg for c in range(len(grid[0]))] for r in range(len(grid))]

def remove_color(grid, color, bg=0):
    return [[bg if grid[r][c] == color else grid[r][c] for c in range(len(grid[0]))] for r in range(len(grid))]

def replace_color(grid, old, new):
    return [[new if grid[r][c] == old else grid[r][c] for c in range(len(grid[0]))] for r in range(len(grid))]


class DSLSolver:
    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        all_v = [v for e in train for row in e["input"] for v in row]
        if not all_v: return None
        bg = Counter(all_v).most_common(1)[0][0]
        colors = sorted(set(all_v) - {bg})
        if not colors: return None

        programs = []
        for c in colors:
            programs.append(lambda g, col=c: keep_only(g, col, bg))
            programs.append(lambda g, col=c: remove_color(g, col, bg))
            programs.append(lambda g, col=c: crop_to_mask(g, select_color(g, col)))
            programs.append(lambda g, col=c: crop_to_mask(g, select_not(g, col)))
            for c2 in colors:
                if c2 != c:
                    programs.append(lambda g, a=c, b=c2: replace_color(g, a, b))
                    programs.append(lambda g, col=c, fill=c2:
                        [[fill if expand_mask(select_color(g, col), 1)[r][cc] and g[r][cc] == bg else g[r][cc]
                          for cc in range(len(g[0]))] for r in range(len(g))])

        programs.append(lambda g: crop_to_mask(g, select_not(g, bg)))

        for prog in programs:
            try:
                if all(prog(e["input"]) == e["output"] for e in train):
                    return prog(test_input)
            except: continue

        simple = [
            lambda g: [r[::-1] for r in g],
            lambda g: g[::-1],
            lambda g: [r[::-1] for r in g[::-1]],
            lambda g: crop_to_mask(g, select_not(g, bg)),
        ]
        for c in colors:
            simple.append(lambda g, col=c: keep_only(g, col, bg))
            simple.append(lambda g, col=c: remove_color(g, col, bg))
            simple.append(lambda g, col=c: crop_to_mask(g, select_color(g, col)))

        for f1 in simple:
            for f2 in simple:
                try:
                    if all(f2(f1(e["input"])) == e["output"] for e in train):
                        return f2(f1(test_input))
                except: continue

        return None
