"""
ARC Program Synthesizer — discovers transform programs through search.

Instead of hardcoded strategies, searches over compositions of atomic
operations to find a program that transforms input → output.

This is how the system LEARNS new strategies — by discovering them.
"""

from __future__ import annotations
from collections import Counter

Grid = list[list[int]]


# ── Atomic Operations ──

def op_hflip(g: Grid) -> Grid: return [r[::-1] for r in g]
def op_vflip(g: Grid) -> Grid: return g[::-1]
def op_rot90(g: Grid) -> Grid:
    R,C=len(g),len(g[0]); return [[g[R-1-r][c] for r in range(R)] for c in range(C)]
def op_rot180(g: Grid) -> Grid: return [r[::-1] for r in g[::-1]]
def op_rot270(g: Grid) -> Grid:
    R,C=len(g),len(g[0]); return [[g[r][C-1-c] for r in range(R)] for c in range(C)]
def op_transpose(g: Grid) -> Grid: return [list(c) for c in zip(*g)]
def op_top(g: Grid) -> Grid: return g[:len(g)//2] if len(g)>=2 else g
def op_bot(g: Grid) -> Grid: return g[len(g)//2:] if len(g)>=2 else g
def op_left(g: Grid) -> Grid: return [r[:len(r)//2] for r in g] if g and len(g[0])>=2 else g
def op_right(g: Grid) -> Grid: return [r[len(r)//2:] for r in g] if g and len(g[0])>=2 else g
def op_dedup_rows(g: Grid) -> Grid:
    r=[g[0]]
    for i in range(1,len(g)):
        if g[i]!=g[i-1]: r.append(g[i])
    return r
def op_dedup_cols(g: Grid) -> Grid:
    if not g: return g
    t=list(map(list,zip(*g))); d=[t[0]]
    for i in range(1,len(t)):
        if t[i]!=t[i-1]: d.append(t[i])
    return list(map(list,zip(*d))) if d else g
def op_sort_rows(g: Grid) -> Grid: return sorted(g)
def op_sort_rows_rev(g: Grid) -> Grid: return sorted(g, reverse=True)
def op_unique_rows(g: Grid) -> Grid:
    s=[]; r=[]
    for row in g:
        if row not in s: s.append(row); r.append(row)
    return r


STATIC_OPS = [
    ("hflip", op_hflip), ("vflip", op_vflip), ("rot90", op_rot90),
    ("rot180", op_rot180), ("rot270", op_rot270), ("transpose", op_transpose),
    ("top", op_top), ("bot", op_bot), ("left", op_left), ("right", op_right),
    ("dedup_rows", op_dedup_rows), ("dedup_cols", op_dedup_cols),
    ("sort_rows", op_sort_rows), ("sort_rows_rev", op_sort_rows_rev),
    ("unique_rows", op_unique_rows),
]


# ── Dynamic Operations (learned from training examples) ──

def make_value_map(train: list[dict]):
    if not train: return None
    e = train[0]; i, o = e["input"], e["output"]
    if len(i) != len(o) or not i or len(i[0]) != len(o[0]): return None
    m = {}
    for r in range(len(i)):
        for c in range(len(i[r])):
            a, b = i[r][c], o[r][c]
            if a != b:
                if a in m and m[a] != b: return None
                m[a] = b
    if not m: return None
    return lambda g: [[m.get(c, c) for c in r] for r in g]


def make_bbox_extract(train: list[dict]):
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    def f(g):
        R, C = len(g), len(g[0]); mr, xr, mc, xc = R, -1, C, -1
        for r in range(R):
            for c in range(C):
                if g[r][c] != bg: mr, xr, mc, xc = min(mr, r), max(xr, r), min(mc, c), max(xc, c)
        return [r[mc:xc+1] for r in g[mr:xr+1]] if xr >= 0 else g
    return f


def make_color_swap(train: list[dict]):
    if not train: return None
    e = train[0]; i, o = e["input"], e["output"]
    if len(i) != len(o) or not i or len(i[0]) != len(o[0]): return None
    pairs = set()
    for r in range(len(i)):
        for c in range(len(i[r])):
            if i[r][c] != o[r][c]:
                pairs.add((min(i[r][c], o[r][c]), max(i[r][c], o[r][c])))
    if len(pairs) != 1: return None
    a, b = list(pairs)[0]
    return lambda g: [[b if c == a else a if c == b else c for c in r] for r in g]


def make_fill_enclosed(train: list[dict]):
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    fcs = set()
    e = train[0]
    for r in range(len(e["input"])):
        for c in range(len(e["input"][r])):
            if e["input"][r][c] == bg and e["output"][r][c] != bg:
                fcs.add(e["output"][r][c])
    if len(fcs) != 1: return None
    fc = list(fcs)[0]
    def f(g):
        R, C = len(g), len(g[0]); res = [r[:] for r in g]; vis = [[0]*C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if not vis[r][c] and g[r][c] == bg:
                    reg = []; q = [(r, c)]; tb = 0
                    while q:
                        cr, cc = q.pop(0)
                        if vis[cr][cc]: continue
                        if g[cr][cc] != bg: continue
                        vis[cr][cc] = 1; reg.append((cr, cc))
                        if cr == 0 or cr == R-1 or cc == 0 or cc == C-1: tb = 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < R and 0 <= nc < C and not vis[nr][nc]: q.append((nr, nc))
                    if not tb:
                        for rr, cc in reg: res[rr][cc] = fc
        return res
    return f



def make_keep_color(train: list[dict]):
    """Keep only the most common non-bg color, replace rest with bg."""
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    non_bg = [c for c in av if c != bg]
    if not non_bg: return None
    keep = Counter(non_bg).most_common(1)[0][0]
    return lambda g: [[c if c == keep else bg for c in row] for row in g]


def make_gravity_down(train: list[dict]):
    """Non-bg cells fall to bottom of each column."""
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    def f(g):
        R, C = len(g), len(g[0]); res = [[bg]*C for _ in range(R)]
        for c in range(C):
            nb = [g[r][c] for r in range(R) if g[r][c] != bg]
            for i, v in enumerate(nb): res[R-len(nb)+i][c] = v
        return res
    return f


def make_gravity_left(train: list[dict]):
    """Non-bg cells slide to left of each row."""
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    def f(g):
        R, C = len(g), len(g[0]); res = [[bg]*C for _ in range(R)]
        for r in range(R):
            nb = [g[r][c] for c in range(C) if g[r][c] != bg]
            for i, v in enumerate(nb): res[r][i] = v
        return res
    return f


def make_surround(train: list[dict]):
    """Surround non-bg cells with a specific color."""
    av = []
    for e in train:
        for r in e["input"]: av.extend(r)
    if not av: return None
    bg = Counter(av).most_common(1)[0][0]
    scs = set()
    for e in train:
        inp, out = e["input"], e["output"]
        R, C = len(inp), len(inp[0])
        for r in range(R):
            for c in range(C):
                if inp[r][c] == bg and out[r][c] != bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C and inp[nr][nc] != bg:
                            scs.add(out[r][c]); break
    if len(scs) != 1: return None
    sc = list(scs)[0]
    def f(g):
        R, C = len(g), len(g[0]); res = [r[:] for r in g]
        for r in range(R):
            for c in range(C):
                if g[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C and g[nr][nc] != bg:
                            res[r][c] = sc; break
        return res
    return f


DYNAMIC_MAKERS = [make_value_map, make_bbox_extract, make_color_swap, make_fill_enclosed, make_keep_color, make_gravity_down, make_gravity_left, make_surround]


class ProgramSynthesizer:
    """Search over compositions of atomic transforms to find programs."""

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth

    def synthesize(self, train: list[dict], test_input: Grid) -> tuple[Grid | None, str | None]:
        """Find a program that transforms training inputs to outputs, apply to test."""
        # Build operator set
        dyn = []
        for maker in DYNAMIC_MAKERS:
            try:
                fn = maker(train)
                if fn:
                    dyn.append((maker.__name__[5:], fn))
            except:
                pass

        ops = STATIC_OPS + dyn

        # Identity
        if all(e["input"] == e["output"] for e in train):
            return [r[:] for r in test_input], "identity"

        # Depth 1
        for n, op in ops:
            try:
                if all(op(e["input"]) == e["output"] for e in train):
                    return op(test_input), n
            except:
                continue

        # Depth 2
        for n1, o1 in ops:
            for n2, o2 in ops:
                try:
                    if all(o2(o1(e["input"])) == e["output"] for e in train):
                        return o2(o1(test_input)), f"{n1}→{n2}"
                except:
                    continue

        # Depth 3 — limited search
        if self.max_depth >= 3:
            spatial = [o for o in ops if o[0] in ("hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "bbox_extract")]
            for n1, o1 in spatial:
                for n2, o2 in ops:
                    if n1 == n2:
                        continue
                    for n3, o3 in spatial:
                        try:
                            if all(o3(o2(o1(e["input"]))) == e["output"] for e in train):
                                return o3(o2(o1(test_input))), f"{n1}→{n2}→{n3}"
                        except:
                            continue

        return None, None
