"""Subgrid pattern matching solver for ARC."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def get_bg(train):
    av = [v for e in train for row in e["input"] for v in row]
    return Counter(av).most_common(1)[0][0] if av else 0

def find_subgrids(grid, bg):
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    subgrids = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                cells = []; queue = [(r,c)]
                while queue:
                    cr,cc = queue.pop(0)
                    if visited[cr][cc] or grid[cr][cc] == bg: continue
                    visited[cr][cc] = True; cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr][nc]: queue.append((nr,nc))
                if cells:
                    mr=min(r for r,c in cells); xr=max(r for r,c in cells)
                    mc=min(c for r,c in cells); xc=max(c for r,c in cells)
                    sg=[[bg]*(xc-mc+1) for _ in range(xr-mr+1)]
                    for cr,cc in cells: sg[cr-mr][cc-mc]=grid[cr][cc]
                    subgrids.append({"grid":sg,"pos":(mr,mc),"size":len(cells),"h":xr-mr+1,"w":xc-mc+1})
    return subgrids

class SubgridSolver:
    def solve(self, train, ti):
        for s in [self._largest, self._smallest, self._nth, self._unique_color, self._most_common]:
            try:
                r = s(train, ti)
                if r is not None: return r
            except: continue
        return None

    def _largest(self, train, ti):
        bg = get_bg(train)
        for e in train:
            sgs = find_subgrids(e["input"], bg)
            if not sgs: return None
            if max(sgs, key=lambda s: s["size"])["grid"] != e["output"]: return None
        sgs = find_subgrids(ti, bg)
        return max(sgs, key=lambda s: s["size"])["grid"] if sgs else None

    def _smallest(self, train, ti):
        bg = get_bg(train)
        for e in train:
            sgs = find_subgrids(e["input"], bg)
            if not sgs: return None
            if min(sgs, key=lambda s: s["size"])["grid"] != e["output"]: return None
        sgs = find_subgrids(ti, bg)
        return min(sgs, key=lambda s: s["size"])["grid"] if sgs else None

    def _nth(self, train, ti):
        bg = get_bg(train)
        for idx in range(10):
            works = True
            for e in train:
                sgs = find_subgrids(e["input"], bg)
                sgs.sort(key=lambda s: (s["pos"][0], s["pos"][1]))
                if idx >= len(sgs) or sgs[idx]["grid"] != e["output"]: works = False; break
            if works:
                sgs = find_subgrids(ti, bg)
                sgs.sort(key=lambda s: (s["pos"][0], s["pos"][1]))
                if idx < len(sgs): return sgs[idx]["grid"]
        return None

    def _unique_color(self, train, ti):
        bg = get_bg(train)
        for e in train:
            sgs = find_subgrids(e["input"], bg)
            if not sgs: return None
            ac = Counter()
            sc = []
            for sg in sgs:
                colors = set(v for row in sg["grid"] for v in row if v != bg)
                sc.append(colors); ac.update(colors)
            target = None
            for i, colors in enumerate(sc):
                if any(ac[c] == 1 for c in colors): target = i; break
            if target is None or sgs[target]["grid"] != e["output"]: return None
        sgs = find_subgrids(ti, bg)
        if not sgs: return None
        ac = Counter()
        sc = []
        for sg in sgs:
            colors = set(v for row in sg["grid"] for v in row if v != bg)
            sc.append(colors); ac.update(colors)
        for i, colors in enumerate(sc):
            if any(ac[c] == 1 for c in colors): return sgs[i]["grid"]
        return None

    def _most_common(self, train, ti):
        bg = get_bg(train)
        for e in train:
            sgs = find_subgrids(e["input"], bg)
            if not sgs: return None
            shapes = Counter(); sm = {}
            for sg in sgs:
                k = str(sg["grid"]); shapes[k] += 1; sm[k] = sg["grid"]
            mc = shapes.most_common(1)[0]
            if mc[1] <= 1 or sm[mc[0]] != e["output"]: return None
        sgs = find_subgrids(ti, bg)
        if not sgs: return None
        shapes = Counter(); sm = {}
        for sg in sgs:
            k = str(sg["grid"]); shapes[k] += 1; sm[k] = sg["grid"]
        mc = shapes.most_common(1)[0]
        return sm[mc[0]] if mc[1] > 1 else None
