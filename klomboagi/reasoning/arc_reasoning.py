"""
Object-level reasoning solver for ARC.
Detects objects, analyzes what changes between input/output,
formulates rules, and applies them.
"""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def bg_of(train):
    av=[v for e in train for r in e["input"] for v in r]
    return Counter(av).most_common(1)[0][0] if av else 0

def detect_objects(grid, bg=0):
    R,C=len(grid),len(grid[0])
    visited=[[False]*C for _ in range(R)]
    objects=[]
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c]!=bg:
                color=grid[r][c]; cells=[]; queue=[(r,c)]
                while queue:
                    cr,cc=queue.pop(0)
                    if visited[cr][cc] or grid[cr][cc]!=color: continue
                    visited[cr][cc]=True; cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr][nc]: queue.append((nr,nc))
                objects.append({"color":color,"cells":set(cells),"size":len(cells),
                    "bbox":(min(r for r,_ in cells),min(c for _,c in cells),
                            max(r for r,_ in cells),max(c for _,c in cells))})
    return objects

class ReasoningSolver:
    def solve(self, train, ti):
        for s in [self._remove_smallest, self._keep_largest, self._remove_border,
                  self._keep_border, self._recolor_nearest, self._remove_by_color_count,
                  self._keep_unique_shape, self._fill_color_bbox, self._fill_color_rows]:
            try:
                r = s(train, ti)
                if r is not None: return r
            except: continue
        return None

    def _remove_smallest(self, train, ti):
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            if not objs: return g
            ms=min(o["size"] for o in objs)
            result=[row[:] for row in g]
            for o in objs:
                if o["size"]==ms:
                    for r,c in o["cells"]: result[r][c]=bg
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _keep_largest(self, train, ti):
        bg = bg_of(train)
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            if not objs: return g
            largest=max(objs,key=lambda o:o["size"])
            result=[[bg]*C for _ in range(R)]
            for r,c in largest["cells"]: result[r][c]=largest["color"]
            return result
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _remove_border(self, train, ti):
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            result=[[bg]*C for _ in range(R)]
            for o in objs:
                if not any(r==0 or r==R-1 or c==0 or c==C-1 for r,c in o["cells"]):
                    for r,c in o["cells"]: result[r][c]=o["color"]
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _keep_border(self, train, ti):
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            result=[[bg]*C for _ in range(R)]
            for o in objs:
                if any(r==0 or r==R-1 or c==0 or c==C-1 for r,c in o["cells"]):
                    for r,c in o["cells"]: result[r][c]=o["color"]
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _recolor_nearest(self, train, ti):
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            if len(objs)<2: return g
            result=[row[:] for row in g]
            for i,o in enumerate(objs):
                cr=sum(r for r,_ in o["cells"])/len(o["cells"])
                cc=sum(c for _,c in o["cells"])/len(o["cells"])
                best_d=float('inf'); best_c=o["color"]
                for j,oth in enumerate(objs):
                    if i==j: continue
                    or2=sum(r for r,_ in oth["cells"])/len(oth["cells"])
                    oc2=sum(c for _,c in oth["cells"])/len(oth["cells"])
                    d=abs(cr-or2)+abs(cc-oc2)
                    if d<best_d: best_d=d; best_c=oth["color"]
                for r,c in o["cells"]: result[r][c]=best_c
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _remove_by_color_count(self, train, ti):
        """Remove objects whose color appears only once."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            color_counts=Counter(o["color"] for o in objs)
            result=[row[:] for row in g]
            for o in objs:
                if color_counts[o["color"]]==1:
                    for r,c in o["cells"]: result[r][c]=bg
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _keep_unique_shape(self, train, ti):
        """Keep only the object with a shape that appears exactly once."""
        bg = bg_of(train)
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            shapes=Counter()
            for o in objs:
                s=tuple(sorted((r-o["bbox"][0],c-o["bbox"][1]) for r,c in o["cells"]))
                shapes[s]+=1
            result=[[bg]*C for _ in range(R)]
            for o in objs:
                s=tuple(sorted((r-o["bbox"][0],c-o["bbox"][1]) for r,c in o["cells"]))
                if shapes[s]==1:
                    for r,c in o["cells"]: result[r][c]=o["color"]
            return result
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _fill_color_bbox(self, train, ti):
        """Fill the bounding box of each color's cells with that color."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0])
            cc={}
            for r in range(R):
                for c in range(C):
                    if g[r][c]!=bg: cc.setdefault(g[r][c],[]).append((r,c))
            result=[row[:] for row in g]
            for color,cells in cc.items():
                if len(cells)<2: continue
                mr=min(r for r,_ in cells); xr=max(r for r,_ in cells)
                mc=min(c for _,c in cells); xc=max(c for _,c in cells)
                for r in range(mr,xr+1):
                    for c in range(mc,xc+1):
                        if result[r][c]==bg: result[r][c]=color
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _fill_color_rows(self, train, ti):
        """Fill between same-color cells in same row/col."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0])
            cc={}
            for r in range(R):
                for c in range(C):
                    if g[r][c]!=bg: cc.setdefault(g[r][c],[]).append((r,c))
            result=[row[:] for row in g]
            for color,cells in cc.items():
                if len(cells)<2: continue
                rg={}
                for r,c in cells: rg.setdefault(r,[]).append(c)
                for r,cols in rg.items():
                    if len(cols)>=2:
                        for c in range(min(cols),max(cols)+1):
                            if result[r][c]==bg: result[r][c]=color
                cg={}
                for r,c in cells: cg.setdefault(c,[]).append(r)
                for c,rows in cg.items():
                    if len(rows)>=2:
                        for r in range(min(rows),max(rows)+1):
                            if result[r][c]==bg: result[r][c]=color
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None
