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
                  self._keep_unique_shape, self._fill_color_bbox, self._fill_color_rows, self._stamp_pattern, self._fill_diagonal, self._recolor_by_rank, self._move_by_color]:
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

    def _stamp_pattern(self, train, ti):
        """Each non-bg color generates a consistent pattern around it."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        
        for radius in [1, 2, 3]:
            patterns={}; ok=True
            for e in train:
                inp,out=e["input"],e["output"]
                R,C=len(inp),len(inp[0])
                for r in range(R):
                    for c in range(C):
                        if inp[r][c]!=bg:
                            color=inp[r][c]; pat=[]
                            for dr in range(-radius,radius+1):
                                for dc in range(-radius,radius+1):
                                    if dr==0 and dc==0: continue
                                    nr,nc=r+dr,c+dc
                                    if 0<=nr<R and 0<=nc<C:
                                        if inp[nr][nc]==bg and out[nr][nc]!=bg:
                                            pat.append((dr,dc,out[nr][nc]))
                            if pat:
                                pk=tuple(sorted(pat))
                                if color in patterns:
                                    if patterns[color]!=pk: ok=False; break
                                else: patterns[color]=pk
                    if not ok: break
                if not ok: break
            
            if ok and patterns:
                def apply_p(g,pats,bg_val):
                    R,C=len(g),len(g[0]); result=[row[:] for row in g]
                    for r in range(R):
                        for c in range(C):
                            if g[r][c] in pats:
                                for dr,dc,col in pats[g[r][c]]:
                                    nr,nc=r+dr,c+dc
                                    if 0<=nr<R and 0<=nc<C and result[nr][nc]==bg_val:
                                        result[nr][nc]=col
                    return result
                try:
                    if all(apply_p(e["input"],patterns,bg)==e["output"] for e in train):
                        return apply_p(ti,patterns,bg)
                except: pass
        return None

    def _fill_diagonal(self, train, ti):
        """Fill diagonal lines between same-color cells."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        def fn(g):
            R,C=len(g),len(g[0]); cc={}
            for r in range(R):
                for c in range(C):
                    if g[r][c]!=bg: cc.setdefault(g[r][c],[]).append((r,c))
            result=[row[:] for row in g]
            for color,cells in cc.items():
                for i in range(len(cells)):
                    for j in range(i+1,len(cells)):
                        r1,c1=cells[i]; r2,c2=cells[j]
                        dr=1 if r2>r1 else (-1 if r2<r1 else 0)
                        dc=1 if c2>c1 else (-1 if c2<c1 else 0)
                        if dr!=0 and dc!=0 and abs(r2-r1)==abs(c2-c1):
                            rr,cc2=r1+dr,c1+dc
                            while (rr,cc2)!=(r2,c2):
                                if 0<=rr<R and 0<=cc2<C and result[rr][cc2]==bg: result[rr][cc2]=color
                                rr+=dr; cc2+=dc
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _recolor_by_rank(self, train, ti):
        """Recolor objects based on their size rank."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        rank_map={}; ok=True
        for e in train:
            objs=detect_objects(e["input"],bg)
            sizes=sorted(set(o["size"] for o in objs))
            for o in objs:
                rank=sizes.index(o["size"])
                r0,c0=list(o["cells"])[0]
                nc=e["output"][r0][c0]
                if rank in rank_map and rank_map[rank]!=nc: ok=False; break
                rank_map[rank]=nc
            if not ok: break
        if not ok or not rank_map: return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            sizes=sorted(set(o["size"] for o in objs))
            result=[row[:] for row in g]
            for o in objs:
                rank=sizes.index(o["size"])
                if rank in rank_map:
                    for r,c in o["cells"]: result[r][c]=rank_map[rank]
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _move_by_color(self, train, ti):
        """Move objects based on their color."""
        bg = bg_of(train)
        for e in train:
            if len(e["input"])!=len(e["output"]): return None
        color_vec={}; ok=True
        for e in train:
            in_objs=detect_objects(e["input"],bg)
            out_objs=detect_objects(e["output"],bg)
            used=set()
            for io in in_objs:
                for j,oo in enumerate(out_objs):
                    if j in used: continue
                    if io["shape"]==oo["shape"]:
                        dr=oo["bbox"][0]-io["bbox"][0]; dc=oo["bbox"][1]-io["bbox"][1]
                        if io["color"] in color_vec:
                            if color_vec[io["color"]]!=(dr,dc): ok=False; break
                        else: color_vec[io["color"]]=(dr,dc)
                        used.add(j); break
            if not ok: break
        if not ok or not color_vec: return None
        def fn(g):
            R,C=len(g),len(g[0]); objs=detect_objects(g,bg)
            result=[[bg]*C for _ in range(R)]
            for o in objs:
                dr,dc=color_vec.get(o["color"],(0,0))
                for r,c in o["cells"]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<R and 0<=nc<C: result[nr][nc]=o["color"]
            return result
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None
