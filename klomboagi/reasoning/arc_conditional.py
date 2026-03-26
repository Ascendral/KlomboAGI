"""
Object-Level Conditional Logic for ARC.

Detects objects, computes properties, learns conditional rules:
- If object is largest → recolor to X
- If object is single cell → remove it
- If object touches border → keep, else remove
- Objects move by vector determined by their size
- Extract the object matching property X
"""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def detect_objects(grid, bg=0):
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    objects = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                color = grid[r][c]; cells = []; queue = [(r,c)]
                while queue:
                    cr,cc = queue.pop(0)
                    if visited[cr][cc] or grid[cr][cc] != color: continue
                    visited[cr][cc] = True; cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr][nc]: queue.append((nr,nc))
                mr=min(r for r,c in cells); xr=max(r for r,c in cells)
                mc=min(c for r,c in cells); xc=max(c for r,c in cells)
                shape=tuple(sorted((r-mr,c-mc) for r,c in cells))
                objects.append({"color":color,"cells":cells,"size":len(cells),
                    "bbox":(mr,mc,xr,xc),"shape":shape,"w":xc-mc+1,"h":xr-mr+1,
                    "is_rect":len(cells)==(xc-mc+1)*(xr-mr+1),
                    "is_single":len(cells)==1,
                    "is_square":(xc-mc)==(xr-mr) and len(cells)==(xc-mc+1)**2,
                    "is_line_h":xr==mr and len(cells)>1,
                    "is_line_v":xc==mc and len(cells)>1,
                    "touches_border":mr==0 or xr==R-1 or mc==0 or xc==C-1,
                    "center_r":(mr+xr)/2,"center_c":(mc+xc)/2})
    return objects

def obj_features(obj, all_objs, R, C):
    return {"color":obj["color"],"size":obj["size"],"is_rect":obj["is_rect"],
        "is_single":obj["is_single"],"is_square":obj["is_square"],
        "is_line_h":obj["is_line_h"],"is_line_v":obj["is_line_v"],
        "touches_border":obj["touches_border"],"w":obj["w"],"h":obj["h"],
        "is_largest":obj["size"]==max(o["size"] for o in all_objs) if all_objs else False,
        "is_smallest":obj["size"]==min(o["size"] for o in all_objs) if all_objs else False}


class ConditionalSolver:
    def solve(self, train, test_input):
        for s in [self._recolor, self._remove, self._move, self._extract]:
            try:
                r = s(train, test_input)
                if r is not None: return r
            except: continue
        return None
    
    def _get_bg(self, train):
        av=[v for e in train for row in e["input"] for v in row]
        return Counter(av).most_common(1)[0][0] if av else 0

    def _recolor(self, train, ti):
        bg = self._get_bg(train)
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        
        fks = ["is_largest","is_smallest","is_rect","is_single","is_square","is_line_h","is_line_v","touches_border"]
        for fk in fks:
            # Find target color for objects with this feature
            target = None
            for e in train:
                R,C=len(e["input"]),len(e["input"][0])
                objs=detect_objects(e["input"],bg)
                for obj in objs:
                    f=obj_features(obj,objs,R,C)
                    if f.get(fk):
                        r0,c0=obj["cells"][0]
                        oc=e["output"][r0][c0]
                        if oc!=obj["color"]:
                            if target is None: target=oc
                            elif target!=oc: target=None; break
                if target is None: break
            
            if target is None: continue
            
            works=True
            for e in train:
                R,C=len(e["input"]),len(e["input"][0])
                objs=detect_objects(e["input"],bg)
                result=[row[:] for row in e["input"]]
                for obj in objs:
                    f=obj_features(obj,objs,R,C)
                    if f.get(fk):
                        for r,c in obj["cells"]: result[r][c]=target
                if result!=e["output"]: works=False; break
            
            if works:
                R,C=len(ti),len(ti[0])
                objs=detect_objects(ti,bg)
                result=[row[:] for row in ti]
                for obj in objs:
                    f=obj_features(obj,objs,R,C)
                    if f.get(fk):
                        for r,c in obj["cells"]: result[r][c]=target
                return result
        
        # Size-based recolor
        size_map={}
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            objs=detect_objects(e["input"],bg)
            for obj in objs:
                r0,c0=obj["cells"][0]
                oc=e["output"][r0][c0]
                if oc!=obj["color"]:
                    if obj["size"] in size_map and size_map[obj["size"]]!=oc: return None
                    size_map[obj["size"]]=oc
        
        if size_map:
            works=True
            for e in train:
                R,C=len(e["input"]),len(e["input"][0])
                objs=detect_objects(e["input"],bg)
                result=[row[:] for row in e["input"]]
                for obj in objs:
                    if obj["size"] in size_map:
                        for r,c in obj["cells"]: result[r][c]=size_map[obj["size"]]
                if result!=e["output"]: works=False; break
            if works:
                R,C=len(ti),len(ti[0])
                objs=detect_objects(ti,bg)
                result=[row[:] for row in ti]
                for obj in objs:
                    if obj["size"] in size_map:
                        for r,c in obj["cells"]: result[r][c]=size_map[obj["size"]]
                return result
        return None

    def _remove(self, train, ti):
        bg=self._get_bg(train)
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        
        fks=["is_largest","is_smallest","is_rect","is_single","is_square","is_line_h","is_line_v","touches_border"]
        for fk in fks:
            for keep_true in [True, False]:
                works=True
                for e in train:
                    R,C=len(e["input"]),len(e["input"][0])
                    objs=detect_objects(e["input"],bg)
                    result=[[bg]*C for _ in range(R)]
                    for obj in objs:
                        f=obj_features(obj,objs,R,C)
                        if (f.get(fk) if keep_true else not f.get(fk)):
                            for r,c in obj["cells"]: result[r][c]=obj["color"]
                    if result!=e["output"]: works=False; break
                if works:
                    R,C=len(ti),len(ti[0])
                    objs=detect_objects(ti,bg)
                    result=[[bg]*C for _ in range(R)]
                    for obj in objs:
                        f=obj_features(obj,objs,R,C)
                        if (f.get(fk) if keep_true else not f.get(fk)):
                            for r,c in obj["cells"]: result[r][c]=obj["color"]
                    return result
        return None

    def _move(self, train, ti):
        bg=self._get_bg(train)
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        
        movements=[]
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            in_objs=detect_objects(e["input"],bg)
            out_objs=detect_objects(e["output"],bg)
            for io in in_objs:
                for oo in out_objs:
                    if io["shape"]==oo["shape"] and io["color"]==oo["color"]:
                        dr=oo["bbox"][0]-io["bbox"][0]; dc=oo["bbox"][1]-io["bbox"][1]
                        if dr!=0 or dc!=0:
                            movements.append({"dr":dr,"dc":dc,"size":io["size"]})
                        break
        if not movements: return None
        
        vecs=set((m["dr"],m["dc"]) for m in movements)
        if len(vecs)==1:
            dr,dc=list(vecs)[0]
            R,C=len(ti),len(ti[0])
            objs=detect_objects(ti,bg)
            result=[[bg]*C for _ in range(R)]
            for obj in objs:
                for r,c in obj["cells"]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<R and 0<=nc<C: result[nr][nc]=obj["color"]
            
            works=True
            for e in train:
                R2,C2=len(e["input"]),len(e["input"][0])
                objs2=detect_objects(e["input"],bg)
                pred=[[bg]*C2 for _ in range(R2)]
                for obj in objs2:
                    for r,c in obj["cells"]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<R2 and 0<=nc<C2: pred[nr][nc]=obj["color"]
                if pred!=e["output"]: works=False; break
            if works: return result
        return None

    def _extract(self, train, ti):
        bg=self._get_bg(train)
        fks=["is_largest","is_smallest","is_rect","is_square"]
        for fk in fks:
            works=True
            for e in train:
                R,C=len(e["input"]),len(e["input"][0])
                objs=detect_objects(e["input"],bg)
                target=None
                for obj in objs:
                    f=obj_features(obj,objs,R,C)
                    if f.get(fk): target=obj; break
                if target is None: works=False; break
                mr,mc,xr,xc=target["bbox"]
                ext=[[bg]*(xc-mc+1) for _ in range(xr-mr+1)]
                for r,c in target["cells"]: ext[r-mr][c-mc]=target["color"]
                if ext!=e["output"]: works=False; break
            if works:
                R,C=len(ti),len(ti[0])
                objs=detect_objects(ti,bg)
                for obj in objs:
                    f=obj_features(obj,objs,R,C)
                    if f.get(fk):
                        mr,mc,xr,xc=obj["bbox"]
                        result=[[bg]*(xc-mc+1) for _ in range(xr-mr+1)]
                        for r,c in obj["cells"]: result[r-mr][c-mc]=obj["color"]
                        return result
        return None
