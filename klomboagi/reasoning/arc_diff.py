"""Diff-based rule induction for ARC: analyze what changed and learn why."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def bg_of(train):
    av=[v for e in train for r in e["input"] for v in r]
    return Counter(av).most_common(1)[0][0] if av else 0

class DiffSolver:
    def solve(self, train, ti):
        if not train: return None
        ir,ic = len(train[0]["input"]), len(train[0]["input"][0])
        or_,oc = len(train[0]["output"]), len(train[0]["output"][0])
        if ir!=or_ or ic!=oc: return None
        bg = bg_of(train)
        
        for fn in [self._bg_n4_fill, self._edge_recolor, self._val_replace_from_diff, self._val_n8_rule, self._val_minmax, self._val_rowcol_count]:
            try:
                r = fn(train, ti, bg)
                if r is not None: return r
            except: continue
        return None
    
    def _bg_n4_fill(self, train, ti, bg):
        """Fill bg cells that have >= N non-bg neighbors."""
        for thresh in [1,2,3]:
            # Find what color they become
            target = None
            for e in train:
                R,C=len(e["input"]),len(e["input"][0])
                for r in range(R):
                    for c in range(C):
                        if e["input"][r][c]==bg and e["output"][r][c]!=bg:
                            n4=sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                  if 0<=r+dr<R and 0<=c+dc<C and e["input"][r+dr][c+dc]!=bg)
                            if n4>=thresh:
                                t=e["output"][r][c]
                                if target is None: target=t
                                elif target!=t: target=None; break
                    if target is None: break
                if target is None: break
            
            if target is None: continue
            
            def apply(g, bg_val=bg, th=thresh, tgt=target):
                R,C=len(g),len(g[0]); r=[row[:] for row in g]
                for i in range(R):
                    for j in range(C):
                        if g[i][j]==bg_val:
                            n4=sum(1 for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]
                                  if 0<=i+di<R and 0<=j+dj<C and g[i+di][j+dj]!=bg_val)
                            if n4>=th: r[i][j]=tgt
                return r
            
            if all(apply(e["input"])==e["output"] for e in train):
                return apply(ti)
        return None
    
    def _edge_recolor(self, train, ti, bg):
        """Recolor edge cells of specific color."""
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            changes={}
            for r in range(R):
                for c in range(C):
                    if e["input"][r][c]!=e["output"][r][c]:
                        if not(r==0 or r==R-1 or c==0 or c==C-1): return None
                        a,b=e["input"][r][c],e["output"][r][c]
                        if a in changes and changes[a]!=b: return None
                        changes[a]=b
        
        if not changes: return None
        def apply(g, m=changes):
            R,C=len(g),len(g[0])
            return [[m.get(g[r][c],g[r][c]) if r==0 or r==R-1 or c==0 or c==C-1 else g[r][c]
                     for c in range(C)] for r in range(R)]
        if all(apply(e["input"])==e["output"] for e in train):
            return apply(ti)
        return None
    
    def _val_replace_from_diff(self, train, ti, bg):
        """Simple value replacement discovered from diffs."""
        m={}
        for e in train:
            for r in range(len(e["input"])):
                for c in range(len(e["input"][r])):
                    a,b=e["input"][r][c],e["output"][r][c]
                    if a!=b:
                        if a in m and m[a]!=b: return None
                        m[a]=b
        if not m: return None
        def apply(g): return [[m.get(v,v) for v in row] for row in g]
        if all(apply(e["input"])==e["output"] for e in train):
            return apply(ti)
        return None

    def _val_n8_rule(self, train, ti, bg):
        """(value, count of non-bg 8-neighbors) → output."""
        lookup={}
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            for r in range(R):
                for c in range(C):
                    n8=sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                          if (dr or dc) and 0<=r+dr<R and 0<=c+dc<C and e["input"][r+dr][c+dc]!=bg)
                    key=(e["input"][r][c],n8)
                    if key in lookup and lookup[key]!=e["output"][r][c]: return None
                    lookup[key]=e["output"][r][c]
        R,C=len(ti),len(ti[0]); result=[row[:] for row in ti]
        for r in range(R):
            for c in range(C):
                n8=sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                      if (dr or dc) and 0<=r+dr<R and 0<=c+dc<C and ti[r+dr][c+dc]!=bg)
                key=(ti[r][c],n8)
                if key not in lookup: return None
                result[r][c]=lookup[key]
        return result

    def _val_minmax(self, train, ti, bg):
        """(value, min 4-neighbor, max 4-neighbor) → output."""
        lookup={}
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            for r in range(R):
                for c in range(C):
                    ns=[e["input"][r+dr][c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0<=r+dr<R and 0<=c+dc<C]
                    if not ns: ns=[-1]
                    key=(e["input"][r][c],min(ns),max(ns))
                    if key in lookup and lookup[key]!=e["output"][r][c]: return None
                    lookup[key]=e["output"][r][c]
        R,C=len(ti),len(ti[0]); result=[row[:] for row in ti]
        for r in range(R):
            for c in range(C):
                ns=[ti[r+dr][c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0<=r+dr<R and 0<=c+dc<C]
                if not ns: ns=[-1]
                key=(ti[r][c],min(ns),max(ns))
                if key not in lookup: return None
                result[r][c]=lookup[key]
        return result

    def _val_rowcol_count(self, train, ti, bg):
        """(value, count in row, count in col) → output."""
        lookup={}
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            for r in range(R):
                for c in range(C):
                    v=e["input"][r][c]
                    rc=e["input"][r].count(v)
                    cc=sum(1 for r2 in range(R) if e["input"][r2][c]==v)
                    key=(v,rc,cc)
                    if key in lookup and lookup[key]!=e["output"][r][c]: return None
                    lookup[key]=e["output"][r][c]
        R,C=len(ti),len(ti[0]); result=[row[:] for row in ti]
        for r in range(R):
            for c in range(C):
                v=ti[r][c]; rc=ti[r].count(v)
                cc=sum(1 for r2 in range(R) if ti[r2][c]==v)
                key=(v,rc,cc)
                if key not in lookup: return None
                result[r][c]=lookup[key]
        return result
