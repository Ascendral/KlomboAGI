"""Context-based pixel prediction for ARC."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

class ContextSolver:
    def solve(self, train, ti):
        for fn in [self._row_context, self._col_context, self._super_context]:
            try:
                r = fn(train, ti)
                if r is not None: return r
            except: continue
        return None
    
    def _row_context(self, train, ti):
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        lookup = {}
        for e in train:
            for r in range(len(e["input"])):
                k=tuple(e["input"][r]); v=tuple(e["output"][r])
                if k in lookup and lookup[k]!=v: return None
                lookup[k]=v
        result=[]
        for row in ti:
            k=tuple(row)
            if k not in lookup: return None
            result.append(list(lookup[k]))
        return result
    
    def _col_context(self, train, ti):
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        lookup = {}
        for e in train:
            R,C=len(e["input"]),len(e["input"][0])
            for c in range(C):
                k=tuple(e["input"][r][c] for r in range(R))
                v=tuple(e["output"][r][c] for r in range(R))
                if k in lookup and lookup[k]!=v: return None
                lookup[k]=v
        R,C=len(ti),len(ti[0])
        result=[[0]*C for _ in range(R)]
        for c in range(C):
            k=tuple(ti[r][c] for r in range(R))
            if k not in lookup: return None
            v=lookup[k]
            for r in range(R): result[r][c]=v[r]
        return result
    
    def _super_context(self, train, ti):
        for e in train:
            if len(e["input"])!=len(e["output"]) or len(e["input"][0])!=len(e["output"][0]): return None
        lookup = {}
        for e in train:
            inp,out=e["input"],e["output"]; R,C=len(inp),len(inp[0])
            for r in range(R):
                for c in range(C):
                    k=(inp[r][c],tuple(inp[r]),tuple(inp[r2][c] for r2 in range(R)))
                    v=out[r][c]
                    if k in lookup and lookup[k]!=v: return None
                    lookup[k]=v
        R,C=len(ti),len(ti[0])
        result=[row[:] for row in ti]
        for r in range(R):
            for c in range(C):
                k=(ti[r][c],tuple(ti[r]),tuple(ti[r2][c] for r2 in range(R)))
                if k in lookup: result[r][c]=lookup[k]
                else: return None
        return result
