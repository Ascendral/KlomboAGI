"""Simple grid transforms: upscale, subsample, invert, swap."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def get_bg(train):
    av = [v for e in train for row in e["input"] for v in row]
    return Counter(av).most_common(1)[0][0] if av else 0

class TransformSolver:
    def solve(self, train, ti):
        if not train or not train[0]["input"] or not train[0]["input"][0]: return None
        ir, ic = len(train[0]["input"]), len(train[0]["input"][0])
        or_, oc = len(train[0]["output"]), len(train[0]["output"][0])
        bg = get_bg(train)
        
        # Upscale
        for s in [2,3,4,5]:
            if or_==ir*s and oc==ic*s:
                fn = lambda g,sc=s: [[g[r//sc][c//sc] for c in range(len(g[0])*sc)] for r in range(len(g)*sc)]
                try:
                    if all(fn(e["input"])==e["output"] for e in train): return fn(ti)
                except: pass
        
        # Subsample
        for s in [2,3,4]:
            if ir%s==0 and ic%s==0 and ir//s==or_ and ic//s==oc:
                fn = lambda g,sc=s: [[g[r][c] for c in range(0,len(g[0]),sc)] for r in range(0,len(g),sc)]
                try:
                    if all(fn(e["input"])==e["output"] for e in train): return fn(ti)
                except: pass
        
        # Invert (2-color)
        if ir==or_ and ic==oc:
            colors = set(v for row in train[0]["input"] for v in row) - {bg}
            if len(colors)==1:
                c = list(colors)[0]
                fn = lambda g,co=c,b=bg: [[b if v==co else co if v==b else v for v in row] for row in g]
                try:
                    if all(fn(e["input"])==e["output"] for e in train): return fn(ti)
                except: pass
        
        # Color swap
        if ir==or_ and ic==oc:
            changes = set()
            for e in train:
                for r in range(min(ir,len(e["input"]))):
                    for c in range(min(ic,len(e["input"][r]))):
                        a,b = e["input"][r][c], e["output"][r][c]
                        if a!=b: changes.add((min(a,b),max(a,b)))
            if len(changes)==1:
                a,b = list(changes)[0]
                fn = lambda g,x=a,y=b: [[y if v==x else x if v==y else v for v in row] for row in g]
                try:
                    if all(fn(e["input"])==e["output"] for e in train): return fn(ti)
                except: pass
        
        # Self-symmetry operations
        if ir==or_ and ic==oc:
            for fn in [self._xor_rot180, self._or_rot180, self._or_hflip, self._or_vflip, self._shift_by_col, self._shift_by_row]:
                try:
                    if all(fn(e['input'],bg)==e['output'] for e in train):
                        return fn(ti,bg)
                except: pass
        
        return None

    # XOR with rotated self
    def _xor_rot180(self, g, bg):
        R,C=len(g),len(g[0])
        rot=[row[::-1] for row in g[::-1]]
        return [[g[r][c] if g[r][c]!=bg else rot[r][c] for c in range(C)] for r in range(R)]

    def _or_rot180(self, g, bg):
        R,C=len(g),len(g[0])
        rot=[row[::-1] for row in g[::-1]]
        return [[g[r][c] if g[r][c]!=bg else rot[r][c] for c in range(C)] for r in range(R)]

    def _or_hflip(self, g, bg):
        R,C=len(g),len(g[0])
        hf=[row[::-1] for row in g]
        return [[g[r][c] if g[r][c]!=bg else hf[r][c] for c in range(C)] for r in range(R)]

    def _or_vflip(self, g, bg):
        R,C=len(g),len(g[0])
        vf=g[::-1]
        return [[g[r][c] if g[r][c]!=bg else vf[r][c] for c in range(C)] for r in range(R)]

    def _shift_by_col(self, g, bg):
        """Each column cyclically shifted down by its column index."""
        R,C=len(g),len(g[0]); result=[[0]*C for _ in range(R)]
        for c in range(C):
            col=[g[r][c] for r in range(R)]
            s=c%R
            shifted=col[s:]+col[:s]
            for r in range(R): result[r][c]=shifted[r]
        return result

    def _shift_by_row(self, g, bg):
        """Each row cyclically shifted right by its row index."""
        return [row[i%len(row):]+row[:i%len(row)] for i,row in enumerate(g)]
