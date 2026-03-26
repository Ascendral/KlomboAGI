"""Grow and shrink strategies for ARC puzzles."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

class GrowShrinkSolver:
    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        strategies = [
            self._remove_dup_rows, self._remove_dup_cols, self._remove_both,
            self._unique_rows, self._double_rows, self._double_cols,
            self._concat_vflip, self._concat_vflip_rev,
            self._concat_hflip, self._concat_hflip_rev,
            self._downsample, self._count_per_row, self._count_per_col,
        ]
        for s in strategies:
            try:
                r = s(train, test_input)
                if r is not None: return r
            except: continue
        return None

    def _remove_dup_rows(self, train, ti):
        def fn(g):
            r=[g[0]]
            for i in range(1,len(g)):
                if g[i]!=g[i-1]: r.append(g[i])
            return r
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _remove_dup_cols(self, train, ti):
        def fn(g):
            t=list(map(list,zip(*g))); d=[t[0]]
            for i in range(1,len(t)):
                if t[i]!=t[i-1]: d.append(t[i])
            return list(map(list,zip(*d)))
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _remove_both(self, train, ti):
        def rd(g):
            r=[g[0]]
            for i in range(1,len(g)):
                if g[i]!=g[i-1]: r.append(g[i])
            return r
        def rc(g):
            t=list(map(list,zip(*g))); d=[t[0]]
            for i in range(1,len(t)):
                if t[i]!=t[i-1]: d.append(t[i])
            return list(map(list,zip(*d)))
        def fn(g): return rc(rd(g))
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _unique_rows(self, train, ti):
        def fn(g):
            s=[]
            for r in g:
                if r not in s: s.append(r)
            return s
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _double_rows(self, train, ti):
        def fn(g):
            r=[]
            for row in g: r.append(row[:]); r.append(row[:])
            return r
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _double_cols(self, train, ti):
        def fn(g): return [r+r for r in g]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _concat_vflip(self, train, ti):
        def fn(g): return g+g[::-1]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _concat_vflip_rev(self, train, ti):
        def fn(g): return g[::-1]+g
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _concat_hflip(self, train, ti):
        def fn(g): return [r+r[::-1] for r in g]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _concat_hflip_rev(self, train, ti):
        def fn(g): return [r[::-1]+r for r in g]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _downsample(self, train, ti):
        for bs in [2,3,4,5]:
            ir,ic = len(train[0]["input"]),len(train[0]["input"][0])
            or_,oc = len(train[0]["output"]),len(train[0]["output"][0])
            if ir%bs==0 and ic%bs==0 and or_==ir//bs and oc==ic//bs:
                def fn(g,b=bs):
                    R,C=len(g),len(g[0]); result=[]
                    for br in range(0,R,b):
                        row=[]
                        for bc in range(0,C,b):
                            vals=[g[r][c] for r in range(br,min(br+b,R)) for c in range(bc,min(bc+b,C))]
                            row.append(Counter(vals).most_common(1)[0][0])
                        result.append(row)
                    return result
                if all(fn(e["input"])==e["output"] for e in train):
                    return fn(ti)
        return None

    def _count_per_row(self, train, ti):
        av=[]
        for e in train:
            for r in e["input"]: av.extend(r)
        bg=Counter(av).most_common(1)[0][0]
        def fn(g): return [[sum(1 for c in row if c!=bg) for row in g]]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None

    def _count_per_col(self, train, ti):
        av=[]
        for e in train:
            for r in e["input"]: av.extend(r)
        bg=Counter(av).most_common(1)[0][0]
        def fn(g):
            C=len(g[0])
            return [[sum(1 for r in range(len(g)) if g[r][c]!=bg)] for c in range(C)]
        if all(fn(e["input"])==e["output"] for e in train):
            return fn(ti)
        return None
