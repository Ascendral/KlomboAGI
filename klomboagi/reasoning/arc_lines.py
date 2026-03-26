"""Line drawing and connection strategies for ARC."""
from __future__ import annotations
from collections import Counter

Grid = list[list[int]]

def get_bg(train):
    av = [v for e in train for row in e["input"] for v in row]
    return Counter(av).most_common(1)[0][0] if av else 0

class LineSolver:
    def solve(self, train, ti):
        bg = get_bg(train)
        for fn in [self._fill_h, self._fill_v, self._fill_per_color_h, self._fill_per_color_v,
                   self._fill_both, self._voronoi, self._draw_cross, self._draw_diag,
                   self._connect_pairs_h, self._connect_pairs_v]:
            try:
                ir,ic=len(train[0]["input"]),len(train[0]["input"][0])
                or_,oc=len(train[0]["output"]),len(train[0]["output"][0])
                if ir!=or_ or ic!=oc: continue
                if all(fn(e["input"],bg)==e["output"] for e in train):
                    r=fn(ti,bg)
                    if r is not None: return r
            except: continue
        return None
    
    def _fill_h(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for i in range(R):
            f=-1;l=-1;color=None
            for c in range(C):
                if g[i][c]!=bg:
                    if f==-1: f=c;color=g[i][c]
                    l=c
            if f!=-1 and l!=f and color:
                for c in range(f,l+1):
                    if r[i][c]==bg: r[i][c]=color
        return r
    
    def _fill_v(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for c in range(C):
            f=-1;l=-1;color=None
            for i in range(R):
                if g[i][c]!=bg:
                    if f==-1: f=i;color=g[i][c]
                    l=i
            if f!=-1 and l!=f and color:
                for i in range(f,l+1):
                    if r[i][c]==bg: r[i][c]=color
        return r
    
    def _fill_per_color_h(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        colors=set(v for row in g for v in row)-{bg}
        for color in colors:
            for i in range(R):
                f=-1;l=-1
                for c in range(C):
                    if g[i][c]==color:
                        if f==-1: f=c
                        l=c
                if f!=-1 and l!=f:
                    for c in range(f,l+1):
                        if r[i][c]==bg: r[i][c]=color
        return r
    
    def _fill_per_color_v(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        colors=set(v for row in g for v in row)-{bg}
        for color in colors:
            for c in range(C):
                f=-1;l=-1
                for i in range(R):
                    if g[i][c]==color:
                        if f==-1: f=i
                        l=i
                if f!=-1 and l!=f:
                    for i in range(f,l+1):
                        if r[i][c]==bg: r[i][c]=color
        return r
    
    def _fill_both(self,g,bg):
        return self._fill_per_color_v(self._fill_per_color_h(g,bg),bg)
    
    def _voronoi(self,g,bg):
        from collections import deque
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        q=deque()
        for i in range(R):
            for c in range(C):
                if g[i][c]!=bg: q.append((i,c,g[i][c]))
        vis=[[g[i][c]!=bg for c in range(C)] for i in range(R)]
        while q:
            i,c,color=q.popleft()
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=i+dr,c+dc
                if 0<=nr<R and 0<=nc<C and not vis[nr][nc]:
                    vis[nr][nc]=True; r[nr][nc]=color; q.append((nr,nc,color))
        return r
    
    def _draw_cross(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for i in range(R):
            for c in range(C):
                if g[i][c]!=bg:
                    color=g[i][c]
                    for c2 in range(C):
                        if r[i][c2]==bg: r[i][c2]=color
                    for i2 in range(R):
                        if r[i2][c]==bg: r[i2][c]=color
        return r
    
    def _draw_diag(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for i in range(R):
            for c in range(C):
                if g[i][c]!=bg:
                    color=g[i][c]
                    for d in range(1,max(R,C)):
                        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr,nc=i+dr*d,c+dc*d
                            if 0<=nr<R and 0<=nc<C and r[nr][nc]==bg:
                                r[nr][nc]=color
        return r
    
    def _connect_pairs_h(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for i in range(R):
            seen={}
            for c in range(C):
                if g[i][c]!=bg:
                    color=g[i][c]
                    if color in seen:
                        for c2 in range(seen[color]+1,c):
                            if r[i][c2]==bg: r[i][c2]=color
                    seen[color]=c
        return r
    
    def _connect_pairs_v(self,g,bg):
        R,C=len(g),len(g[0]); r=[row[:] for row in g]
        for c in range(C):
            seen={}
            for i in range(R):
                if g[i][c]!=bg:
                    color=g[i][c]
                    if color in seen:
                        for i2 in range(seen[color]+1,i):
                            if r[i2][c]==bg: r[i2][c]=color
                    seen[color]=i
        return r
