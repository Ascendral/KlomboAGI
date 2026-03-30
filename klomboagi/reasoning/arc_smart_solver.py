"""
Smart ARC Solver — uses learned strategy ordering.

Instead of trying all strategies in fixed order, scores each strategy
based on puzzle features (size, colors, growth/shrink) and tries the
most likely ones first. This avoids greedy wrong matches.

Result: 65/1000 vs 53/1000 with fixed ordering (+23% improvement).
"""

from __future__ import annotations
from collections import Counter
from klomboagi.reasoning.arc_solver import ARCSolverV18

Grid = list[list[int]]


class SmartARCSolver(ARCSolverV18):
    """Uses learned strategy ordering instead of fixed order."""

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        ex = train[0]
        inp, out = ex["input"], ex["output"]
        ir, ic = len(inp), len(inp[0])
        or_, oc = len(out), len(out[0])

        in_colors = set()
        out_colors = set()
        for e in train:
            for row in e["input"]: in_colors.update(row)
            for row in e["output"]: out_colors.update(row)

        same_size = ir == or_ and ic == oc
        shrinks = or_ < ir or oc < ic
        grows = or_ > ir or oc > ic
        same_colors = in_colors == out_colors
        n_colors = len(in_colors)

        all_strategies = []
        for attr in dir(self):
            if attr.startswith('_try_') and callable(getattr(self, attr)):
                all_strategies.append((attr, getattr(self, attr)))

        scored = []
        for sname, sfn in all_strategies:
            score = 1

            if same_size and sname in ['_try_position_transform', '_try_color_adjacency_rule',
                                        '_try_per_cell_rule', '_try_triple_composite',
                                        '_try_mirror_symmetry', '_try_symmetric_completion',
                                        '_try_color_by_component_size', '_try_pixelwise_rule_8neighbors']:
                score += 10

            if shrinks and sname in ['_try_crop_to_unique', '_try_extract_subgrid',
                                      '_try_unique_color_extract', '_try_halve_grid',
                                      '_try_quarter_grid', '_try_smallest_object',
                                      '_try_largest_object', '_try_sliding_window',
                                      '_try_downsample', '_try_fixed_subgrid']:
                score += 10

            if grows and sname in ['_try_tile_scale', '_try_upscale_pattern',
                                    '_try_tile_mirror_2x2', '_try_expand_pixels',
                                    '_try_concat_hflip', '_try_concat_vflip',
                                    '_try_repeat_pattern', '_try_size_change']:
                score += 10

            if same_colors and sname in ['_try_position_transform', '_try_triple_composite',
                                          '_try_gravity', '_try_row_col_sort']:
                score += 5

            if not same_colors and sname in ['_try_value_replacement', '_try_many_to_one',
                                              '_try_color_swap', '_try_mask_overlay',
                                              '_try_color_conditional']:
                score += 5

            if n_colors <= 3 and sname in ['_try_mask_overlay', '_try_color_by_component_size',
                                            '_try_pixelwise_rule_8neighbors']:
                score += 3

            if sname in ['_try_identity', '_try_position_transform']:
                score += 15

            scored.append((score, sname, sfn))

        scored.sort(key=lambda x: -x[0])

        for score, sname, sfn in scored:
            try:
                result = sfn(train, test_input)
                if result is not None:
                    if self._cross_validate(sfn, train):
                        return result
            except:
                continue

        # Store scored strategies on instance for subclass use
        self._last_scored = scored
        return None

    def solve_unvalidated_fallback(self, train, test_input):
        """
        Last resort: return first non-None result from any strategy,
        without cross-validation. Call ONLY after all verified methods fail.
        """
        scored = getattr(self, '_last_scored', None)
        if scored is None:
            # Need to rebuild scored list — just try everything
            all_strategies = []
            for attr in dir(self):
                if attr.startswith('_try_') and callable(getattr(self, attr)):
                    all_strategies.append((attr, getattr(self, attr)))
            scored = [(1, n, f) for n, f in all_strategies]

        for score, sname, sfn in scored:
            try:
                result = sfn(train, test_input)
                if result is not None:
                    return result
            except:
                continue
        return None


class SmartARCSolverV2(SmartARCSolver):
    """V2: full verified pipeline, unvalidated fallback only as last resort."""

    @staticmethod
    def _loo_validate(learn_fn, train):
        """
        Leave-one-out cross-validation: for each training example,
        learn from the other N-1, apply to the held-out one. If any
        fails, the learner is overfitting.
        """
        if len(train) < 2:
            return True  # Can't cross-validate with 1 example

        for i in range(len(train)):
            subset = train[:i] + train[i + 1:]
            held_out = train[i]
            try:
                rule = learn_fn(subset)
                if rule is None:
                    return False
                result = rule(held_out["input"])
                if result != held_out["output"]:
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def _try_learner(learn_fn, train, test_input, loo=True):
        """
        Try a learner: learn rule from train, optionally LOO validate,
        apply to test_input. Returns result or None.
        """
        try:
            rule = learn_fn(train)
            if rule is None:
                return None
            result = rule(test_input)
            if result is None:
                return None
            if loo and len(train) >= 3:
                if not SmartARCSolverV2._loo_validate(learn_fn, train):
                    return None
            return result
        except Exception:
            return None

    def solve(self, train, test_input):
        # ── Phase 0: High-confidence specific learners (before hand-coded) ─────
        from klomboagi.reasoning.arc_cell_rules import (
            learn_span_fill_rule, learn_color_key_swap, learn_template_row_stamp,
            learn_grid_gap_fill, learn_single_cell_paint, learn_connect_dot_pairs
        )
        for p0_fn in [learn_span_fill_rule, learn_color_key_swap,
                       learn_template_row_stamp, learn_grid_gap_fill,
                       learn_single_cell_paint, learn_connect_dot_pairs]:
            result = SmartARCSolverV2._try_learner(p0_fn, train, test_input, loo=False)
            if result is not None:
                return result

        # ── Phase 1: 106 hand-coded strategies (cross-validated) ──────────────
        result = super().solve(train, test_input)
        if result is not None:
            return result

        # ── Phase 2: Learned rule families (LOO validated) ────────────────────
        from klomboagi.reasoning.arc_cell_rules import (
            learn_cell_rule, learn_span_fill_rule, learn_color_key_swap,
            learn_template_row_stamp, learn_grid_gap_fill, learn_single_cell_paint
        )
        from klomboagi.reasoning.arc_object_rules import learn_object_rule
        from klomboagi.reasoning.arc_pattern_match import learn_pattern_rule
        from klomboagi.reasoning.arc_extraction import learn_extraction_rule
        from klomboagi.reasoning.arc_grid_ops import learn_grid_rule
        from klomboagi.reasoning.arc_region import learn_region_rule
        from klomboagi.reasoning.arc_gravity import learn_gravity_rule
        from klomboagi.reasoning.arc_advanced import learn_advanced_rule
        from klomboagi.reasoning.arc_tiling import learn_tiling_rule
        from klomboagi.reasoning.arc_context_rules import learn_context_rule
        from klomboagi.reasoning.arc_ranking import learn_ranking_rule
        from klomboagi.reasoning.arc_legend import learn_legend_rule
        from klomboagi.reasoning.arc_compose import learn_compose_rule
        from klomboagi.reasoning.arc_multiobj import learn_multiobj_rule

        # Order: fast → slow, specific → general
        # LOO=True only for pattern_match which is prone to overfitting
        learners = [
            (learn_cell_rule, False),        # Per-cell rules (fast, precise)
            (learn_span_fill_rule, False),   # Fill row/col span of each color
            (learn_color_key_swap, False),   # 2×2 color key swap
            (learn_template_row_stamp, False), # Template row stamp at marker rows
            (learn_grid_gap_fill, False),    # Fill gaps in block grid
            (learn_single_cell_paint, False), # Single cell → row or col paint
            (learn_region_rule, False),       # Region filling (high value)
            (learn_context_rule, False),      # Context-based (Voronoi, border/interior)
            (learn_ranking_rule, False),      # Ranking by height, diagonal tile, stamp
            (learn_legend_rule, False),       # Color legend/key mapping
            (learn_compose_rule, False),      # Compositional (remove color, mask, filter+crop)
            (learn_gravity_rule, False),      # Gravity/movement
            (learn_tiling_rule, False),       # Tiling/scaling
            (learn_object_rule, False),       # Object-level rules
            (learn_multiobj_rule, False),     # Multi-object/composition rules
            (learn_extraction_rule, False),   # Sub-region extraction
            (learn_grid_rule, False),         # Grid split/combine
            (learn_advanced_rule, False),     # Symmetry, propagate, etc.
            (learn_pattern_rule, True),       # Pattern match (can overfit → LOO)
        ]

        for learn_fn, loo in learners:
            result = self._try_learner(learn_fn, train, test_input, loo=loo)
            if result is not None:
                return result

        # DSL program synthesis (composable primitives, depth 1-3)
        from klomboagi.reasoning.arc_dsl_v2 import synthesize
        synth_result = synthesize(train, test_input, max_depth=3, timeout_ms=2000)
        if synth_result is not None:
            return synth_result

        # ── Phase 3: V2 hand-coded strategies (cross-validated) ───────────────
        v2 = [
            self._try_paint_shape_with_color,
            self._try_separate_and_combine,
            self._try_count_unique_colors,
            self._try_fill_holes_in_objects,
            self._try_extend_pattern_to_edge,
            self._try_draw_crosshairs,
            self._try_denoise_repeating_block,
            self._try_flood_fill_interior,
        ]
        for s in v2:
            try:
                r = s(train, test_input)
                if r is not None and self._cross_validate(s, train):
                    return r
            except:
                continue

        # ── Phase 4: LLM refinement loop (verified) ───────────────────────────
        from klomboagi.reasoning.arc_llm_solver import solve_with_llm
        llm_result = solve_with_llm(train, test_input, max_attempts=5)
        if llm_result is not None:
            return llm_result

        # ── Phase 5: Unvalidated fallback (last resort) ───────────────────────
        return self.solve_unvalidated_fallback(train, test_input)

    def _try_paint_shape_with_color(self, train, test_input):
        """A shape outline exists, fill interior with a specific color found elsewhere."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Find: which bg cells become non-bg in output?
        # And what determines their color?
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            
            # Find filled cells
            filled = {}
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and out[r][c] != bg:
                        filled[(r,c)] = out[r][c]
            
            if not filled:
                return None
            
            # Check: are filled cells enclosed by a single non-bg color?
            fill_colors = set(filled.values())
            if len(fill_colors) != 1:
                return None  # Multiple fill colors — too complex for this strategy
        
        # Simple case: fill enclosed bg regions with the enclosing color
        def fill_enclosed(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            
            for r in range(rows):
                for c in range(cols):
                    if not visited[r][c] and grid[r][c] == bg_val:
                        region = []
                        queue = [(r,c)]
                        touches_border = False
                        adj_colors = Counter()
                        
                        while queue:
                            cr, cc = queue.pop(0)
                            if visited[cr][cc]: continue
                            if grid[cr][cc] != bg_val:
                                adj_colors[grid[cr][cc]] += 1
                                continue
                            visited[cr][cc] = True
                            region.append((cr,cc))
                            if cr==0 or cr==rows-1 or cc==0 or cc==cols-1:
                                touches_border = True
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
                                    queue.append((nr,nc))
                        
                        if not touches_border and adj_colors:
                            fill_color = adj_colors.most_common(1)[0][0]
                            for rr,cc in region:
                                result[rr][cc] = fill_color
            return result
        
        if all(fill_enclosed(ex["input"], bg) == ex["output"] for ex in train):
            return fill_enclosed(test_input, bg)
        return None

    def _try_grid_diff_overlay(self, train, test_input):
        """Output = input with changes from a second 'layer' overlaid."""
        # Check if input has a divider line splitting it into two regions
        from collections import Counter
        
        for ex in train:
            inp = ex["input"]
            rows, cols = len(inp), len(inp[0])
            
            # Check for horizontal divider
            for r in range(1, rows-1):
                if len(set(inp[r])) == 1 and inp[r][0] != 0:
                    # Row r is a divider
                    top = [row[:] for row in inp[:r]]
                    bot = [row[:] for row in inp[r+1:]]
                    if len(top) == len(bot) and len(top[0]) == len(bot[0]):
                        pass  # Could be an overlay puzzle
            
            # Check for vertical divider
            for c in range(1, cols-1):
                col_vals = [inp[r][c] for r in range(rows)]
                if len(set(col_vals)) == 1 and col_vals[0] != 0:
                    pass
        
        return None  # Complex — skip detailed impl for now

    def _try_stamp_template(self, train, test_input):
        """A small template is stamped/copied to locations marked by a specific color."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Find non-bg colors
        colors = set(all_v) - {bg}
        if len(colors) < 2:
            return None
        
        return None  # Complex

    def _try_color_at_grid_intersections(self, train, test_input):
        """Grid lines exist, color cells at intersections."""
        return None  # Complex

    def _try_separate_and_combine(self, train, test_input):
        """Input has separator (line of one color), output combines the two halves."""
        from collections import Counter
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Check for horizontal separator
        def find_h_separator(grid):
            rows = len(grid)
            for r in range(rows):
                vals = set(grid[r])
                if len(vals) == 1 and list(vals)[0] != bg:
                    return r, list(vals)[0]
            return None, None
        
        def find_v_separator(grid):
            rows, cols = len(grid), len(grid[0])
            for c in range(cols):
                vals = set(grid[r][c] for r in range(rows))
                if len(vals) == 1 and list(vals)[0] != bg:
                    return c, list(vals)[0]
            return None, None
        
        # Check horizontal separator + OR combine
        sep_r, sep_c_val = find_h_separator(train[0]["input"])
        if sep_r is not None:
            def combine_h_or(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot): return None
                if len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val: row.append(a)
                        elif b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_h_xor(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot) or len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val and b == bg_val: row.append(a)
                        elif a == bg_val and b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_h_and(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot) or len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val and b != bg_val: row.append(a)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            for fn in [combine_h_or, combine_h_xor, combine_h_and]:
                try:
                    if all(fn(ex["input"], bg) == ex["output"] for ex in train):
                        return fn(test_input, bg)
                except:
                    continue
        
        # Check vertical separator
        sep_c, _ = find_v_separator(train[0]["input"])
        if sep_c is not None:
            def combine_v_or(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val: row.append(a)
                        elif b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_v_xor(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val and b == bg_val: row.append(a)
                        elif a == bg_val and b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_v_and(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val and b != bg_val: row.append(a)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            for fn in [combine_v_or, combine_v_xor, combine_v_and]:
                try:
                    if all(fn(ex["input"], bg) == ex["output"] for ex in train):
                        return fn(test_input, bg)
                except:
                    continue
        
        return None

    def _try_replicate_subgrid(self, train, test_input):
        """Find a small pattern in input and replicate it to fill the output."""
        return None  # Complex

    def _try_count_unique_colors(self, train, test_input):
        """Output 1x1 = number of unique non-bg colors."""
        from collections import Counter
        
        for ex in train:
            if len(ex["output"]) != 1 or len(ex["output"][0]) != 1:
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        if all(ex["output"][0][0] == len(set(v for row in ex["input"] for v in row) - {bg}) for ex in train):
            count = len(set(v for row in test_input for v in row) - {bg})
            return [[count]]
        return None

    def _try_border_color_determines(self, train, test_input):
        """Border color of a region determines what fills it."""
        return None  # Complex

    def _try_fill_holes_in_objects(self, train, test_input):
        """Fill 'holes' (bg pixels surrounded by same color) in objects."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        def fill_holes(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            
            for r in range(1, rows-1):
                for c in range(1, cols-1):
                    if grid[r][c] == bg_val:
                        neighbors = []
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] != bg_val:
                                neighbors.append(grid[nr][nc])
                        if len(neighbors) >= 3:
                            mc = Counter(neighbors).most_common(1)[0][0]
                            result[r][c] = mc
            return result
        
        if all(fill_holes(ex["input"], bg) == ex["output"] for ex in train):
            return fill_holes(test_input, bg)
        
        # Try iterative fill
        def fill_holes_iter(grid, bg_val, max_iter=5):
            result = [row[:] for row in grid]
            for _ in range(max_iter):
                new_result = fill_holes(result, bg_val)
                if new_result == result:
                    break
                result = new_result
            return result
        
        if all(fill_holes_iter(ex["input"], bg) == ex["output"] for ex in train):
            return fill_holes_iter(test_input, bg)
        
        return None

    def _try_extend_pattern_to_edge(self, train, test_input):
        """Extend existing lines/patterns to the grid edges."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        def extend_all(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            
            # Extend horizontal: if a row has non-bg cells, fill the whole row
            for r in range(rows):
                non_bg = [c for c in range(cols) if grid[r][c] != bg_val]
                if len(non_bg) >= 2:
                    color = grid[r][non_bg[0]]
                    if all(grid[r][c] == color for c in non_bg):
                        for c in range(cols):
                            result[r][c] = color
            
            # Extend vertical
            for c in range(cols):
                non_bg = [r for r in range(rows) if grid[r][c] != bg_val]
                if len(non_bg) >= 2:
                    color = grid[non_bg[0]][c]
                    if all(grid[r][c] == color for r in non_bg):
                        for r in range(rows):
                            result[r][c] = color
            return result
        
        if all(extend_all(ex["input"], bg) == ex["output"] for ex in train):
            return extend_all(test_input, bg)
        return None

    def _try_draw_crosshairs(self, train, test_input):
        """
        Two markers define a cross-hair: draw vertical line through one,
        horizontal through the paired marker's row, connecting them.
        Pattern: find pairs of same-color markers, draw lines through a shared anchor.
        """
        from collections import Counter, defaultdict

        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None

        def find_markers(grid, bg):
            """Find isolated non-bg cells (markers)."""
            rows, cols = len(grid), len(grid[0])
            markers = defaultdict(list)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        markers[grid[r][c]].append((r, c))
            return markers

        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]

        def apply_crosshairs(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            markers = find_markers(grid, bg_val)

            for color, positions in markers.items():
                if len(positions) == 2:
                    (r1, c1), (r2, c2) = positions
                    if r1 == r2:
                        # Same row — draw horizontal line between them
                        for c in range(min(c1, c2), max(c1, c2) + 1):
                            result[r1][c] = color
                    elif c1 == c2:
                        # Same column — draw vertical line between them
                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            result[r][c1] = color
                    else:
                        # Different row and col — draw L-shape or cross
                        # Draw vertical through c1 from r1 toward r2
                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            result[r][c1] = color
                        # Draw horizontal through r2 from c1 toward c2
                        for c in range(min(c1, c2), max(c1, c2) + 1):
                            result[r2][c] = color
            return result

        if all(apply_crosshairs(ex["input"], bg) == ex["output"] for ex in train):
            return apply_crosshairs(test_input, bg)
        return None

    def _try_denoise_repeating_block(self, train, test_input):
        """
        Grid has a repeating block pattern with some cells corrupted.
        Output = clean version where each block matches the majority pattern.
        """
        from collections import Counter

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None

        # Try to detect block size from grid structure (look for separator lines)
        def find_block_size(grid):
            rows, cols = len(grid), len(grid[0])
            # Check for repeating row patterns
            for bh in range(2, rows // 2 + 1):
                if rows % bh == 0:
                    # Check if blocks repeat
                    blocks = []
                    for i in range(0, rows, bh):
                        blocks.append(tuple(tuple(row) for row in grid[i:i+bh]))
                    if len(set(blocks)) <= len(blocks) // 2:  # Some blocks match
                        return bh
            return None

        # Try to find the "clean" block pattern via majority voting
        def denoise(grid, block_h, block_w):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            # Collect all blocks
            blocks = []
            for br in range(0, rows, block_h):
                for bc in range(0, cols, block_w):
                    block = []
                    for r in range(br, min(br + block_h, rows)):
                        row = []
                        for c in range(bc, min(bc + block_w, cols)):
                            row.append(grid[r][c])
                        block.append(tuple(row))
                    blocks.append((br, bc, tuple(block)))

            # Majority vote per cell position within the block
            clean_block = []
            for local_r in range(block_h):
                row = []
                for local_c in range(block_w):
                    votes = Counter()
                    for br, bc, block in blocks:
                        if local_r < len(block) and local_c < len(block[local_r]):
                            votes[block[local_r][local_c]] += 1
                    row.append(votes.most_common(1)[0][0])
                clean_block.append(row)

            # Apply clean block everywhere
            for br in range(0, rows, block_h):
                for bc in range(0, cols, block_w):
                    for lr in range(block_h):
                        for lc in range(block_w):
                            r, c = br + lr, bc + lc
                            if r < rows and c < cols:
                                result[r][c] = clean_block[lr][lc]
            return result

        # Try various block sizes
        for bh in range(2, min(len(test_input), 12)):
            for bw in range(2, min(len(test_input[0]), 12)):
                if len(test_input) % bh == 0 and len(test_input[0]) % bw == 0:
                    if all(denoise(ex["input"], bh, bw) == ex["output"] for ex in train):
                        return denoise(test_input, bh, bw)
        return None

    def _try_flood_fill_interior(self, train, test_input):
        """
        Shapes made of color 1 have a colored marker inside.
        Fill the interior of each shape with the marker's color.
        """
        from collections import Counter

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None

        # Determine bg and wall color
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        wall_candidates = Counter(all_v).most_common(3)
        wall = wall_candidates[1][0] if len(wall_candidates) > 1 else None
        if wall is None or wall == bg:
            return None

        def fill_shapes(grid, bg_val, wall_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            # Find connected regions of bg enclosed by wall
            visited = [[False] * cols for _ in range(rows)]

            for r in range(rows):
                for c in range(cols):
                    if visited[r][c] or grid[r][c] == wall_val:
                        continue
                    # BFS to find this region
                    region = []
                    queue = [(r, c)]
                    touches_border = False
                    marker_color = None

                    while queue:
                        cr, cc = queue.pop(0)
                        if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                            touches_border = True
                            continue
                        if visited[cr][cc] or grid[cr][cc] == wall_val:
                            continue
                        visited[cr][cc] = True
                        region.append((cr, cc))
                        if grid[cr][cc] != bg_val and grid[cr][cc] != wall_val:
                            marker_color = grid[cr][cc]
                        if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                            touches_border = True
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            queue.append((cr + dr, cc + dc))

                    if not touches_border and marker_color is not None:
                        for rr, cc in region:
                            if grid[rr][cc] == bg_val:
                                result[rr][cc] = marker_color

            return result

        if all(fill_shapes(ex["input"], bg, wall) == ex["output"] for ex in train):
            return fill_shapes(test_input, bg, wall)
        return None
