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
            learn_grid_gap_fill, learn_single_cell_paint, learn_connect_dot_pairs,
            learn_cross_from_dots, learn_diamond_expand, learn_arrow_ray,
            learn_lshape_concavity, learn_conditional_span_fill,
            learn_ushape_gap_drop, learn_template_stamp_at_marker
        )
        for p0_fn in [learn_span_fill_rule, learn_conditional_span_fill,
                       learn_color_key_swap,
                       learn_template_row_stamp, learn_grid_gap_fill,
                       learn_single_cell_paint, learn_connect_dot_pairs,
                       learn_cross_from_dots, learn_diamond_expand,
                       learn_arrow_ray, learn_lshape_concavity,
                       learn_ushape_gap_drop, learn_template_stamp_at_marker]:
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
        #
        # Family name mapping for classifier-guided reordering
        family_learner_map = {
            "cell_rule": (learn_cell_rule, False),
            "region": (learn_region_rule, False),
            "context": (learn_context_rule, False),
            "ranking": (learn_ranking_rule, False),
            "legend": (learn_legend_rule, False),
            "compose": (learn_compose_rule, False),
            "gravity": (learn_gravity_rule, False),
            "tiling": (learn_tiling_rule, False),
            "object_rule": (learn_object_rule, False),
            "multiobj": (learn_multiobj_rule, False),
            "extraction": (learn_extraction_rule, False),
            "grid_ops": (learn_grid_rule, False),
            "advanced": (learn_advanced_rule, False),
            "pattern": (learn_pattern_rule, True),
        }

        default_order = [
            "cell_rule", "region", "context", "ranking", "legend",
            "compose", "gravity", "tiling", "object_rule", "multiobj",
            "extraction", "grid_ops", "advanced", "pattern",
        ]

        # ── Classifier-guided reordering ──
        # If classifier predicts a specific family, try it first
        try:
            from klomboagi.reasoning.arc_classifier import predict_family_proba
            ranked = predict_family_proba(train)
            if ranked:
                # Get top predicted families (non-"none", probability > 0.1)
                priority = [fam for fam, prob in ranked
                            if fam != "none" and fam in family_learner_map
                            and prob > 0.1]
                if priority:
                    # Reorder: priority families first, then rest in default order
                    rest = [f for f in default_order if f not in priority]
                    ordered = priority + rest
                else:
                    ordered = default_order
            else:
                ordered = default_order
        except Exception:
            ordered = default_order

        # Also include span_fill variants in the loop
        extra_learners = [
            (learn_span_fill_rule, False),
            (learn_color_key_swap, False),
            (learn_template_row_stamp, False),
            (learn_grid_gap_fill, False),
            (learn_single_cell_paint, False),
        ]

        # Try extra learners first (fast, specific)
        for learn_fn, loo in extra_learners:
            result = self._try_learner(learn_fn, train, test_input, loo=loo)
            if result is not None:
                return result

        # Try classifier-ordered families
        for family in ordered:
            learn_fn, loo = family_learner_map[family]
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
            self._try_slide_object_to_anchor,
            self._try_tile_complement,
            self._try_fractal_mark_tile,
            self._try_paint_shape_with_color,
            self._try_separate_and_combine,
            self._try_count_unique_colors,
            self._try_fill_holes_in_objects,
            self._try_extend_pattern_to_edge,
            self._try_draw_crosshairs,
            self._try_denoise_repeating_block,
            self._try_flood_fill_interior,
            self._try_hollow_rectangles,
            self._try_connect_same_rowcol_pairs,
            self._try_outer_corners_of_rectangle,
            self._try_most_frequent_at_center_bottom,
            self._try_ring_at_crosshair_intersection,
            self._try_fill_to_nearest_corner,
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

        # ── Phase 5: Disabled — unvalidated fallback produces too many wrong
        # results (pattern_rule alone fires 176 times incorrectly without LOO).
        # Return None to avoid polluting results.
        return None

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

    def _try_slide_object_to_anchor(self, train, test_input):
        """One object slides toward a fixed 'anchor' object until adjacent."""
        from collections import Counter

        def get_objects(grid, bg):
            """Return dict: color -> set of (r,c) cells."""
            rows, cols = len(grid), len(grid[0])
            objs = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        if v not in objs:
                            objs[v] = []
                        objs[v].append((r, c))
            return objs

        def bbox(cells):
            rs = [r for r,_ in cells]
            cs = [c for _,c in cells]
            return min(rs), min(cs), max(rs), max(cs)

        def slide_to_adjacent(grid, mover_color, anchor_color, bg):
            """Slide mover_color cells toward anchor_color until touching."""
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            mover = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == mover_color]
            anchor = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == anchor_color]
            if not mover or not anchor:
                return None

            mr0, mc0, mr1, mc1 = bbox(mover)
            ar0, ac0, ar1, ac1 = bbox(anchor)

            # Gap in each dimension (positive = mover before anchor, negative = after, 0 = overlap)
            v_dist = (ar0 - mr1) if ar0 > mr1 else (-(mr0 - ar1) if mr0 > ar1 else 0)
            h_dist = (ac0 - mc1) if ac0 > mc1 else (-(mc0 - ac1) if mc0 > ac1 else 0)

            # Determine direction: slide along the axis with larger gap
            if abs(v_dist) >= abs(h_dist) and v_dist != 0:
                if v_dist > 0:
                    # Anchor is below mover — slide down
                    shift_r, shift_c = ar0 - mr1 - 1, 0
                else:
                    # Anchor is above mover — slide up
                    shift_r, shift_c = ar1 + 1 - mr0, 0
            elif h_dist != 0:
                if h_dist > 0:
                    # Anchor is right of mover — slide right
                    shift_r, shift_c = 0, ac0 - mc1 - 1
                else:
                    # Anchor is left of mover — slide left
                    shift_r, shift_c = 0, ac1 + 1 - mc0
            else:
                return None  # Already adjacent or overlapping

            if shift_r == 0 and shift_c == 0:
                return None

            # Move mover cells
            for r, c in mover:
                result[r][c] = bg
            for r, c in mover:
                nr, nc = r + shift_r, c + shift_c
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = mover_color
                else:
                    return None
            return result

        # Find consistent anchor (doesn't move) and mover (slides) colors
        if not train:
            return None
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]

        # Find colors that move vs stay
        colors = sorted(set(all_v) - {bg})
        if len(colors) != 2:
            return None  # Only handle 2-object case

        c1, c2 = colors[0], colors[1]

        # Check which stays (anchor) and which moves
        for anchor_c, mover_c in [(c1, c2), (c2, c1)]:
            def apply_slide(inp):
                return slide_to_adjacent(inp, mover_c, anchor_c, bg)

            valid = True
            for ex in train:
                result = apply_slide(ex["input"])
                if result != ex["output"]:
                    valid = False
                    break
            if valid:
                return apply_slide(test_input)

        return None

    def _try_tile_complement(self, train, test_input):
        """Output is the complement (bg<->fg) of the input, tiled k×k times."""
        from collections import Counter

        def find_bg(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        factors = set()
        for ex in train:
            ir, ic = len(ex["input"]), len(ex["input"][0])
            or_, oc = len(ex["output"]), len(ex["output"][0])
            if or_ % ir != 0 or oc % ic != 0:
                return None
            fr, fc = or_ // ir, oc // ic
            if fr != fc:
                return None
            factors.add(fr)

        if len(factors) != 1:
            return None
        k = list(factors)[0]
        if k <= 1:
            return None

        for ex in train:
            inp = ex["input"]
            out = ex["output"]
            ir, ic = len(inp), len(inp[0])
            bg = find_bg(inp)
            # Find the only other color
            non_bg = sorted(set(v for row in inp for v in row) - {bg})
            if len(non_bg) != 1:
                return None
            fg = non_bg[0]
            # Build complement
            comp = [[fg if inp[r][c] == bg else bg for c in range(ic)] for r in range(ir)]
            # Check all tiles in output are comp
            for ti in range(k):
                for tj in range(k):
                    tile = [out[ti*ir+r][tj*ic:tj*ic+ic] for r in range(ir)]
                    if tile != comp:
                        return None

        # Apply to test
        ir, ic = len(test_input), len(test_input[0])
        bg = find_bg(test_input)
        non_bg = sorted(set(v for row in test_input for v in row) - {bg})
        if len(non_bg) != 1:
            return None
        fg = non_bg[0]
        comp = [[fg if test_input[r][c] == bg else bg for c in range(ic)] for r in range(ir)]
        result = []
        for ti in range(k):
            for r in range(ir):
                result.append(comp[r] * k)
        return result

    def _try_fractal_mark_tile(self, train, test_input):
        """Output tiles input at positions where input == mark_color; 0 elsewhere.

        mark_color is the most common non-background color in the input.
        """
        from collections import Counter

        def find_bg(grid):
            vals = [v for row in grid for v in row]
            c = Counter(vals)
            return c.most_common(1)[0][0]

        factors = set()
        for ex in train:
            ir, ic = len(ex["input"]), len(ex["input"][0])
            or_, oc = len(ex["output"]), len(ex["output"][0])
            if ir != ic:
                return None  # Only square inputs
            if or_ != ir * ir or oc != ic * ic:
                return None
            factors.add(ir)

        if len(factors) != 1:
            return None
        N = list(factors)[0]

        for ex in train:
            inp = ex["input"]
            out = ex["output"]
            bg = 0

            # Find mark color: most common non-bg value
            vals = [v for row in inp for v in row if v != bg]
            if not vals:
                return None
            mark = Counter(vals).most_common(1)[0][0]

            # Verify: tile at (i,j) is inp if inp[i,j]==mark, else all bg
            valid = True
            for i in range(N):
                for j in range(N):
                    tile = [out[i*N+r][j*N:j*N+N] for r in range(N)]
                    if inp[i][j] == mark:
                        if tile != inp:
                            valid = False; break
                    else:
                        if any(tile[r][c] != bg for r in range(N) for c in range(N)):
                            valid = False; break
                if not valid:
                    break
            if not valid:
                return None

        # Apply to test
        N = len(test_input)
        if N != len(test_input[0]):
            return None
        bg = 0
        vals = [v for row in test_input for v in row if v != bg]
        if not vals:
            return None
        mark = Counter(vals).most_common(1)[0][0]

        result = [[bg] * (N * N) for _ in range(N * N)]
        for i in range(N):
            for j in range(N):
                if test_input[i][j] == mark:
                    for r in range(N):
                        for c in range(N):
                            result[i*N+r][j*N+c] = test_input[r][c]
        return result

    def _try_remove_isolated_pixels(self, train, test_input):
        """Remove non-background cells that have no 4-connected same-color neighbor."""
        def _remove_isolated(grid):
            rows, cols = len(grid), len(grid[0])
            bg = 0
            result = [row[:] for row in grid]
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v == bg:
                        continue
                    neighbors = [
                        grid[r-1][c] if r > 0 else bg,
                        grid[r+1][c] if r < rows-1 else bg,
                        grid[r][c-1] if c > 0 else bg,
                        grid[r][c+1] if c < cols-1 else bg,
                    ]
                    if v not in neighbors:
                        result[r][c] = bg
            return result

        # Validate on training examples
        for ex in train:
            expected = ex["output"]
            predicted = _remove_isolated(ex["input"])
            if predicted != expected:
                return None
            # Must actually change something (avoid no-op pass-through)
            if predicted == list(list(row) for row in ex["input"]):
                return None

        return _remove_isolated(test_input)

    def _try_hollow_rectangles(self, train, test_input):
        """Hollow out solid-color rectangles: keep outer border, set interior to bg."""
        from collections import deque

        def _hollow(grid):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            visited = [[False] * cols for _ in range(rows)]
            bg = 0
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    comp = []
                    q = deque([(sr, sc)])
                    visited[sr][sc] = True
                    while q:
                        r, c = q.popleft()
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    r0 = min(p[0] for p in comp)
                    r1 = max(p[0] for p in comp)
                    c0 = min(p[1] for p in comp)
                    c1 = max(p[1] for p in comp)
                    if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                        continue  # not a solid rectangle
                    for r in range(r0 + 1, r1):
                        for c in range(c0 + 1, c1):
                            result[r][c] = bg
            return result

        for ex in train:
            predicted = _hollow(ex["input"])
            if predicted != ex["output"]:
                return None
            if predicted == [list(row) for row in ex["input"]]:
                return None

        return _hollow(test_input)

    def _try_connect_same_rowcol_pairs(self, train, test_input):
        """Connect pairs of same-color marker cells that share a row or column with a fill color."""
        from collections import Counter

        def _find_fill_color(grid_in, grid_out, bg):
            rows, cols = len(grid_in), len(grid_out[0])
            fills = set()
            for r in range(rows):
                for c in range(cols):
                    if grid_in[r][c] == bg and grid_out[r][c] != bg:
                        fills.add(grid_out[r][c])
            return list(fills)

        def _apply(grid, marker_color, fill_color, bg):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            # Group marker cells by row and column
            by_row = {}
            by_col = {}
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == marker_color:
                        by_row.setdefault(r, []).append(c)
                        by_col.setdefault(c, []).append(r)
            # Connect pairs in same row
            for r, cols_list in by_row.items():
                if len(cols_list) == 2:
                    c1, c2 = sorted(cols_list)
                    for c in range(c1 + 1, c2):
                        if result[r][c] == bg:
                            result[r][c] = fill_color
            # Connect pairs in same column
            for c, rows_list in by_col.items():
                if len(rows_list) == 2:
                    r1, r2 = sorted(rows_list)
                    for r in range(r1 + 1, r2):
                        if result[r][c] == bg:
                            result[r][c] = fill_color
            return result

        # Determine bg, marker color, fill color from training examples
        bg = 0
        marker_color = None
        fill_color = None

        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            # Find bg as most common color in input
            all_in = [inp[r][c] for r in range(rows) for c in range(cols)]
            bg_candidate = Counter(all_in).most_common(1)[0][0]
            if marker_color is None:
                bg = bg_candidate
                # Marker = non-bg color in input
                m_colors = set(v for v in all_in if v != bg)
                if len(m_colors) != 1:
                    return None
                marker_color = list(m_colors)[0]
            elif bg_candidate != bg:
                return None

            fills = _find_fill_color(inp, out, bg)
            if not fills:
                # No fills in this example — OK if there are no pairs
                # Verify the output equals the input
                if inp != out:
                    return None
                continue
            if len(fills) > 1:
                return None
            f = fills[0]
            if f == marker_color or f == bg:
                return None
            if fill_color is None:
                fill_color = f
            elif fill_color != f:
                return None

        if marker_color is None or fill_color is None:
            return None

        # Validate on all training examples
        for ex in train:
            predicted = _apply(ex["input"], marker_color, fill_color, bg)
            if predicted != ex["output"]:
                return None

        return _apply(test_input, marker_color, fill_color, bg)

    def _try_outer_corners_of_rectangle(self, train, test_input):
        """Mark the 8 outer-corner positions of each hollow rectangular border with a new color.

        A hollow rectangular border: cells form only the perimeter of a bounding box
        (top row, bottom row, left col, right col fully filled, interior is background).

        For a frame with bounding box (r0,c0)-(r1,c1):
          outer corners = (r0-1,c0),(r0,c0-1),(r0-1,c1),(r0,c1+1),
                          (r1+1,c0),(r1,c0-1),(r1+1,c1),(r1,c1+1)
        """
        from collections import Counter, deque

        def _find_rect_shapes(grid):
            """Return list of (color, r0, c0, r1, c1) for hollow frames or solid rectangles."""
            rows, cols = len(grid), len(grid[0])
            bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            visited = [[False] * cols for _ in range(rows)]
            rects = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    comp = []
                    q = deque([(sr, sc)])
                    visited[sr][sc] = True
                    while q:
                        r, c = q.popleft()
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    r0 = min(p[0] for p in comp)
                    r1 = max(p[0] for p in comp)
                    c0 = min(p[1] for p in comp)
                    c1 = max(p[1] for p in comp)
                    if r1 == r0 and c1 == c0:
                        continue  # single cell
                    # Only square shapes get outer corners
                    if (r1 - r0) != (c1 - c0):
                        continue
                    # Check if solid rectangle
                    solid_size = (r1-r0+1) * (c1-c0+1)
                    if len(comp) == solid_size and r1 > r0 and c1 > c0:
                        rects.append((color, r0, c0, r1, c1))
                        continue
                    # Check if hollow frame (perimeter-only, ≥3×3 bounding box)
                    if r1 - r0 < 2 or c1 - c0 < 2:
                        continue
                    perimeter = set()
                    for c in range(c0, c1+1):
                        perimeter.add((r0, c)); perimeter.add((r1, c))
                    for r in range(r0+1, r1):
                        perimeter.add((r, c0)); perimeter.add((r, c1))
                    if set(comp) == perimeter:
                        rects.append((color, r0, c0, r1, c1))
            return rects, bg

        def _outer_corner_cells(r0, c0, r1, c1):
            return [
                (r0-1, c0), (r0, c0-1),   # top-left outer corner
                (r0-1, c1), (r0, c1+1),   # top-right outer corner
                (r1+1, c0), (r1, c0-1),   # bottom-left outer corner
                (r1+1, c1), (r1, c1+1),   # bottom-right outer corner
            ]

        def _apply(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            rects, bg = _find_rect_shapes(grid)
            if not rects:
                return None
            result = [row[:] for row in grid]
            for _, r0, c0, r1, c1 in rects:
                for r, c in _outer_corner_cells(r0, c0, r1, c1):
                    if 0 <= r < rows and 0 <= c < cols and result[r][c] == bg:
                        result[r][c] = fill_color
            return result

        # Determine fill_color from training examples
        fill_color = None
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            rects, bg = _find_rect_shapes(inp)
            if not rects:
                return None
            # Find what color was added in the output
            added = {}
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and out[r][c] != bg:
                        added[out[r][c]] = added.get(out[r][c], 0) + 1
            if not added:
                return None
            if len(added) != 1:
                return None
            fc = list(added.keys())[0]
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None
            # Verify all added cells match outer corner positions
            expected_cells = set()
            for _, r0, c0, r1, c1 in rects:
                for r, c in _outer_corner_cells(r0, c0, r1, c1):
                    if 0 <= r < rows and 0 <= c < cols and inp[r][c] == bg:
                        expected_cells.add((r, c))
            actual_cells = {(r, c) for r in range(rows) for c in range(cols)
                            if inp[r][c] == bg and out[r][c] != bg}
            if actual_cells != expected_cells:
                return None

        if fill_color is None:
            return None

        result = _apply(test_input, fill_color)
        if result is None:
            return None
        if result == [list(row) for row in test_input]:
            return None
        return result

    def _try_most_frequent_at_center_bottom(self, train, test_input):
        """Grid divided by a row of 5s: top has colored cells, bottom is zeros.
        Output: place most-frequent color from top section at (last_row, center_col).
        """
        from collections import Counter

        def _find_divider(grid):
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                if all(int(grid[r][c]) == 5 for c in range(cols)):
                    return r
            return None

        def _apply(grid, color):
            rows, cols = len(grid), len(grid[0])
            div = _find_divider(grid)
            if div is None:
                return None
            # Bottom section must be all zeros (below divider)
            for r in range(div + 1, rows):
                for c in range(cols):
                    if int(grid[r][c]) != 0:
                        return None
            result = [list(map(int, row)) for row in grid]
            last_row = rows - 1
            center_col = (cols - 1) // 2
            if result[last_row][center_col] != 0:
                return None
            result[last_row][center_col] = color
            return result

        # Validate on all training examples (color varies per example — computed from top section)
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            div = _find_divider(inp)
            if div is None or div == 0:
                return None
            # Bottom must be zeros
            bottom_ok = all(int(inp[r][c]) == 0 for r in range(div+1, rows) for c in range(cols))
            if not bottom_ok:
                return None
            # Find the single change
            changes = [(r, c, int(out[r][c])) for r in range(rows) for c in range(cols)
                       if int(inp[r][c]) != int(out[r][c])]
            if len(changes) != 1:
                return None
            cr, cc, cv = changes[0]
            if cr != rows - 1 or cc != (cols - 1) // 2:
                return None
            # Verify color matches most frequent in top section
            top_cells = [int(inp[r][c]) for r in range(div) for c in range(cols) if int(inp[r][c]) != 5]
            if not top_cells:
                return None
            most_common = Counter(top_cells).most_common(1)[0][0]
            if cv != most_common:
                return None

        # For test, compute most frequent color from top section
        test_grid = [list(map(int, row)) for row in test_input]
        div = _find_divider(test_grid)
        if div is None:
            return None
        top_cells = [test_grid[r][c] for r in range(div) for c in range(len(test_grid[0])) if test_grid[r][c] != 5]
        if not top_cells:
            return None
        test_color = Counter(top_cells).most_common(1)[0][0]
        return _apply(test_grid, test_color)

    def _try_ring_at_crosshair_intersection(self, train, test_input):
        """Find two perpendicular lines (one row, one col of same/different color).
        Draw a 3x3 ring of color 4 around their intersection (center unchanged).
        """
        def _find_lines(grid):
            """Return (row_idx, col_idx) for the pair of perpendicular full lines."""
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            # Find rows and cols that are fully non-bg (a complete line)
            full_rows = [r for r in range(rows) if all(int(grid[r][c]) != bg for c in range(cols))]
            full_cols = [c for c in range(cols) if all(int(grid[r][c]) != bg for r in range(rows))]
            if len(full_rows) == 1 and len(full_cols) == 1:
                return full_rows[0], full_cols[0], bg
            return None, None, bg

        def _apply(grid, ring_color):
            rows, cols = len(grid), len(grid[0])
            row_idx, col_idx, bg = _find_lines(grid)
            if row_idx is None:
                return None
            result = [list(map(int, r)) for r in grid]
            ir, ic = row_idx, col_idx
            # Draw 3x3 ring (8 cells) around intersection (ir, ic)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # leave center unchanged
                    nr, nc = ir + dr, ic + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = ring_color
            return result

        # Validate on training examples
        ring_color = None
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            row_idx, col_idx, bg = _find_lines(inp)
            if row_idx is None:
                return None
            # Find what color was added (ring color)
            added = set()
            for r in range(rows):
                for c in range(cols):
                    if int(inp[r][c]) != int(out[r][c]):
                        added.add(int(out[r][c]))
            if len(added) != 1:
                return None
            rc = list(added)[0]
            if ring_color is None:
                ring_color = rc
            elif ring_color != rc:
                return None
            # Verify: expected ring cells match actual changes
            expected = set()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row_idx + dr, col_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        expected.add((nr, nc))
            actual = {(r, c) for r in range(rows) for c in range(cols)
                      if int(inp[r][c]) != int(out[r][c])}
            if actual != expected:
                return None

        if ring_color is None:
            return None

        result = _apply(test_input, ring_color)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_fill_to_nearest_corner(self, train, test_input):
        """Single non-bg pixel in a uniform grid: fill rectangle from pixel to nearest corner."""
        from collections import Counter

        def _get_pixel(grid):
            rows, cols = len(grid), len(grid[0])
            bg = Counter(int(grid[r][c]) for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            pixels = [(r, c, int(grid[r][c])) for r in range(rows) for c in range(cols) if int(grid[r][c]) != bg]
            if len(pixels) != 1:
                return None, None, None, bg
            r0, c0, color = pixels[0]
            return r0, c0, color, bg

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            r0, c0, color, bg = _get_pixel(grid)
            if r0 is None:
                return None
            # Find nearest corner by Chebyshev distance
            corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
            nearest = min(corners, key=lambda p: max(abs(p[0]-r0), abs(p[1]-c0)))
            cr, cc = nearest
            rmin, rmax = min(r0, cr), max(r0, cr)
            cmin, cmax = min(c0, cc), max(c0, cc)
            result = [list(map(int, row)) for row in grid]
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax+1):
                    result[r][c] = color
            return result

        # Validate on training examples
        for ex in train:
            inp, out = ex["input"], ex["output"]
            r0, c0, color, bg = _get_pixel(inp)
            if r0 is None:
                return None
            # All cells in input must be bg except the one pixel
            rows, cols = len(inp), len(inp[0])
            if any(int(inp[r][c]) != bg and not (r == r0 and c == c0)
                   for r in range(rows) for c in range(cols)):
                return None
            predicted = _apply(inp)
            if predicted is None or predicted != [list(map(int, row)) for row in out]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result
