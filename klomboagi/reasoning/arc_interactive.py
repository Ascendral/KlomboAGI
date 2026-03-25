"""
ARC Interactive Solver — human nudges guide the solver.

The baby brain meets the puzzle solver. You can:
- Watch it attempt a puzzle
- Nudge it when it's going the wrong direction
- Teach it new concepts by explaining what you see
- It remembers your nudges and applies them to future puzzles
"""

from __future__ import annotations

import json
import os
from klomboagi.reasoning.arc_solver import ARCSolverV10, Grid
from klomboagi.reasoning.arc_synthesizer import ProgramSynthesizer
from klomboagi.reasoning.arc_learner import ARCLearner, analyze_grid


NUDGE_MEMORY = "/Volumes/AIStorage/AI/klomboagi/memory/arc_nudges.json"


class ARCInteractiveSolver:
    """
    Combines V10 strategies + program synthesis + learner + human nudges.
    
    The human can intervene at any point:
    - "try symmetry" → boosts symmetry strategies
    - "it's about objects" → boosts object detection
    - "the small shape is a template" → teaches a concept
    - "wrong — look at the colors not the positions" → redirects
    """

    def __init__(self) -> None:
        self.solver = ARCSolverV10()
        self.synthesizer = ProgramSynthesizer(max_depth=3)
        self.learner = ARCLearner()
        self.nudge_history: list[dict] = []
        self._load_nudges()

    def _load_nudges(self) -> None:
        if os.path.exists(NUDGE_MEMORY):
            try:
                with open(NUDGE_MEMORY, 'r') as f:
                    self.nudge_history = json.load(f)
            except:
                pass

    def _save_nudges(self) -> None:
        os.makedirs(os.path.dirname(NUDGE_MEMORY), exist_ok=True)
        with open(NUDGE_MEMORY, 'w') as f:
            json.dump(self.nudge_history[-500:], f, indent=2)

    def attempt(self, puzzle_id: str, train: list[dict], test_input: Grid,
                expected: Grid | None = None) -> dict:
        """
        Full attempt with explanation of reasoning.
        Returns a report of what it tried and what worked.
        """
        report = {
            "puzzle_id": puzzle_id,
            "input_props": analyze_grid(train[0]["input"]) if train else {},
            "output_props": analyze_grid(train[0]["output"]) if train else {},
            "attempts": [],
            "solved": False,
            "answer": None,
            "program": None,
        }

        # Step 1: Try V10 strategies
        result = self.solver.solve(train, test_input)
        if result is not None:
            correct = (result == expected) if expected else None
            report["attempts"].append({
                "method": "V10 strategies",
                "result": "found answer",
                "correct": correct,
            })
            if correct is None or correct:
                report["solved"] = True
                report["answer"] = result
                report["program"] = "V10"
                return report

        # Step 2: Try program synthesis
        synth_result, program = self.synthesizer.synthesize(train, test_input)
        if synth_result is not None:
            correct = (synth_result == expected) if expected else None
            report["attempts"].append({
                "method": "program synthesis",
                "program": program,
                "result": "found answer",
                "correct": correct,
            })
            if correct is None or correct:
                report["solved"] = True
                report["answer"] = synth_result
                report["program"] = program
                return report

        # Step 3: Check if any past nudges apply
        applicable_nudges = self._find_applicable_nudges(report["input_props"])
        if applicable_nudges:
            for nudge in applicable_nudges:
                self.learner.nudge(nudge["hint"])
            # Retry with nudge-adjusted ordering
            result2, episode = self.learner.solve_and_learn(
                puzzle_id, train, test_input, expected
            )
            if episode.solved:
                report["solved"] = True
                report["answer"] = result2
                report["program"] = f"nudge-guided ({nudge['hint']})"
                report["attempts"].append({
                    "method": "nudge-guided retry",
                    "nudge": nudge["hint"],
                    "correct": True,
                })
                return report

        report["attempts"].append({"method": "all methods exhausted", "result": "unsolved"})
        return report

    def nudge(self, puzzle_id: str, hint: str, train: list[dict],
              test_input: Grid, expected: Grid | None = None) -> dict:
        """
        Human nudge — "try symmetry", "look at objects", "it's a color swap + flip".
        
        Adjusts strategy ordering and retries.
        """
        input_props = analyze_grid(train[0]["input"]) if train else {}

        # Record the nudge
        self.nudge_history.append({
            "puzzle_id": puzzle_id,
            "hint": hint,
            "input_props": input_props,
        })
        self._save_nudges()

        # Apply nudge to learner
        self.learner.nudge(hint)

        # Retry with adjusted strategies
        result, episode = self.learner.solve_and_learn(
            puzzle_id, train, test_input, expected
        )

        return {
            "nudge": hint,
            "solved": episode.solved,
            "answer": result,
            "strategy_used": episode.strategy_used,
        }

    def _find_applicable_nudges(self, input_props: dict) -> list[dict]:
        """Find past nudges that worked on puzzles with similar properties."""
        applicable = []
        for nudge in self.nudge_history:
            past_props = nudge.get("input_props", {})
            # Simple similarity check
            matches = 0
            for key in ["is_square", "num_colors", "h_symmetric", "v_symmetric"]:
                if key in past_props and key in input_props:
                    if past_props[key] == input_props[key]:
                        matches += 1
            if matches >= 2:
                applicable.append(nudge)
        return applicable[:3]  # Top 3

    def explain_puzzle(self, train: list[dict]) -> str:
        """Describe what the system observes about a puzzle."""
        if not train:
            return "No training examples."

        lines = []
        for i, ex in enumerate(train):
            inp, out = ex["input"], ex["output"]
            ip = analyze_grid(inp)
            op = analyze_grid(out)

            lines.append(f"Example {i+1}:")
            lines.append(f"  Input:  {ip['rows']}x{ip['cols']}, {ip['num_colors']} colors, "
                        f"{'square' if ip['is_square'] else 'rectangle'}, "
                        f"sparsity={ip['sparsity']}")
            lines.append(f"  Output: {op['rows']}x{op['cols']}, {op['num_colors']} colors")

            # Size change?
            if ip['rows'] != op['rows'] or ip['cols'] != op['cols']:
                lines.append(f"  → Size changes: {ip['rows']}x{ip['cols']} → {op['rows']}x{op['cols']}")

            # Color change?
            in_c = set(c for row in inp for c in row)
            out_c = set(c for row in out for c in row)
            new = out_c - in_c
            gone = in_c - out_c
            if new:
                lines.append(f"  → New colors appear: {new}")
            if gone:
                lines.append(f"  → Colors removed: {gone}")

            # Symmetry?
            if ip['h_symmetric']:
                lines.append("  → Input is horizontally symmetric")
            if ip['v_symmetric']:
                lines.append("  → Input is vertically symmetric")

        return "\n".join(lines)

    def show_grid(self, grid: Grid) -> str:
        """Pretty-print a grid."""
        color_map = {0: '·', 1: '█', 2: '▓', 3: '▒', 4: '░', 5: '◆', 6: '◇', 7: '○', 8: '●', 9: '△'}
        lines = []
        for row in grid:
            lines.append(" ".join(color_map.get(c, str(c)) for c in row))
        return "\n".join(lines)
