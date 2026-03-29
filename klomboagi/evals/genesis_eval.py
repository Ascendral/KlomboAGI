"""
Evals for KlomboAGI Genesis — does it actually reason or just retrieve?

Categories:
1. RETRIEVAL — can it recall what it was taught? (baseline)
2. INFERENCE — can it derive facts it was never told? (deduction chains)
3. ANALOGY — can it map structure across domains? (A:B::C:?)
4. COUNTERFACTUAL — can it reason about hypotheticals? (what if?)
5. COMPUTATION — can it do math? (not just store formulas)
6. SELF-AWARENESS — does it know what it knows and doesn't?
7. LEARNING — does it actually learn from new input?
8. GENERATION — can it construct explanations it never saw?
9. SURPRISE — does it detect contradictions?
10. IDENTITY — does it know what it is?

Each eval returns pass/fail + explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class EvalResult:
    """Result of a single eval."""
    name: str
    category: str
    passed: bool
    expected: str
    actual: str
    explanation: str = ""
    duration_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual[:200],
            "explanation": self.explanation,
        }


@dataclass
class EvalReport:
    """Full eval report."""
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    by_category: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Genesis Eval: {self.passed}/{self.total} passed"]
        for cat, stats in sorted(self.by_category.items()):
            p = stats["passed"]
            t = stats["total"]
            bar = "█" * p + "░" * (t - p)
            lines.append(f"  [{bar}] {cat}: {p}/{t}")
        lines.append("")
        for r in self.results:
            status = "✓" if r.passed else "✗"
            lines.append(f"  {status} {r.name}")
            if not r.passed:
                lines.append(f"      Expected: {r.expected[:80]}")
                lines.append(f"      Got: {r.actual[:80]}")
        return "\n".join(lines)


def run_evals(genesis) -> EvalReport:
    """Run all evals on a Genesis instance."""
    report = EvalReport()

    # Teach everything first
    genesis.teach_everything()

    evals = []
    evals.extend(_retrieval_evals(genesis))
    evals.extend(_inference_evals(genesis))
    evals.extend(_analogy_evals(genesis))
    evals.extend(_counterfactual_evals(genesis))
    evals.extend(_computation_evals(genesis))
    evals.extend(_self_awareness_evals(genesis))
    evals.extend(_learning_evals(genesis))
    evals.extend(_generation_evals(genesis))
    evals.extend(_surprise_evals(genesis))
    evals.extend(_identity_evals(genesis))

    report.results = evals
    report.total = len(evals)
    report.passed = sum(1 for e in evals if e.passed)
    report.failed = report.total - report.passed

    # Group by category
    for r in evals:
        if r.category not in report.by_category:
            report.by_category[r.category] = {"total": 0, "passed": 0}
        report.by_category[r.category]["total"] += 1
        if r.passed:
            report.by_category[r.category]["passed"] += 1

    return report


def _check(name: str, category: str, response: str, expected_words: list[str]) -> EvalResult:
    """Check if response contains expected words."""
    response_lower = response.lower()
    found = [w for w in expected_words if w.lower() in response_lower]
    passed = len(found) >= len(expected_words) // 2 + 1  # majority match
    return EvalResult(
        name=name, category=category, passed=passed,
        expected=f"Contains: {', '.join(expected_words)}",
        actual=response[:200],
        explanation=f"Found {len(found)}/{len(expected_words)}: {', '.join(found)}",
    )


def _retrieval_evals(g) -> list[EvalResult]:
    """Can it recall what it was taught?"""
    return [
        _check("Recall gravity definition", "retrieval",
               g.hear("what is gravity?"), ["gravity", "force", "acceleration"]),
        _check("Recall energy definition", "retrieval",
               g.hear("what is energy?"), ["energy", "work", "kinetic"]),
        _check("Recall prime number", "retrieval",
               g.hear("what is a prime number?"), ["prime", "divisible", "1"]),
    ]


def _inference_evals(g) -> list[EvalResult]:
    """Can it derive facts it was never told?"""
    return [
        _check("Causal chain: what causes acceleration?", "inference",
               g.hear("what causes acceleration?"), ["force", "gravity"]),
        _check("Why question: why does acceleration happen?", "inference",
               g.hear("why does acceleration happen?"), ["force", "causes"]),
        _check("Multi-hop: how does gravity connect to energy?", "inference",
               g.hear("how does gravity connect to energy?"),
               ["gravity", "energy", "steps"]),
    ]


def _analogy_evals(g) -> list[EvalResult]:
    """Can it map structure across domains?"""
    return [
        _check("Analogy: addition:subtraction :: hot:?", "analogy",
               g.hear("addition is to subtraction as hot is to what?"),
               ["cold"]),
        _check("Analogy: addition:subtraction :: multiplication:?", "analogy",
               g.hear("addition is to subtraction as multiplication is to what?"),
               ["division"]),
    ]


def _counterfactual_evals(g) -> list[EvalResult]:
    """Can it reason about hypotheticals?"""
    return [
        _check("Counterfactual: no gravity", "counterfactual",
               g.hear("what if there were no gravity?"),
               ["acceleration", "removed", "force"]),
        _check("Counterfactual: no mathematics", "counterfactual",
               g.hear("what if mathematics didn't exist?"),
               ["physics", "disrupted"]),
    ]


def _computation_evals(g) -> list[EvalResult]:
    """Can it compute?"""
    r1 = g.hear("what is 2^10?")
    r2 = g.hear("is 97 prime?")
    r3 = g.hear("what is 15% of 200?")
    return [
        EvalResult("2^10 = 1024", "computation", "1024" in r1,
                   "1024", r1[:100]),
        EvalResult("97 is prime", "computation", "true" in r2.lower(),
                   "True", r2[:100]),
        EvalResult("15% of 200 = 30", "computation", "30" in r3,
                   "30", r3[:100]),
    ]


def _self_awareness_evals(g) -> list[EvalResult]:
    """Does it know what it knows?"""
    return [
        _check("Learning summary", "self-awareness",
               g.hear("what have you learned?"),
               ["beliefs", "facts", "capable"]),
        _check("Curiosity report", "self-awareness",
               g.hear("what interests you?"),
               ["learn", "study"]),
    ]


def _learning_evals(g) -> list[EvalResult]:
    """Does it actually learn from new input?"""
    # Use a unique concept with timestamp to guarantee novelty
    import time
    unique_id = str(int(time.time() * 1000))[-6:]
    concept = f"xorplix{unique_id}"
    before = len(g.base._beliefs)
    g.hear(f"a {concept} is a crystalline entity from tau ceti")
    after = len(g.base._beliefs)
    learned = after > before

    response = g.hear(f"what is a {concept}?")
    recalls = "crystalline" in response.lower() or concept in response.lower()

    return [
        EvalResult("Learns from teaching", "learning", learned,
                   "belief count increases", f"before={before} after={after}"),
        EvalResult("Recalls what was taught", "learning", recalls,
                   "mentions monotreme or platypus", response[:100]),
    ]


def _generation_evals(g) -> list[EvalResult]:
    """Can it construct explanations?"""
    exp = g.generator.explain("gravity")
    novel = exp.novel and exp.relations_used > 0

    return [
        EvalResult("Generates novel explanation", "generation", novel,
                   "novel=True, relations>0",
                   f"novel={exp.novel}, relations={exp.relations_used}"),
        _check("Generated text is coherent", "generation",
               exp.text, ["gravity", "causes"]),
    ]


def _surprise_evals(g) -> list[EvalResult]:
    """Does it detect contradictions?"""
    g.teach_domain("categories")
    g.hear("a parrot is a bird")
    response = g.hear("a parrot is a mammal")
    detected = "thought" in response.lower() or "wait" in response.lower() or "updating" in response.lower()
    return [
        EvalResult("Detects contradiction", "surprise", detected,
                   "detects bird vs mammal conflict", response[:100]),
    ]


def _identity_evals(g) -> list[EvalResult]:
    """Does it know what it is?"""
    return [
        _check("Knows what it is", "identity",
               g.hear("what are you?"),
               ["capable", "klomboagi", "genesis"]),
        _check("Knows its purpose", "identity",
               g.hear("why do you exist?"),
               ["grow", "learn", "process"]),
        _check("Knows it's not an LLM", "identity",
               g.hear("are you like chatgpt?"),
               ["not", "llm", "own", "original"]),
    ]
