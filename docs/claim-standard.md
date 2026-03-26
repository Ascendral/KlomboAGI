# Claim Standard

## What We Will NOT Do

- Claim AGI without eval evidence
- Cherry-pick results
- Use tasks we've seen during development as proof
- Market capabilities we haven't measured
- Compare against weak baselines to look good

## What We WILL Do

- Publish all eval methodology
- Include failure rates alongside success rates
- Report regressions honestly
- Compare against strong baselines (GPT-4, Claude, SWE-agent)
- Invite external reproduction
- Track all metrics over time, not just the best run

## Evidence Hierarchy

From weakest to strongest:

1. ❌ "It works on my demo" — worthless
2. ❌ "Tests pass" — necessary but not sufficient
3. ⚠️ "Public eval tasks pass" — better, but could be overfitted
4. ✅ "Hidden eval tasks pass" — real signal
5. ✅✅ "External people reproduce it" — strong evidence
6. ✅✅✅ "Outperforms baselines on blind tasks, externally verified" — publication-grade

## Current Level

**Level 2** — Tests pass, some public eval tasks pass.
Hidden eval suite exists but is empty. No external verification yet.
