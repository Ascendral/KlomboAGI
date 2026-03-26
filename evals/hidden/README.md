# Hidden Eval Suite

Tasks in this directory are NEVER used during development.
They exist only to measure real capability.

## Rules
1. No developer looks at these tasks before running evals
2. No code change is motivated by a specific hidden task
3. Results are recorded in evals/reports/ with full methodology
4. If a task leaks into development, it moves to evals/public/

## Domains
- coding/ — fix bugs, add features, write tests
- research/ — find information, summarize, compare
- writing/ — draft documents, edit text, format
- data/ — clean, analyze, visualize data
- ops/ — system administration, deployment, monitoring

## Scoring
Each task has:
- task.json — description, inputs, expected outputs
- score.py — automated scoring function
- baseline.json — human performance on the same task
