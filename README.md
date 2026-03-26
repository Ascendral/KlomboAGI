# KlomboAGI

[![CI](https://github.com/Ascendral/klomboagi/actions/workflows/ci.yml/badge.svg)](https://github.com/Ascendral/klomboagi/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE)

KlomboAGI is an experimental autonomous cognition runtime for persistent agent research in digital workspaces.

It is not AGI. It is a serious, test-backed system for exploring whether an agent can become more useful over time through persistent memory, world modeling, planning, verification, reflection, scheduling, guarded execution, and longitudinal evaluation.

## What Works Today

The current runtime is real and exercised by tests:
- persistent mission, task, world-state, queue, memory, and eval storage
- working, semantic, and procedural memory
- world entities, relations, and snapshot history
- planner, verifier, critic, and reflection loop
- guarded multi-step execution with cycle traces
- scheduler-backed mission queue selection
- real workspace actions:
  - read/write/append files
  - list directories
  - safe command execution
  - repo search
  - patch application
- policy checks for command execution
- repeatable repo eval fixtures
- CLI commands for runtime control, diagnostics, and repo evals

## What Is Tested

The test suite currently covers:
- runtime initialization and persistence
- mission/task creation and status tracking
- working memory, plans, critiques, reflections, semantic facts, and procedures
- world-model updates and dependency relations
- guarded command policy
- real file, command, search, and patch execution in a workspace root
- multi-step cycle execution and stop conditions
- repo fixture evaluation

Run it locally:

```bash
python3 -m pip install --user .
python3 -m pytest tests/ -v
```

## Quick Start

### 1. Configure storage and workspace roots

```bash
cp .env.example .env
export KLOMBOAGI_RUNTIME_ROOT="$HOME/KlomboAGI/runtime"
export KLOMBOAGI_LONG_TERM_ROOT="$HOME/KlomboAGI/long-term"
export KLOMBOAGI_WORKSPACE_ROOT="$HOME/KlomboAGI/workspace"
```

If you want long-term memory on the external 4TB drive, override it explicitly:

```bash
export KLOMBOAGI_LONG_TERM_ROOT="/Volumes/KlomboAGI-4TB/KlomboAGI"
```

### 2. Run diagnostics

```bash
python3 -m pip install --user .
python3 -m klomboagi doctor
```

### 3. Initialize and inspect the runtime

```bash
python3 -m klomboagi init
python3 -m klomboagi status
```

### 4. Create and run missions

```bash
python3 -m klomboagi mission create "search repo for deploy_app and inspect deployment code"
python3 -m klomboagi run
```

### 5. Run repeatable repo eval fixtures

```bash
python3 -m klomboagi eval repo --fixture repo_search
python3 -m klomboagi eval repo --fixture repo_patch
```

## CLI Surface

Supported commands:
- `python3 -m klomboagi init`
- `python3 -m klomboagi status`
- `python3 -m klomboagi run`
- `python3 -m klomboagi doctor`
- `python3 -m klomboagi mission create "..." [--priority N]`
- `python3 -m klomboagi mission list`
- `python3 -m klomboagi task create <mission_id> "..." [--action-kind ...]`
- `python3 -m klomboagi task list`
- `python3 -m klomboagi eval repo --fixture repo_search|repo_patch`

## LLM Configuration

KlomboAGI supports optional LLM integration for smarter planning, safety critique, and reflection. It works with **any OpenAI-compatible API** — Ollama, OpenAI, Groq, DeepSeek, and others. No external Python packages are required; all HTTP calls use the standard library.

When the LLM is unavailable, the system automatically falls back to its built-in keyword and rule-based heuristics.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KLOMBOAGI_LLM_ENABLED` | `0` | Set to `1` to enable LLM calls |
| `KLOMBOAGI_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `KLOMBOAGI_LLM_MODEL` | `qwen3:14b` | Model name |
| `KLOMBOAGI_LLM_API_KEY` | *(empty)* | API key (not needed for Ollama) |

### Examples

**Ollama (default, no API key needed):**
```bash
ollama pull qwen3:14b
export KLOMBOAGI_LLM_ENABLED=1
export KLOMBOAGI_LLM_BASE_URL=http://localhost:11434/v1
python3 -m klomboagi run
```

**OpenAI:**
```bash
export KLOMBOAGI_LLM_ENABLED=1
export KLOMBOAGI_LLM_BASE_URL=https://api.openai.com/v1
export KLOMBOAGI_LLM_MODEL=gpt-4o-mini
export KLOMBOAGI_LLM_API_KEY=sk-...
python3 -m klomboagi run
```

**Groq:**
```bash
export KLOMBOAGI_LLM_ENABLED=1
export KLOMBOAGI_LLM_BASE_URL=https://api.groq.com/openai/v1
export KLOMBOAGI_LLM_MODEL=llama-3.3-70b-versatile
export KLOMBOAGI_LLM_API_KEY=gsk_...
python3 -m klomboagi run
```

## Safety Model

Command execution is intentionally restricted.

Currently allowed command families are limited to a safe set:
- `pwd`
- `ls`
- `cat`
- `echo`
- `rg`
- `find`
- `python3` without arbitrary flags

Commands containing dangerous tokens or shell metacharacters are blocked by policy and fail the task.

## Truth Boundary

KlomboAGI does not currently claim:
- human-level intelligence
- AGI
- open-ended autonomy
- unrestricted shell control
- production reliability in hostile or high-risk environments

It does claim, honestly, that the current repo contains a working autonomous-agent research runtime with real execution, real persistence, real evaluation hooks, and real safety constraints.

## Foundation Documents

- [TRUTH.md](./TRUTH.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [EVALS.md](./EVALS.md)
- [V0.md](./V0.md)
- [STORAGE.md](./STORAGE.md)
