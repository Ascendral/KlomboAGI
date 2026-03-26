# KlomboAGI — Metrics

## Scoreboard

Every run gets scored on these 7 metrics. If a change doesn't move one, it doesn't count.

### 1. Task Success Rate
- Binary: did the system produce the correct output?
- Measured per-task and aggregated per-domain
- Target: >50% on unseen tasks (Stage 1), >70% (Stage 3)

### 2. Intervention Count
- How many times did a human need to step in?
- 0 = fully autonomous, 1+ = assisted
- Target: <0.5 interventions per task average

### 3. Autonomy Horizon
- How many sequential steps can the system take without human input?
- Measured as max consecutive autonomous actions before failure or stall
- Target: 10+ steps (Stage 1), 50+ (Stage 4)

### 4. Failure Recovery
- When the system fails, does it detect the failure and try a different approach?
- Binary per failure event + quality of recovery
- Target: >50% recovery rate

### 5. Transfer Score
- Performance on domain B after training only on domain A
- Measured as ratio: success_on_B / success_on_B_with_direct_training
- Target: >0.3 transfer ratio

### 6. Memory Usefulness
- Did retrieving a memory/skill change the decision?
- Did the changed decision improve the outcome?
- Measured as: decisions_improved_by_memory / total_memory_retrievals
- Target: >30% usefulness rate

### 7. Regression Count
- How many previously-passing tasks broke after a code change?
- Must be 0 for any release
- Tracked per commit via CI

## Reporting

Every eval run produces a JSON report:
```json
{
  "run_id": "...",
  "timestamp": "...",
  "tasks_attempted": 50,
  "tasks_succeeded": 23,
  "interventions": 12,
  "avg_autonomy_horizon": 7.3,
  "failure_recoveries": 4,
  "failure_total": 27,
  "transfer_score": 0.21,
  "memory_retrievals": 15,
  "memory_useful": 5,
  "regressions": 0
}
```
