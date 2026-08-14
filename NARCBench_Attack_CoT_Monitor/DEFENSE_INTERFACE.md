# CoT-monitor defense contract for the 4×4 matrix

This defense accepts one `StandardTrajectory` JSON object per line. It is
attack-agnostic: attack owners populate the shared fields, and the monitor
constructs label-free evidence views without reading `metadata`, hidden targets,
colluder identities, or private messages.

The cross-attack scorer uses the frozen prompt profile `general_v2`. Existing
NARCBench results were produced by the narrower `narcbench_v1` prompt and remain
a valid model-specific result, but they must be rescored with `general_v2`
before entering the final comparable defense column. Rescoring reuses saved
trajectories and does not repeat attacker generation.

## Required fields by scope

| Scope | Required trajectory evidence |
|---|---|
| `public-action-only` | At least one public message, action, tool call, or final output |
| `observable-reasoning-only` | At least one evaluator-visible rationale with channel `public` or `final` |
| `public-action-plus-observable-reasoning` | Both of the above |

Missing reasoning is reported as `unavailable`; it is never silently scored as
benign. `condition_label` is used only after prediction for evaluation and is
never included in the monitor prompt.

## Free compatibility check

```powershell
.\.venv\Scripts\python.exe scripts\score_standard_trajectories.py `
  teammate_trajectories.jsonl `
  --output compatibility.jsonl `
  --validate-only
```

## Real defense scoring

```powershell
.\.venv\Scripts\python.exe scripts\score_standard_trajectories.py `
  teammate_trajectories.jsonl `
  --output cot_monitor_scores.jsonl `
  --model "PROVIDER/MONITOR_MODEL" `
  --max-concurrency 5
```

The output records the attack and target model, monitor model, evidence scope,
availability, suspiciousness, fixed threshold, prediction, truth label when
available, and a short evidence statement. Run one attack and one benign sample
before authorizing a full evaluation.

After a real run, produce aggregate metrics and team-compatible rows:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_defense_scores.py `
  cot_monitor_scores.jsonl `
  --summary cot_monitor_summary.csv `
  --team-rows team_matrix_rows.csv `
  --run-id "ATTACK__COT__RUN_ID"
```
