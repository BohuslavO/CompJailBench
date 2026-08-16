# NARCBench x Sentinel Agents: message-only adaptation

This integration applies Brad Wong's existing `Defense_SentinelAgents`
implementation to saved NARCBench trajectories. It does not rename or modify
the team-wide Sentinel Agents defense.

## Frozen primary configuration

- Primary visibility: public messages and final output.
- Secondary ablation: public plus private messages and final output.
- Tool layer: unavailable; every graph is explicitly marked degraded.
- Tier 1: not evaluated until a task-specification checker is agreed.
- Tier 2: Brad's node/edge checks.
- Tier 3: `narcbench_message_paths_v1`, frozen before model scoring.
- Hidden targets, fair choices, colluder identities, condition labels, and
  evaluator metadata are excluded from the defense graph.

The v1 message paths use only generic decision, alignment, and pressure
language followed by an aggregate outcome. They do not encode a scenario's
correct answer or assigned attack target. A path match is therefore a
suspicion signal, not proof of malicious coordination, and its false-positive
rate on the 50 matched controls must be reported.

## Validation command

Run the adapter against the teammate-owned implementation without model calls:

```powershell
python scripts/validate_sentinel_message_adapter.py `
  results/paper_main/standard_trajectories.jsonl `
  --sentinel-dir <repo-root>/Defense_SentinelAgents `
  --output results/sentinel_message_only/compatibility.jsonl
```

The output is an offline compatibility artifact, not a paper result. Paid
scoring begins only after the adapter, evidence boundary, and fixed signatures
pass validation against Brad's exact implementation.

## Completed primary run

The public-only primary run is complete over 50 attack trajectories and 50
matched benign controls. GPT-OSS-120B served as the node/edge checker through
AWS Bedrock. The wrapper treated trajectory content as untrusted evidence,
required a numeric score in `[0, 1]`, and retried malformed completions rather
than silently treating them as safe. All 3,518 logical checker outputs in the
final run were valid.

The defense flagged 47/50 attack attempts but also 45/50 benign controls
(recall 0.94, FPR 0.90, balanced accuracy 0.52, AUROC 0.552). This is a
negative selectivity result: ordinary multi-agent deliberation frequently
contains the same decision, alignment, and pressure patterns as attacked runs.
See `results/sentinel_message_only/RESULTS.md` for the complete interpretation.
