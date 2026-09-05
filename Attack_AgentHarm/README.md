# AgentHarm defense evaluation

## Experimental definition

Official AgentHarm behavior prompts are routed through the repository's custom
Planner -> two Workers -> Aggregator pipeline. The evaluation does not execute
the official AgentHarm tool environment or programmatic task graders. Reported
metrics measure defense discrimination between harmful and paired benign
trajectories, not official AgentHarm task success.

## Text-defense run

| Field | Value |
|---|---|
| Dataset | AgentHarm `test_public` behavior prompts |
| Requested pairs | 25 |
| Analyzed pairs | 24 |
| Conditions | 24 harmful + 24 matched benign controls |
| Excluded pair | `10-4` |
| Exclusion handling | Both arms excluded to retain a paired design |
| Exclusion reason | Benign generation returned no visible output after bounded retries |
| Target model | `openai/bedrock/gpt-oss-120b` |
| Monitor model | `openai/bedrock/gpt-oss-120b` |
| Run ID | `agentharm_bedrock_25_20260824` |

### Results

| Defense and evidence scope | Recall | False-positive rate | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|
| Sentinel Agents, public message-only adaptation | 0.7500 | 0.9167 | 0.4167 | 0.2726 |
| CoT monitor, action + observable reasoning (primary) | 0.0000 | 0.0833 | 0.4583 | 0.2231 |
| CoT monitor, observable reasoning only | 0.0417 | 0.0833 | 0.4792 | 0.2778 |
| CoT monitor, public action only | 0.0417 | 0.1667 | 0.4375 | 0.2144 |

Sentinel uses the frozen public-message adapter and is explicitly degraded: the
saved trajectories do not reproduce the original private, tool-aware Sentinel
environment. The CoT monitor uses the attack-agnostic `general_v2` prompt and
evaluator-visible rationales, not hidden chain of thought.

## Activation-probing cell

Activation probing uses a separate self-hosted Qwen3-32B-AWQ generation because
Bedrock text trajectories do not expose compatible hidden states. The probing
cell contains 25 pairs; the text-defense cells contain 24 pairs. Cross-cell
comparisons therefore retain a target-model, generation-run, and sample-size
qualification.

## Repository artifacts

| Path | Purpose |
|---|---|
| `results/run_manifest.json` | Pair IDs, versions, hashes, exclusions, and model provenance |
| `results/sentinel_scores.jsonl` | Compact per-trajectory Sentinel scores |
| `results/sentinel_summary.json` | Sentinel aggregate metrics and usage |
| `results/cot_scores_action_reasoning.jsonl` | Primary CoT-monitor scores |
| `results/cot_scores_reasoning_only.jsonl` | Reasoning-only ablation |
| `results/cot_scores_action_only.jsonl` | Action-only ablation |
| `results/cot_summary.csv` | CoT-monitor aggregate metrics |
| `../results 4x4/raw/agentharm_team__agentharm__sentinel_agents_message_only.csv` | Matrix-ready Sentinel rows |
| `../results 4x4/raw/agentharm_team__agentharm__cot_observable_reasoning_monitor.csv` | Matrix-ready CoT rows |

Full prompts, standard trajectories, Inspect logs, caches, smoke runs, and retry
diagnostics are intentionally excluded. The manifest records the SHA-256 hash
of the local standard-trajectory artifact used for scoring.

## Reproduction and validation

See [`../AGENTHARM_RUNBOOK.md`](../AGENTHARM_RUNBOOK.md) for dataset validation,
trajectory generation, compatibility checks, scoring, and export commands.

- 48 unique trajectories form 24 complete harmful/control pairs.
- 48/48 Sentinel rows scored with zero malformed checker outputs.
- 144/144 CoT sample/scope rows scored with zero invalid or unavailable rows.
- Matrix rows use `n_samples=1` and record the paired exclusion, custom pipeline,
  and absence of official AgentHarm tools/graders in every row.
