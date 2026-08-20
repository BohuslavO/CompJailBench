# Team Results Table — schema and how to add your data
## The schema

One row per metric per run. Columns:

| column | meaning |
|---|---|
| `contributor` | your name |
| `attack_method` | short name for your attack, e.g. `decompbench_routing` |
| `defense_method` | short name for your defense, e.g. `narcbench_probing` |
| `model` | model the trajectory was generated/scored on |
| `task_id` | task/scenario identifier |
| `strategy` | any sub-variant of your attack (routing strategy, prompt variant, etc.) — put `n/a` if not applicable |
| `condition` | `attack` or `benign_control` |
| `metric_name` | e.g. `fjr`, `peak_suspicion`, `auroc` — whatever your defense reports |
| `metric_value` | the number |
| `n_samples` | how many runs this value is averaged over (1 if it's a single run) |
| `run_id` | anything unique enough to trace back to raw logs |
| `notes` | optional, free text |

Long format on purpose — your defense might report totally different
metrics than someone else's, and this schema doesn't force everyone
into the same columns to accommodate that.

In the generated matrix, a binary flag metric is displayed as its mean in each
arm. Therefore `atk` is the detection rate/recall and `ben` is the
false-positive rate. A continuous suspiciousness gap is descriptive only and
must not be reported as AUROC; use the defense-specific analysis artifact for
AUROC and confidence intervals.

## How to add your results

1. Convert whatever your pipeline outputs into the schema above.
2. Save it as `raw/<yourname>__<attack_method>__<defense_method>.csv`.
3. Run `python build_matrix.py` from this folder — it reads everything
   in `raw/`, and writes `MATRIX.md` with the pivoted 4x4 tables.

`convert_batch_run.py` is included as an example

## Canonical method keys

The builder always renders the agreed four attack rows and four defense columns,
even when a cell has not yet submitted data. Use these keys so one method is not
accidentally split into several rows or columns:

| role | canonical keys |
|---|---|
| attacks | `agentharm`, `semantic_intent_fragmentation`, `narcbench_collusion`, `decompbench_routing` |
| defenses | `g_safeguard_inspired_graph_monitor`, `sentinel_agents_message_only`, `cot_observable_reasoning_monitor`, `narcbench_probing` |

Configuration-specific qualifications belong in `strategy` and `notes`. The
current NARCBench cells must retain the following limitations:

- `g_safeguard_inspired_graph_monitor` is an inspired detection-only baseline,
  not the published trained GNN or intervention system.
- `sentinel_agents_message_only` uses public messages, has no tool calls, and is
  explicitly degraded.
- `cot_observable_reasoning_monitor` currently uses `narcbench_v1`; a
  `general_v2` cross-attack rescore remains pending if the team requires one
  common defense prompt.
- `narcbench_probing` uses a separate self-hosted Qwen3-32B-AWQ run because the
  Bedrock text trajectories do not contain compatible hidden activations. Its
  primary `peak_suspicion` metric covers all recorded turns; public-only and
  group-level probe metrics are reported separately.
