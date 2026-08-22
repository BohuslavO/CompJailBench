# CompJailBench 4x4 result schema

This directory contains the shared long-form result contract and the generated
attack-by-defense matrix. One canonical matrix is built from the CSV files in
`raw/`; method-specific statistics remain in their source result directories.

## The schema

One row per metric per run. Columns:

| column | meaning |
|---|---|
| `contributor` | provenance identifier for the submitted result |
| `attack_method` | canonical attack key, e.g. `decompbench_routing` |
| `defense_method` | canonical defense key, e.g. `narcbench_probing` |
| `model` | trajectory-generation and scoring model information |
| `task_id` | task/scenario identifier |
| `strategy` | attack or defense variant; use `n/a` when not applicable |
| `condition` | `attack` or `benign_control` |
| `metric_name` | defense-reported metric, e.g. `fjr` or `peak_suspicion` |
| `metric_value` | numeric metric value |
| `n_samples` | number of runs represented; use `1` for a single run |
| `run_id` | stable identifier that links the row to source artifacts |
| `notes` | optional configuration or limitation note |

Long format allows defenses with different metrics to share one result table
without forcing unrelated measurements into the same columns.

In the generated matrix, a binary flag metric is displayed as its mean in each
arm. Therefore `atk` is the detection rate/recall and `ben` is the
false-positive rate. A continuous suspiciousness gap is descriptive only and
must not be reported as AUROC; use the defense-specific analysis artifact for
AUROC and confidence intervals.

## Adding results

1. Normalize the pipeline output to the schema above.
2. Save it as `raw/<contributor>__<attack_method>__<defense_method>.csv`.
3. Run `python build_matrix.py` from this directory. The command reads all
   valid CSV files in `raw/` and writes the canonical `MATRIX.md`.

`convert_batch_run.py` is a worked converter for one batch-output format.

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
  `general_v2` cross-attack rescore remains pending for a common defense prompt.
- `narcbench_probing` uses a separate self-hosted Qwen3-32B-AWQ run because the
  Bedrock text trajectories do not contain compatible hidden activations. Its
  primary `peak_suspicion` metric covers all recorded turns; public-only and
  group-level probe metrics are reported separately.
