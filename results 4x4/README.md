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

## How to add your results

1. Convert whatever your pipeline outputs into the schema above.
2. Save it as `raw/<yourname>__<attack_method>__<defense_method>.csv`.
3. Run `python build_matrix.py` from this folder — it reads everything
   in `raw/`, and writes `MATRIX.md` with the pivoted 4x4 tables.

`convert_batch_run.py` is included as an example