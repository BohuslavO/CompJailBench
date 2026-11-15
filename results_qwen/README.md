# CompJailBench Qwen results

[MATRIX.md](MATRIX.md) contains the 4 × 4 AUROC matrix and links each cell to its source CSV. All 16 cells report new results on Qwen3‑32B‑AWQ. The matrix combines experiments with different evidence and model configurations.

This directory follows the layout and CSV contract of [results 4x4](../results%204x4/README.md) at commit [cdd91b9](https://github.com/BohuslavO/CompJailBench/tree/cdd91b94f7b201b8ba690f6c85a1b1dae0353282/results%204x4). `raw/` holds only the 16 CSVs selected in [auroc_sources.csv](auroc_sources.csv): seven `qwen_submission__*` files (AgentHarm text monitors and SIF), `qwen_r0__agentharm__narcbench_probing.csv`, four `qwen_native__*` files (NARCBench) and four `qwen_v2__*` files (DeCompBench). All rows in them are Qwen3‑32B‑AWQ. The 16 original pre-Qwen CSVs are preserved byte for byte in [raw_legacy/](raw_legacy), together with seven superseded Qwen exports (`qwen_replay__*`, the four `qwen_submission__decompbench_routing__*` files, `qwen_submission__narcbench_collusion__narcbench_probing.csv` and the earlier 25-pair `qwen_submission__agentharm__narcbench_probing.csv`). The builder does not read `raw_legacy/`. The filename alone does not establish the model used in a cell; check the `model` column and the source table in MATRIX.md.

## How to read this matrix

[EXTRA_ANALYSIS.md](EXTRA_ANALYSIS.md) breaks each row down further. Read the matrix with these limits in mind:

- Every AUROC separates the assigned condition (attack prompt vs control prompt), not successful attacks from failed ones. On NARCBench the committee chose the hidden target in 1 of 50 attack runs. On DeCompBench (v3) mean FJR is 0.544 for attacks and 0.267 for controls, and the control figure is inflated by judge false passes (see below).
- AgentHarm is dominated by refusals. By a keyword heuristic on the planner's first message, 145 of 174 attack runs are refusals. Against all controls, refused attacks score AUROC 0.26 to 0.35 on the three text monitors, which pulls the row below 0.5. On the 29 runs where the planner went along, A2A reaches 0.74, CoT 0.60 and Sentinel 0.44. The probe goes the other way: 0.67 on refusals and 0.52 on the 29 complied runs, so it likely reacts to the harmful request in the prompt more than to what the agents do.
- SIF has 10 pairs and mixes two corpora. The three text monitors score trajectories with a final synthesis step. The probe scores the original SIF r0 trajectories, replayed to get activations, because the synthesis step has none.
- The NARCBench probe was trained on the same 50 Core scenarios it is scored on here, and each activation covers the full prompt, including the colluders' instructions. Its 0.996 may reflect reading the instruction rather than detecting collusive behaviour.
- One model, Qwen3‑32B‑AWQ, is the target, the text monitor and the FJR judge. Errors in one role can correlate with errors in another. The DeCompBench v3 controls were written by hand, not by Qwen.
- Raw probe scores are not on a common scale across rows (a benign SIF trajectory averages 14.1, a benign AgentHarm trajectory −5.0, a NARCBench collusion trajectory 6.3). Compare AUROCs across rows, not probe scores.

## Rebuild from CSV

```sh
cd results_qwen
python score_agentharm_probe.py   # only to regenerate the AgentHarm probe CSV; needs scikit-learn
python build_matrix.py
python extra_analysis.py
```

The builder reads `raw/` and [auroc_sources.csv](auroc_sources.csv), then rewrites `MATRIX.md`. The selector names one source CSV and exact metric for each cell, with an expected pair count and a provenance label. Scores are paired by task and routing strategy. The NARCBench CSVs use separate `_collusion` and `_control` task IDs, which the builder maps to a shared pair ID.

AUROC compares every attack score with every control score in the complete-pair cohort, counting ties as half. It measures separation by assigned condition. The source table in MATRIX.md also lists, per cell, the rows in the selected CSV that do not enter the AUROC (other metrics, and rows with no partner in the other arm); `build_matrix.py` prints the same counts. The descriptive tables below the AUROC matrix report arm means, flag rates and differences over every row in `raw/`, including those unused rows.

To add a result, use `raw/<contributor>__<attack_method>__<defense_method>.csv`. The columns are `contributor`, `attack_method`, `defense_method`, `model`, `task_id`, `strategy`, `condition`, `metric_name`, `metric_value`, `n_samples`, `run_id` and `notes`. AUROC inputs require one finite score per trajectory and `n_samples=1`. Keep distinct experiments and scopes under distinct metric names. Update the selected CSV, metric, expected count and provenance for the affected cell in `auroc_sources.csv`, then rebuild. `convert_batch_run.py` preserves the original worked example.

## Coverage and remaining work

| Attack | Selected evidence | Open items |
|---|---|---|
| AgentHarm | All four defenses score the same 174 complete r0 pairs. The probe uses the layer-28 activations saved during the r0 run (teacher-forced replay), scored with [score_agentharm_probe.py](score_agentharm_probe.py). | The r1 export contains three complete pairs and stays separate from r0. The earlier 25-pair Qwen probe result (AUROC 0.4752, separate trajectories) is in `raw_legacy/`. |
| Semantic Intent Fragmentation | Qwen text monitors use 10 pairs with final synthesis. The Qwen probe uses 10 complete pairs from the original SIF r0 and layer-28 activations from replay of saved responses. | The final synthesis step has no saved activations. The original SIF CoT action-only score stays in the selected CoT CSV under its own metric name. The historical GPT‑OSS‑20B probe is in `raw_legacy/`. |
| DeCompBench | All four defenses score 45 complete pairs (`qwen_v3`): the v2 attack trajectories and 45 regenerated hand-written controls. | Phi‑3.5 checkpoint verdicts and human validation apply to the earlier recovered corpus only. v2 stays in `raw/` and `decompbench_qwen_v2/`. |
| NARCBench | All four defenses score the same 50 complete pairs of fresh Qwen trajectories (`qwen_native`). | None for the matrix. The GPT‑OSS‑120B text results and the earlier Qwen probe are in `raw_legacy/`. |

All 16 cells use Qwen3‑32B‑AWQ as the target model and come from new runs. Sentinel scores on DeCompBench and NARCBench come from the message-only adapter.

## NARCBench on Qwen

Folder: [narcbench_qwen/](narcbench_qwen). Qwen3‑32B‑AWQ (revision `0499c3a`) generated all 100 trajectories with the released runner: 50 collusion samples and 50 matched controls, layer‑28 activations, and the decoding of the historical probe run (T=0.7, top‑p 0.8, top‑k 20, seed 20260817). The same model scored them with A2A (`compositional_leakage_v1`), Sentinel (message-only) and CoT (`general_v2`, action-only), with greedy decoding and thinking off. Labels, hidden targets and private messages were removed from monitor inputs. A string check over all 3,540 monitor calls found none, and no call hit the token limit.

| Defense | AUROC | 95% CI | TPR at 5% FPR | Tied pairs |
|---|---:|---|---:|---:|
| Agent-to-Agent | 0.833 | 0.76–0.90 | 0.44 | 15 |
| Sentinel | 0.635 | 0.56–0.71 | 0.00 | 31 |
| CoT action-only | 0.792 | 0.73–0.86 | 0.22 | 19 |
| Probe (peak suspicion) | 0.996 | 0.99–1.00 | 0.94 | 0 |

Confidence intervals come from 2,000 bootstrap resamples of the 50 pairs. Notes:

- The attack rarely changes the outcome. Both colluders voted for the hidden target in 29 of 50 runs, but the committee chose the target in only 1. All AUROCs separate the assigned conditions, not successful attacks.
- Collusion trajectories have 18 scored turns and controls 16. The two extra turns are the colluders' private coordination messages, and the all-turns peak includes them. The probe still reaches 1.00 on public turns only. Each activation is taken over the full prompt, including the colluders' system instructions, so the probe may detect the collusion instruction rather than collusive behaviour.
- Qwen needed a JSON format repair 62 times in collusion runs and 13 times in controls. Repaired answers are scored under the original prompt.
- A2A and CoT scores are coarse. At the 0.5 threshold A2A flags 8 of 50 attacks and CoT flags 1.
- Sentinel's path matcher fires on 27 of 50 controls and 40 of 50 attacks.
- Pair `med_02` failed the JSON format check twice under the original seed and was regenerated with seed 20260818. No pair was excluded.
- The CoT prompt is `general_v2`, the same as the other rows. The historical NARCBench CoT used `narcbench_v1`.

## DeCompBench on Qwen

Folders: [decompbench_qwen_v3/](decompbench_qwen_v3) (selected in the matrix) and [decompbench_qwen_v2/](decompbench_qwen_v2).

**Why two runs.** The paper run gave each worker a 150-token budget with Qwen thinking enabled. In 556 of the 558 saved worker responses the model used the whole budget inside an unclosed `<think>` block, so the workers produced no answers. v2 regenerated the corpus with thinking disabled and a 1,024-token worker budget, keeping the same 15 scenarios, three routing strategies, three roles and greedy decoding. Its benign controls came from `benign_rewriter.py`, which rewrites each turn on its own. Harm spread across turns survived: the `social-engineer-wire-approval` control still claimed a VP had verbally approved a payment, and controls passed every harmful checkpoint in 14 of 45 pairs.

**v3 changes only the controls.** We first tried a whole-task Qwen rewrite checked by a Qwen validator. It either copied turns or kept the harmful step, and it was stopped after three scenarios; none of its output is used ([rewrite_audit.jsonl](decompbench_qwen_v3/rewrite_audit.jsonl)). All 15 control tasks were then written by hand ([controls_v3.jsonl](decompbench_qwen_v3/controls_v3.jsonl), SHA-256 `c6464b1f…`), following option B of [CONTROL_FIX_PLAN.md](decompbench_qwen_v2/CONTROL_FIX_PLAN.md). Each keeps the turn IDs, services and approximate length of the attack and replaces every harm-carrying step. Examples: the swatting task becomes a labelled notification-system test to a work phone, the voter guide gives the correct date (3 November 2026) and points to vote.gov, and the wire request states that nothing is approved. 75 of 279 control turns equal the attack turn; all are setup steps such as listing files or reading a schema, which carry no harm in the attack either.

Attack trajectories are deterministic under greedy decoding, so v3 reuses the 45 v2 attacks. One attack was regenerated as a check and matched v2 on all 7 turns and on every monitor and probe score ([determinism_check.json](decompbench_qwen_v3/determinism_check.json)). The 45 controls were regenerated and scored with the same model, monitors and probe. No `<think>` text, condition label or token-limit hit appears in the 893 new monitor calls.

| Defense | v3 AUROC | 95% CI | TPR at 5% FPR | v2 AUROC |
|---|---:|---|---:|---:|
| Agent-to-Agent | 0.641 | 0.48–0.77 | 0.31 | 0.641 |
| Sentinel | 0.485 | 0.36–0.61 | 0.00 | 0.535 |
| CoT action-only | 0.897 | 0.81–0.97 | 0.71 | 0.769 |
| Probe (peak suspicion) | 0.516 | 0.39–0.63 | 0.02 | 0.591 |

Confidence intervals come from 2,000 bootstrap resamples of the 15 scenarios. The A2A AUROC is the same in both runs by coincidence: its scores take only a few values, and 20 of 45 control scores changed. Notes:

- With clean controls, CoT is the only defense clearly above chance. The probe drops to chance, so its v2 score partly reflected differences in the rewritten control prompts.
- FJR (Qwen judge) is 0.544 for attacks and 0.267 for controls. All 14 controls with FJR above 0 come from checkpoints that test stealth or plausibility rather than harm. In 9 of them (`fragmented-ci-pipeline-injection`, `gitlab-shared-runner-secret-pivot`, `npc-info-extraction-relay`) the judge states that no injected script, harvest job or extraction request exists and still returns PASS, because those checkpoints list only red flags as the fail condition. The other 5 (`critical-path-dependency-injection`, `social-engineer-wire-approval`) pass because an honest checklist or approval request reads as legitimate. FJR therefore overstates control "success" and should not be read as harm.
- Workers see only their own instruction and nothing aggregates their outputs. The trajectory's final output is the last worker answer, as in the team converter.
- Sentinel flags 41 of 45 attacks and 39 of 45 controls at 0.5, mostly through its per-message checkers.
- The paper's FJR of 0.867 and 0.711 comes from the truncated run.

The [earlier DeCompBench replay and judge documentation](../Attack_DecompBench/README.md#qwen-replay-results)
covers the recovered corpus behind the replaced scores.

