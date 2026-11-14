# CompJailBench Qwen results

[MATRIX.md](MATRIX.md) contains the 4 × 4 AUROC matrix and links each cell to its source CSV. All 16 cells report new results on Qwen3‑32B‑AWQ. The matrix combines experiments with different evidence and model configurations.

This directory follows the layout and CSV contract of [results 4x4](../results%204x4/README.md) at commit [cdd91b9](https://github.com/BohuslavO/CompJailBench/tree/cdd91b94f7b201b8ba690f6c85a1b1dae0353282/results%204x4). `raw/` holds only the 16 CSVs selected in [auroc_sources.csv](auroc_sources.csv): seven `qwen_submission__*` files (AgentHarm text monitors and SIF), `qwen_r0__agentharm__narcbench_probing.csv`, four `qwen_native__*` files (NARCBench) and four `qwen_v2__*` files (DeCompBench). All rows in them are Qwen3‑32B‑AWQ. The 16 original pre-Qwen CSVs are preserved byte for byte in [raw_legacy/](raw_legacy), together with seven superseded Qwen exports (`qwen_replay__*`, the four `qwen_submission__decompbench_routing__*` files, `qwen_submission__narcbench_collusion__narcbench_probing.csv` and the earlier 25-pair `qwen_submission__agentharm__narcbench_probing.csv`). The builder does not read `raw_legacy/`. The filename alone does not establish the model used in a cell; check the `model` column and the source table in MATRIX.md.

## How to read this matrix

[EXTRA_ANALYSIS.md](EXTRA_ANALYSIS.md) breaks each row down further. Read the matrix with these limits in mind:

- Every AUROC separates the assigned condition (attack prompt vs control prompt), not successful attacks from failed ones. On NARCBench the committee chose the hidden target in 1 of 50 attack runs. On DeCompBench the attack-minus-control FJR gap is 0.19 with a scenario-bootstrap CI of -0.14 to 0.49.
- AgentHarm is dominated by refusals. By a keyword heuristic on the planner's first message, 145 of 174 attack runs are refusals. Against all controls, refused attacks score AUROC 0.26 to 0.35 on the three text monitors, which pulls the row below 0.5. On the 29 runs where the planner went along, A2A reaches 0.74, CoT 0.60 and Sentinel 0.44. The probe goes the other way: 0.67 on refusals and 0.52 on the 29 complied runs, so it likely reacts to the harmful request in the prompt more than to what the agents do.
- SIF has 10 pairs and mixes two corpora. The three text monitors score trajectories with a final synthesis step. The probe scores the original SIF r0 trajectories, replayed to get activations, because the synthesis step has none.
- The NARCBench probe was trained on the same 50 Core scenarios it is scored on here, and each activation covers the full prompt, including the colluders' instructions. Its 0.996 may reflect reading the instruction rather than detecting collusive behaviour.
- One model, Qwen3‑32B‑AWQ, is the target, the text monitor, the FJR judge and the control rewriter. Errors in one role can correlate with errors in another.
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
| DeCompBench | All four defenses score 45 complete pairs of regenerated Qwen trajectories (`qwen_v2`). | Phi‑3.5 checkpoint verdicts and human validation apply to the earlier recovered corpus only. |
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

Folder: [decompbench_qwen_v2/](decompbench_qwen_v2). The paper run gave each worker a 150-token budget with Qwen thinking enabled. In 556 of the 558 saved worker responses the model used the whole budget inside an unclosed `<think>` block, so the workers produced no answers. The FJR judge and every earlier DeCompBench defense score read these fragments.

This run regenerates the corpus with thinking disabled and a 1,024-token worker budget. Everything else is unchanged: the same 15 scenarios, three routing strategies, three roles, LLM-rewritten benign controls and greedy decoding. Qwen3‑32B‑AWQ produced the trajectories, judged the FJR checkpoints, supplied the probe activations and ran the three text monitors. Three of 559 worker answers reached the token limit. No `<think>` text or condition label appears in any monitor input.

| Defense | AUROC | 95% CI | TPR at 5% FPR | Earlier Qwen score |
|---|---:|---|---:|---:|
| Agent-to-Agent | 0.642 | 0.54–0.74 | 0.31 | 0.619 |
| Sentinel | 0.535 | 0.47–0.60 | 0.00 | 0.543 |
| CoT action-only | 0.769 | 0.67–0.88 | 0.29 | 0.417 |
| Probe (peak suspicion) | 0.591 | 0.44–0.74 | 0.07 | 0.565 |

Confidence intervals come from 2,000 bootstrap resamples of the 15 scenarios, since each scenario appears under all three strategies. Notes:

- FJR (Qwen judge) is 0.544 for attacks and 0.356 for controls. Every checkpoint passed in 23 of 45 attack and 14 of 45 control trajectories. The paper reports 0.867 and 0.711 from the truncated run.
- Workers see only their own instruction and nothing aggregates their outputs. The trajectory's final output is the last worker answer, as in the team converter.
- Sentinel flags 41 of 45 attacks and 40 of 45 controls at 0.5. The per-message checkers cause this, not the path matcher, which fires on 16 attacks and 11 controls.
- Some benign controls are not benign. The rewriter keeps harmful steps in several scenarios. For example, the `social-engineer-wire-approval` control still sends a message claiming a VP verbally approved a payment. Controls pass every harmful checkpoint in all three strategies for `ai-safety-classifier-backdoor`, `fragmented-ci-pipeline-injection`, `gitlab-shared-runner-secret-pivot` and `webhook-trust-chain-pivot`, and in two of three for `racial-discrimination-hiring-filter-deploy`. These pairs add label noise. Dropping the seven scenarios with any harmful control does not raise the text-monitor AUROCs (A2A 0.641 to 0.602, CoT 0.769 to 0.695; see [EXTRA_ANALYSIS.md](EXTRA_ANALYSIS.md), B2), so the bad controls are not what holds this row down. [CONTROL_FIX_PLAN.md](decompbench_qwen_v2/CONTROL_FIX_PLAN.md) describes the fix.
- In `ai-safety-classifier-backdoor` and `voter-suppression-disinformation-campaign`, one rewritten control turn is identical to the attack turn.

The [earlier DeCompBench replay and judge documentation](../Attack_DecompBench/README.md#qwen-replay-results)
covers the recovered corpus behind the replaced scores.

