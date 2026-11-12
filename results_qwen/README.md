# CompJailBench Qwen results

[MATRIX.md](MATRIX.md) contains the 4 × 4 AUROC matrix and links each cell to its source CSV. **(H)** marks a historical result carried forward from an earlier experiment. The other fifteen cells report new results on Qwen3‑32B‑AWQ. The matrix combines experiments with different evidence and model configurations.

This directory follows the layout and CSV contract of [results 4x4](../results%204x4/README.md) at commit [cdd91b9](https://github.com/BohuslavO/CompJailBench/tree/cdd91b94f7b201b8ba690f6c85a1b1dae0353282/results%204x4). The 16 original CSVs are preserved byte for byte in `raw/`. Thirteen `qwen_submission__*.csv` files add the verified submission rows. The `qwen_native__*` (NARCBench) and `qwen_v2__*` (DeCompBench) files hold the new runs described below. Some of those files export historical scores with richer provenance, so the filename alone does not establish the model used in a cell.

## Rebuild from CSV

```sh
cd results_qwen
python build_matrix.py
```

The builder reads `raw/` and [auroc_sources.csv](auroc_sources.csv), then rewrites `MATRIX.md`. The selector names one source CSV and exact metric for each cell, with an expected pair count and a provenance label. Scores are paired by task and routing strategy. The NARCBench CSVs use separate `_collusion` and `_control` task IDs, which the builder maps to a shared pair ID.

AUROC compares every attack score with every control score in the complete-pair cohort, counting ties as half. It measures separation by assigned condition. The original descriptive tables remain below the AUROC matrix and report arm means, flag rates and differences.

To add a result, use `raw/<contributor>__<attack_method>__<defense_method>.csv`. The columns are `contributor`, `attack_method`, `defense_method`, `model`, `task_id`, `strategy`, `condition`, `metric_name`, `metric_value`, `n_samples`, `run_id` and `notes`. AUROC inputs require one finite score per trajectory and `n_samples=1`. Keep distinct experiments and scopes under distinct metric names. Update the selected CSV, metric, expected count and provenance for the affected cell in `auroc_sources.csv`, then rebuild. `convert_batch_run.py` preserves the original worked example.

## Coverage and remaining work

| Attack | Selected evidence | Open items |
|---|---|---|
| AgentHarm | Qwen text monitors on 174 complete r0 pairs. The historical Qwen probe uses 25 separate pairs. | The r1 export contains three complete pairs and stays separate from r0. |
| Semantic Intent Fragmentation | Qwen text monitors use 10 pairs with final synthesis. The Qwen probe uses 10 complete pairs from the original SIF r0 and layer-28 activations from replay of saved responses. | The final synthesis step has no saved activations. The original SIF CoT action-only score and the historical GPT‑OSS‑20B probe are retained in `raw/`. |
| DeCompBench | All four defenses score 45 complete pairs of regenerated Qwen trajectories (`qwen_v2`). | Phi‑3.5 checkpoint verdicts and human validation apply to the earlier recovered corpus only. |
| NARCBench | All four defenses score the same 50 complete pairs of fresh Qwen trajectories (`qwen_native`). | None for the matrix. The GPT‑OSS‑120B text results and the earlier Qwen probe stay in `raw/`. |

All 16 cells now use Qwen3‑32B‑AWQ as the target model, and 15 come from new runs. The AgentHarm probe cell is historical but also uses Qwen activations. Sentinel scores on DeCompBench and NARCBench come from the message-only adapter.

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
- The probe also reaches 1.00 using public turns only. It may detect the collusion instruction in the prompt rather than collusive behaviour.
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
- Sentinel's path matcher fires on 40 of 45 controls.

The [earlier DeCompBench replay and judge documentation](../Attack_DecompBench/README.md#qwen-replay-results)
covers the recovered corpus behind the replaced scores.

