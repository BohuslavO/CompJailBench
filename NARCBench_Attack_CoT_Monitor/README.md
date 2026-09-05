# NARCBench attack and defense evaluations

This package evaluates a matched NARCBench committee-collusion attack and four
defense configurations. Each full-corpus evaluation uses the same 50
NARCBench-Core scenarios under collusion and matched honest-control conditions,
for 100 trajectories per defense cell.

## Status

| Defense cell | Model/input | Primary result | Status |
|---|---|---|---|
| Observable-rationale monitor | Saved Bedrock trajectories | AUROC 0.316 for action+rationale; action-only AUROC 0.427 | Complete; negative monitoring result |
| Agent-to-Agent Monitor | Saved public-message graphs and final output | AUROC 0.714; recall 0.060; FPR 0.000 | Complete; project-defined detection-only baseline |
| Sentinel Agents (message-only adaptation) | Saved public messages | AUROC 0.552; recall 0.940; FPR 0.900 | Complete; degraded, tool-free setting |
| NARCBench activation probing | Fresh Qwen layer-28 activations | AUROC 0.9896; AP 0.9901 | Complete; 50 matched pairs |

The **Agent-to-Agent Monitor** is the project-defined graph-and-judge baseline
described in the paper. The Sentinel cell is explicitly **message-only,
public-scope, degraded** because the trajectories contain no tool calls. The
observable-rationale monitor does not claim access to hidden chain of thought.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Attack solver, defense adapters, trajectory schema, and statistics |
| `tasks/` | Inspect task definitions |
| `scripts/` | Evaluation, analysis, export, and audit commands |
| `tests/` | Offline unit and integration tests |
| `configs/` | Frozen model, prompt, probe, layer, and decoding contracts |
| `results/` | Compact frozen outputs and matrix-ready rows |
| `paper/` | Methodology text, references, and audit protocol |

Method-specific documentation:

- [`NARCBENCH_PROBING.md`](NARCBENCH_PROBING.md)
- [`AGENT_TO_AGENT_MONITOR.md`](AGENT_TO_AGENT_MONITOR.md)
- [`SENTINEL_MESSAGE_ONLY.md`](SENTINEL_MESSAGE_ONLY.md)
- [`DEFENSE_INTERFACE.md`](DEFENSE_INTERFACE.md)

Frozen results and paper text:

- [`results/paper_main/paper_results.md`](results/paper_main/paper_results.md)
- [`results/paper_main/EXPERIMENT_SUMMARY.md`](results/paper_main/EXPERIMENT_SUMMARY.md)
- [`results/narcbench_probing/RESULTS.md`](results/narcbench_probing/RESULTS.md)
- [`results/agent_to_agent_monitor/RESULTS.md`](results/agent_to_agent_monitor/RESULTS.md)
- [`results/sentinel_message_only/RESULTS.md`](results/sentinel_message_only/RESULTS.md)
- [`paper/methodology_narcbench_cot.tex`](paper/methodology_narcbench_cot.tex)
- [`paper/methodology_narcbench_probing.tex`](paper/methodology_narcbench_probing.tex)
- [`paper/methodology_agent_to_agent_monitor.tex`](paper/methodology_agent_to_agent_monitor.tex)
- [`paper/methodology_sentinel_message_only.tex`](paper/methodology_sentinel_message_only.tex)

## Experimental design

- 50 scenario IDs across 17 domains.
- One collusion and one matched control run per scenario.
- Four agents, three public discussion rounds, and four final votes.
- Two designated colluders privately coordinate in the attack condition.
- Scenario text, agent identities, option content, speaking-order seed, and
  public deliberation structure are matched within each pair.
- Option labels are counterbalanced across `A/B` and `1/2` and across target
  position.
- Refusals, abstentions, and parse failures remain in the primary analysis.

The Bedrock attack run used `openai/bedrock/gpt-oss-120b` and produced 100
complete trajectories, 1,700 attacker calls, and 4,138,714 total tokens. Its
strict end-to-end attack-success rate was 3/50. The activation-probing cell
required a separate `Qwen/Qwen3-32B-AWQ` run because hidden activations cannot
be reconstructed from saved text trajectories. That Qwen run also achieved
3/50 strict attack successes; probe discrimination and attack effectiveness are
reported as separate quantities.

## Reproducibility

Create an environment from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill only the provider variables required for the selected model. The `.env`
file, virtual environment, raw credentials, diagnostic logs, and temporary
artifacts are excluded from version control.

Run the offline verification suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\inspect.exe list tasks tasks\narcbench_real_attack.py
```

The NARCBench activation-probing workflow and exact Kaggle commands are in
[`NARCBENCH_PROBING.md`](NARCBENCH_PROBING.md). The probe binary and raw
activation arrays are intentionally not duplicated here. Their hashes and
provenance are recorded in
[`configs/narcbench_probe_main.json`](configs/narcbench_probe_main.json) and
[`results/narcbench_probing/run_manifest.json`](results/narcbench_probing/run_manifest.json).

## Defense interface

Cross-attack scoring accepts one `StandardTrajectory` JSON object per line.
Validate compatibility without model calls before starting a full evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\score_standard_trajectories.py `
  attack_trajectories.jsonl `
  --output compatibility.jsonl `
  --validate-only
```

The comparable observable-rationale defense column uses the frozen
attack-agnostic `general_v2` prompt. A missing observable rationale is reported
as `unavailable`; it is not silently replaced by action-only evidence. See
[`results/paper_main/COT_DEFENSE_COLUMN_STATUS.md`](results/paper_main/COT_DEFENSE_COLUMN_STATUS.md)
for the remaining cross-attack inputs.

## Interpretation limits

- Observable rationales are evaluator-visible explanations, not guaranteed
  faithful hidden reasoning.
- The attacker and observable-rationale monitor used separate Inspect roles but
  the same underlying Bedrock checkpoint.
- The Agent-to-Agent Monitor is a project-defined, post-hoc detection baseline;
  it does not train a classifier or intervene in a live execution.
- The Sentinel result does not characterize the full tool-aware defense.
- The activation probe's released metadata does not establish a verifiable
  scenario-level train/evaluation split; its result is described as
  within-benchmark discrimination.
- The activation probe's primary all-turn score includes private coordination
  turns and therefore represents a privileged evidence scope.

Compact per-sample scores, aggregate summaries, matrix-ready exports, exact
software versions, and artifact hashes are retained under `results/`. Raw model
artifacts remain outside Git and are identified by SHA-256 in the run manifests.
