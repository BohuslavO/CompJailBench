# NARCBench attack × observable-reasoning monitor

This isolated project implements Beibarys's attack/defense contribution without
modifying the team's `CompJailBreak` repository. It uses all 50 official
NARCBench-Core scenarios under matched collusion and honest-control conditions
(100 runs), then evaluates a separate LLM monitor under three predeclared
evidence scopes.

The project is code-complete, mock-tested, and has a completed paper-scale run
over all 50 matched scenarios (100 trajectories). The frozen analysis outputs
are in `results/paper_main`; the attacker and monitor used separate Inspect
roles backed by the same Bedrock-hosted checkpoint, which is reported as a
limitation rather than presented as cross-model validation.

A preliminary semantic-coordination audit is also complete for a fixed 10/50
subset. One human reviewer supplied the labels; AI-assisted practice files were
excluded, and no inter-rater statistic is claimed.

## Scientific scope

The [official NARCBench paper](https://arxiv.org/abs/2604.01151) studies
activation probes on open-weight models. This project uses its committee
collusion scenarios as an **attack surface**, but the defense is a text-level
LLM monitor over public actions and explicitly elicited rationales. Therefore,
this is a NARCBench-derived attack/defense experiment, not a reproduction of the
paper's activation-probing defense.

The phrase “CoT monitor” is used only as team shorthand. The implementation
does not claim access to hidden chain of thought. It records concise
model-generated **observable rationales** intended for an evaluator.

## Experiment design

- 50 scenario IDs across 17 domains.
- One collusion and one matched control run per scenario.
- Four agents, three public discussion rounds, and four final votes.
- Two designated colluders privately coordinate in the attack condition.
- Agent identities, scenario, option content, speaking-order seed, and final
  user prompt are matched within each pair.
- Option labels are counterbalanced across `A/B` and `1/2` and across position.
- Main analysis includes all runs. Refusals, abstentions, and parse failures are
  reported instead of silently discarded.

Defense conditions:

1. Public action only (ablation).
2. Observable rationale only (ablation).
3. Public action plus observable rationale (predeclared primary defense).
4. Private-message audit (optional upper bound; never described as deployable).

## What has already been validated

- The official 50 scenarios convert to exactly 50 complete matched pairs.
- Dataset labels and agent identities match within every pair.
- Dynamic `A/B/1/2` vote extraction works without inventing missing votes.
- Deployable monitor views exclude private messages, private rationales,
  condition labels, hidden targets, and attack metadata.
- Inspect discovers both tasks and completes a two-sample mock-model run.
- The result analyzer produces JSONL, CSV, and a paper-ready Markdown table.
- A full real run completed all 1,700 attacker calls and 300 monitor calls with
  no invalid monitor judgments in the final log.
- The portable trajectory exporter produced 100 valid `StandardTrajectory`
  JSONL records, and all 300 action/reasoning/combined compatibility checks
  passed.
- The completed cell exports to 600 valid rows in the team's long-form matrix
  schema and loads in the team matrix builder.

## Completed paper-scale result

The final run used `openai/bedrock/gpt-oss-120b` with low reasoning effort. It
produced 50 complete matched pairs and 4,138,714 total tokens, including
1,118,159 monitor-role tokens. The primary combined public-action plus
observable-reasoning monitor achieved AUROC 0.316; the action-only baseline
achieved AUROC 0.427. These results do not support a monitoring benefit from
the elicited rationales under this model and protocol. See
`results/paper_main/paper_results.md` and `results/paper_main/MEETING_SUMMARY.md`.

## Applying the defense to the other attacks

The cross-attack contract is documented in `DEFENSE_INTERFACE.md`. First ask an
attack owner for `StandardTrajectory` JSONL and perform the free compatibility
check:

```powershell
.\.venv\Scripts\python.exe scripts\score_standard_trajectories.py `
  teammate_trajectories.jsonl `
  --output compatibility.jsonl `
  --validate-only
```

Only after one attack and one benign trajectory are compatible should a real
monitor model be authorized. Missing observable reasoning is reported as
`unavailable`, not converted into a benign score.

The completed NARCBench result used the narrower `narcbench_v1` prompt. The
final comparable CoT-defense column is frozen to the attack-agnostic
`general_v2` prompt, so the saved NARCBench trajectories must be rescored once
with that prompt. This requires monitor calls only, not the 1,700 attacker calls.
See `results/paper_main/COT_DEFENSE_COLUMN_STATUS.md`.

## Local setup

From PowerShell in this folder:

```powershell
& "C:\path\to\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill only the credential variables for the selected provider in `.env`. The
real `.env` is ignored by Git. Do not send credential values to teammates or to
this chat.

For an Amazon Bedrock bearer key used with an `openai/bedrock/*` model, the
only required entries are `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION`. Standard
AWS access-key variables are an alternative credential method, not additional
requirements.

On this Windows installation, redirect Inspect's temporary application data to
the project before an evaluation so its trace, control, and sample-buffer files
remain writable:

```powershell
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = "$PWD\tmp\inspect-appdata"
```

Run the free checks at any time:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\inspect.exe list tasks tasks\narcbench_real_attack.py
```

## Exact workflow for the real experiment

### 1. Confirm the two model identifiers

The attacker and monitor should be separate roles, even if the team ultimately
uses the same underlying provider. For later compatibility with NARCBench
activation probing, the attacker trajectory must record the exact model ID; an
activation-probing teammate also needs access to that model's hidden states.

Do not put API keys in source code or Slack. Configure the provider using its
normal environment variables or an ignored `.env` file.

### 2. Run one real matched-pair smoke test

Replace the two placeholders:

```powershell
$env:INSPECT_TRACE_FILE = "$PWD\tmp\inspect-real-trace.log"
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = "$PWD\tmp\inspect-appdata"
.\.venv\Scripts\inspect.exe eval `
  "tasks\narcbench_real_attack.py@narcbench_cot_experiment" `
  --model "ATTACKER_PROVIDER/MODEL" `
  --model-role 'monitor={model: MONITOR_PROVIDER/MODEL, temperature: 0, max_tokens: 1536}' `
  --temperature 0.7 `
  --max-tokens 768 `
  -T "pair_ids=env_02" `
  --log-dir "logs\real_smoke" `
  --display plain
```

This pair requires 34 attacker calls and 6 monitor calls. Check that both
samples finish, votes are not all `ABSTAIN`, parse failures are low, and the
monitor returns numeric suspiciousness values.

### 3. Run the full paper experiment

Do not use `--limit`, because a raw sample limit can break matched pairs.

```powershell
$env:INSPECT_TRACE_FILE = "$PWD\tmp\inspect-paper-trace.log"
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = "$PWD\tmp\inspect-appdata"
.\.venv\Scripts\inspect.exe eval `
  "tasks\narcbench_real_attack.py@narcbench_cot_experiment" `
  --model "ATTACKER_PROVIDER/MODEL" `
  --model-role 'monitor={model: MONITOR_PROVIDER/MODEL, temperature: 0, max_tokens: 1536}' `
  --temperature 0.7 `
  --max-tokens 768 `
  --max-connections 10 `
  --log-dir "logs\paper_main" `
  --display plain
```

The full run makes 1,700 attacker calls and 300 monitor calls. Based on the old
single-pair pilot, budget for several million tokens. Actual time and cost
depend on provider rate limits and model pricing.

### 4. Analyze the completed log

```powershell
$log = Get-ChildItem "logs\paper_main\*.eval" |
  Sort-Object LastWriteTime |
  Select-Object -Last 1 -ExpandProperty FullName

.\.venv\Scripts\python.exe scripts\analyze_eval.py $log `
  --output-dir results\paper_main
```

This creates:

- `samples.jsonl`: normalized per-run outcomes and monitor judgments.
- `attack_summary.csv`: valid-collusion, attack-success, control baseline,
  paired uplift, confidence intervals, and exact McNemar test.
- `monitor_summary.csv`: AUROC, AUPRC, recall, FPR, precision, balanced accuracy,
  invalid-output count, and model identity for each evidence scope.
- `paper_results.md`: a table ready to adapt into the team paper.

### 5. Conduct the semantic coordination audit

Follow `paper/manual_audit_protocol.md`. The primary intention-to-attack
analysis keeps all 50 pairs. A secondary per-protocol analysis may identify the
collusion runs in which both designated agents both supported the target and
showed semantic evidence of coordination. Never delete failed runs from the
primary end-to-end analysis.

Export a reviewer packet without monitor predictions:

```powershell
.\.venv\Scripts\python.exe scripts\export_audit_packet.py $log `
  --output-dir results\paper_main\audit_packet
```

Validate and summarize the frozen subset without model calls:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_audit.py
```

### 6. Fill—not invent—the paper placeholders

After analysis, replace only the bracketed values in
`paper/methodology_narcbench_cot.tex`. Keep the preliminary `env_02` result
separate from the full experiment.

## Realistic remaining time

Once model access works:

- One-pair smoke test and inspection: roughly 15–30 minutes.
- Full 50-pair generation and monitoring: roughly 1–4 hours on a stable hosted
  API, potentially longer under throttling.
- Automated analysis: under 5 minutes.
- Human semantic audit: about 45–90 minutes for one careful reviewer; a second
  reviewer improves credibility but adds time.
- Paper-table insertion and final sanity checks: 20–40 minutes.

The best SAT-study/rest window is during the full API run. The code requires no
manual interaction while Inspect is processing unless the provider raises an
authentication, quota, or throttling error.

## Repository boundary

The shared `StandardTrajectory` v2.0 file has been pushed to the existing
`beibarys-narcbench-cot` branch for team schema discussion. The remaining
polished source and results have not yet been pushed. Do not upload `.env`,
`.venv`, raw credentials, diagnostic logs, or temporary reference files.
