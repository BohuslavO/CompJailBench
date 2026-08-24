# AgentHarm 4x4 handoff

This runner generates paired harmful/benign trajectories once and then sends
the same saved trajectories to the frozen cross-attack defense adapters. It
does not alter any completed result under `results 4x4/raw/`.

## Experimental definition

This is an **AgentHarm behavior-prompt multi-agent attack**: it loads the
official paired AgentHarm prompts, routes each prompt through the custom
Planner -> Workers -> Aggregator pipeline, and evaluates the resulting
communication trace. It does not execute AgentHarm's official tool sandbox or
programmatic task graders. Keep that qualification in the paper and in every
matrix row's `notes` field.

The NARCBench activation-probing cell is separate. Azure/Bedrock generations
do not expose compatible hidden activations; that cell needs a self-hosted
Qwen3-32B-AWQ generation run using the same AgentHarm pair IDs.

## 1. Install

From the repository root in CMD:

```bat
set PYTHONUTF8=1
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Configure Azure locally

Create a repository-root `.env` (already ignored by Git):

```dotenv
AZUREAI_OPENAI_API_KEY=YOUR_SECRET
AZUREAI_OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/
AZUREAI_OPENAI_API_VERSION=YOUR_API_VERSION
COMPJAILBENCH_AGENT_MODEL=openai/azure/YOUR_DEPLOYMENT_NAME
COMPJAILBENCH_JUDGE_MODEL=openai/azure/YOUR_DEPLOYMENT_NAME
```

For the tested Bedrock GPT-OSS-120B path, use:

```dotenv
COMPJAILBENCH_AGENT_MODEL=openai/bedrock/gpt-oss-120b
COMPJAILBENCH_JUDGE_MODEL=openai/bedrock/gpt-oss-120b
COMPJAILBENCH_REASONING_EFFORT=low
```

Never commit or paste the key into chat/Slack.

For the later frozen-defense scoring scripts, place the same ignored `.env`
inside `NARCBench_Attack_CoT_Monitor\` as well; those scripts deliberately load
their own sub-project environment. Keep both files local.

## 3. Free checks

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\validate_agentharm_setup.py --split val --limit 2
.venv\Scripts\python.exe eval.py --dry-run --condition harmful --split val --limit 1 --run-id agentharm_smoke
```

The validator must say `official AgentHarm`, show equal harmful/benign counts,
and say that ordered pair IDs match. Any placeholder or mismatch stops the run.

## 4. Paid 1+1 Azure smoke generation

Use the same run ID for both paired arms:

```bat
.venv\Scripts\python.exe eval.py --defenses none --condition harmful --split val --limit 1 --run-id agentharm_smoke
.venv\Scripts\python.exe eval.py --defenses none --condition benign --split val --limit 1 --run-id agentharm_smoke
.venv\Scripts\python.exe scripts\export_agentharm_trajectories.py --trace-root results\traces\agentharm_smoke --output results\runs\agentharm_smoke\standard_trajectories.jsonl
```

Inspect both generated traces before continuing. `none` is intentional: it
captures each attack trajectory only once without spending extra judge calls.

## 5. Free defense compatibility checks

From `NARCBench_Attack_CoT_Monitor`:

```bat
..\.venv\Scripts\python.exe scripts\score_standard_trajectories.py ..\results\runs\agentharm_smoke\standard_trajectories.jsonl --output ..\results\runs\agentharm_smoke\cot_compatibility.jsonl --validate-only
..\.venv\Scripts\python.exe scripts\score_g_safeguard_inspired.py ..\results\runs\agentharm_smoke\standard_trajectories.jsonl --output ..\results\runs\agentharm_smoke\g_compatibility.jsonl --validate-only
..\.venv\Scripts\python.exe scripts\validate_sentinel_message_adapter.py ..\results\runs\agentharm_smoke\standard_trajectories.jsonl --sentinel-dir ..\Defense_SentinelAgents --output ..\results\runs\agentharm_smoke\sentinel_compatibility.jsonl
```

Only after these pass should the three monitor-model smoke scores be run.

## 6. Full run

Repeat section 4 with a dated full-run ID, `--split test_public`, and explicit
`--limit 176` on both generation commands. The default remains 10 as a cost
safety guard. Official Inspect Evals 0.18.0 currently provides 176 harmful and
176 exactly paired benign public samples. Record the generated manifest and
package versions.

Before the full generation, confirm the complete pairing for free:

```bat
.venv\Scripts\python.exe scripts\validate_agentharm_setup.py --split test_public --limit 176
```

Score the one exported JSONL with the frozen scripts:

- `score_g_safeguard_inspired.py`
- `score_sentinel_message_only.py` with `--scope public`
- `score_standard_trajectories.py` with the cross-attack `general_v2` profile

Run one harmful and one benign monitor score first, then use each scorer's
resume/checkpoint option for the full set. Their analysis scripts create the
canonical long-form rows. Copy only reviewed rows into `results 4x4/raw/`, run
`results 4x4/build_matrix.py`, and commit the four new AgentHarm cells plus the
regenerated matrix.

## Recovery and ownership

All work is isolated on branch `codex/agentharm-runner-fix`. Atharv can fetch
and continue the branch at any point. No merge into `main` is required to
inspect or run it, and deleting the branch would restore the repository to the
pre-change state.
