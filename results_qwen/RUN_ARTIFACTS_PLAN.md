# Publishing the run artifacts (preparation only)

Nothing in this file has been run. The two new runs (`decompbench_qwen_v2`, `narcbench_qwen`) wrote their raw outputs to the Modal volume `compjailbench-agentharm-scores`. Only the per-sample summaries are in the repo. This file lists what to download, where it goes, and which out-of-repo paths `run_modal.py` depends on.

## 1. List the volume first

The paths below are read from the `run_modal.py` code, not from the volume itself. Run these and compare before downloading:

```sh
modal volume ls compjailbench-agentharm-scores /decompbench_qwen_v2
modal volume ls compjailbench-agentharm-scores /decompbench_qwen_v2/activations | head
modal volume ls compjailbench-agentharm-scores /narcbench_qwen_native_v1
modal volume ls compjailbench-agentharm-scores /narcbench_qwen_native_v1/inputs
modal volume ls compjailbench-agentharm-scores /narcbench_qwen_native_v1/gen
```

The NARCBench activations are written by the released runner (`run_narcbench_probe_kaggle.py --output .../gen`), so their file names under `gen/` are not visible in `run_modal.py`. The `ls` of `gen/` will show them.

## 2. Files to download and where they go

The two runs write different files. `decompbench_qwen_v2` writes one `results.jsonl` that holds the turns, checkpoint verdicts and monitor rows together. It has no separate `scores.jsonl`, `inputs/` or `attack_outcomes.jsonl`. `narcbench_qwen` writes no `generation_calls.jsonl`; generation is logged by the runner under `gen/`.

### DeCompBench → `results_qwen/decompbench_qwen_v2/run/`

| Volume path | Repo path | In git |
|---|---|---|
| `/decompbench_qwen_v2/results.jsonl` | `run/results.jsonl` | yes, see section 4 |
| `/decompbench_qwen_v2/monitor_calls.jsonl` | `run/monitor_calls.jsonl` | yes |
| `/decompbench_qwen_v2/generation_calls.jsonl` | `run/generation_calls.jsonl` | yes |
| `/decompbench_qwen_v2/manifest.json` | `run/manifest.json` | yes (compare with the existing `manifest.json`) |
| `/decompbench_qwen_v2/pilot.json` | `run/pilot.json` | yes |
| `/decompbench_qwen_v2/complete.json` | `run/complete.json` | yes |
| `/decompbench_qwen_v2/activations/*.npy` (90 files) | outside the repo | no; record SHA-256 in `run/ACTIVATIONS_SHA256.txt` |

### NARCBench → `results_qwen/narcbench_qwen/run/`

| Volume path | Repo path | In git |
|---|---|---|
| `/narcbench_qwen_native_v1/inputs/standard_trajectories.jsonl` | `run/inputs/standard_trajectories.jsonl` | yes; SHA-256 must equal `9d54c8354ebdf29912e901e11d8c2ffa8810790102b7a552c9dcdb2b8bd85c81` (`manifest.json` → `inputs.input_sha256`) |
| `/narcbench_qwen_native_v1/inputs_manifest.json` | `run/inputs_manifest.json` | yes |
| `/narcbench_qwen_native_v1/scores.jsonl` | `run/scores.jsonl` | yes |
| `/narcbench_qwen_native_v1/monitor_calls.jsonl` | `run/monitor_calls.jsonl` | yes |
| `/narcbench_qwen_native_v1/attack_outcomes.jsonl` | `run/attack_outcomes.jsonl` | yes |
| `/narcbench_qwen_native_v1/manifest.json` | `run/manifest.json` | yes |
| `/narcbench_qwen_native_v1/pilot_generation.json`, `generation_exceptions.json`, `generation_complete.json`, `complete.json` | `run/` | yes |
| `/narcbench_qwen_native_v1/gen/probe_scores.jsonl` | `run/gen/probe_scores.jsonl` | yes |
| `/narcbench_qwen_native_v1/gen/inspect_logs/**/*.eval` | `run/gen/inspect_logs/` | decide after checking total size; Git LFS or a release asset if large |
| `/narcbench_qwen_native_v1/gen/completed_pairs/*.json` | `run/gen/completed_pairs/` | yes if small |
| activation files under `/narcbench_qwen_native_v1/gen/` | outside the repo | no; record SHA-256 in `run/ACTIVATIONS_SHA256.txt` |

## 3. Commands

Download outside the repo first, check, then copy in. `modal volume get` copies a file or a directory.

```sh
STAGE=~/cjb_volume_stage
mkdir -p $STAGE/decomp $STAGE/narc

# DeCompBench (whole run directory, including activations)
modal volume get compjailbench-agentharm-scores /decompbench_qwen_v2 $STAGE/decomp

# NARCBench (whole run directory, including gen/ and activations)
modal volume get compjailbench-agentharm-scores /narcbench_qwen_native_v1 $STAGE/narc

# Integrity checks
sha256sum $STAGE/narc/narcbench_qwen_native_v1/inputs/standard_trajectories.jsonl   # expect 9d54c835...
wc -l $STAGE/decomp/decompbench_qwen_v2/results.jsonl           # expect 90
wc -l $STAGE/decomp/decompbench_qwen_v2/monitor_calls.jsonl     # expect 1577 (summary.json)
wc -l $STAGE/narc/narcbench_qwen_native_v1/monitor_calls.jsonl  # expect 3540 (summary.json)
du -sh $STAGE/*/*/activations $STAGE/narc/narcbench_qwen_native_v1/gen

# Activation hashes (kept out of git)
( cd $STAGE/decomp/decompbench_qwen_v2 && find activations -type f -print0 | sort -z | xargs -0 sha256sum ) \
  > results_qwen/decompbench_qwen_v2/run/ACTIVATIONS_SHA256.txt
```

The exact directory layout `modal volume get` creates inside `$STAGE` can differ between Modal client versions; adjust the paths after the first download. Then copy the files listed in section 2 into `run/`, and add `results_qwen/**/activations/` and `*.npy` to `.gitignore`.

## 4. Check before committing trajectories

The DeCompBench task files come from a gated dataset whose terms say not to redistribute it (`Attack_DecompBench/DecompBench x NARCBench Dataset Files/LICENSE_AND_ACCESS.md`). `results.jsonl` and `monitor_calls.jsonl` contain the attack prompts verbatim (`turns[].user_prompt`) and Qwen's responses to them. Publishing them in a public repo may count as redistribution. Options: keep them in a private repo or gated release, publish them with prompts replaced by hashes, or get the dataset owners' agreement. NARCBench trajectories do not have this issue as far as this repo shows, but check the NARCBench licence too.

## 5. Out-of-repo paths in `run_modal.py`

`ROOT` is the folder of each `run_modal.py` on your machine, so `ROOT.parents[1]` and `ROOT.parent` point outside this repo.

| Used by | Out-of-repo path | What it is | Proposed in-repo equivalent |
|---|---|---|---|
| both | `ROOT.parents[1]/sources/CompJailBench/NARCBench_Attack_CoT_Monitor/{src,scripts}` (NARCBench also `data`, `configs`) | a separate local checkout of this repo | `REPO = Path(__file__).resolve().parents[2]`; `REPO/'NARCBench_Attack_CoT_Monitor'/...`. Record the commit hash of the checkout used in the run. |
| both | `ROOT.parents[1]/sources/CompJailBench/Defence_NARCBench Probing` | probe bundle and score pipeline | `REPO/'Defence_NARCBench Probing'` |
| both | `ROOT.parent/decompbench_parallel/qwen_remaining_scoring/score_qwen_pilot_monitors.py` | monitor scorer; overwrites the in-repo copy inside the image | `REPO/'NARCBench_Attack_CoT_Monitor/scripts/score_qwen_pilot_monitors.py'` only if identical. The in-repo file (commit 6baac3c) has SHA-256 `a0e6ef48402dafafa80adb7ec0f70b2ffc5ceeabac0d66345bf9198afcba307b`. If yours differs, commit yours as e.g. `results_qwen/shared/score_qwen_pilot_monitors.py`. |
| both | `modal.Image.from_id('im-VOAEtmK389l59IYjXjnrji')` | prebuilt image; also the only source of `/opt/CompJailBench/Defense_SentinelAgents` (not added with `add_local_dir`) | an explicit `modal.Image.debian_slim().pip_install(...)` with pinned versions, plus `add_local_dir(REPO/'Defense_SentinelAgents', ...)`. Get the package list with `modal shell --image im-VOAEtmK389l59IYjXjnrji -c "pip freeze"` (or from the code that built the image). |
| both | volume `compjailbench-qwen-cache` | model weights | none needed; the model and revision `0499c3a` are recorded in the manifests |
| DeCompBench | `ROOT.parent/imports/decompbench_previous/dataset/{task_loader,routing,benign_rewriter,checkpoint_scorer}.py` | DeCompBench code | `REPO/'Attack_DecompBench'/<name>.py`, but the repo has two different copies of `task_loader.py`, `routing.py` and `checkpoint_scorer.py` (top level and `DecompBench x NARCBench Dataset Files/`). Hash yours and use the matching one, or commit yours under `results_qwen/decompbench_qwen_v2/src/`. |
| DeCompBench | `ROOT.parent/imports/decompbench_previous/dataset/tasks/tasks/<slug>/` for the 15 scenarios | gated task data | do not commit. Add a loader that reads them from a path given by an env var (e.g. `DECOMPBENCH_TASKS`), and commit a `tasks_sha256.txt` listing the hash of every file used. |
| DeCompBench | `ROOT.parents[1]/sources/CompJailBench/Attack_DecompBench/results_qwen/trajectory_index.jsonl` | paper trajectory index | `REPO/'Attack_DecompBench/results_qwen/trajectory_index.jsonl'` (in repo, SHA-256 `dfabb22a...`) |

Hash your local copies so we can decide which in-repo file matches:

```sh
sha256sum <your path>/decompbench_parallel/qwen_remaining_scoring/score_qwen_pilot_monitors.py \
          <your path>/imports/decompbench_previous/dataset/{task_loader,routing,benign_rewriter,checkpoint_scorer}.py
( cd <your path>/sources/CompJailBench && git rev-parse HEAD && git status --short )
```

`run_modal.py` is unchanged until you confirm the mapping.
