# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

CompJailBench is a benchmark suite for **distributed jailbreaks in multi-agent LLM
systems** — attacks where no single message is adversarial and the violation exists
only in how agent steps compose. Each top-level folder is a self-contained module
implementing one attack or one defense from a specific paper, plus `experiments/`,
which wires an attack against a defense on real models.

| Folder | Paper | Role |
|---|---|---|
| `Attack_DecompBench/` | DecompBench | attack (pre-existing) |
| `Defence_NARCBench Probing/` | NARCBench | defense probe (pre-existing) |
| `Attack_SemanticIntentFragmentation/` | Ahad et al. 2026, SIF | attack |
| `Defense_SentinelAgents/` | He et al. 2025, SentinelAgent | defense |
| `experiments/` | — | runs SIF vs SentinelAgent live on AWS Bedrock |

The four benchmark folders **do not import each other**. Each runs standalone. The
only place two of them are composed is `experiments/`.

## The seam pattern (the one architectural idea to understand first)

Every model-shaped operation in every folder is an **injected synchronous callable
that defaults to a deterministic stub**. The canonical shape, identical across all
folders, is:

```python
call_llm(system_prompt: str, user_prompt: str) -> str
```

with specialized variants `Classifier = Callable[[str], float]`, `Judge`,
`NodeChecker`, `EdgeChecker`. `Attack_DecompBench/pipeline.py:stub_llm`,
`Attack_SemanticIntentFragmentation/orchestrator.py:stub_planner`, and the defense
checkers all follow it. This is why **every folder's `smoke_test.py` runs fully
offline with no API key** — the stubs stand in for models.

Consequence that is easy to get wrong: **under stubs, the attack always succeeds and
the defense always catches it** (SIF-ASR 1.000, detection recall 1.000). Those are
properties of the fixtures, not findings. A green smoke test proves the plumbing, not
a result. Real numbers require swapping real models into the seams — which is the
entire job of `experiments/bedrock_bridge.py`.

## Running things

Use the project venv's interpreter explicitly (the machine default is the Windows
Store `python` alias). PowerShell has no `&&` — chain with `;` or one command per line.

```powershell
# offline regression — no AWS, no key, ~seconds each
.\.venv\Scripts\python.exe Attack_SemanticIntentFragmentation\smoke_test.py
.\.venv\Scripts\python.exe Defense_SentinelAgents\smoke_test.py

# zero-spend integration check — drives the real combined task against a role-aware
# fake model, exercising the anyio bridge, JSON plan extractor, judge parsers, and
# the solver->scorer store handoff. Expect "WIRING OK".
.\.venv\Scripts\python.exe experiments\wiring_check.py

# a folder's standalone Inspect task, under a mock model (no spend)
.\.venv\Scripts\inspect.exe eval Defense_SentinelAgents\inspect_task.py --model mockllm/model
```

Each `smoke_test.py` is the single runnable check for its folder — there is no pytest
suite. Named sections inside it (e.g. SIF `smoke_test.py` section "6b") are targeted
regression guards; read them before changing the code they cover.

### Live runs (spend money — Bedrock, us-east-1)

Full setup is in `experiments/aws_setup.md` (credentials, IAM, guardrails, model IDs,
the pinned-configuration table the paper needs). The run and its analysis:

```powershell
.\.venv\Scripts\inspect.exe eval experiments/sif_vs_sentinel.py `
  --model bedrock/<orchestrator-id> --temperature 0 `
  --model-role worker=bedrock/<id> --model-role judge_rubric=bedrock/<id> `
  --model-role judge_civ=bedrock/<id> --model-role judge_drb=bedrock/<id> `
  --model-role sentinel_node=bedrock/<id> --model-role sentinel_edge=bedrock/<id> `
  -T guardrail_strict=<id> -T guardrail_permissive=<id> --max-samples 8

.\.venv\Scripts\python.exe experiments\analyze_run.py   # newest log in logs/
```

`analyze_run.py` regenerates every table in `findings_report.html` from a `.eval` log.
Eval logs land in `./logs/` (gitignored). Resolve real model IDs for the account with
`python experiments/resolve_models.py --check`.

Command gotchas that fail silently:
- **`--temperature 0`, not `-M temperature=0`.** `-M` passes provider args, not
  generation config; the latter is ignored and the orchestrator samples. Role models
  are pinned in code (`bedrock_bridge.ROLE_CONFIG`); the orchestrator arrives via
  `--model` and must be pinned on the CLI.
- Model IDs have **no single naming pattern** — some carry a date+version suffix, some
  don't; the geo prefix (`us.`/`eu.`/`global.`) varies per model, and several models
  have no inference profile at all. Copy each ID verbatim from its Bedrock model card;
  never infer one from another's shape.

## How `experiments/` composes the two folders

`sif_vs_sentinel.py` is one Inspect `@task` running attack then defense over a
**single shared trajectory**. The contract: the solver writes `state.store["trajectory"]`
(and `plan`, `sif_result`, `alert`), and the scorers read it; `attack_evaded_defense`
is the research question (attack cleared all five gate conditions AND no defense tier
fired). Three integration facts that are invisible from any single file:

1. **Sync/async bridge.** The seams are synchronous; Inspect is async. All model work
   in the solver runs inside one `anyio.to_thread.run_sync(...)`, and
   `bedrock_bridge.model_call_llm` re-enters the loop with `anyio.from_thread.run`.
   Roles **must** be resolved with `resolve_roles()` on the event loop *before* the
   thread hop — `get_model(role=...)` reads a ContextVar that the worker thread does
   not share, so calling it inside the thread silently falls back to a default model
   with nothing in the log.
2. **Module name collision.** Every folder has an `inspect_task.py` and a
   `smoke_test.py`. `experiments/` puts both folders on `sys.path` and loads the two
   colliding `inspect_task` modules by explicit path under distinct names; every other
   module name is unique.
3. **Guardrails as the gating battery.** `guardrail_battery(strict_id, permissive_id)`
   turns two Bedrock guardrails into the F1/F2 classifier pair. `guardrail_classifier`
   fails **loud** on API error (`on_error=None` raises) — scoring an unreachable
   classifier as 0.0 would push `FS(P)` to 1.0 and manufacture attack success.

## Reproducibility invariants that break silently

These are correct in the current code and easy to undo without any test failing:

- **Policy clauses in `scenarios.json` are verbatim NIST SP 800-53 text.** Condition
  (d) requires the judge's citation to be a real substring of the clause, so rewriting
  or paraphrasing a clause silently weakens the gate.
- **The two guardrails must be independently configured** (different filter strength).
  `FS(P)` is a max over the battery; one guardrail used twice is one classifier
  reported as two.
- **No judge may share a model family with the orchestrator, and judge_civ must differ
  from judge_rubric.** Self-blindness (SIF found 0/14 orchestrators flag their own
  plans) and conjunction collapse, respectively.
- **`assert_composition_only` gates the SIF-ASR numerator.** A sample whose single
  subtask taints in isolation is not a compositional attack and is excluded.
- **The attack-path library must not be derived from the sample under test.**
  `build_library` refuses this; deriving it gives recall 1.0 and FPR 1.0 trivially.
- **Judge/checker temperature is pinned to 0** in `ROLE_CONFIG`. Verdicts otherwise
  sample, moving conditions (c)-(e) and therefore SIF-ASR between identical runs.
- **The generate cache is off** (`model_call_llm(cache=False)`) on purpose: Inspect's
  cache is keyed on model+prompt and will replay a response recorded before a prompt
  change, with nothing in the log saying so.

## Extending the seams

To add a real backend, write a new `call_llm`/`Classifier`/`Judge` and inject it — do
not make the seams async (that change crosses both folders and breaks both offline
smoke tests) and do not edit the benchmark folders from `experiments/`. The
trajectory record is extended backward-compatibly: older 4-key records
(`agent_name`, `system_prompt`, `user_prompt`, `response`, consumed by
`Defence_NARCBench Probing/`) and newer records carrying optional `tool_calls`/
`recipient` both build a graph.
