# Running SIF vs SentinelAgent on AWS Bedrock

Setup runbook for the combined experiment. Fill in the **Pinned configuration**
table at the bottom as you go — the paper needs to name exact model versions and
region, and those are not recoverable from the eval logs alone.

Nothing here needs a GPU, a SageMaker endpoint, or a running instance. Cost is
per-token plus per-text-unit for guardrail calls; there is no idle spend.

---

## 1. Budget alarm — do this first

AWS Budgets → Create budget → Cost budget → monthly cap, alerts at 50/80/100%.
Set it before enabling model access, not after.

## 2. Region

`us-east-1` or `us-west-2` carry the widest model coverage. Everything below
assumes one region; using two splits your guardrails from your models and the
`ApplyGuardrail` call will fail with a confusing not-found.

```powershell
winget install --id Amazon.AWSCLI -e     # then reopen the terminal for PATH
aws configure                            # key, secret, us-east-1, json
aws sts get-caller-identity              # must return your account ARN
```

Create the key first: **IAM → Users → Create user**, attach the policy in step 4,
then **Security credentials → Create access key → Command Line Interface**.

The CLI is a convenience, not a dependency. `boto3` is what actually runs:
`resolve_models.py` and Inspect's `bedrock/` provider both call it directly. The
CLI only earns its place for `aws configure` and the ad-hoc `list-*` commands
below. Without it, three environment variables do the same job:

```powershell
$env:AWS_ACCESS_KEY_ID     = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_DEFAULT_REGION    = "us-east-1"
```

Inspect uses standard boto3 credential resolution — env vars,
`~/.aws/credentials`, SSO, or an instance role. There is no Inspect-specific
credential variable.

## 3. Model access

**There is no longer a "Model access" page to click through.** AWS retired it in
late 2025 along with the `PutFoundationModelEntitlement` permission. Access to
all Bedrock foundation models is enabled by default in commercial regions; the
console path is now **Bedrock console → Model catalog**. (The old page survives
only in GovCloud `us-gov-west-1`, under *Bedrock configurations → Model access*.)

Three prerequisites replace the old request flow. Get them out of the way before
the first run, because the failure mode is confusing:

1. **AWS Marketplace permissions** — see step 4. On first invocation of a
   third-party model Bedrock auto-initiates the Marketplace subscription in the
   background. During that window (up to ~15 min) calls may succeed *temporarily*
   and then start returning `AccessDeniedException` when the subscription fails
   for want of permissions. An eval that dies partway through with
   AccessDenied almost always means this, not a bad model id.

2. **Anthropic models require a one-time First Time Use form.** Select any
   Anthropic model in the Model catalog and submit use case details, or call
   `PutUseCaseForModelAccess`. Access is granted immediately on submit, once per
   account — or once at an AWS Organization's management account, inherited by
   its members. The form wants a description of intended use and a website URL;
   a GitHub profile or project URL is acceptable if you have no company site.

3. **A valid payment method** on the account for Marketplace purchases.

**Do not trust `get-foundation-model-availability` as the gate.** Observed
2026-08-07: all three Anthropic entries reported `agreementAvailability:
NOT_AVAILABLE` while `authorizationStatus: AUTHORIZED` and
`entitlementAvailability: AVAILABLE` — and every one of them invoked fine. No
use-case form was ever presented, because none was required. Chasing that field
costs an afternoon looking for a console page that will not appear.

The only check that predicts whether the eval runs is a real call:

```python
boto3.client("bedrock-runtime", region_name="us-east-1").converse(
    modelId="<id>", inferenceConfig={"maxTokens": 5},
    messages=[{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}])
```

Fractions of a cent for the whole lineup, and the exception message names the
real problem when there is one. Note this *is* the EULA-acceptance mechanism —
see the closing note in this step.

You still want at least **five distinct families** to satisfy the role
constraints in step 6: two orchestrators (strong and weak) and three judges
sharing a family with neither.

Note for the paper's reproducibility section: invoking a third-party model
constitutes agreement to its EULA. If that needs review first, block
`bedrock:InvokeModel` by SCP or IAM until the terms are accepted — denying
`aws-marketplace:Subscribe` alone does **not** stop the first invocation,
because Bedrock auto-initiates the subscription.

## 4. IAM

**IAM → Users →** *your user* **→ Permissions → Add permissions ▾ → Create inline
policy → JSON**, paste, name it, create.

Not during the Create-user wizard — that step only attaches *existing* managed
policies and has no inline option. Create the user with no permissions first,
then add this from the user's own page.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "CompJailBenchBedrock",
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:ApplyGuardrail",
      "bedrock:ListFoundationModels",
      "bedrock:ListInferenceProfiles",
      "bedrock:GetFoundationModelAvailability",
      "bedrock:ListGuardrails",
      "bedrock:GetGuardrail",
      "bedrock:CreateGuardrail",
      "bedrock:CreateGuardrailVersion",
      "aws-marketplace:Subscribe",
      "aws-marketplace:Unsubscribe",
      "aws-marketplace:ViewSubscriptions"
    ],
    "Resource": "*"
  }]
}
```

In the JSON policy editor, select all and replace the skeleton — the "Add new
statement" button builds a second statement rather than filling this one in.

The four `*Guardrail*` actions are for step 7. Drop `CreateGuardrail` and
`CreateGuardrailVersion` once the two guardrails exist; the runs only need
`ApplyGuardrail`.

The `aws-marketplace` actions are only needed the **first** time a given model is
invoked in the account. Once enabled, other users in the account can invoke it
without them. If your role cannot hold Marketplace permissions, someone who can
must invoke each model once as a one-time step.

Models from Amazon, Meta, Mistral, DeepSeek, Qwen and OpenAI are not sold through
Marketplace and have no product ID, so `aws-marketplace:*` does not apply to them
— relevant if you are scoping permissions tightly and wondering why a condition
key won't match.

## 5. Find the real model IDs

**Do not copy model IDs out of any document, including this one.** The Bedrock
catalog moves, and most current models must be addressed through a cross-region
inference profile (`us.`, `eu.`, `global.` prefix) rather than the bare model id.

```bash
aws bedrock list-inference-profiles --region "$AWS_DEFAULT_REGION" \
  --query 'inferenceProfileSummaries[].{id:inferenceProfileId,name:inferenceProfileName}' \
  --output table

aws bedrock list-foundation-models --region "$AWS_DEFAULT_REGION" \
  --by-output-modality TEXT \
  --query 'modelSummaries[].{id:modelId,provider:providerName}' --output table
```

Inspect addresses them as `bedrock/<id>`.

**Copy each ID verbatim — do not infer it from another model's shape.** There is
no single naming pattern, and guessing produces a `ValidationException` that
reads like a permissions problem. Two real examples that coexist today:

```
us.anthropic.claude-sonnet-4-6                          # newer: undated
global.anthropic.claude-haiku-4-5-20251001-v1:0         # older: date + version
```

Newer releases dropped the date stamp and `-v1:0` suffix that older ones carry,
and the geo prefix varies per model (`us.`, `eu.`, `apac.`, `global.`) — a Global
profile routes to all commercial regions, a geo-scoped one never changes its
destination list.

**Not every model has an inference profile.** Several in this lineup are
in-Region only; their model cards read *Geo inference ID: Not supported /
Global inference ID: Not supported*, and there is no profile section to copy
from. Use the bare model ID and stay in one region:

```
openai.gpt-oss-20b-1:0                   # no profile; in-Region only
mistral.mistral-large-3-675b-instruct    # no profile, no date, no version
```

Each card's **Programmatic Access** table has one row per endpoint, and the IDs
can differ between them — GPT-OSS is `openai.gpt-oss-20b-1:0` on
`bedrock-runtime` but `openai.gpt-oss-20b` on `bedrock-mantle`. Inspect's
`bedrock/` provider calls **bedrock-runtime**, so take that row. The mantle ID
fails validation in a way that looks like the model does not exist.

Or let the script do it, which also emits the eval command with roles filled in:

```bash
python experiments/resolve_models.py --check
```

Record what you pick in the table at the bottom. Where an ID is undated, note the
date you pinned it — an undated profile can be repointed at a newer snapshot, so
the ID alone will not identify what you actually ran.

## 6. Assign model roles

Eight roles. An unset role falls back to `--model`, so a first run can point
everything at one model — but the numbers are not reportable until the roles are
split, for two reasons that are easy to violate by accident:

- **No judge may share a family with either orchestrator.** The SIF paper found
  0/14 orchestrators flagged their own plans. Condition (d) exists to avoid that
  self-blindness, and it cannot do so if the judge is the planner.
- **`judge_civ` must differ from `judge_rubric`.** They are the precision and
  recall halves of the conjunctive gate; on one model the conjunction collapses
  and the 0% false-positive result goes with it.

The defense checkers (`sentinel_node`, `sentinel_edge`) should also differ from
the orchestrator, or the system under test is grading its own homework.

| role | what it is | constraint |
|---|---|---|
| `orchestrator` | the system under test | the capability ablation varies this |
| `worker` | executes subtasks | cheap; content only matters to edge checks |
| `judge_rubric` | L1 policy rubric, condition (c) | ≠ orchestrator family |
| `judge_civ` | CIV plan judge, condition (d) | ≠ orchestrator AND ≠ judge_rubric |
| `judge_drb` | genuine-harm judge, condition (e) | ≠ orchestrator family |
| `sentinel_node` | Granite Guardian's slot | ≠ orchestrator family |
| `sentinel_edge` | LlamaFirewall's slot | ≠ orchestrator family |
| `sentinel_path` | path semantics; off by default | ≠ orchestrator family |

## 7. Create two guardrails

Bedrock console → **Safeguards → Guardrails** → create two.

**Configure content filters and nothing else.** `guardrail_classifier` grades on
`assessments[].contentPolicy.filters[].confidence`; every other policy can only
contribute a binary 1.0 via `GUARDRAIL_INTERVENED`, which adds noise to `FS(P)`
without adding resolution. So: skip denied topics, word filters, sensitive
information, and contextual grounding (that last one needs a source document
anyway). The blocked-message text is never read — the masked output is discarded.

The two differ only in filter strength, on the **prompt/input** side (the call
uses `source="INPUT"`; response-side settings never execute):

| guardrail | Hate / Insults / Sexual / Violence / Misconduct | Prompt attacks |
|---|---|---|
| `cjb-strict` | High | On, High |
| `cjb-permissive` | Low | Off |

Higher strength filters more, so High is the strict end. Already handled in code
and not yours to set: `qualifiers: ["guard_content"]` (required for the Prompt
attacks filter to fire at all) and `outputScope="FULL"` (required for
confidences to appear when nothing intervened).

Two separately-configured guardrails give F1/F2 two genuine operating points.
One guardrail called twice is one classifier reported as two, and `FS(P)` is a
max over the battery — so a duplicated member changes nothing and overstates the
evidence.

**Then publish a version of each.** The ID is generated on save and shown on the
guardrail's detail page (short alphanumeric, e.g. `abc123def456`), or:

```bash
aws bedrock list-guardrails \
  --query 'guardrails[].{id:id,name:name,version:version}' --output table
```

A fresh guardrail is at `DRAFT`, which is **mutable** — editing a filter changes
what `DRAFT` resolves to, with nothing in the eval log recording that the battery
moved between runs. Click **Create version** on each to freeze it, and pass the
number:

```python
guardrail_battery(strict_id="abc123def456", permissive_id="def456ghi789",
                  version="1")   # not the DRAFT default
```

Both IDs and both version numbers go in the pinned-config table. Use the ID when
guardrails and models share a region — the full ARN is only required across
regions, which step 2 rules out anyway.

**Substitution to disclose in the paper:** this replaces the source paper's
LlamaGuard-7b + KoaAI gating pair (and ToxBERT/Aegis/WildGuard as informational).
`FS(P)` measured against Bedrock Guardrails is not directly comparable to their
reported 1.00.

## 8. Install

A venv, not the system interpreter — on Windows the default `python` is the
Store alias, and installing 90 packages into it is how you get a broken
interpreter later. Run from the repo root (`CompJailBreak/`); `.venv` is already
gitignored.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install inspect_ai langgraph aioboto3 boto3
```

Then either activate it (`.\.venv\Scripts\Activate.ps1`) or call
`.\.venv\Scripts\python.exe` explicitly in every command below.

**PowerShell 5.1 has no `&&`.** Chain with `;`, or run one command per line. The
bash snippets in this file need translating; the `inspect eval` invocations are
unaffected apart from line continuations (`` ` `` instead of `\`).

Verified 2026-08-07 on Python 3.14.6 / `inspect_ai` 0.3.252 — all wheels
available, no build step.

---

## Running

### Step 1 — offline regression, no AWS

```bash
cd Attack_SemanticIntentFragmentation && python smoke_test.py
cd ../Defense_SentinelAgents          && python smoke_test.py
```

### Step 2 — wiring check, zero spend

```bash
python experiments/wiring_check.py
```

Drives the real task with a role-aware fake. Exercises the anyio sync/async
bridge, the JSON plan extractor, the judge parsers, and the solver→scorer store
handoff. Expect `WIRING OK`, 33 samples, ~2 seconds.

On Windows this logs `Control server failed to start ... module 'socket' has no
attribute 'AF_UNIX'`. Harmless — Inspect's optional control surface needs Unix
domain sockets. The eval itself is unaffected.

### Step 3 — two live samples, then read the log

Pass `--temperature 0`. The role models are pinned in code (`ROLE_CONFIG` in
`bedrock_bridge.py`), but the orchestrator arrives via `--model` and is not.
Note it is `--temperature 0`, **not** `-M temperature=0` — `-M` passes provider
arguments, not generation config, so the latter is silently ignored and the
orchestrator samples. `analyze_run.py` prints the effective temperature per
model, read off the model events, and will say so.

```bash
inspect eval experiments/sif_vs_sentinel.py --limit 2 \
  --model bedrock/<orchestrator-id> --temperature 0 \
  --model-role judge_civ=bedrock/<other-family-id> \
  --model-role judge_rubric=bedrock/<third-family-id> \
  --model-role judge_drb=bedrock/<fourth-family-id> \
  --model-role sentinel_node=bedrock/<fifth-family-id> \
  --model-role sentinel_edge=bedrock/<fifth-family-id> \
  --max-samples 8
```

Read the transcript before going further. Confirm the orchestrator actually
returned a plan and that the judges cited the clause.

### Step 4 — the pipeline-proof run

Same command without `--limit`. 33 samples.

### Capability ablation

```bash
--model-role orchestrator=bedrock/<weaker-model-id>
```

No code changes.

---

## Cost

The wiring check measures **~31 model calls per sample**, so 33 samples is
roughly **1,000 calls** on short prompts. Single-digit dollars.

The defense dominates: the LLM node and edge checkers fire once per graph
element (~16), and `strict_recheck` runs them a second time — about 32 of the 31
average. Two ways to cut it if that matters more than fidelity:

```bash
-T llm_checkers=false     # lexicon stubs for node/edge status; ~4 calls/sample
-T strict_recheck=false   # halves the checker calls, and is an ablation you want anyway
```

`llm_path_semantics` is **off** by default; turning it on adds one call per
pattern per sample.

**GPT-OSS is a reasoning model**, so the sentinel checkers cost far more than
their output suggests: a reply of `"0.0"` measured 270 output tokens, and the
checkers fire ~32 times per sample. Budget on ~300 output tokens per check, not
on the 3 characters that come back, and expect wall-clock to be dominated by
this rather than by the orchestrator.

Verified safe, but worth knowing why it nearly wasn't: `llm_node_checker` parses
with `re.search(r"[01](?:\.\d+)?", raw)`, which takes the *first* number
anywhere in the string — and the model's own reasoning quotes the prompt's
phrase "Rate 0.0-1.0". Had reasoning reached the parser, every check would have
returned `0.0` and tier 2 would have been silently dead with nothing in the log.
Inspect strips `reasoningContent` from `output.completion`, so it does not. If
you ever swap in a provider that does not strip it, anchor the regex.

---

## Reading the result

The number this experiment exists to produce is `attack_evaded_defense` on the
attack arms: the attack cleared all five gate conditions and no defense tier
fired.

**Signals the run is broken rather than interesting:**

| symptom | meaning |
|---|---|
| `fragmentation_score` 0.0 everywhere | guardrails too strict; the premise is that subtasks pass |
| every metric 1.000, FPR 0.000 | the stub signature — confirm a real model id was reached |
| `composition_only` < 1.0 on attack arms | **expected and reportable.** The live planner emitted a plan with an already-tainted subtask, so that sample is not a compositional attack and does not belong in the SIF-ASR numerator |
| detection recall 1.0 with `human_review_rate` 1.0 | recall bought entirely with escalations, not detection |
| results identical across code changes | the **generate cache** is serving stale responses |

On that last one: `model_call_llm` sets `cache=False` for exactly this reason.
Inspect's cache is keyed on model and prompt, so it will replay a response
recorded before you changed a prompt, with nothing in the log saying so. If you
ever enable it:

```bash
inspect cache clear      # note: `clear` is a subcommand, not a --clear flag
```

---

## Pinned configuration

Fill in before reporting. Recoverable from nowhere else. Paste each ID exactly as
copied, **without** the `bedrock/` prefix — that prefix is Inspect's provider
routing, not part of the ID. Example of a filled row:

| `orchestrator` (strong) | `us.anthropic.claude-sonnet-4-6` | 2026-08-07 |

| item | value | date pinned |
|---|---|---|
| Region | us-east-1 | 2026-08-07 |
| `orchestrator` (strong) | us.anthropic.claude-sonnet-4-6 | 2026-08-07 |
| `orchestrator` (weak, ablation) | us.anthropic.claude-haiku-4-5-20251001-v1:0 | 2026-08-07 |
| `worker` | us.anthropic.claude-haiku-4-5-20251001-v1:0 | 2026-08-07 |
| `judge_rubric` | us.meta.llama3-3-70b-instruct-v1:0 | 2026-08-07 |
| `judge_civ` | mistral.mistral-large-3-675b-instruct | 2026-08-07 |
| `judge_drb` | us.amazon.nova-pro-v1:0 | 2026-08-07 |
| `sentinel_node` / `sentinel_edge` | openai.gpt-oss-20b-1:0 | 2026-08-07 |
| Guardrail (strict) id + version | fq62giqasusw | 1 |
| Guardrail (permissive) id + version | g8e7b3bwc0py | 1 |
| `inspect_ai` version | | |

Verified against the model cards on 2026-08-07: all seven resolve in `us-east-1`
and all support the Converse API, which is what Inspect's `bedrock/` provider
calls. Notes the paper should carry:

- **Sonnet 4.6 and Haiku 4.5 have no In-Region availability in us-east-1** — only
  Geo and Global. The `us.` prefix is mandatory for both, not a preference.
- **`us.anthropic.claude-sonnet-4-6` is undated**, so the date column above is the
  only record of which snapshot ran. The Haiku and Llama IDs are self-identifying.
- **`worker` and the weak `orchestrator` are the same model.** Fine for the main
  run, but in the capability ablation the planner and the executors collapse onto
  one model. Disclose it, or point `worker` at a third Anthropic model.
- Nova Pro (Dec 2024) and Llama 3.3 70B (Dec 2024) are the oldest entries. Nova
  Pro's card gives an EOL of "no sooner than 2025-12-04" and still reads Active —
  re-check before a final run rather than assuming it survives the review cycle.
