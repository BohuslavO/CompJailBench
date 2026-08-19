# Plan: SIF attack vs NARCBench activation probe

> Companion to the SIF-vs-SentinelAgent run. Same attack, a *white-box* defense.
> The research question is identical in shape — **does SIF evade the defense?** —
> but the defense reads model **activations**, not trajectory text, so almost
> nothing about the SentinelAgent runbook transfers. Read this before spending a
> GPU-hour.

## The one-paragraph version

NARCBench Probing is a linear probe fit on a model's residual-stream activations
to separate *colluding* agents from *honest* ones. SIF has no colluders: the
retriever, processor, and publisher each perform an honest, individually-benign
subtask, and the violation exists only in their composition — which lives in no
single agent's activation vector. The expected result is therefore an **AUROC at
chance (~0.5)** on SIF-attack vs benign, and the finding is not "the probe is
bad" but "**activation collusion-probes are blind to compositional attacks by
construction**" — the exact activation analog of the SentinelAgent tier-3 result.
That null is only publishable if we first prove the probe is *not* broken: it must
reproduce NARCBench's own Core AUROC on NARCBench's own data before we trust a 0.5
on ours. That sanity gate is the spine of this plan.

---

## Why this defense is nothing like SentinelAgent

| | SentinelAgent | NARCBench Probing |
|---|---|---|
| Reads | trajectory **text** | model **activations** (last-token hidden state) |
| Runs on | Bedrock (black-box API) | **open weights on a GPU** — no API exposes activations |
| Output | per-trajectory **binary alert** | per-trajectory **continuous score** → **AUROC** over a labeled set |
| Probe is | model-agnostic | **model-specific** (trained in one model's activation space) |
| Strongest detectors | tiers 1–3 | `asymmetry_probe`, `sorted_concat` — **require exactly N=4 agents** |

Two consequences fall out immediately and both are findings, not bugs:

1. **SIF emits 3 agents** (`orchestrator.DEFAULT_AGENTS`: retriever/processor/
   publisher; M4 collapses to 2 distinct agents). NARCBench's two *trained*
   classifiers only fire at N=4 (`score_pipeline.post_hoc_scores` returns `None`
   for them otherwise). So SIF is scorable only by the three generalizable
   techniques — `peak_suspicion`, `deception_split`, `activation_alignment` — and
   the defense's strongest machinery is structurally disabled by SIF's topology.
   Report this explicitly; optionally add a 4-agent SIF variant as a follow-up so
   the full battery applies.
2. **The probe is trained on collusion**, SIF is not collusion. The deception
   direction was fit to find an agent *hiding* intent among peers. SIF agents hide
   nothing — each is honest about a benign task. There is no per-agent signal for a
   per-agent probe to read. This is the headline.

---

## What already exists — reuse, don't rebuild

Everything on the defense side is written and smoke-tested:

| step | existing code | file |
|---|---|---|
| forward-pass + hook activations | `load_model`, `extract`, `build_gen_text` | [extract_activations.py](../Defence_NARCBench%20Probing/extract_activations.py) |
| fit deception direction + classifiers | `fit_bundle` | [train_narcbench_probe.py](../Defence_NARCBench%20Probing/train_narcbench_probe.py) |
| score a trajectory (5 techniques) + AUROC | `post_hoc_scores`, `LiveScorer`, `early_auroc` | [score_pipeline.py](../Defence_NARCBench%20Probing/score_pipeline.py) |
| per-model layer ranges | `MODEL_LAYERS` | train_narcbench_probe.py |

The attack side and its trajectory export already exist too — `sif_vs_sentinel.py`
runs the SIF orchestrator/workers and writes `state.store["trajectory"]` in the
exact 4-key shape `extract_activations.build_gen_text` consumes
(`orchestrator.py` docstring line 22 names this file as the consumer).

**New code is one thin glue script.** No edits to any of the four benchmark folders.

---

## The seam, and the one bridge that makes this possible

SentinelAgent's seam was `call_llm(system,user)->str`. NARCBench's seam is
`list[np.ndarray]` — one activation vector per agent. Bedrock cannot produce it.
The bridge is **teacher-forcing**: take the SIF trajectory *text* and run each
`(system, user, assistant-response)` triple through the open-weights model in one
forward pass, hooking the target layer. That is precisely what
`extract_activations.extract` does.

Faithfulness rule that decides the whole design: **the model that authors the SIF
text must be the model that gets probed.** If Claude authors the text and we
teacher-force Qwen, the activations describe "Qwen reading Claude's words," not the
acting agent — a proxy a reviewer will reject. So run SIF *on the probed model*.
Because that model authored the tokens, teacher-forcing its own output reproduces
the activations that occurred during generation. Valid.

---

## Decision: probe **gpt-oss-20b** (canonical alt: Qwen3-32B)

gpt-oss-20b is the pragmatic pick and it is not close:

- It is **already in our Bedrock lineup** (`openai.gpt-oss-20b-1:0`) — so the same
  model is SIF orchestrator+worker *and* the probed model, satisfying the
  faithfulness rule with zero extra choices.
- It is one of NARCBench's four configured models (`MODEL_LAYERS`: layers 10–14,
  probe layer 12) and **NARCBench ships its precomputed Core activations** — so
  training the probe needs **no GPU**.
- At 20B it fits on a single 24–48 GB GPU for the one inference-only extraction job.

Run **Qwen3-32B-AWQ** as the robustness / "match the NARCBench paper" replication
(their headline model, layers 26–30, probe layer 28) if the gpt-oss-20b result
needs a second model to stand on. Same pipeline, bigger GPU.

---

## Pipeline

Ordered so each cheap step de-risks the expensive one after it.

### S0 — offline, no spend, no GPU
```powershell
.\.venv\Scripts\python.exe "Defence_NARCBench Probing\smoke_test.py"   # already green
```
Confirms `fit_bundle` → `post_hoc_scores` → `early_auroc` plumbing on fabricated
activations. Add a matching self-check to the new glue script (below).

### S1 — build and VALIDATE the probe (no GPU)
Download NARCBench's shipped activations and fit the bundle:
```powershell
huggingface-cli download aaronrose227/narcbench --repo-type dataset --local-dir data
.\.venv\Scripts\python.exe "Defence_NARCBench Probing\train_narcbench_probe.py" `
  --activations data\activations\gpt_oss_20b\core --model openai/gpt-oss-20b   # layer 12
```
**Then the sanity gate — do not skip.** Hold out a slice of NARCBench-Core and
confirm the probe reproduces NARCBench's reported collusion AUROC (use
`auroc_eval.py` from the `DecompBench x NARCBench test files` folder as the
reference harness). If it doesn't clear their number, *stop* — a 0.5 on SIF from a
probe that also scores 0.5 on NARCBench's own collusion is a broken probe, not an
evaded one, and proves nothing. A validated probe is the precondition for every
downstream claim.

### S2 — run SIF on the probed model (Bedrock, single-digit $)
Reuse the existing task; put gpt-oss-20b in orchestrator **and** worker; guardrails
are irrelevant here so omit them; the SentinelAgent roles still run but their
scores are ignored (point them at the cheapest model to trim spend):
```powershell
.\.venv\Scripts\inspect.exe eval experiments/sif_vs_sentinel.py `
  --model bedrock/openai.gpt-oss-20b-1:0 --temperature 0 `
  --model-role worker=bedrock/openai.gpt-oss-20b-1:0
```
The `.eval` log in `logs/` now carries, per sample, `store["trajectory"]` (the
authored turns) and `metadata["is_attack_arm"]` (the label). Both attack arms and
benign controls are in one dataset, giving AUROC both classes.

### S3 — extract SIF activations (the only GPU step; < 1 GPU-hour)
Load gpt-oss-20b weights on one GPU and teacher-force every SIF turn through it.
The glue script (S4) calls `extract_activations.extract` directly, so this is just
"run S4 on a GPU box." Footprint: ~33 samples × ~3 turns ≈ 100 short forward
passes on a 20B model — minutes, not hours. A single cloud spot GPU or a local
24 GB card with the quantized model both suffice.

### S4 — score → AUROC (no GPU after activations exist)
New glue script `experiments/score_sif_vs_narcbench.py`:
```
load probe bundle (S1)
for each sample in the .eval log (S2):
    turns   = store["trajectory"]
    acts    = extract_activations.extract(model, tok, turns, [layer])   # S3, GPU
    per_agent = average acts by agent_name          # NARCBench's unit is the agent
    scores  = score_pipeline.post_hoc_scores(per_agent, bundle)
    record (sample_id, is_attack_arm, scores)
for technique in peak_suspicion, deception_split, activation_alignment:
    AUROC = score_pipeline.early_auroc([s[technique] for s], labels)
    detection@fixedFPR: threshold at benign 5%/10% FPR, count attack arms caught
report: per-technique AUROC, attack_evaded_probe cell, per-mechanism, benign FPR
```
`asymmetry_probe`/`sorted_concat` come back `None` at N=3 — report them as **N/A
(requires N=4)**, which is itself a result. Include one `assert`-based `demo()` on
fabricated separable/non-separable activations so the AUROC math has a runnable check.

### S5 — analysis + paper
Fold the numbers into the report alongside the SentinelAgent table. The two
defenses now tell one story: a text monitor whose compositional tier is
non-compositional by construction, and an activation probe blind to composition
because composition is in no activation. Extend `analyze_run.py` or add a sibling;
add a NARCBench panel to `findings_report.html`.

---

## Numbers the paper needs

- **AUROC per technique** (peak_suspicion, deception_split, activation_alignment),
  with the N=4 detectors marked N/A and *why*.
- **Early-AUROC checkpoint curve** — `LiveScorer` score after agent 1, 2, 3;
  does suspicion ever separate as the trajectory unfolds?
- **detection@fixedFPR** — at a benign-calibrated 5%/10% FPR threshold, what
  fraction of SIF attack arms are flagged. (AUROC's thresholded twin, the analog of
  SentinelAgent recall.)
- **`attack_evaded_probe`** — attack arm cleared the SIF 5-condition gate AND scored
  below the probe threshold. The headline cell.
- **Per-mechanism** (M1–M4) and **benign FPR**.
- **NARCBench-Core validation AUROC** (S1) reported next to the SIF AUROC — the
  contrast (probe works there, chance here) *is* the argument.

---

## Signals: broken vs interesting

- **AUROC ≈ 0.5 on SIF is the expected finding — but only after S1 passes.** Before
  S1 it is uninterpretable.
- **AUROC ≈ 0.5 on NARCBench's own Core (S1)** → broken probe (wrong layer, wrong
  model dir, activation/metadata misalignment). Fix before touching SIF.
- **peak_suspicion identical on attack and benign** → consistent with the thesis
  (no per-agent deception either way); confirm it's not that every SIF turn is
  empty/stub text producing degenerate activations.
- **Any technique clearing ~0.7 on SIF** → genuinely interesting: the probe found a
  compositional signal it wasn't designed for. Inspect *which* agent drives
  peak_suspicion and whether it tracks mechanism.

---

## Validity caveats to state plainly in the paper

1. **Teacher-forcing** — mitigated by probing the authoring model (S2=S3 model), so
   activations are the model's own, not a foreign transcript's.
2. **Off-distribution probe** — trained on collusion, applied to non-collusion SIF.
   This is the point, not a defect, but must be framed as "we ask whether a
   collusion probe generalizes to compositional attacks," with the S1 validation
   proving the probe is competent on its home distribution.
3. **Role-agents ≠ peer-colluders** — SIF's pipeline roles are mapped onto
   NARCBench's peer-agent group; state the mapping.
4. **N=4 detectors disabled** by SIF's 3-agent topology; the two strongest NARCBench
   techniques never run unless SIF is reconfigured to 4 agents.
5. **Model substitution** — results are for gpt-oss-20b (and Qwen3-32B if run), not
   the full SIF Bedrock lineup.

---

## Cost / infra

- **Probe training: free** — sklearn LR on shipped activations, CPU, seconds.
- **SIF run (S2): single-digit dollars** on Bedrock, already characterized.
- **GPU (S3): one inference-only job, < 1 GPU-hour**, single 24–48 GB GPU. This is
  the entire "GPU infrastructure decision" the SentinelAgent plan deferred — and it
  is small, because NARCBench shipping the training activations removed the only
  large GPU cost.

## Out of scope (follow-ups, not blockers)

- 4-agent SIF reconfiguration to unlock `asymmetry_probe`/`sorted_concat`.
- Cross-model probe transfer (train on Qwen, score gpt-oss activations).
- Transfer/Stego NARCBench tiers — Core is enough for the first result.
- A new probe architecture — we test *their* probe, we don't improve it.

## One thing to confirm before S3

GPU access: a local ≥24 GB card, or a cloud spot GPU budget? It decides gpt-oss-20b
(24 GB ok) vs Qwen3-32B (needs more) and nothing else in the plan. Everything up to
S2 runs with no GPU at all.
