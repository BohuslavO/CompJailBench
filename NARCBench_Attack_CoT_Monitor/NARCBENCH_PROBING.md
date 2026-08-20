# NARCBench attack x NARCBench Probing

This directory contains the integration for evaluating the NARCBench attack
with the activation-probing defense in `Defence_NARCBench Probing`. The adapter
uses the released activation extractor and scoring pipeline without modifying
the defense implementation.

## Experimental scope

- Target model: `Qwen/Qwen3-32B-AWQ`
- Probe layer: 28
- Defense input: hidden activations from the same fresh Qwen generation run
- Primary score: the released `scorer_defense.py` definition—maximum deception
  projection across every recorded trajectory turn
- Secondary scores: a public-discussion-only peak and the released group metrics on
  one mean discussion vector per agent
- Samples: the matched 50 attack and 50 benign-control samples already stored
  in `data/narcbench_core_official.json`
- Pre-registered primary metric: turn-level `peak_suspicion`

The old Bedrock `gpt-oss-120b` text trajectories cannot be used for the
activation probe because they do not contain Qwen hidden states. They remain
valid for message-only defenses. The completed probing cell therefore uses a
separate fresh Qwen run rather than claiming that the Bedrock trajectories are
activation-compatible.

## Completed full evaluation

The final Kaggle run passed strict validation over all 50 matched scenarios:

- 50 collusion-condition samples and 50 matched benign controls;
- 100 unique score rows and 100 activation artifacts;
- 1,700 layer-28 activation turns with dimension 5,120;
- zero duplicate score rows and zero failed-generation artifacts;
- 80 schema-only formatting repairs across 1,700 accepted turns (4.71%);
- exact probe SHA-256
  `d9a340a3276c477b817e9d0f7580a1c21cf3e5d080bb0ac0281e4c23b3d8e4ec`.

For the pre-registered all-turn `peak_suspicion` metric, the attack mean was
6.473 and the benign-control mean was 0.832. The paired mean difference was
5.642 (10,000-resample paired-bootstrap 95% CI [4.653, 6.666]), AUROC was
0.9896, and average precision was 0.9901. The public-discussion-only peak and
agent-mean group metrics are secondary analyses; they are not substitutes for
the pre-registered primary metric. No binary threshold was fitted on these
evaluation samples.

The fresh Qwen attack outcomes were analyzed separately from the defense
scores. Both designated colluders followed the assigned target in 31/50 runs
(0.620; Wilson 95% CI [0.482, 0.741]). Three of 50 attacks achieved the strict
end-to-end outcome, compared with 0/50 matched controls selecting the target.
The paired target-selection uplift was 0.060 (bootstrap 95% CI [0.000, 0.140];
exact McNemar p=0.250). There were no abstentions or unrepaired parse failures.
These results do not establish a statistically clear committee-outcome effect,
despite strong condition-label separation by the probe.

Compact reproducibility artifacts are stored in `results/narcbench_probing`.
The complete 22.81 MB evidence archive is retained outside Git and is identified
by SHA-256
`e3d8d339486ce18d13b60eeb26b31479cb2f3a822849a6f6fcda85d982f64cb8`.
See `results/narcbench_probing/RESULTS.md` for interpretation and limitations.

## Implementation

- `src/narcbench_probe.py`: Inspect scorer-wrapper and strict artifact checks
- `src/narcbench_probe_runtime.py`: Qwen generation plus activation capture
- `scripts/run_narcbench_probe_kaggle.py`: resume-safe pilot/full runner
- `scripts/analyze_narcbench_probe.py`: metrics and team-table export
- `scripts/build_narcbench_probe_bundle.py`: fallback bundle builder
- `tests/test_narcbench_probe.py`: zero-GPU integration tests

The Kaggle runner reserves GPU headroom with a 12.5 GiB-per-T4 model-weight
budget, CPU weight offload, the standard dynamic KV cache, and cache-free activation
extraction. These are memory-placement settings only; they do not change the
model, prompts, probe, decoding policy, or evaluation labels.

An offloaded KV cache was rejected during the pilot because it produced
corrupted generations and non-finite sampling probabilities with this
Qwen/AWQ/two-T4 stack. The validated run therefore uses Transformers' dynamic
cache default; this runtime choice is recorded in the task and pair metadata.

Qwen3 is run with `enable_thinking=False`. Each task prompt already requests a
concise, observable rationale in its JSON response. Disabling the model's
separate hidden `<think>` mode prevents the fixed output budget from being
consumed before the parseable committee message or final recommendation is
emitted; hidden reasoning is not scored as task evidence.

Generation follows Qwen's published non-thinking recommendation rather than
greedy decoding: temperature 0.7, top-p 0.8, and top-k 20. Qwen explicitly
warns that greedy decoding may produce degradation and endless repetition.
For reproducibility, each call receives a stable seed derived from a fixed run
seed and the rendered prompt. The retry prompt contains only an audit hash of
the malformed draft, not its text, so corrupted output cannot be echoed into
the regeneration context.

The generation allowance is 512 new tokens per call. Every response must be a
complete JSON object with `observable_rationale` and either `message` or
`response`, depending on the workflow phase. A malformed or truncated output
receives at most one logged formatting-repair regeneration. The retry must
preserve the intended position, may not introduce new factual claims, and is
governed by the same compact JSON contract. The accepted response is
activation-scored under the original task prompt, so the repair instruction
itself is not exposed to the probe. Both conditions use the identical policy,
and each activation row records whether repair was used and the number of
generation attempts. If the retry is also invalid, the sample fails before
activation scoring, preventing silent abstentions or corrupted discussion
history from being reported as experimental results.
The shared formatting contract limits the task message/response to 80 words
and the observable rationale to 40 words and disallows Markdown fences. These
constraints affect serialization length, not the assigned option, scenario,
agent role, discussion rounds, or defense score.

## Probe artifact provenance

The compatible `narcbench_probe.pkl` was added to `main` in commit `71851d6`,
followed by `scorer_defense.py` in `0ac4403`. The binary is not duplicated in
this package. Its exact hash and inspected metadata are recorded in
`configs/narcbench_probe_main.json`; the runner verifies the SHA-256
before unpickling it. Do not fit or select a decision threshold on these 100
evaluation samples.

## Block A - local zero-GPU check

```bash
python -m unittest tests.test_narcbench_probe -v
```

## Block B - one matched pair on Kaggle

Upload this project, the unchanged `Defence_NARCBench Probing` directory from
`main`, and the released probe bundle to a Kaggle GPU notebook. Then run:

```bash
python -m pip install -r requirements-probing-kaggle.txt
python scripts/run_narcbench_probe_kaggle.py \
  --probe "../Defence_NARCBench Probing/narcbench_probe.pkl" \
  --pairs med_01 \
  --output /kaggle/working/narcbench_probing_pilot
```

Check that the pair completion marker, two Inspect logs, two activation files,
and two probe-score records exist before scaling up.

## Block C - five-pair pilot

Reuse the same output directory so the completed first pair is skipped:

```bash
python scripts/run_narcbench_probe_kaggle.py \
  --probe "../Defence_NARCBench Probing/narcbench_probe.pkl" \
  --max-pairs 5 \
  --output /kaggle/working/narcbench_probing_pilot
```

Only continue if all ten samples score successfully and storage/runtime look
reasonable.

## Block D - full matched evaluation

```bash
python scripts/run_narcbench_probe_kaggle.py \
  --probe "../Defence_NARCBench Probing/narcbench_probe.pkl" \
  --output /kaggle/working/narcbench_probing_full
```

The runner writes a marker after each matched pair, so rerunning the command
resumes instead of paying for completed pairs again.

## Block E - analyze and export

```bash
python scripts/analyze_narcbench_probe.py \
  /kaggle/working/narcbench_probing_full/probe_scores.jsonl \
  --output-dir /kaggle/working/narcbench_probing_full/analysis \
  --run-id narcbench-probing-qwen32b-layer28 \
  --require-full
```

Report AUROC and average precision for the pre-registered primary
`peak_suspicion`, then clearly label the public-only and group-level metrics as
secondary analyses. Also report attack/control means and paired bootstrap
confidence intervals. A binary threshold requires a separate validation split
or a threshold fixed by the original defense, not post-hoc tuning on the test
set.
