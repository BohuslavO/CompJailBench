# Defence_NARCBench Probing

Activation-probing defense: our contribution to CompJailBench's defense set. Detects distributed
jailbreaks using agent activation vectors, reusing NARCBench's pretrained deception probe rather
than training a new one.

## What this defense does

NARCBench [Rose et al., 2026] detects collusion in multi-agent systems using agent activation
vectors. This parallels CompJailBench, where no individual agent's output reveals the system's
overall harmful goal. Unlike NARCBench's colluding agents, which are given explicit hidden
instructions to collude, agents under a distributed-decomposition attack are not aware that a
harmful request is being assembled — each only sees its own benign-looking subtask. We address
this by evaluating the whole trajectory against ground truth (attack vs. benign-control), rather
than requiring an explicit per-agent collusion label. We reuse NARCBench's pretrained deception
probe, fit on a Qwen3-32B model.

We use NARCBench's five probing techniques:

- **Activation Alignment** — mean pairwise cosine similarity across all agents' activation
  vectors; the negative of this value is the anomaly signal.
- **Peak Suspicion** — the maximum deception score across all agents; the group is flagged if any
  single agent's projection onto the deception direction is high.
- **Sorted Concatenation** — agents sorted by deception score, standardized activation vectors
  concatenated, dimensionality reduced via PCA, and a logistic regression classifier trained on
  the result.
- **Deception Split** — the average of the top two deception scores minus the average of the
  bottom two; a large positive value indicates a suspicious 2-agent subgroup.
- **Asymmetry Probe** — variance, range, the Deception Split statistic, and the mean, combined
  into a single logistic regression classifier.

## Files

- `extract_activations.py` — runs the target model on a trajectory's turns with forward hooks on
  the chosen layer(s), extracting each agent's activation vector (`build_gen_text` matches
  NARCBench's own prompt construction exactly, since the probe was trained on that construction).
- `train_narcbench_probe.py` — (re)fits NARCBench's deception probe on activation data; only
  needed if retraining rather than reusing the shipped `narcbench_probe.pkl`.
- `narcbench_probe.pkl` — the pretrained probe bundle (scaler + classifier weights) used at score
  time. Must match the target model's hidden size — a probe trained for a different model will
  fail silently informative (a shape mismatch error), not produce wrong-but-plausible scores.
- `score_pipeline.py` — runs the five probing techniques over a set of extracted activations and
  produces per-sample scores (Peak Suspicion, etc.).
- `scorer_defense.py` — wraps this defense as an Inspect `Scorer`, so it can be attached to any
  attack's `Task` and read the trajectory the attack's `Solver` left in the sample.
- `smoke_test.py` — sanity-check script to confirm activation extraction and scoring work
  end-to-end on a small number of trajectories before a full run.

## Running it

1. Confirm `narcbench_probe.pkl` matches the target model you're extracting activations from
   (same hidden size / same base model family it was trained on).
2. Run `smoke_test.py` against a handful of trajectories first.
3. For a full run, `scorer_defense.py` is the entry point to attach to an attack's Inspect
   `Task` — see `CompJailBench_Inspect/` for how DeCompBench's attack wires this in via
   `run_kaggle.py`.
