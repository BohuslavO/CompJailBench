# CompJailBench_Inspect

Shared Inspect integration and cross-attack/cross-defense trajectory converters. This is the
glue layer that lets any contributor's attack be scored by any contributor's defense without a
custom harness for every pair — it does not implement an attack or a defense itself.

## Why this folder exists

Without a shared harness, evaluating every attack against every defense would require a custom
evaluation path per pair. AISI Inspect organizes an evaluation around tasks, samples, solvers,
scorers, and metrics; this folder is where our attacks and defenses are wired into that shared
structure, plus the trajectory-format converters that let one contributor's raw output become
another contributor's expected input.

## Files

- `task.py` — defines the Inspect `Task` for the DeCompBench attack: dataset + attack solver, so
  a teammate's `Scorer` can be plugged in directly to score our attack's trajectories against
  their own defense.
- `dataset.py` — builds an Inspect `Dataset` of `Sample`s from DeCompBench tasks. One dataset per
  fixed (condition, strategy) combination; each `Sample`'s metadata carries `task_dir`,
  `condition` (`attack`/`benign_control`), and `strategy`, and its `target` carries the task's
  checkpoints for LLM-judged scoring.
- `standard_trajectory.py` — the shared `StandardTrajectory` schema used as the common
  interchange format between attacks and defenses that don't share a native trajectory shape.
- `trajectory_converters.py`, `postprocess_to_standard_trajectory.py` — convert a raw attack
  trajectory (our own `pipeline.py` output, or another attack's native format) into
  `StandardTrajectory` JSONL.
- `to_execution_trace.py` — converts our trajectory format into Atharv's `execution_trace`
  node/edge dict shape, so our attack's trajectories can be scored by the G-Safeguard-inspired
  graph monitor without modifying that defense's code.
- `run_kaggle.py` — the real driver script: loads the target model, runs the DeCompBench attack
  across all (strategy, condition) combinations via Inspect `eval()`, and writes `.eval` logs
  (this is what actually produces the raw results behind our row of `results 4x4/MATRIX.md`).
- `inspect_smoke_test.py` — confirms Inspect discovers the task and can complete a minimal
  (e.g. mock-model) run before spending real compute/API budget on a full batch.

## How this fits into the shared results matrix

1. `run_kaggle.py` produces `.eval` logs for the DeCompBench attack.
2. A converter in this folder (`to_execution_trace.py`, or `trajectory_converters.py` /
   `postprocess_to_standard_trajectory.py`) turns those logs into whatever format the target
   defense expects.
3. The target defense's own scoring entry point (e.g. `Defense_SentinelAgents/inspect_task.py`,
   `compjailbench/monitors.py`, `NARCBench_Attack_CoT_Monitor/scripts/score_standard_trajectories.py`)
   is run against the converted trajectories, real model, unmodified.
4. Results are exported into the schema in `results 4x4/README.md` and saved to
   `results 4x4/raw/`.

If you're testing your own attack against someone else's defense, check that defense's own
README or `DEFENSE_INTERFACE.md` first for its exact expected input shape before writing a new
converter here — several defenses already accept `StandardTrajectory` directly.
