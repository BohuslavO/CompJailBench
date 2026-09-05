# CompJailBench

CompJailBench is a benchmark for distributed jailbreaks in multi-agent LLM systems: a harmful
request is adversarially decomposed across agents so that each subtask looks safe in isolation
while the composed output is unsafe. Each team member owns one attack and one defense; every
attack is evaluated against all four defenses under a common long-format results schema. See
`results 4x4/MATRIX.md` for the current state of that 4x4 matrix and `results 4x4/README.md`
for the schema itself.

## Scope

This repository is the shared implementation and results store for the project. It is not a
single unified codebase; each attack and each defense was built independently against a common
trajectory/schema contract, and cross-tested against the others' work. Where an evaluation uses a
custom baseline or constrained evidence scope (for example, the project-defined Agent-to-Agent
Monitor, message-only Sentinel Agents, or a non-default monitor prompt), that qualification is
recorded in the corresponding CSV's `notes` column and must be preserved in any write-up.

## Repository layout

- `Attack_DecompBench/` — DeCompBench-based task-routing attack.
- `Attack_AgentHarm/` — AgentHarm behavior-prompt evaluation, compact defense
  scores, aggregate summaries, and run manifest.
- `Defence_NARCBench Probing/` — activation-probing defense, reused from NARCBench.
- `Attack_SemanticIntentFragmentation/` — Semantic Intent Fragmentation attack.
- `Defense_SentinelAgents/` — graph-based SentinelAgent defense.
- `compjailbench/` — shared AgentHarm baseline code and the
  Agent-to-Agent Monitor implementation.
- `NARCBench_Attack_CoT_Monitor/` — isolated project for the NARCBench collusion attack and the
  CoT/observable-rationale monitor. Has its own README and a `DEFENSE_INTERFACE.md`
  contract that other attacks use to test against this defense.
- `CompJailBench_Inspect/` — shared Inspect integration and trajectory converters
  (`StandardTrajectory`, `execution_trace`) used to run one contributor's attack through
  another contributor's defense without a custom harness per pair.
- `results 4x4/` — the shared results matrix. `raw/` holds one CSV per
  (contributor, attack, defense) submission; `build_matrix.py` reads all of `raw/` and
  regenerates `MATRIX.md`.
- `sif_vs_defenses/`, `data/activations/` — supporting data and intermediate artifacts for the
  Semantic Intent Fragmentation attack.

## Adding a result

For AgentHarm reproduction and additional defense cells, follow
[`AGENTHARM_RUNBOOK.md`](AGENTHARM_RUNBOOK.md). Frozen text-defense results and
limitations are documented in [`Attack_AgentHarm/README.md`](Attack_AgentHarm/README.md).

1. Convert your attack's saved trajectories into the format the target defense expects (see
   `CompJailBench_Inspect/` for existing converters, or that defense's own
   `DEFENSE_INTERFACE.md`/README if one exists).
2. Run a free compatibility check first where the defense supports one (e.g.
   `--validate-only` in `NARCBench_Attack_CoT_Monitor/scripts/score_standard_trajectories.py`),
   then one attack sample and one benign sample against the real model, before authorizing a
   full run.
3. Run the defense and export one CSV row per metric in the schema documented in
   `results 4x4/README.md`.
4. Save it as `results 4x4/raw/<yourname>__<attack_method>__<defense_method>.csv`, using the
   canonical attack/defense keys listed in that README.
5. Run `python build_matrix.py` from inside `results 4x4/` and commit both the new CSV and the
   regenerated `MATRIX.md` together.

## Repository notes

- Raw Inspect `.eval` logs and Python `__pycache__` are intentionally not tracked (see
  `.gitignore`); regenerate them locally by re-running the relevant attack/defense scripts.
- Some sub-projects (e.g. `NARCBench_Attack_CoT_Monitor/`) keep their own `.env`/`.venv` and
  local setup instructions rather than sharing one root environment, since attacker and monitor
  model access differ per defense.
