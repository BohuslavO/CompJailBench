# Attack_DecompBench

DeCompBench-based task-routing attack: our contribution to CompJailBench's distributed-jailbreak
attack surface. A harmful request is pulled from the DeCompBench dataset, decomposed into
non-harmful-looking subtasks, and those subtasks are routed across a small multi-agent pipeline
so that no single agent sees enough of the request to refuse it.

This folder contains the attack implementation and the Qwen replay artifacts. To run it as an Inspect task, score it
against our own NARCBench Probing defense, or score it against another contributor's defense,
see `CompJailBench_Inspect/`.

## Qwen replay results

The recovered cohort contains 90 saved trajectories across 45 attack/control pairs:
15 scenarios under each of the three routing strategies. The [Qwen results matrix](../results_qwen/MATRIX.md)
reports AUROC by assigned attack/control condition. CoT action-only scored **0.4170**,
Sentinel response-content Tier 2 scored **0.5432**, and the new Qwen activation probe
scored **0.5654**. The historical A2A AUROC of **0.6188** is retained; its monitor
model is unconfirmed. The historical probe AUROC of **0.6020** remains in the raw CSVs.

The activation replay uses Qwen3‑32B‑AWQ revision
`0499c3ac83fdef8810b907a23894ba91e95eddd8`. It extracts 558 last-token vectors
of dimension 5120 at layer 28 by teacher-forcing the saved inputs and responses.
The frozen probe aggregates turn scores by maximum. See the [replay summary](results_qwen/probe_replay_summary.json)
and [trajectory index](results_qwen/trajectory_index.jsonl). Full logs, replay text
and activation arrays are kept locally. The [upstream dataset access terms](https://huggingface.co/datasets/decompositionbench/DeCompBench)
restrict redistribution of the source dataset.

### Independent checkpoint judge

The historical checkpoint judge used Qwen, the same model family as the trajectory
generator. A separate **Phi‑3.5‑mini‑instruct** judge has now scored all **108 textual
checkpoints** on the recovered trajectories. All verdicts passed format validation.

| Assigned condition | PASS | FAIL | Insufficient evidence | Trajectories with every checkpoint passed |
|---|---:|---:|---:|---:|
| Attack | 16 | 1 | 37 | 13/45 |
| Benign control | 6 | 4 | 44 | 6/45 |

These verdicts are provisional. Human validation on 42 held-out checkpoints is
pending. The judge assesses evidence in saved response text; environment execution
was not observed. A validated independent FJR and success-conditioned AUROC are
still pending. See the [Phi summary](results_qwen/phi_checkpoint_summary.json)
and [judge configuration](results_qwen/phi_judge_config.json).

## What this attack does

We use the DeCompBench dataset and decomposition framework [Kothamasu et al., 2026] to
decompose a harmful request into multiple non-harmful subtasks. DeCompBench does not itself
target multi-agent LLM systems, so we implement our own methodology for distributing decomposed
requests across agents.

The original DeCompBench dataset includes capability tags for each decomposed request. Using
these tags, we split tasks across agents based on role (e.g. a technical request is routed to a
domain-specialist agent). We route the most dangerous fragment of a decomposed request to either
the least-informed agent (one with limited memory or no access to outside/global context) or to
the agent whose role is most likely to override safety training. We test each of these routing
strategies (`least_informed_routing`, `role_override_routing`) against a control condition in
which roles are assigned at random (`random_baseline`), using DJR/FJR.

We compare Monolithic and Decomposed request success rates, and additionally evaluate against a
Benign-Control condition, in which a separate "safe control" LLM generates structurally matched
but non-harmful requests, to measure whether the system incorrectly flags innocent requests as
harmful and whether a downstream defense is well calibrated.

## Files

- `task_loader.py` — loads a `DeCompTask` (title, checkpoints, harm category, services, per-turn
  subtask prompts) from a task directory in the downloaded dataset.
- `routing.py` — the three routing strategies (`random_baseline`, `least_informed_routing`,
  `role_override_routing`), each a function that maps a task's turns to agent roles.
- `pipeline.py` — the LangGraph orchestrator-worker skeleton that actually runs a task: assigns
  roles via a chosen routing strategy, calls the target model turn by turn, and records the full
  trajectory (`agent_name`, `turn_id`, `label`, `system_prompt`, `user_prompt`, `response` per
  turn).
- `checkpoint_scorer.py` — scores a completed trajectory against the task's checkpoints (a set of
  pass/fail rule-based or LLM-judged criteria), yielding the per-task pass count used to compute
  DJR (full/partial/no success) and FJR (pooled checkpoint pass rate).
- `solver_attack.py` — wraps `pipeline.py` as an Inspect `Solver`, so the attack can run inside
  an Inspect `Task` alongside any Inspect `Scorer`.
- `standard_trajectory.py` — the shared `StandardTrajectory` schema (see
  `CompJailBench_Inspect/` for the canonical copy and converters); kept here so this folder has
  no hard dependency on that folder's location.
- `run_first_test.py`, `smoke_test.py` — sanity-check scripts to confirm the pipeline runs
  end-to-end on a small number of tasks before a full batch run.

## Dataset

The dataset subfolder is the downloaded DeCompBench release (250 tasks; see Kothamasu et al.,
2026). It is gated (CC-BY-4.0, access-request required) — the `README.md` inside that subfolder
is the dataset's own license/access card, not documentation of this attack's code.

## Running it

1. Download/place the DeCompBench dataset under this folder (see the dataset subfolder's own
   access instructions).
2. Run `smoke_test.py` first against a small number of tasks to confirm the pipeline, routing
   strategies, and checkpoint scorer all work against your target model before a full run.
3. For a full batch run (all tasks x all strategies x attack/benign-control), use the driver in
   `CompJailBench_Inspect/run_kaggle.py`, which wraps this folder's solver and dataset loader
   into a real Inspect `eval()` run and writes `.eval` logs.
