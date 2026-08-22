# Attack_DecompBench

DeCompBench-based task-routing attack: our contribution to CompJailBench's distributed-jailbreak
attack surface. A harmful request is pulled from the DeCompBench dataset, decomposed into
non-harmful-looking subtasks, and those subtasks are routed across a small multi-agent pipeline
so that no single agent sees enough of the request to refuse it.

This folder contains the attack implementation only. To run it as an Inspect task, score it
against our own NARCBench Probing defense, or score it against another contributor's defense,
see `CompJailBench_Inspect/`.

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
