"""
Composition root. This is the ONE file allowed to import from both
compjailbench_attack/ and compjailbench_defense/ -- neither of those
should import the other directly (Marco's "no hardcoded references
either direction" note on Phase 4). Everything on either side only
needs to agree on the shared contract (state.metadata["trajectory"],
target.text = checkpoints.md), not on each other's internals.

This is also the composition point where a defense Scorer can be attached to
an attack Solver. Swap the `solver=` / `scorer=` arguments for another
compatible Solver/Scorer pair and the rest is unchanged, since both sides only depend on the
Sample/TaskState contract below.
"""
from inspect_ai import Task, task

from dataset import build_dataset


@task
def compjailbench(
    solver,
    scorer,
    tasks_root: str = "tasks/tasks",
    condition: str = "attack",
    strategy: str = "role_override_routing",
    slugs: list = None,
) -> Task:
    return Task(
        dataset=build_dataset(tasks_root, condition, strategy, slugs),
        solver=solver,
        scorer=scorer,
    )
