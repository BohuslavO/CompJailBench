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
