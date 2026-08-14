from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def install_check() -> Task:
    """Small test that checks whether Inspect can run an evaluation."""

    dataset = MemoryDataset(
        [
            Sample(
                input="Reply with the word OK.",
                target="OK",
                id="install-check-1",
            )
        ]
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),
    )