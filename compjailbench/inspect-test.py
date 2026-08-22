from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

@task
def hello():
    return task(
        dataset = [
            Sample(
                input="What is the capital of France?",
                target="Paris",
            )
        ],
        solver=generate(),
    )