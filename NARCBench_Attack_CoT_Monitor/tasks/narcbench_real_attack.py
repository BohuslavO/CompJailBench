# Add the project root to Python's import path.
# This lets Inspect find our src/ package when the task is run.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Inspect task objects.
from inspect_ai import Task, task


# Our NARCBench dataset loader.
from src.narcbench_dataset import load_narcbench_dataset


# Our REAL solver.
# This is the code that calls the selected LLM instead of
# using predetermined dry-run messages and votes.
from src.narcbench_real_solver import narcbench_real_solver


# Our NARCBench attack scorer.
from src.narcbench_scorer import narcbench_attack_scorer


@task
def narcbench_real_attack() -> Task:
    """
    Real NARCBench collusion/control evaluation.

    Dataset:
        env_02_collusion
        env_02_control

    Solver:
        Real model-generated multi-agent interaction.

    Scorer:
        Calculates the NARCBench attack outcome.
    """

    return Task(
        dataset=load_narcbench_dataset(),
        solver=narcbench_real_solver(),
        scorer=narcbench_attack_scorer(),
    )