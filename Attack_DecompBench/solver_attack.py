import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline import run_task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compjailbench_inspect"))
from trajectory_converters import from_decompbench_trajectory
from standard_trajectory import StandardTrajectory


@solver
def decompbench_routing_solver(base_roles: list, call_llm: Callable, model_name: str = "qwen3-32b-awq") -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_dir = state.metadata["task_dir"]
        condition = state.metadata.get("condition", "attack")
        strategy = state.metadata.get("strategy", "role_override_routing")

        trajectory = run_task(task_dir, base_roles, strategy, call_llm=call_llm, condition=condition)

        state.metadata["trajectory"] = trajectory

        std = state.store_as(StandardTrajectory)
        converted = from_decompbench_trajectory(
            trajectory,
            sample_id=str(state.sample_id),
            condition_label=condition,
            model_name=model_name,
            extra_metadata={"strategy": strategy},
        )
        for field in converted.model_fields:
            setattr(std, field, getattr(converted, field))

        state.completed = True
        return state

    return solve
