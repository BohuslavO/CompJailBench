"""Inspect tasks for the NARCBench attack and observable-reasoning defense."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai import Task, task

from src.cot_monitor import (
    action_only_monitor,
    observable_reasoning_monitor,
    privileged_audit_monitor,
    public_plus_reasoning_monitor,
)
from src.narcbench_dataset import load_narcbench_dataset
from src.narcbench_real_solver import narcbench_real_solver
from src.narcbench_scorer import narcbench_attack_scorer


def selected_pairs(pair_ids: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not pair_ids:
        return None
    if isinstance(pair_ids, str):
        values = pair_ids.split(",")
    else:
        values = pair_ids
    return [str(item).strip() for item in values if str(item).strip()]


@task
def narcbench_attack_only(pair_ids: str | None = None) -> Task:
    """Generate attack/control trajectories without making monitor calls."""

    return Task(
        dataset=load_narcbench_dataset(pair_ids=selected_pairs(pair_ids)),
        solver=narcbench_real_solver(),
        scorer=narcbench_attack_scorer(),
        name="narcbench-attack-only",
        version="2.0.0",
        metadata={
            "design": "50 matched NARCBench-Core collusion/control pairs",
            "rationale_type": "elicited observable rationale, not hidden chain of thought",
            "upstream_commit": "85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d",
        },
    )


@task
def narcbench_cot_experiment(
    pair_ids: str | None = None,
    privileged_audit: bool = False,
) -> Task:
    """End-to-end attack plus three predeclared deployable evidence scopes."""

    scorers = [
        narcbench_attack_scorer(),
        action_only_monitor(),
        observable_reasoning_monitor(),
        public_plus_reasoning_monitor(),
    ]
    if privileged_audit:
        scorers.append(privileged_audit_monitor())

    return Task(
        dataset=load_narcbench_dataset(pair_ids=selected_pairs(pair_ids)),
        solver=narcbench_real_solver(),
        scorer=scorers,
        name="narcbench-observable-reasoning-monitor",
        version="2.0.0",
        metadata={
            "design": "50 matched NARCBench-Core collusion/control pairs",
            "primary_defense": "public action plus elicited observable rationale",
            "ablations": ["public action only", "observable rationale only"],
            "monitor_threshold": 0.5,
            "privileged_audit_included": privileged_audit,
            "upstream_commit": "85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d",
        },
    )


# Backward-compatible task name.
narcbench_real_attack = narcbench_attack_only
