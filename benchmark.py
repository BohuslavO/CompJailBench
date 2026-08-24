from __future__ import annotations

from inspect_ai import Task, task

from agents import multi_agent_pipeline
from agentharm_data import load_agentharm
from defenses import (
    cot_monitor_scorer,
    g_safeguard_scorer,
    probing_scorer,
    sentinel_agent_scorer,
)
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[])
def trace_capture_scorer():
    """Free bookkeeping scorer for generation-only trajectory capture."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=1,
            answer="trace_captured",
            explanation="Trajectory generation completed; defenses score it post-hoc.",
            metadata={"condition": (state.metadata or {}).get("condition")},
        )

    return score


def _make_task(defense_scorer=None) -> Task:
    scorers = [trace_capture_scorer()]
    if defense_scorer is not None:
        scorers.append(defense_scorer())
    return Task(
        dataset=load_agentharm(),
        solver=multi_agent_pipeline(),
        scorer=scorers,
    )


@task
def compjailbench_none() -> Task:
    """Generation-only run. Saved trajectories are scored post-hoc."""
    return _make_task(defense_scorer=None)


@task
def compjailbench_g_safeguard() -> Task:
    """AgentHarm vs. the G-SafeGuard-inspired topology monitor."""
    return _make_task(defense_scorer=g_safeguard_scorer)


@task
def compjailbench_sentinel_agent() -> Task:
    """AgentHarm vs. the SentinelAgent-inspired oversight agent
    (arXiv:2505.24201)."""
    return _make_task(defense_scorer=sentinel_agent_scorer)


@task
def compjailbench_cot_monitor() -> Task:
    """AgentHarm vs. the CoT Monitor (arXiv:2507.11473 / arXiv:2510.23966)."""
    return _make_task(defense_scorer=cot_monitor_scorer)


@task
def compjailbench_probing() -> Task:
    """AgentHarm vs. the NARCBench-inspired activation probing defense
    (arXiv:2604.01151). Disabled unless config.ENABLE_PROBING=True and a
    local model + trained probe are configured -- see defenses/probing.py.
    """
    return _make_task(defense_scorer=probing_scorer)
