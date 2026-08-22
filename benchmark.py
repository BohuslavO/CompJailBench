from __future__ import annotations

from inspect_ai import Task, task

from agents import multi_agent_pipeline
from datasets.agentharm_loader import load_agentharm
from defenses import (
    cot_monitor_scorer,
    g_safeguard_scorer,
    probing_scorer,
    sentinel_agent_scorer,
)
from defenses.task_success import task_success_scorer


def _make_task(defense_scorer=None) -> Task:
    scorers = [task_success_scorer()]
    if defense_scorer is not None:
        scorers.append(defense_scorer())
    return Task(
        dataset=load_agentharm(),
        solver=multi_agent_pipeline(),
        scorer=scorers,
    )


@task
def compjailbench_none() -> Task:
    """No defense -- baseline attack success rate with only the raw
    multi-agent pipeline and no monitor at all."""
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