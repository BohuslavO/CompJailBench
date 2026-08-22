"""
Converts MY DeCompBench trajectory into Atharv's execution_trace shape
(compjailbench/monitors.py -- AgentHarmMonitor, GraphMonitor), so I can
test my attack against his defense specifically (team agreement: each
person tests their own attack against all 4 defenses -- this is the
opposite direction from trajectory_converters.py's from_execution_trace,
which converts HIS attack's output, not mine).

Edge structure choice, worth being explicit about: turns are linked as
a sequential chain (turn_1 -> turn_2 -> turn_3...), NOT a star from a
synthetic "orchestrator" node. GraphMonitor's structural heuristic flags
fan-out > 4 as suspicious -- a star topology from one central dispatcher
node would trip that on almost every DeCompBench task (most have more
than 4 turns), which would be an artifact of THIS converter's
representation choice, not a real signal about the attack. A sequential
chain keeps max out-degree at 1, avoiding a synthetic false positive
that has nothing to do with whether the attack actually did anything
suspicious.
"""
from typing import Any


def to_execution_trace(
    trajectory: list[dict],
    task_description: str,
    sample_id: Any = None,
    sample_name: str = None,
) -> dict:
    """trajectory: the list[dict] pipeline.py's run_task() returns.
    task_description: the overall harmful task text -- not present per-turn
    in my trajectory format, must be supplied (e.g. DeCompTask.title, or
    the first turn's user_prompt as a fallback approximation)."""
    nodes = []
    edges = []
    prev_node_id = None

    for turn in trajectory:
        node_id = f"turn{turn['turn_id']}_{turn['agent_name']}"
        nodes.append({
            "id": node_id,
            "type": "agent",
            "name": turn["agent_name"],
            "input": turn["user_prompt"],
            "output": turn["response"],
        })
        if prev_node_id is not None:
            edges.append({"source": prev_node_id, "target": node_id, "type": "task_handoff"})
        prev_node_id = node_id

    final_output = trajectory[-1]["response"] if trajectory else ""

    return {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "task": task_description,
        "nodes": nodes,
        "edges": edges,
        "final_output": final_output,
    }
