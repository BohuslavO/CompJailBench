from typing import Any

from standard_trajectory import StandardTrajectory


def from_decompbench_trajectory(
    trajectory: list[dict],
    sample_id: str,
    condition_label: str,
    model_name: str,
    attack_name: str = "decompbench_routing",
    extra_metadata: dict = None,
) -> StandardTrajectory:
    messages = []
    actions = []
    agent_ids = []
    edges = []
    for turn in trajectory:
        agent_name = turn["agent_name"]
        if agent_name not in agent_ids:
            agent_ids.append(agent_name)
        messages.append({
            "message_id": f"turn_{turn['turn_id']}",
            "sender": agent_name,
            "recipients": [],
            "round": turn["turn_id"],
            "channel": "public",
            "content": turn["response"],
            "instruction": turn["user_prompt"],
            "label": turn["label"],
        })
        actions.append({
            "agent_id": agent_name,
            "action": "complete_turn",
            "turn_id": turn["turn_id"],
            "label": turn["label"],
        })


        edges.append({"source": "orchestrator", "target": agent_name, "type": "turn_assignment", "turn_id": turn["turn_id"]})

    final_output = trajectory[-1]["response"] if trajectory else ""

    metadata = {"strategy": None, "n_turns": len(trajectory)}
    if extra_metadata:
        metadata.update(extra_metadata)

    return StandardTrajectory(
        sample_id=sample_id,
        attack_name=attack_name,
        condition_label=condition_label,
        messages=messages,
        reasoning_traces={},
        actions=actions,
        tool_calls=[],
        final_output=final_output,
        model_name=model_name,
        agent_ids=agent_ids,
        communication_edges=edges,
        activation_refs=[],
        metadata=metadata,
    )


def from_execution_trace(
    execution_trace: dict,
    sample_id: str = None,
    condition_label: str = "",
    model_name: str = "",
    attack_name: str = "agentharm",
) -> StandardTrajectory:
    messages = []
    actions = []
    agent_ids = []
    for node in execution_trace.get("nodes", []):
        node_id = node.get("id", node.get("name", "unknown"))
        if node_id not in agent_ids:
            agent_ids.append(node_id)
        messages.append({
            "message_id": node_id,
            "sender": node_id,
            "recipients": [],
            "round": None,
            "channel": "public",
            "content": node.get("output", ""),
            "instruction": node.get("input", ""),
            "node_type": node.get("type"),
        })
        actions.append({
            "agent_id": node_id,
            "action": node.get("type", "unknown"),
            "duration_seconds": node.get("duration_seconds"),
        })

    edges = [
        {"source": e.get("source"), "target": e.get("target"), "type": e.get("type")}
        for e in execution_trace.get("edges", [])
    ]

    return StandardTrajectory(
        sample_id=sample_id or str(execution_trace.get("sample_id", "")),
        attack_name=attack_name,
        condition_label=condition_label,
        messages=messages,
        reasoning_traces={},
        actions=actions,
        tool_calls=[],
        final_output=execution_trace.get("final_output", ""),
        model_name=model_name,
        agent_ids=agent_ids,
        communication_edges=edges,
        activation_refs=[],
        metadata={"task": execution_trace.get("task"), "sample_name": execution_trace.get("sample_name")},
    )
