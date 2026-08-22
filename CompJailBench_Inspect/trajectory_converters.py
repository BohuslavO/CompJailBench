"""
Converters from each team member's own attack trajectory format into
the shared StandardTrajectory (standard_trajectory.py), so any
StandardTrajectory-reading defense -- e.g. cot_narcbench's
cot_monitor.py -- can score any team attack's output, not just its own.

Two converters here, with very different fidelity:

from_decompbench_trajectory(): full fidelity. Every field DeCompBench
routing trajectories have maps cleanly onto StandardTrajectory.

from_execution_trace(): LOSSY, by construction, not by a bug in this
converter. Atharv's pipeline (compjailbench/agents.py::_timed_call)
discards system_prompt before it ever reaches execution_trace -- no
converter can recover data that was never captured at the source. This
converter carries that gap forward honestly (reasoning_traces stays
empty, messages won't include what instruction each node was actually
given) rather than silently inventing something plausible-looking.
"""
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
    """trajectory: the list[dict] pipeline.py's run_task() returns --
    each turn has agent_name/turn_id/label/system_prompt/user_prompt/response.
    condition_label: "attack" or "benign_control", whatever this run used.

    Known gap, not a bug: reasoning_traces stays empty. DeCompBench
    routing doesn't capture a separate chain-of-thought from the final
    response -- call_llm returns one string, not response+reasoning
    split apart. cot_narcbench's cot_only_monitor condition will see
    nothing for trajectories converted this way. If Qwen3's thinking
    output ever gets captured separately, wire it in here.
    """
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
        # No real inter-agent communication happens in this attack --
        # a central pipeline assigns each turn to an agent, agents don't
        # message each other. Representing this honestly as
        # orchestrator -> agent edges, not synthetic agent -> agent
        # chatter that didn't happen.
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
        reasoning_traces={},  # see docstring -- genuinely empty, not omitted by mistake
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
    attack_name: str = "atharv_decomposition",
) -> StandardTrajectory:
    """execution_trace: the dict benchmark.py builds -- {sample_id,
    sample_name, task, started_at, completed_at, nodes, edges, final_output}.
    Each node: {id, type, name, input, output, started_at, completed_at,
    duration_seconds}. Each edge: {source, target, type}.

    LOSSY, structurally, not something this converter can fix:
    - system_prompt: never captured at the source (agents.py::_timed_call
      discards it after the model call). Every message here will be
      missing what instruction each agent was actually working under
      beyond its task input -- this needs fixing in Atharv's code, not here.
    - reasoning_traces: his pipeline doesn't separate reasoning from
      output at all -- stays empty, same real gap as the DeCompBench
      converter, for a different underlying reason.
    - condition_label / sample_id: execution_trace doesn't carry a
      ground-truth attack/benign label at all (his pipeline doesn't have
      a benign_control condition), so these must be passed in by the
      caller rather than read off the trace.
    """
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
            "instruction": node.get("input", ""),  # NOT the real system_prompt -- see docstring
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
        reasoning_traces={},  # not captured by this pipeline at all
        actions=actions,
        tool_calls=[],
        final_output=execution_trace.get("final_output", ""),
        model_name=model_name,
        agent_ids=agent_ids,
        communication_edges=edges,
        activation_refs=[],
        metadata={"task": execution_trace.get("task"), "sample_name": execution_trace.get("sample_name")},
    )
