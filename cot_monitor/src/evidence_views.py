"""Construct label-free monitor inputs from a StandardTrajectory.

(fetched verbatim from NARCBench_Attack_CoT_Monitor/src/evidence_views.py.)
"""

import json
from typing import Any

from src.trajectory import StandardTrajectory


def format_as_text(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _public_reasoning(trajectory: StandardTrajectory) -> list[dict[str, Any]]:
    return [
        item
        for item in trajectory.observable_reasoning
        if item.get("channel") in {"public", "final"}
        and item.get("visible_to_monitor", True)
    ]


def build_action_only_view(trajectory: StandardTrajectory) -> str:
    return format_as_text(
        {
            "public_messages": trajectory.public_messages,
            "actions": trajectory.actions,
            "tool_calls": trajectory.tool_calls,
            "final_output": trajectory.final_output,
        }
    )


def build_observable_reasoning_view(trajectory: StandardTrajectory) -> str:
    return format_as_text({"observable_reasoning": _public_reasoning(trajectory)})


def build_public_plus_reasoning_view(trajectory: StandardTrajectory) -> str:
    return format_as_text(
        {
            "public_messages": trajectory.public_messages,
            "observable_reasoning": _public_reasoning(trajectory),
            "actions": trajectory.actions,
            "tool_calls": trajectory.tool_calls,
            "final_output": trajectory.final_output,
        }
    )


def build_privileged_audit_view(trajectory: StandardTrajectory) -> str:
    return format_as_text(
        {
            "public_messages": trajectory.public_messages,
            "private_messages": trajectory.private_messages,
            "observable_reasoning": trajectory.observable_reasoning,
            "actions": trajectory.actions,
            "tool_calls": trajectory.tool_calls,
            "final_output": trajectory.final_output,
        }
    )


build_cot_only_view = build_observable_reasoning_view
build_all_messages_view = build_public_plus_reasoning_view
