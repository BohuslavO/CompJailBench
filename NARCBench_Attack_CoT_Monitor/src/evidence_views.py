import json
from typing import Any

from src.trajectory import StandardTrajectory


def format_as_text(data: dict[str, Any]) -> str:
    """
    Convert structured Python data into readable JSON text.

    The future monitor LLM will receive this text as evidence.
    """
    return json.dumps(
        data,
        ensure_ascii=False,  # Preserve non-English characters correctly.
        indent=2,            # Make the evidence easier to read and debug.
    )


def build_action_only_view(trajectory: StandardTrajectory) -> str:
    """
    Give the monitor observable behavior but no reasoning traces.
    """
    return format_as_text(
        {
            "messages": trajectory.messages,
            "actions": trajectory.actions,
            "tool_calls": trajectory.tool_calls,
            "final_output": trajectory.final_output,
        }
    )


def build_cot_only_view(trajectory: StandardTrajectory) -> str:
    """
    Give the monitor observable reasoning traces only.
    """
    return format_as_text(
        {
            "reasoning_traces": trajectory.reasoning_traces,
        }
    )


def build_all_messages_view(trajectory: StandardTrajectory) -> str:
    """
    Give the monitor both observable reasoning and behavior.
    """
    return format_as_text(
        {
            "messages": trajectory.messages,
            "reasoning_traces": trajectory.reasoning_traces,
            "actions": trajectory.actions,
            "tool_calls": trajectory.tool_calls,
            "final_output": trajectory.final_output,
        }
    )