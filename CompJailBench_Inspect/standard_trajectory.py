from typing import Any

from inspect_ai.util import StoreModel
from pydantic import Field


class StandardTrajectory(StoreModel):
    sample_id: str = ""
    attack_name: str = ""
    condition_label: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_traces: dict[str, str] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_output: str = ""
    model_name: str = ""
    agent_ids: list[str] = Field(default_factory=list)
    communication_edges: list[dict[str, Any]] = Field(default_factory=list)
    activation_refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
