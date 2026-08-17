"""Attack/defense interchange schema used by the CompJailBreak experiments.

(fetched verbatim from NARCBench_Attack_CoT_Monitor/src/trajectory.py, main
branch, schema_version 2.0 -- not modified.)
"""

from typing import Any

from inspect_ai.util import StoreModel
from pydantic import Field


class StandardTrajectory(StoreModel):
    schema_version: str = "2.0"
    sample_id: str = ""
    pair_id: str = ""
    attack_name: str = ""
    condition_label: str = ""

    public_task: dict[str, Any] = Field(default_factory=dict)

    messages: list[dict[str, Any]] = Field(default_factory=list)
    public_messages: list[dict[str, Any]] = Field(default_factory=list)
    private_messages: list[dict[str, Any]] = Field(default_factory=list)

    observable_reasoning: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_traces: dict[str, str] = Field(default_factory=dict)

    actions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_output: str = ""
    model_name: str = ""
    agent_ids: list[str] = Field(default_factory=list)
    communication_edges: list[dict[str, Any]] = Field(default_factory=list)
    activation_refs: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
