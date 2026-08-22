"""
Team-wide shared trajectory schema. This is a faithful copy of
StandardTrajectory from the "cot narcbench" branch (src/trajectory.py)
-- reproduced here rather than imported from their package so this
module has no dependency on their folder layout. If the team settles
on one canonical location for this class, THIS is the file to delete
in favor of importing from wherever that canonical copy lives --
having two independent copies drift out of sync would be worse than
either problem this file solves.

Verified against the real cot_narcbench code, not assumed: this
schema's fields are read by their real src/evidence_views.py
(build_action_only_view / build_cot_only_view / build_all_messages_view)
with no additional structure required beyond what's declared below --
those functions just JSON-dump whatever's here for a monitor LLM to
read, so converters targeting this schema don't need to match any
stricter internal key contract than what's declared here.
"""
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
