"""Attack/defense interchange schema used by the CompJailBreak experiments."""

from typing import Any

from inspect_ai.util import StoreModel
from pydantic import Field


class StandardTrajectory(StoreModel):
    """Structured trajectory with explicit observability boundaries.

    ``messages`` and ``reasoning_traces`` are retained for compatibility with
    the team's earlier schema. New defenses should consume the channel-specific
    fields so private coordination cannot accidentally leak into a deployable
    monitor condition.
    """

    schema_version: str = "2.0"
    sample_id: str = ""
    pair_id: str = ""
    attack_name: str = ""
    condition_label: str = ""

    # Label-free task information visible to every participant. This must not
    # contain attack targets, colluder identities, or evaluation labels.
    public_task: dict[str, Any] = Field(default_factory=dict)

    # Compatibility aggregate. It may contain private data, so deployable
    # monitors must use public_messages instead.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    public_messages: list[dict[str, Any]] = Field(default_factory=list)
    private_messages: list[dict[str, Any]] = Field(default_factory=list)

    # Elicited, evaluator-visible rationales—not hidden chain of thought.
    observable_reasoning: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_traces: dict[str, str] = Field(default_factory=dict)

    actions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_output: str = ""
    model_name: str = ""
    agent_ids: list[str] = Field(default_factory=list)
    communication_edges: list[dict[str, Any]] = Field(default_factory=list)
    activation_refs: list[dict[str, Any]] = Field(default_factory=list)

    # Defenses must not read this field: it contains labels and hidden targets.
    metadata: dict[str, Any] = Field(default_factory=dict)
