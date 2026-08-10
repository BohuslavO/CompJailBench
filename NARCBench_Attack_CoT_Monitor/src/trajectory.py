from typing import Any

from inspect_ai.util import StoreModel
from pydantic import Field


class StandardTrajectory(StoreModel):
    """
    Common data format shared by every attack and defense.

    Attack solver:
        writes messages, reasoning, actions, and results here.

    Defense scorer:
        reads the same information without depending on one specific attack.
    """

    # Identifies the exact benchmark case.
    sample_id: str = ""

    # Examples: "narcbench", "sif", "agentharm".
    attack_name: str = ""

    # Ground truth, normally "attack" or "benign".
    condition_label: str = ""

    # Publicly observable messages exchanged by agents.
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # Only observable reasoning or generated rationales.
    # Do not store or claim access to hidden internal thoughts.
    reasoning_traces: dict[str, str] = Field(default_factory=dict)

    # Decisions, votes, file operations, or other agent actions.
    actions: list[dict[str, Any]] = Field(default_factory=list)

    # Tool calls made by agents, when the attack uses tools.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Final result produced by the full multi-agent workflow.
    final_output: str = ""

    # Name of the model that generated this trajectory.
    # Important for reproducibility and activation-based probing.
    model_name: str = ""

    # IDs of all agents that participated in this run.
    agent_ids: list[str] = Field(default_factory=list)

    # Communication structure between agents.
    # G-SafeGuard and SentinelAgent can use this to construct graphs.
    communication_edges: list[dict[str, Any]] = Field(default_factory=list)

    # References to saved hidden-state activations, when available.
    # Do NOT put giant activation arrays directly inside the Inspect log.
    # Instead, save them separately and store their file references here.
    activation_refs: list[dict[str, Any]] = Field(default_factory=list)

    # Information needed only by a particular attack.
    # For NARCBench: colluders, hidden target, votes, committee result.
    metadata: dict[str, Any] = Field(default_factory=dict)