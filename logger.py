from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

NodeType = Literal["planner", "worker", "aggregator", "tool", "memory"]
EdgeType = Literal["delegation", "tool_call", "memory_write", "memory_read", "aggregation"]


@dataclass
class AgentNode:
    node_id: str
    node_type: NodeType
    agent_name: str
    # Final visible output of this node (what gets passed downstream).
    output: str = ""
    # Explicit reasoning / chain-of-thought this node produced before acting,
    # if PIPELINE_CONFIG.require_visible_reasoning is True. This is the
    # primary artifact the CoT Monitor defense scores.
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEdge:
    edge_id: str
    edge_type: EdgeType
    source: str  # node_id
    target: str  # node_id
    payload: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionTrace:
    """Full record of one multi-agent run against one AgentHarm sample."""

    trace_id: str
    sample_id: str
    original_prompt: str
    nodes: list[AgentNode] = field(default_factory=list)
    edges: list[AgentEdge] = field(default_factory=list)
    final_output: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: AgentNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: AgentEdge) -> None:
        self.edges.append(edge)

    def all_reasoning(self) -> str:
        """Concatenate every node's visible reasoning, in execution order.

        This is what the CoT Monitor defense reads.
        """
        parts = []
        for n in self.nodes:
            if n.reasoning:
                parts.append(f"[{n.node_type}:{n.agent_name}]\n{n.reasoning}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "sample_id": self.sample_id,
            "original_prompt": self.original_prompt,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "final_output": self.final_output,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
        }


def new_trace(sample_id: str, original_prompt: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        sample_id=sample_id,
        original_prompt=original_prompt,
    )


def save_trace(trace: ExecutionTrace, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{trace.trace_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)
    return out_path


def load_trace(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
