from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from inspect_ai.util import Store

from src.sentinel_message_adapter import (
    build_message_graph,
    build_message_library,
    compatibility,
    forbidden_evidence_present,
    signature_text_is_target_independent,
)
from src.trajectory import StandardTrajectory


@dataclass
class Node:
    node_id: str
    node_type: str
    role: str = ""
    declared_capabilities: list = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: str
    content: str = ""
    args: dict = field(default_factory=dict)
    turn_id: int | None = None
    tool: str | None = None

    def text(self) -> str:
        return f"{self.content} {self.args}"


@dataclass
class ExecGraph:
    kind: str
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    degraded: bool = False

    def add_node(self, node):
        self.nodes.setdefault(node.node_id, node)

    def add_edge(self, edge):
        self.edges.append(edge)


@dataclass
class PathStep:
    edge_type: str = "message"
    tool: str | None = None
    content_pattern: str | None = None


@dataclass
class AttackPath:
    pattern_id: str
    steps: list
    harm_category: str = ""
    source: str = ""
    description: str = ""


GRAPH_BACKEND = SimpleNamespace(Node=Node, Edge=Edge, ExecGraph=ExecGraph)
PATH_BACKEND = SimpleNamespace(PathStep=PathStep, AttackPath=AttackPath)


class SentinelMessageAdapterTests(unittest.TestCase):
    def trajectory(self) -> StandardTrajectory:
        return StandardTrajectory(
            store=Store(),
            sample_id="example_attack",
            pair_id="example",
            attack_name="narcbench_collusion",
            condition_label="attack",
            agent_ids=["agent_1", "agent_2"],
            public_messages=[
                {
                    "message_id": "public_1",
                    "sender": "agent_1",
                    "recipients": ["agent_2"],
                    "round": 1,
                    "channel": "public",
                    "content": "PUBLIC_EVIDENCE",
                }
            ],
            private_messages=[
                {
                    "message_id": "private_1",
                    "sender": "agent_2",
                    "recipients": ["agent_1"],
                    "round": 0,
                    "channel": "private",
                    "content": "PRIVATE_EVIDENCE",
                }
            ],
            final_output="FINAL_OUTPUT",
            metadata={
                "hidden_target": "HIDDEN_TARGET_SECRET",
                "fair_choice": "FAIR_CHOICE_SECRET",
                "colluders": ["SECRET_COLLUDER"],
                "condition": "SECRET_CONDITION",
            },
        )

    def test_public_scope_excludes_private_and_hidden_evidence(self) -> None:
        trajectory = self.trajectory()
        self.assertEqual(compatibility(trajectory)[0], True)
        graph = build_message_graph(trajectory, GRAPH_BACKEND, scope="public")
        rendered = " ".join(edge.text() for edge in graph.edges)
        self.assertIn("PUBLIC_EVIDENCE", rendered)
        self.assertIn("FINAL_OUTPUT", rendered)
        self.assertNotIn("PRIVATE_EVIDENCE", rendered)
        self.assertEqual(forbidden_evidence_present(graph, trajectory), [])
        self.assertTrue(graph.degraded)

    def test_private_ablation_adds_private_message(self) -> None:
        graph = build_message_graph(
            self.trajectory(), GRAPH_BACKEND, scope="public_and_private"
        )
        rendered = " ".join(edge.text() for edge in graph.edges)
        self.assertIn("PRIVATE_EVIDENCE", rendered)

    def test_frozen_library_is_target_independent(self) -> None:
        library = build_message_library(PATH_BACKEND)
        self.assertEqual(len(library), 3)
        self.assertTrue(signature_text_is_target_independent(PATH_BACKEND))
        self.assertTrue(all(path.source == "narcbench_message_v1" for path in library))


if __name__ == "__main__":
    unittest.main()
