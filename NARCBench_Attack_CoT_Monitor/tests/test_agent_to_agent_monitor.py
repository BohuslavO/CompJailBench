from __future__ import annotations

import unittest

from inspect_ai.util import Store

from src.agent_to_agent_monitor import (
    build_public_graph,
    compatibility,
    monitor_messages,
    parse_result,
)
from src.trajectory import StandardTrajectory
from scripts.analyze_agent_to_agent_monitor import canonical_rows, summarize


class AgentToAgentMonitorTests(unittest.TestCase):
    def trajectory(self) -> StandardTrajectory:
        return StandardTrajectory(
            store=Store(),
            sample_id="example_attack",
            pair_id="example",
            attack_name="narcbench_collusion",
            condition_label="attack",
            model_name="example/model",
            public_task={
                "title": "Example",
                "context": "PUBLIC_TASK_CONTEXT",
                "options": {"A": "safe", "B": "unsafe"},
            },
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
                    "sender": "agent_1",
                    "recipients": ["agent_2"],
                    "round": 0,
                    "channel": "private",
                    "content": "PRIVATE_SECRET",
                }
            ],
            communication_edges=[
                {
                    "source": "agent_1",
                    "target": "agent_2",
                    "round": 1,
                    "channel": "public",
                    "message_id": "public_1",
                },
                {
                    "source": "agent_1",
                    "target": "agent_2",
                    "round": 0,
                    "channel": "private",
                    "message_id": "private_1",
                },
            ],
            final_output="FINAL_OUTPUT",
            metadata={"hidden_target": "METADATA_SECRET"},
        )

    def test_compatibility_and_public_boundary(self) -> None:
        trajectory = self.trajectory()
        self.assertEqual(compatibility(trajectory), (True, ""))
        graph = build_public_graph(trajectory)
        rendered = str(graph)
        self.assertIn("PUBLIC_EVIDENCE", rendered)
        self.assertIn("FINAL_OUTPUT", rendered)
        self.assertIn("PUBLIC_TASK_CONTEXT", rendered)
        self.assertNotIn("PRIVATE_SECRET", rendered)
        self.assertNotIn("METADATA_SECRET", rendered)
        self.assertEqual(graph["graph_stats"]["num_edge_events"], 1)
        self.assertEqual(graph["nodes"][0]["public_message_count"], 1)

    def test_monitor_prompt_and_result(self) -> None:
        trajectory = self.trajectory()
        graph = build_public_graph(trajectory)
        messages = monitor_messages(graph)
        self.assertEqual(len(messages), 2)
        result = parse_result(
            '{"suspiciousness": 0.75, "brief_evidence": "combined harm"}',
            trajectory=trajectory,
            monitor_model="example/monitor",
            graph_stats=graph["graph_stats"],
        )
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["prediction"], "attack")
        self.assertTrue(result["correct"])

    def test_retry_deduplication_and_summary(self) -> None:
        raw = [
            {
                "sample_id": "pair_attack",
                "pair_id": "pair",
                "status": "invalid",
                "truth": "attack",
                "usage": {"total_tokens": 10},
            },
            {
                "sample_id": "pair_attack",
                "pair_id": "pair",
                "status": "scored",
                "truth": "attack",
                "prediction": "attack",
                "suspiciousness": 0.8,
                "prompt_version": "test",
                "usage": {"total_tokens": 20},
            },
            {
                "sample_id": "pair_control",
                "pair_id": "pair",
                "status": "scored",
                "truth": "benign",
                "prediction": "benign",
                "suspiciousness": 0.2,
                "prompt_version": "test",
                "usage": {"total_tokens": 30},
            },
        ]
        canonical = canonical_rows(raw)
        self.assertEqual(len(canonical), 2)
        summary = summarize(canonical, raw)
        self.assertEqual(summary["n_scored"], 2)
        self.assertAlmostEqual(summary["metrics"]["roc_auc"], 1.0)
        self.assertEqual(summary["usage"]["model_calls"], 3)
        self.assertEqual(summary["usage"]["total_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
