from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai.dataset import MemoryDataset, Sample

from agentharm_data.loader import load_agentharm, load_paired_agentharm
from agents.pipeline import _split_reasoning_output
from scripts.export_agentharm_trajectories import load_and_convert, trace_to_standard


def fake_loader(condition: str, split: str, force_redownload: bool) -> MemoryDataset:
    samples = [
        Sample(
            id=sample_id,
            input=f"{condition} prompt {sample_id}",
            target="",
            metadata={"id": sample_id, "category": "Fraud"},
        )
        for sample_id in ("2-1", "2-2")
    ]
    return MemoryDataset(samples=samples, name=f"fake_{condition}_{split}")


def fake_trace(condition: str, pair_id: str = "2-1") -> dict:
    return {
        "trace_id": f"trace-{condition}",
        "sample_id": pair_id,
        "original_prompt": f"{condition} request",
        "nodes": [
            {
                "node_id": "planner",
                "node_type": "planner",
                "agent_name": "PlannerAgent",
                "output": "SUBTASK: research",
                "reasoning": "I will decompose the request.",
                "tool_calls": [],
            },
            {
                "node_id": "aggregator",
                "node_type": "aggregator",
                "agent_name": "AggregatorAgent",
                "output": "final",
                "reasoning": "I will combine the worker results.",
                "tool_calls": [],
            },
        ],
        "edges": [
            {
                "source": "planner",
                "target": "aggregator",
                "edge_type": "aggregation",
                "payload": "result",
            }
        ],
        "final_output": "final",
        "metadata": {
            "run_id": "test-run",
            "condition_label": condition,
            "pair_id": pair_id,
            "agentharm_condition": "harmful" if condition == "attack" else "benign",
            "agentharm_split": "val",
            "category_original": "Fraud",
            "agent_model": "mockllm/model",
            "is_placeholder": False,
        },
    }


class AgentHarmLoaderTests(unittest.TestCase):
    def test_harmful_and_benign_are_normalised_and_paired(self) -> None:
        harmful, benign = load_paired_agentharm(split="val", limit=2, loader=fake_loader)
        self.assertEqual([s.metadata["pair_id"] for s in harmful], ["2-1", "2-2"])
        self.assertEqual([s.metadata["pair_id"] for s in benign], ["2-1", "2-2"])
        self.assertEqual(harmful[0].metadata["condition_label"], "attack")
        self.assertEqual(benign[0].metadata["condition_label"], "benign_control")
        self.assertEqual(benign[0].metadata["category"], "benign")
        self.assertFalse(harmful[0].metadata["is_placeholder"])

    def test_loader_fails_closed(self) -> None:
        def broken_loader(condition: str, split: str, force_redownload: bool):
            raise ImportError("simulated failure")

        with self.assertRaisesRegex(RuntimeError, "stopped before any model call"):
            load_agentharm("harmful", "val", 1, loader=broken_loader)

    def test_huggingface_datasets_is_not_shadowed(self) -> None:
        module = importlib.import_module("datasets.exceptions")
        self.assertIn("site-packages", str(Path(module.__file__)))


class PipelineParsingTests(unittest.TestCase):
    def test_reasoning_without_output_is_not_used_as_action(self) -> None:
        reasoning, output = _split_reasoning_output(
            "<reasoning>analysis only</reasoning>"
        )
        self.assertEqual(reasoning, "analysis only")
        self.assertEqual(output, "")

    def test_plain_refusal_remains_a_visible_output(self) -> None:
        reasoning, output = _split_reasoning_output("I cannot help with that request.")
        self.assertEqual(reasoning, "")
        self.assertEqual(output, "I cannot help with that request.")


class AgentHarmExporterTests(unittest.TestCase):
    def test_export_scope_and_labels(self) -> None:
        converted = trace_to_standard(fake_trace("attack"))
        self.assertEqual(converted["schema_version"], "2.0")
        self.assertEqual(converted["sample_id"], "attack__2-1")
        self.assertEqual(converted["pair_id"], "2-1")
        self.assertEqual(converted["public_task"]["context"], "attack request")
        self.assertEqual(len(converted["public_messages"]), 2)
        self.assertEqual(len(converted["observable_reasoning"]), 2)
        self.assertEqual(converted["communication_edges"][0]["channel"], "public")
        self.assertFalse(converted["metadata"]["official_agentharm_tools_executed"])

    def test_export_requires_matched_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attack.json").write_text(json.dumps(fake_trace("attack")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched harmful and benign"):
                load_and_convert(root)
            (root / "benign.json").write_text(
                json.dumps(fake_trace("benign_control")), encoding="utf-8"
            )
            converted = load_and_convert(root)
            self.assertEqual(len(converted), 2)


if __name__ == "__main__":
    unittest.main()
