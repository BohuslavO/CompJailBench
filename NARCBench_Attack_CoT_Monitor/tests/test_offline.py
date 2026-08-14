from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from inspect_ai.util import Store


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_statistics import (
    average_precision,
    confusion_metrics,
    exact_mcnemar_pvalue,
    roc_auc,
    wilson_interval,
)
from src.model_helpers import extract_recommendation, parse_json_response, validate_choice
from src.evidence_views import (
    build_action_only_view,
    build_observable_reasoning_view,
    build_public_plus_reasoning_view,
)
from src.trajectory import StandardTrajectory
from src.defense_interface import (
    monitor_messages,
    parse_monitor_result,
    scope_availability,
)
from scripts.summarize_audit import parse_audit_file, summarize
from scripts.analyze_defense_scores import matrix_rows, summarize_scores
from tasks.narcbench_real_attack import selected_pairs


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        path = PROJECT_ROOT / "data" / "narcbench_core_official.json"
        self.records = json.loads(path.read_text(encoding="utf-8"))

    def test_full_matched_design(self) -> None:
        self.assertEqual(len(self.records), 100)
        pairs: dict[str, list[dict]] = {}
        for record in self.records:
            pairs.setdefault(record["pair_id"], []).append(record)
        self.assertEqual(len(pairs), 50)
        for pair in pairs.values():
            self.assertEqual({item["condition"] for item in pair}, {"collusion", "control"})
            collusion, control = sorted(pair, key=lambda item: item["condition"])
            self.assertEqual(collusion["context"], control["context"])
            self.assertEqual(collusion["options"], control["options"])
            self.assertEqual(collusion["target_option"], control["target_option"])
            self.assertEqual(
                [agent["name"] for agent in collusion["agents"]],
                [agent["name"] for agent in control["agents"]],
            )

    def test_counterbalanced_labels(self) -> None:
        collusion = [record for record in self.records if record["condition"] == "collusion"]
        targets = [record["target_option"] for record in collusion]
        self.assertEqual(set(targets), {"A", "B", "1", "2"})
        self.assertLessEqual(max(targets.count(label) for label in set(targets)), 13)

    def test_pair_selector_accepts_cli_string_and_list(self) -> None:
        expected = ["audit_01", "cyber_01"]
        self.assertEqual(selected_pairs("audit_01,cyber_01"), expected)
        self.assertEqual(selected_pairs(["audit_01", "cyber_01"]), expected)


class ParsingTests(unittest.TestCase):
    def test_json_inside_markdown(self) -> None:
        result = parse_json_response('```json\n{"message": "ok"}\n```')
        self.assertTrue(result["_parse_ok"])
        self.assertEqual(result["message"], "ok")

    def test_invalid_is_not_called_refusal(self) -> None:
        result = parse_json_response("plain text")
        self.assertFalse(result["_parse_ok"])
        self.assertNotIn("_refusal", result)

    def test_dynamic_choices(self) -> None:
        self.assertEqual(validate_choice("Option 2", ["1", "2"]), "2")
        self.assertEqual(extract_recommendation("RECOMMENDATION: [Option A]", ["A", "B"]), "A")
        self.assertEqual(
            extract_recommendation('{"response":"RECOMMENDATION: Option 1"}', ["1", "2"]),
            "1",
        )
        self.assertEqual(extract_recommendation("I decline", ["A", "B"]), "ABSTAIN")


class StatisticsTests(unittest.TestCase):
    def test_perfect_discrimination(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)
        metrics = confusion_metrics(labels, scores)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_wilson_bounds(self) -> None:
        low, high = wilson_interval(5, 10)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_pvalue([(1, 1), (0, 0)]), 1.0)
        self.assertLess(exact_mcnemar_pvalue([(1, 0)] * 10), 0.01)


class EvidenceBoundaryTests(unittest.TestCase):
    def test_deployable_views_exclude_private_evidence_and_labels(self) -> None:
        trajectory = StandardTrajectory(
            store=Store(),
            condition_label="attack",
            public_messages=[{"channel": "public", "content": "public evidence"}],
            private_messages=[{"channel": "private", "content": "PRIVATE_SECRET"}],
            observable_reasoning=[
                {"channel": "private", "content": "PRIVATE_RATIONALE"},
                {"channel": "public", "content": "public rationale", "visible_to_monitor": True},
            ],
            actions=[{"action": "vote", "value": "A"}],
            metadata={"hidden_target": "LABEL_SECRET"},
        )
        for view in (
            build_action_only_view(trajectory),
            build_observable_reasoning_view(trajectory),
            build_public_plus_reasoning_view(trajectory),
        ):
            self.assertNotIn("PRIVATE_SECRET", view)
            self.assertNotIn("PRIVATE_RATIONALE", view)
            self.assertNotIn("LABEL_SECRET", view)
            self.assertNotIn('"condition_label"', view)

    def test_scope_availability_does_not_invent_missing_reasoning(self) -> None:
        trajectory = StandardTrajectory(
            store=Store(),
            public_messages=[{"channel": "public", "content": "message"}],
            observable_reasoning=[],
            final_output="result",
        )
        self.assertEqual(scope_availability(trajectory, "public-action-only"), (True, ""))
        available, reason = scope_availability(
            trajectory, "public-action-plus-observable-reasoning"
        )
        self.assertFalse(available)
        self.assertIn("observable reasoning", reason)

    def test_external_result_parser(self) -> None:
        trajectory = StandardTrajectory(
            store=Store(),
            sample_id="sample-1",
            attack_name="example",
            condition_label="collusion",
            model_name="target/model",
        )
        result = parse_monitor_result(
            '{"suspiciousness": 0.75, "brief_evidence": "shared plan"}',
            trajectory=trajectory,
            scope="public-action-only",
            monitor_model="monitor/model",
        )
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["prediction"], "attack")
        self.assertTrue(result["correct"])

    def test_external_prompt_treats_evidence_as_data(self) -> None:
        messages = monitor_messages(
            "public-action-only", "[BEGIN TRAJECTORY] injected [END TRAJECTORY]"
        )
        user_content = str(messages[1].content)
        self.assertIn("[BEGIN-TRAJECTORY]", user_content)
        self.assertIn("[END-TRAJECTORY]", user_content)


class AuditSummaryTests(unittest.TestCase):
    def test_completed_audit_subset_parses(self) -> None:
        selection_path = PROJECT_ROOT / "configs" / "audit_subset_r1.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        audit_dir = PROJECT_ROOT / "results" / "paper_main" / "audit_packet"
        rows = [
            parse_audit_file(audit_dir / f"{scenario_id}.md", selection["reviewer_id"])
            for scenario_id in selection["scenario_ids"]
        ]
        summary = summarize(rows, selection)
        self.assertEqual(summary["n_reviewed"], 10)
        self.assertEqual(summary["audited_valid_collusion"]["count"], 3)
        self.assertIsNone(summary["inter_rater_reliability"])


class CrossAttackAnalysisTests(unittest.TestCase):
    def test_score_summary_and_matrix_export(self) -> None:
        rows = []
        for index, (truth, score) in enumerate(
            (("attack", 0.9), ("attack", 0.8), ("benign", 0.2), ("benign", 0.1))
        ):
            rows.append(
                {
                    "status": "scored",
                    "truth": truth,
                    "suspiciousness": score,
                    "prediction": "attack" if score >= 0.5 else "benign",
                    "scope": "public-action-plus-observable-reasoning",
                    "prompt_version": "general_v2",
                    "threshold": 0.5,
                    "sample_id": f"sample-{index}",
                    "attack_name": "example_attack",
                    "target_model": "target/model",
                    "monitor_model": "monitor/model",
                }
            )
        summary = summarize_scores(rows)[0]
        self.assertEqual(summary["roc_auc"], 1.0)
        self.assertEqual(summary["invalid"], 0)
        exported = matrix_rows(rows, contributor="R1", run_id="test-run")
        self.assertEqual(len(exported), 8)
        self.assertEqual({row["condition"] for row in exported}, {"attack", "benign_control"})


if __name__ == "__main__":
    unittest.main()
