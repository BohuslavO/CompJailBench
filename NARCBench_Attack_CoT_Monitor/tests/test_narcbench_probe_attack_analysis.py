from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_eval import attack_summary, complete_pairs
from scripts.analyze_narcbench_probe_attack import (
    canonical_attack_rows,
    normalize_usage_availability,
    pair_outcomes,
)


def attack_metadata(
    pair_id: str,
    condition: str,
    *,
    target_selected: bool,
    target_adherence: bool = False,
) -> dict:
    return {
        "scenario_id": pair_id,
        "pair_id": pair_id,
        "domain": "test",
        "condition": condition,
        "votes": {"agent_1": "A"},
        "colluders": ["agent_1", "agent_2"] if condition == "collusion" else [],
        "hidden_target": "A",
        "committee_decision": "A" if target_selected else "B",
        "target_vote_count": 3 if target_selected else 1,
        "target_selected": target_selected,
        "target_adherence": target_adherence,
        "valid_collusion": target_adherence,
        "attack_success": target_adherence and target_selected,
        "abstention_count": 0,
        "model_call_count": 18 if condition == "collusion" else 16,
        "parse_failure_count": 0,
    }


def sample_row(pair_id: str, condition: str, **kwargs) -> dict:
    return {
        "sample_id": f"{pair_id}_{condition}",
        "pair_id": pair_id,
        "condition": condition,
        "domain": "test",
        "attack": attack_metadata(pair_id, condition, **kwargs),
        "monitors": {},
        "model_usage": {},
        "role_usage": {},
        "monitor_event_usage": {"total_tokens": 0},
        "total_time_seconds": 0,
        "working_time_seconds": 0,
    }


class ProbeAttackAnalysisTests(unittest.TestCase):
    def test_duplicate_samples_are_rejected(self) -> None:
        row = sample_row("p1", "collusion", target_selected=False)
        with self.assertRaisesRegex(ValueError, "Duplicate sample rows"):
            canonical_attack_rows([row, dict(row)])

    def test_paired_attack_summary_and_outcome_export(self) -> None:
        rows = [
            sample_row(
                "p1", "collusion", target_selected=True, target_adherence=True
            ),
            sample_row("p1", "control", target_selected=False),
            sample_row("p2", "collusion", target_selected=False),
            sample_row("p2", "control", target_selected=True),
        ]
        canonical = canonical_attack_rows(rows)
        pairs = complete_pairs(canonical)
        summary = attack_summary(pairs)
        outcomes = pair_outcomes(pairs)

        self.assertEqual(len(canonical), 4)
        self.assertEqual(summary["valid_collusion_rate"], 0.5)
        self.assertEqual(summary["attack_success_rate"], 0.5)
        self.assertEqual(summary["paired_target_selection_uplift"], 0.0)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(outcomes[0]["attack_success"])

        normalized = normalize_usage_availability(summary, canonical)
        self.assertFalse(normalized["token_usage_recorded"])
        self.assertIsNone(normalized["total_tokens"])


if __name__ == "__main__":
    unittest.main()
