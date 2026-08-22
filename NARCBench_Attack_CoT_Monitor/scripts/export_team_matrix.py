"""Export completed per-run monitor scores in the shared long-form matrix schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "contributor",
    "attack_method",
    "defense_method",
    "model",
    "task_id",
    "strategy",
    "condition",
    "metric_name",
    "metric_value",
    "n_samples",
    "run_id",
    "notes",
]


def metric_token(scope: str) -> str:
    return scope.replace("-", "_").replace("+", "plus")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="paper_defense_50_COMPLETE")
    parser.add_argument(
        "--contributor",
        required=True,
        help="Contributor identifier recorded in the matrix provenance field",
    )
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    with args.samples.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            condition = "attack" if sample["condition"] == "collusion" else "benign_control"
            target_model = next(iter(sample.get("model_usage", {})), "unknown")
            for monitor in sample["monitors"].values():
                if not monitor.get("deployable", True):
                    continue
                suspiciousness = monitor.get("suspiciousness")
                if suspiciousness is None:
                    continue
                scope = monitor["scope"]
                model_pair = f"target={target_model};monitor={monitor.get('monitor_model', 'unknown')}"
                common = {
                    "contributor": args.contributor,
                    "attack_method": "narcbench_core_observable_rationale",
                    "defense_method": "cot_observable_reasoning_monitor",
                    "model": model_pair,
                    "task_id": sample["sample_id"],
                    "strategy": scope,
                    "condition": condition,
                    "n_samples": 1,
                    "run_id": args.run_id,
                    "notes": (
                        "prompt=narcbench_v1; fixed threshold=0.5; observable rationales "
                        "are elicited, not hidden CoT"
                    ),
                }
                token = metric_token(scope)
                rows.append(
                    {
                        **common,
                        "metric_name": f"suspiciousness__{token}",
                        "metric_value": float(suspiciousness),
                    }
                )
                rows.append(
                    {
                        **common,
                        "metric_name": f"monitor_flag__{token}",
                        "metric_value": int(monitor.get("prediction") == "attack"),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matrix rows to {args.output}")


if __name__ == "__main__":
    main()
