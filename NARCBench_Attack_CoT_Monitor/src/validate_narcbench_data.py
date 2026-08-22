"""Standalone validation for the converted official NARCBench-Core data."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "narcbench_core_official.json"


def load_samples() -> list[dict[str, Any]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    samples = load_samples()
    if len(samples) != 100:
        raise ValueError(f"Expected 100 samples, found {len(samples)}")

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if len(sample["agents"]) != 4:
            raise ValueError(f"{sample['sample_id']} must contain four agents")
        if sample["condition"] not in {"collusion", "control"}:
            raise ValueError(f"{sample['sample_id']} has an invalid condition")
        if sample["target_option"] not in sample["options"]:
            raise ValueError(f"{sample['sample_id']} has an invalid target")
        if sample["condition"] == "collusion" and len(sample["colluders"]) != 2:
            raise ValueError(f"{sample['sample_id']} needs two colluders")
        if sample["condition"] == "control" and sample["colluders"]:
            raise ValueError(f"{sample['sample_id']} control contains colluders")
        pairs[sample["pair_id"]].append(sample)

    if len(pairs) != 50:
        raise ValueError(f"Expected 50 pairs, found {len(pairs)}")
    for pair_id, pair in pairs.items():
        if {item["condition"] for item in pair} != {"collusion", "control"}:
            raise ValueError(f"{pair_id} is not a complete matched pair")
        first, second = pair
        for key in ("context", "options", "target_option", "fair_choice"):
            if first[key] != second[key]:
                raise ValueError(f"{pair_id} differs on {key}")

    print("NARCBench-Core validation passed: 100 samples, 50 matched pairs, 17 domains.")


if __name__ == "__main__":
    main()
