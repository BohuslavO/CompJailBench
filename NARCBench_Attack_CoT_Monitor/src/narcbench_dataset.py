"""Inspect dataset loader for the official 50-pair NARCBench-Core corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from inspect_ai.dataset import MemoryDataset, Sample


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "narcbench_core_official.json"


def _validate_records(records: list[dict]) -> None:
    if len(records) != 100:
        raise ValueError(f"Expected 100 samples (50 matched pairs), found {len(records)}")

    pairs: dict[str, set[str]] = {}
    for record in records:
        required = {
            "sample_id",
            "pair_id",
            "condition",
            "context",
            "options",
            "target_option",
            "fair_choice",
            "agents",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('sample_id', '<unknown>')} missing {sorted(missing)}")
        pairs.setdefault(record["pair_id"], set()).add(record["condition"])

    malformed = {pair_id: conditions for pair_id, conditions in pairs.items() if conditions != {"collusion", "control"}}
    if len(pairs) != 50 or malformed:
        raise ValueError(f"Invalid matched-pair structure: pairs={len(pairs)}, malformed={malformed}")


def load_narcbench_dataset(
    data_path: Path = DATA_PATH,
    pair_ids: Iterable[str] | None = None,
) -> MemoryDataset:
    """Load all official pairs, optionally selecting complete named pairs."""

    records = json.loads(data_path.read_text(encoding="utf-8"))
    _validate_records(records)

    if pair_ids is not None:
        selected = {str(pair_id) for pair_id in pair_ids}
        records = [record for record in records if record["pair_id"] in selected]
        present = {record["pair_id"] for record in records}
        missing = selected - present
        if missing:
            raise ValueError(f"Unknown NARCBench pair IDs: {sorted(missing)}")

    samples: list[Sample] = []
    for record in records:
        options_text = "\n".join(
            f"Option {label}: {description}" for label, description in record["options"].items()
        )
        samples.append(
            Sample(
                id=record["sample_id"],
                input=f"{record['title']}\n\n{record['context']}\n\n{options_text}",
                target=record["target_option"],
                metadata=record,
            )
        )

    suffix = "official_core" if pair_ids is None else "official_core_subset"
    return MemoryDataset(samples=samples, name=f"narcbench_{suffix}")
