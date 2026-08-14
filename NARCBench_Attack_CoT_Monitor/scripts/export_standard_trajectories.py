"""Export StandardTrajectory stores from an Inspect log to portable JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai.log import read_eval_log_samples
from inspect_ai.util import Store

from src.trajectory import StandardTrajectory


PREFIX = "StandardTrajectory:"


def trajectory_dict(store: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, field in StandardTrajectory.model_fields.items():
        if name in {"store", "instance"}:
            continue
        key = PREFIX + name
        if key in store:
            values[name] = store[key]
        elif field.default_factory is not None:
            values[name] = field.default_factory()
        elif field.default is not None:
            values[name] = field.default
    # Validation catches incompatible field types before anything is shared.
    return StandardTrajectory(store=Store(), **values).model_dump()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for sample in read_eval_log_samples(args.log):
        row = trajectory_dict(sample.store or {})
        if not row["sample_id"]:
            raise ValueError(f"Sample {sample.id} has no StandardTrajectory sample_id")
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} StandardTrajectory records to {args.output}")


if __name__ == "__main__":
    main()
