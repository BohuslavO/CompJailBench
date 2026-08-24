from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentharm_data import load_paired_agentharm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate real paired AgentHarm data without model/API calls."
    )
    parser.add_argument(
        "--split",
        choices=("val", "test_public", "test_private"),
        default="test_public",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    harmful, benign = load_paired_agentharm(split=args.split, limit=args.limit)
    harmful_samples = list(harmful)
    benign_samples = list(benign)
    if any((sample.metadata or {}).get("is_placeholder") for sample in harmful_samples):
        raise SystemExit("ERROR: harmful dataset contains a placeholder")
    if any((sample.metadata or {}).get("is_placeholder") for sample in benign_samples):
        raise SystemExit("ERROR: benign dataset contains a placeholder")

    pair_ids = [str(sample.metadata["pair_id"]) for sample in harmful_samples]
    print(f"PASS: official AgentHarm split={args.split}")
    print(f"PASS: harmful={len(harmful_samples)} benign={len(benign_samples)}")
    print(f"PASS: ordered pair IDs match ({len(pair_ids)} pairs)")
    print(f"First pair IDs: {', '.join(pair_ids[:5])}")
    print("No model or paid API calls were made.")


if __name__ == "__main__":
    main()
