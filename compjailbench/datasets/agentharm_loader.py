"""
Loads the public AgentHarm dataset (AI Safety Institute) and converts it
into CompJailBench's sample schema.

AgentHarm is released specifically for evaluating model/agent robustness
against harmful agentic tasks — using it here means our numbers are
comparable to the original AgentHarm baseline and to other papers that
build on it, rather than inventing an ad-hoc harmful task set.

Requires: `pip install datasets` and network access (this needs to run on
your machine, not in a sandboxed environment).

Usage:
    python datasets/agentharm_loader.py --n 20 --split test_public --out datasets/agentharm_samples.json
"""

import argparse
import json


def load_agentharm(n=20, split="test_public", subset="harmful"):
    from datasets import load_dataset

    ds = load_dataset("ai-safety-institute/AgentHarm", subset, split=split)

    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break

        samples.append({
            "id": row.get("id", i),
            "name": row.get("name", f"agentharm_{i}"),
            # "task" is what our pipeline actually feeds to the planner
            "task": row["prompt"],
            "original_intent": row["prompt"],
            "attack_category": row.get("category"),
            "expected_unsafe_objective": row.get("target_functions", None),
            "source": "AgentHarm",
        })

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--split", type=str, default="test_public")
    parser.add_argument("--subset", type=str, default="harmful")
    parser.add_argument("--out", type=str, default="datasets/agentharm_samples.json")
    args = parser.parse_args()

    samples = load_agentharm(n=args.n, split=args.split, subset=args.subset)

    with open(args.out, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} samples to {args.out}")