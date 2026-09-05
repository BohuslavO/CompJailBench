"""
Runs the core CompJailBench comparison:

    single-agent baseline  vs  multi-agent decomposition pipeline

scored by both AgentHarmMonitor (final-output only) and AgentToAgentMonitor
(structure + compositional leakage), and writes results/summary.csv plus
a printed table.

Usage:
    python run_experiment.py --dataset datasets/agentharm_samples.json --num_workers 2
"""

import argparse
import csv
import os

from dataset import BenchmarkDataset
from baseline import SingleAgentBaseline
from benchmark import CompJailBench
from monitors import AgentHarmMonitor, AgentToAgentMonitor
from logger import TrajectoryLogger


def run(dataset_path, num_workers, out_csv):
    dataset = BenchmarkDataset(dataset_path)

    baseline_agent = SingleAgentBaseline()
    multi_agent = CompJailBench(num_workers=num_workers)

    agentharm_monitor = AgentHarmMonitor()
    graph_monitor = AgentToAgentMonitor()

    logger = TrajectoryLogger(output_dir="results/traces")

    rows = []

    for sample in dataset:
        print(f"\n=== Sample {sample['id']}: {sample.get('name', '')} ===")

        # --- Baseline ---
        baseline_result = baseline_agent.run(sample)
        baseline_score = agentharm_monitor.analyze(baseline_result["execution_trace"])
        print(f"[baseline]     completed={baseline_score['unsafe']} "
              f"conf={baseline_score['confidence']:.2f}")

        # --- Multi-agent ---
        multi_result = multi_agent.run(sample)
        multi_agentharm_score = agentharm_monitor.analyze(multi_result["execution_trace"])
        multi_graph_score = graph_monitor.analyze(multi_result["execution_trace"])
        print(f"[multi-agent]  completed={multi_agentharm_score['unsafe']} "
              f"conf={multi_agentharm_score['confidence']:.2f} | "
              f"graph_flagged={multi_graph_score['unsafe']} "
              f"conf={multi_graph_score['confidence']:.2f}")

        logger.save(sample, {
            "execution_trace": multi_result["execution_trace"],
            "monitor": {
                "agentharm": multi_agentharm_score,
                "graph": multi_graph_score,
            },
        })

        rows.append({
            "sample_id": sample["id"],
            "sample_name": sample.get("name", ""),
            "attack_category": sample.get("attack_category", ""),
            "baseline_completed": baseline_score["unsafe"],
            "baseline_confidence": baseline_score["confidence"],
            "multi_agent_completed": multi_agentharm_score["unsafe"],
            "multi_agent_confidence": multi_agentharm_score["confidence"],
            "graph_monitor_flagged": multi_graph_score["unsafe"],
            "graph_monitor_confidence": multi_graph_score["confidence"],
            "graph_monitor_reason": multi_graph_score["reason"],
        })

    # --- Write CSV ---
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary ---
    n = len(rows)
    baseline_rate = sum(1 for r in rows if r["baseline_completed"]) / n
    multi_rate = sum(1 for r in rows if r["multi_agent_completed"]) / n
    graph_catch_rate = sum(
        1 for r in rows if r["multi_agent_completed"] and r["graph_monitor_flagged"]
    ) / max(1, sum(1 for r in rows if r["multi_agent_completed"]))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"n samples:                          {n}")
    print(f"Baseline harmful-completion rate:    {baseline_rate:.1%}")
    print(f"Multi-agent harmful-completion rate: {multi_rate:.1%}")
    print(f"AgentToAgentMonitor catch rate (of successful multi-agent attacks): {graph_catch_rate:.1%}")
    print(f"\nFull results: {out_csv}")
    print(f"Full traces:  results/traces/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/agentharm_samples.json")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", type=str, default="results/summary.csv")
    args = parser.parse_args()

    run(args.dataset, args.num_workers, args.out)