from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from inspect_ai import eval as inspect_eval
from tabulate import tabulate

import config
from benchmark import (
    compjailbench_cot_monitor,
    compjailbench_g_safeguard,
    compjailbench_none,
    compjailbench_probing,
    compjailbench_sentinel_agent,
)

TASK_REGISTRY = {
    "none": compjailbench_none,
    "g_safeguard": compjailbench_g_safeguard,
    "sentinel_agent": compjailbench_sentinel_agent,
    "cot_monitor": compjailbench_cot_monitor,
    "probing": compjailbench_probing,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=list(TASK_REGISTRY.keys()),
        default=[d for d in config.DEFENSES],
        help="Which defense conditions to run (default: everything enabled in config.py).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override config.AGENTHARM_CONFIG.limit for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without calling any model.",
    )
    return parser.parse_args()


def print_config_summary() -> None:
    print("=" * 70)
    print("CompJailBench eval configuration")
    print("=" * 70)
    print(f"AGENT_MODEL: {config.AGENT_MODEL}")
    print(f"JUDGE_MODEL: {config.JUDGE_MODEL}")
    print(f"Defenses to run: {config.DEFENSES}")
    print(f"AgentHarm split: {config.AGENTHARM_CONFIG.split}")
    print(f"Sample limit: {config.AGENTHARM_CONFIG.limit}")
    print("API keys present:")
    for k, present in config.api_keys_present().items():
        print(f"  {k}: {'SET' if present else 'NOT SET'}")
    print("=" * 70)


def run_all(defense_names: list[str]) -> dict[str, list]:
    """Runs one Inspect eval per defense and returns {defense_name: eval_logs}."""
    results = {}
    for name in defense_names:
        if name == "probing" and not config.ENABLE_PROBING:
            print(f"[skip] '{name}' -- config.ENABLE_PROBING is False (see defenses/probing.py).")
            continue
        task_fn = TASK_REGISTRY[name]
        print(f"\n>>> Running AgentHarm vs. {name} ...")
        logs = inspect_eval(
            task_fn(),
            model=config.AGENT_MODEL,
            log_dir=str(config.LOGS_DIR / name),
        )
        results[name] = logs
    return results


def build_comparison_table(results: dict[str, list]) -> pd.DataFrame:
    rows = []
    for defense_name, logs in results.items():
        for log in logs:
            if log.status != "success" or log.results is None:
                rows.append(
                    {
                        "defense": defense_name,
                        "scorer": None,
                        "metric": "status",
                        "value": log.status,
                    }
                )
                continue
            for score in log.results.scores:
                for metric_name, metric in score.metrics.items():
                    rows.append(
                        {
                            "defense": defense_name,
                            "scorer": score.name,
                            "metric": metric_name,
                            "value": metric.value,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.limit is not None:
        config.AGENTHARM_CONFIG.limit = args.limit

    print_config_summary()

    if args.dry_run:
        print("\n[dry run] Config looks resolved. Not calling any model. "
              "Re-run without --dry-run once your API key is set.")
        return

    if not any(config.api_keys_present().values()):
        print(
            "\nNo API keys detected in the environment. Set at least one of "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY / AZURE_OPENAI_API_KEY before "
            "running for real. Exiting (use --dry-run to check config without "
            "this check)."
        )
        sys.exit(1)

    results = run_all(args.defenses)

    df = build_comparison_table(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / "comparison.csv"
    md_path = config.RESULTS_DIR / "comparison.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w") as f:
        f.write(tabulate(df, headers="keys", tablefmt="github", showindex=False))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Full Inspect logs under: {config.LOGS_DIR}")


if __name__ == "__main__":
    main()