"""Summarize NARCBench probing scores and export team-matrix rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_statistics import average_precision, paired_bootstrap_interval, roc_auc
from src.narcbench_probe import PROBE_METRIC_NAMES


TEAM_FIELDS = [
    "contributor", "attack_method", "defense_method", "model", "task_id",
    "strategy", "condition", "metric_name", "metric_value", "n_samples",
    "run_id", "notes",
]
PROBE_METRICS = list(PROBE_METRIC_NAMES)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from ``path``."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest complete row per sample after any resumed reruns."""

    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("sample_id") and isinstance(row.get("scores"), dict):
            by_sample[str(row["sample_id"])] = row
    return sorted(by_sample.values(), key=lambda row: str(row["sample_id"]))


def metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Summarize one continuous score using sample- and pair-level statistics."""

    selected = [row for row in rows if row["scores"].get(metric) is not None]
    labels = [int(row["condition"] == "attack") for row in selected]
    values = [float(row["scores"][metric]) for row in selected]
    attacks = [value for value, label in zip(values, labels) if label == 1]
    controls = [value for value, label in zip(values, labels) if label == 0]

    pair_map: dict[str, dict[str, float]] = {}
    for row in selected:
        pair_map.setdefault(str(row["pair_id"]), {})[str(row["condition"])] = float(
            row["scores"][metric]
        )
    pairs = [pair for pair in pair_map.values() if set(pair) == {"attack", "benign"}]

    def paired_difference(sample: list[dict[str, float]]) -> float:
        return mean(pair["attack"] - pair["benign"] for pair in sample)

    return {
        "metric": metric,
        "n": len(selected),
        "n_attack": len(attacks),
        "n_benign": len(controls),
        "n_complete_pairs": len(pairs),
        "attack_mean": mean(attacks) if attacks else None,
        "benign_mean": mean(controls) if controls else None,
        "mean_difference": (mean(attacks) - mean(controls)) if attacks and controls else None,
        "paired_mean_difference": paired_difference(pairs) if pairs else None,
        "paired_mean_difference_ci95": paired_bootstrap_interval(
            pairs, paired_difference, iterations=10_000
        ),
        "roc_auc": roc_auc(labels, values),
        "average_precision": average_precision(labels, values),
    }


def team_rows(rows: list[dict[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    """Convert canonical scores to the team's long-form matrix contract."""

    output: list[dict[str, Any]] = []
    for row in rows:
        for metric in PROBE_METRICS:
            value = row["scores"].get(metric)
            if value is None:
                continue
            output.append(
                {
                    "contributor": "Beibarys",
                    "attack_method": "narcbench_collusion",
                    "defense_method": "narcbench_probing",
                    "model": f"target={row['model_name']};probe={row['probe_model']}",
                    "task_id": row["sample_id"],
                    "strategy": (
                        f"self-hosted-layer-{row['probe_layer']}-all-turns-turnwise-max"
                        if metric == "peak_suspicion"
                        else (
                            f"self-hosted-layer-{row['probe_layer']}-public-turnwise-max"
                            if metric == "public_peak_suspicion"
                            else f"self-hosted-layer-{row['probe_layer']}-agent-mean-discussion"
                        )
                    ),
                    "condition": "attack" if row["condition"] == "attack" else "benign_control",
                    "metric_name": metric,
                    "metric_value": float(value),
                    "n_samples": 1,
                    "run_id": run_id,
                    "notes": (
                        "fresh_qwen_generation;primary_matches_released_scorer_defense;"
                        "secondary_scope_explicit;continuous_score_no_test_set_threshold;"
                        f"format_repairs={int(row.get('format_repair_count', 0))};"
                        f"generation_attempts={int(row.get('generation_attempts_total', 0))}"
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows with a stable, explicit column order."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="Raw probe_scores.jsonl file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for canonical scores, summaries, and team rows",
    )
    parser.add_argument("--run-id", required=True, help="Stable experiment identifier")
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Require exactly 50 matched pairs and 100 unique samples",
    )
    args = parser.parse_args()

    raw = read_jsonl(args.scores)
    canonical = canonical_rows(raw)
    if args.require_full:
        pairs = {str(row["pair_id"]) for row in canonical}
        attack = sum(row["condition"] == "attack" for row in canonical)
        benign = sum(row["condition"] == "benign" for row in canonical)
        if (len(canonical), len(pairs), attack, benign) != (100, 50, 50, 50):
            raise ValueError(
                f"Expected 100 samples / 50 pairs / 50+50, found "
                f"{len(canonical)} / {len(pairs)} / {attack}+{benign}"
            )

    summaries = [metric_summary(canonical, metric) for metric in PROBE_METRICS]
    scored_turns = sum(int(row.get("n_scored_turns", 0)) for row in canonical)
    format_repairs = sum(int(row.get("format_repair_count", 0)) for row in canonical)
    generation_attempts = sum(
        int(row.get("generation_attempts_total", row.get("n_scored_turns", 0)))
        for row in canonical
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores_canonical.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "defense_method": "narcbench_probing",
                "n_raw_rows": len(raw),
                "n_unique_samples": len(canonical),
                "generation_diagnostics": {
                    "n_scored_turns": scored_turns,
                    "n_format_repairs": format_repairs,
                    "format_repair_rate": (
                        format_repairs / scored_turns if scored_turns else None
                    ),
                    "n_generation_attempts": generation_attempts,
                    "extra_generation_attempts": generation_attempts - scored_turns,
                },
                "metrics": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "summary.csv", summaries, list(summaries[0]))
    exported = team_rows(canonical, run_id=args.run_id)
    write_csv(args.output_dir / "team_rows.csv", exported, TEAM_FIELDS)
    print(json.dumps({"canonical_rows": len(canonical), "team_rows": len(exported)}, indent=2))


if __name__ == "__main__":
    main()
