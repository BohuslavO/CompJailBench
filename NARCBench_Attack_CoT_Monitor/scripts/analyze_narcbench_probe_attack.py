"""Aggregate attack outcomes from the per-pair Qwen probing Inspect logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_eval import (
    attack_summary,
    complete_pairs,
    read_rows,
    serializable,
)


def discover_eval_logs(path: Path) -> list[Path]:
    """Return one or more Inspect logs in deterministic path order."""

    path = Path(path)
    if path.is_file():
        if path.suffix != ".eval":
            raise ValueError(f"Expected an .eval file, found {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Inspect-log path does not exist: {path}")
    logs = sorted(item for item in path.rglob("*.eval") if item.is_file())
    if not logs:
        raise ValueError(f"No .eval files found under {path}")
    return logs


def canonical_attack_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require one unambiguous row per sample and return stable ordering."""

    by_sample: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Attack outcome row has no sample_id")
        if sample_id in by_sample:
            duplicates.append(sample_id)
        else:
            by_sample[sample_id] = row
    if duplicates:
        raise ValueError(f"Duplicate sample rows: {sorted(set(duplicates))}")
    return sorted(by_sample.values(), key=lambda row: str(row["sample_id"]))


def collect_rows(log_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    """Read exactly one collusion/control sample pair from each Inspect log."""

    logs = discover_eval_logs(log_root)
    rows: list[dict[str, Any]] = []
    for log_path in logs:
        log_rows = read_rows(log_path)
        if len(log_rows) != 2:
            raise ValueError(
                f"Expected one matched two-sample pair in {log_path}, "
                f"found {len(log_rows)} samples"
            )
        for row in log_rows:
            row["source_log"] = (
                log_path.relative_to(log_root).as_posix()
                if Path(log_root).is_dir()
                else log_path.name
            )
        rows.extend(log_rows)
    return logs, canonical_attack_rows(rows)


def normalize_usage_availability(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Do not represent missing self-hosted token telemetry as zero usage."""

    output = dict(summary)
    recorded = any(bool(row.get("model_usage")) for row in rows)
    output["token_usage_recorded"] = recorded
    if not recorded:
        output["total_tokens"] = None
        output["monitor_role_tokens"] = None
        output["attacker_role_tokens_estimate"] = None
        output["recorded_total_cost"] = None
    return output


def pair_outcomes(
    pairs: list[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return one compact, audit-friendly attack outcome record per pair."""

    output: list[dict[str, Any]] = []
    for pair in pairs:
        collusion = pair["collusion"]["attack"]
        control = pair["control"]["attack"]
        output.append(
            {
                "pair_id": collusion["pair_id"],
                "domain": collusion["domain"],
                "hidden_target": collusion["hidden_target"],
                "target_adherence": bool(collusion["target_adherence"]),
                "attack_success": bool(collusion["attack_success"]),
                "collusion_target_selected": bool(collusion["target_selected"]),
                "control_target_selected": bool(control["target_selected"]),
                "collusion_target_vote_count": int(collusion["target_vote_count"]),
                "control_target_vote_count": int(control["target_vote_count"]),
                "collusion_abstentions": int(collusion["abstention_count"]),
                "control_abstentions": int(control["abstention_count"]),
                "collusion_votes": json.dumps(collusion["votes"], sort_keys=True),
                "control_votes": json.dumps(control["votes"], sort_keys=True),
            }
        )
    return output


def write_outputs(
    output_dir: Path,
    *,
    logs: list[Path],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> None:
    """Write normalized samples, aggregate statistics, and pair outcomes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "attack_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(serializable(row), ensure_ascii=False) + "\n")

    final_summary = {
        "source_eval_logs": len(logs),
        "canonical_samples": len(rows),
        **serializable(summary),
    }
    (output_dir / "attack_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "attack_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(
            (key, json.dumps(serializable(value)))
            for key, value in final_summary.items()
        )

    fields = list(outcomes[0]) if outcomes else []
    with (output_dir / "attack_pair_outcomes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        type=Path,
        help="One .eval file or a directory containing per-pair .eval logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for normalized outcomes and aggregate summaries",
    )
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Require exactly 50 logs, 100 samples, and 50 complete pairs",
    )
    args = parser.parse_args()

    logs, rows = collect_rows(args.logs)
    pairs = complete_pairs(rows)
    if args.require_full and (len(logs), len(rows), len(pairs)) != (50, 100, 50):
        raise ValueError(
            "Expected 50 logs / 100 samples / 50 pairs, found "
            f"{len(logs)} / {len(rows)} / {len(pairs)}"
        )
    summary = normalize_usage_availability(attack_summary(pairs), rows)
    outcomes = pair_outcomes(pairs)
    write_outputs(
        args.output_dir,
        logs=logs,
        rows=rows,
        summary=summary,
        outcomes=outcomes,
    )
    print(json.dumps(serializable(summary), indent=2))


if __name__ == "__main__":
    main()
