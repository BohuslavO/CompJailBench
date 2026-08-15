"""
Post-hoc converter: reads already-completed .eval logs (produced by
the OLD solver_attack.py, before it was updated to dual-write
StandardTrajectory live) and converts each sample's
state.metadata["trajectory"] into a StandardTrajectory JSON file.

Use this instead of re-running the whole Kaggle batch just to get
StandardTrajectory data -- it reads the trajectory that's ALREADY in
the log (that write was never removed, only the extra StandardTrajectory
write was added later) and converts it after the fact.

Usage (run from wherever your inspect_logs/ folder is, e.g. on Kaggle
after run_kaggle.py finishes, or locally after downloading the logs):
  python postprocess_to_standard_trajectory.py --log-dir inspect_logs --out standard_trajectories
"""
import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from trajectory_converters import from_decompbench_trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="inspect_logs")
    ap.add_argument("--out", default="standard_trajectories")
    ap.add_argument("--model-name", default="qwen3-32b-awq")
    args = ap.parse_args()

    log_paths = list(Path(args.log_dir).rglob("*.eval"))
    print(f"Found {len(log_paths)} .eval log files under {args.log_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_converted = 0
    n_skipped = 0

    for log_path in log_paths:
        log = read_eval_log(str(log_path))
        if log.status != "success":
            print(f"  SKIP {log_path.name}: log status={log.status}")
            continue

        # condition/strategy fall back to the parent folder name
        # (run_kaggle.py writes to inspect_logs/{strategy}__{condition}/...)
        # in case a sample's own metadata is missing either key.
        folder_name = log_path.parent.name
        fallback_strategy, _, fallback_condition = folder_name.partition("__")

        for sample in log.samples:
            trajectory = (sample.metadata or {}).get("trajectory")
            if not trajectory:
                print(f"  SKIP sample {sample.id}: no trajectory in metadata (probably a failed run)")
                n_skipped += 1
                continue

            condition = (sample.metadata or {}).get("condition", fallback_condition or "attack")
            strategy = (sample.metadata or {}).get("strategy", fallback_strategy or "unknown")

            std = from_decompbench_trajectory(
                trajectory,
                sample_id=str(sample.id),
                condition_label=condition,
                model_name=args.model_name,
                extra_metadata={"strategy": strategy, "source_log": log_path.name},
            )

            out_path = out_dir / f"{sample.id}__{strategy}__{condition}.json"
            out_path.write_text(std.model_dump_json(indent=2))
            n_converted += 1

    print(f"\nConverted {n_converted} trajectories to StandardTrajectory JSON in {out_dir}/")
    if n_skipped:
        print(f"Skipped {n_skipped} samples with no trajectory data.")


if __name__ == "__main__":
    main()
