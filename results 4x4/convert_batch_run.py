"""
Converts a batch_run.py-shaped CSV (task,strategy,condition,n_checkpoints,
n_passed,fjr,peak_suspicion,n_turns,error) into the shared result schema
defined in README.md.

This is a worked example for one batch-output format. Adapt the field mapping
for other attack and defense pipelines.

Usage:
  python convert_batch_run.py results.csv \
      --contributor example-contributor \
      --attack-method decompbench_routing \
      --defense-method narcbench_probing \
      --model qwen3-32b-awq \
      --out raw/example__decompbench_routing__narcbench_probing.csv
"""
import argparse
import csv


def convert(in_path, out_path, contributor, attack_method, defense_method, model):
    rows_out = []
    with open(in_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue  # failed runs have no metric to report
            run_id = f"{r['task']}|{r['strategy']}|{r['condition']}"
            base = {
                "contributor": contributor,
                "attack_method": attack_method,
                "defense_method": defense_method,
                "model": model,
                "task_id": r["task"],
                "strategy": r["strategy"],
                "condition": r["condition"],
                "n_samples": 1,
                "run_id": run_id,
                "notes": "",
            }
            if r.get("fjr") not in (None, ""):
                rows_out.append({**base, "metric_name": "fjr", "metric_value": r["fjr"]})
            if r.get("peak_suspicion") not in (None, ""):
                rows_out.append({**base, "metric_name": "peak_suspicion", "metric_value": r["peak_suspicion"]})

    fieldnames = ["contributor", "attack_method", "defense_method", "model", "task_id",
                  "strategy", "condition", "metric_name", "metric_value", "n_samples", "run_id", "notes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--contributor", required=True)
    ap.add_argument("--attack-method", required=True)
    ap.add_argument("--defense-method", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.csv_path, args.out, args.contributor, args.attack_method, args.defense_method, args.model)
