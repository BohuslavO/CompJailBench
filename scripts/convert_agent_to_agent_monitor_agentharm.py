import json
import csv
from pathlib import Path

INPUT = Path("results/runs/agentharm_full_20260824/agent_to_agent_monitor_scores.jsonl")
OUTPUT = Path("results 4x4/raw/slava__agentharm__agent_to_agent_monitor.csv")

MODEL = "target=google/gemini-3.6-flash;monitor=google/gemini-3.6-flash"
NOTES = "prompt_version=compositional_leakage_v1;threshold=0.5;split=test_public;scope=public-utterance-graph-plus-final-output"

rows = []
skipped = 0
with INPUT.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "scored":
            skipped += 1
            continue
        pair_id = row["pair_id"]
        condition = "attack" if row["truth"] == "attack" else "benign_control"
        suspiciousness = row["suspiciousness"]
        flag = 1 if row["prediction"] == "attack" else 0
        run_id = f"{pair_id}|standard|{condition}"
        rows.append([
            "slava", "agentharm", "agent_to_agent_monitor", MODEL,
            pair_id, "standard", condition, "suspiciousness__graph", suspiciousness,
            1, run_id, NOTES,
        ])
        rows.append([
            "slava", "agentharm", "agent_to_agent_monitor", MODEL,
            pair_id, "standard", condition, "monitor_flag__graph", flag,
            1, run_id, NOTES,
        ])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "contributor", "attack_method", "defense_method", "model", "task_id",
        "strategy", "condition", "metric_name", "metric_value", "n_samples",
        "run_id", "notes",
    ])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows ({len(rows) // 2} trajectories, {skipped} skipped) to {OUTPUT}")