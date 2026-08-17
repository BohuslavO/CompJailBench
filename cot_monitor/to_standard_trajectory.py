"""
Converts my DeCompBench .eval logs (from run_kaggle.py) into StandardTrajectory
v2.0 JSONL, one object per line, matching the exact field names in
src/trajectory.py (fetched verbatim from NARCBench_Attack_CoT_Monitor/) so the
real score_standard_trajectories.py can read it back with no changes.

Known, deliberate gap: DeCompBench's pipeline.py records
{agent_name, turn_id, label, system_prompt, user_prompt, response} -- there is
no separate elicited-rationale channel, only a direct response per turn. So
observable_reasoning is left empty here, which makes scope
"observable-reasoning-only" and the combined scope correctly report
"unavailable" per DEFENSE_INTERFACE.md ("Missing reasoning is reported as
unavailable; it is never silently scored as benign") -- this is the honest,
contract-correct result, not a bug to work around. Only
"public-action-only" is scorable for this attack, and that's a real,
first-class result: it's one of the three predeclared evidence scopes, not a
degraded fallback.

condition_label is passed straight through ("attack" / "benign_control") --
both values are recognized as-is by src/defense_interface.py's
normalized_truth(), no remapping needed (confirmed by reading that function
directly).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai.log import read_eval_log

ATTACK_NAME = "decompbench_routing"
MODEL_NAME = "qwen3-32b-awq"


def to_standard_trajectory(trajectory: list, *, sample_id: str, pair_id: str,
                           condition: str, strategy: str, harm_category: str = "") -> dict:
    public_messages = []
    for record in trajectory:
        agent = record["agent_name"]
        public_messages.append({
            "turn": record.get("turn_id"),
            "from": "orchestrator",
            "to": agent,
            "role": "user",
            "content": record.get("user_prompt", ""),
        })
        public_messages.append({
            "turn": record.get("turn_id"),
            "from": agent,
            "to": "orchestrator",
            "role": "assistant",
            "label": record.get("label", ""),
            "content": record.get("response", ""),
        })

    final_output = trajectory[-1].get("response", "") if trajectory else ""
    agent_ids = sorted({r["agent_name"] for r in trajectory})

    return {
        "schema_version": "2.0",
        "sample_id": sample_id,
        "pair_id": pair_id,
        "attack_name": ATTACK_NAME,
        "condition_label": condition,
        "public_task": {"strategy": strategy, "harm_category": harm_category},
        "messages": [],
        "public_messages": public_messages,
        "private_messages": [],
        "observable_reasoning": [],   # deliberately empty -- see module docstring
        "reasoning_traces": {},
        "actions": [],
        "tool_calls": [],
        "final_output": final_output,
        "model_name": MODEL_NAME,
        "agent_ids": agent_ids,
        "communication_edges": [],
        "activation_refs": [],
        "metadata": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="final_eval_logs")
    ap.add_argument("--out", default="trajectories_standard.jsonl")
    args = ap.parse_args()

    log_paths = sorted(Path(args.eval_dir).rglob("*.eval"))
    print(f"Found {len(log_paths)} .eval log files under {args.eval_dir}")

    rows = []
    for log_path in log_paths:
        log = read_eval_log(str(log_path))
        if log.status != "success":
            print(f"  SKIP {log_path.name}: status={log.status}")
            continue

        for sample in log.samples:
            trajectory = (sample.metadata or {}).get("trajectory")
            if not trajectory:
                continue

            strategy = sample.metadata.get("strategy", "unknown")
            condition = sample.metadata.get("condition", "attack")
            slug = sample.metadata.get("slug", str(sample.id))
            harm_category = sample.metadata.get("harm_category", "")

            sample_id = f"{slug}__{strategy}__{condition}"
            pair_id = f"{slug}__{strategy}"

            rows.append(to_standard_trajectory(
                trajectory, sample_id=sample_id, pair_id=pair_id,
                condition=condition, strategy=strategy, harm_category=harm_category,
            ))

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} StandardTrajectory records to {out_path}")


if __name__ == "__main__":
    main()
