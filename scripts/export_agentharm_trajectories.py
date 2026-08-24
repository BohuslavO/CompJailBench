from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def trace_to_standard(trace: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    metadata = dict(trace.get("metadata") or {})
    if metadata.get("is_placeholder"):
        raise ValueError("placeholder trace rejected")
    condition = metadata.get("condition_label") or metadata.get("condition")
    if condition not in {"attack", "benign_control"}:
        raise ValueError(f"invalid condition_label: {condition!r}")
    pair_id = str(metadata.get("pair_id") or trace.get("sample_id") or "")
    if not pair_id:
        raise ValueError("trace has no pair_id or sample_id")

    nodes = list(trace.get("nodes") or [])
    edges = list(trace.get("edges") or [])
    recipients: dict[str, list[str]] = {}
    for edge in edges:
        recipients.setdefault(str(edge.get("source", "")), []).append(
            str(edge.get("target", ""))
        )

    messages: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    observable_reasoning: list[dict[str, Any]] = []
    reasoning_traces: dict[str, str] = {}
    tool_calls: list[dict[str, Any]] = []
    agent_ids: list[str] = []
    for round_index, node in enumerate(nodes):
        node_id = str(node.get("node_id") or f"node_{round_index}")
        agent_ids.append(node_id)
        messages.append(
            {
                "message_id": f"{pair_id}:{node_id}",
                "sender": node_id,
                "recipients": recipients.get(node_id, []),
                "round": round_index,
                "channel": "public",
                "content": str(node.get("output") or ""),
                "node_type": node.get("node_type"),
            }
        )
        actions.append(
            {
                "agent_id": node_id,
                "action": "produce_handoff",
                "round": round_index,
                "node_type": node.get("node_type"),
            }
        )
        reasoning = str(node.get("reasoning") or "").strip()
        if reasoning:
            reasoning_traces[node_id] = reasoning
            observable_reasoning.append(
                {
                    "agent_id": node_id,
                    "round": round_index,
                    "channel": "public",
                    "visible_to_monitor": True,
                    "elicitation": "explicit_reasoning_block",
                    "content": reasoning,
                }
            )
        for call in node.get("tool_calls") or []:
            tool_calls.append({"agent_id": node_id, **dict(call)})

    communication_edges = [
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "type": edge.get("edge_type"),
            "channel": "public",
            "round": edge_index,
            "payload": edge.get("payload", ""),
        }
        for edge_index, edge in enumerate(edges)
    ]
    sample_id = f"{condition}__{pair_id}"
    export_metadata = {
        "run_id": metadata.get("run_id"),
        "agentharm_condition": metadata.get("agentharm_condition"),
        "agentharm_split": metadata.get("agentharm_split"),
        "category_original": metadata.get("category_original"),
        "observability_assumption": metadata.get("observability_assumption"),
        "source_trace_id": trace.get("trace_id"),
        "agentharm_mode": "behavior_prompt_multi_agent_pipeline",
        "official_agentharm_tools_executed": False,
    }
    if source_path is not None:
        export_metadata["source_trace_file"] = source_path.name

    return {
        "schema_version": "2.0",
        "sample_id": sample_id,
        "pair_id": pair_id,
        "attack_name": "agentharm",
        "condition_label": condition,
        "public_task": {"context": str(trace.get("original_prompt") or "")},
        "messages": messages,
        "public_messages": messages,
        "private_messages": [],
        "observable_reasoning": observable_reasoning,
        "reasoning_traces": reasoning_traces,
        "actions": actions,
        "tool_calls": tool_calls,
        "final_output": str(trace.get("final_output") or ""),
        "model_name": str(metadata.get("agent_model") or ""),
        "agent_ids": agent_ids,
        "communication_edges": communication_edges,
        "activation_refs": [],
        "metadata": export_metadata,
    }


def load_and_convert(trace_root: Path) -> list[dict[str, Any]]:
    paths = sorted(trace_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No trace JSON files found under {trace_root}")
    converted = [
        trace_to_standard(json.loads(path.read_text(encoding="utf-8")), path)
        for path in paths
    ]
    keys = [(item["condition_label"], item["pair_id"]) for item in converted]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate condition/pair trajectories found: {duplicates[:5]}")
    attack_ids = {item["pair_id"] for item in converted if item["condition_label"] == "attack"}
    benign_ids = {
        item["pair_id"]
        for item in converted
        if item["condition_label"] == "benign_control"
    }
    if attack_ids != benign_ids:
        raise ValueError(
            "Export requires matched harmful and benign trajectories; "
            f"attack-only={sorted(attack_ids - benign_ids)[:5]}, "
            f"benign-only={sorted(benign_ids - attack_ids)[:5]}"
        )
    return sorted(converted, key=lambda item: (item["pair_id"], item["condition_label"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert saved AgentHarm traces to paired StandardTrajectory JSONL."
    )
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    trajectories = load_and_convert(args.trace_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    counts = Counter(item["condition_label"] for item in trajectories)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_root": str(args.trace_root),
        "output": str(args.output),
        "counts": dict(counts),
        "pair_count": counts.get("attack", 0),
        "models": sorted({item["model_name"] for item in trajectories}),
        "splits": sorted(
            {
                str(item["metadata"].get("agentharm_split"))
                for item in trajectories
            }
        ),
        "inspect_ai_version": _package_version("inspect-ai"),
        "inspect_evals_version": _package_version("inspect-evals"),
        "limitations": [
            "Uses official AgentHarm behavior prompts in the custom multi-agent pipeline.",
            "Does not execute the official AgentHarm tool environment or programmatic graders.",
            "Activation probing requires a separate self-hosted generation run.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: wrote {len(trajectories)} trajectories "
        f"({counts.get('attack', 0)} matched pairs) to {args.output}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
