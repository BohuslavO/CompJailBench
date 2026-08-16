"""Offline validation for NARCBench x Sentinel Agents message-only mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai.util import Store

from src.sentinel_message_adapter import (
    DEFENSE_METHOD,
    SIGNATURE_VERSION,
    build_message_graph,
    build_message_library,
    compatibility,
    forbidden_evidence_present,
    load_sentinel_backend,
    signature_text_is_target_independent,
)
from src.trajectory import StandardTrajectory


def read_trajectories(path: Path) -> list[StandardTrajectory]:
    return [
        StandardTrajectory(store=Store(), **json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--sentinel-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trajectories = read_trajectories(args.input)
    backend = load_sentinel_backend(args.sentinel_dir)
    library = build_message_library(backend.attack_paths)
    if not signature_text_is_target_independent(backend.attack_paths):
        raise RuntimeError("message signature library contains forbidden target information")

    rows = []
    for trajectory in trajectories:
        available, reason = compatibility(trajectory)
        for scope in ("public", "public_and_private"):
            if not available:
                rows.append(
                    {
                        "sample_id": trajectory.sample_id,
                        "scope": scope,
                        "status": "unavailable",
                        "reason": reason,
                    }
                )
                continue
            graph = build_message_graph(
                trajectory, backend.graph_builder, scope=scope
            )
            leaks = forbidden_evidence_present(graph, trajectory)
            if leaks:
                raise RuntimeError(
                    f"{trajectory.sample_id}/{scope}: hidden evidence leaked: {leaks}"
                )
            alert = backend.analyzer.analyze(
                graph,
                library,
                config=backend.analyzer.SentinelConfig(enable_tier1=False),
            )
            rows.append(
                {
                    "sample_id": trajectory.sample_id,
                    "pair_id": trajectory.pair_id,
                    "truth": "attack" if trajectory.condition_label == "attack" else "benign",
                    "defense_method": DEFENSE_METHOD,
                    "signature_version": SIGNATURE_VERSION,
                    "scope": scope,
                    "status": "ready_degraded",
                    "degraded": graph.degraded,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "stub_detected": alert.detected,
                    "stub_tiers_fired": alert.tiers_fired,
                    "stub_matched_paths": [item.pattern_id for item in alert.matched_paths],
                    "note": "offline compatibility only; not a paper result",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "trajectories": len(trajectories),
        "rows": len(rows),
        "ready_degraded": sum(row["status"] == "ready_degraded" for row in rows),
        "hidden_evidence_leaks": 0,
        "signature_version": SIGNATURE_VERSION,
        "paper_result": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
