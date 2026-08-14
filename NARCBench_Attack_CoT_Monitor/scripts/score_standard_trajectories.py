"""Validate or score StandardTrajectory JSONL from any CompJailBreak attack."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai.model import get_model
from inspect_ai.util import Store
from dotenv import load_dotenv

from src.defense_interface import SCOPES, scope_availability, score_trajectory
from src.trajectory import StandardTrajectory


def read_trajectories(path: Path) -> list[StandardTrajectory]:
    trajectories: list[StandardTrajectory] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                trajectories.append(StandardTrajectory(store=Store(), **data))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid StandardTrajectory: {exc}") from exc
    if not trajectories:
        raise ValueError(f"{path}: no trajectories found")
    return trajectories


def compatibility_rows(
    trajectories: list[StandardTrajectory], scopes: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for scope in scopes:
            available, reason = scope_availability(trajectory, scope)
            rows.append(
                {
                    "sample_id": trajectory.sample_id,
                    "attack_name": trajectory.attack_name,
                    "target_model": trajectory.model_name,
                    "scope": scope,
                    "status": "ready" if available else "unavailable",
                    "reason": reason,
                }
            )
    return rows


async def score_all(
    trajectories: list[StandardTrajectory],
    scopes: list[str],
    model_name: str,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    monitor = get_model(model_name, temperature=0, max_tokens=1536)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(trajectory: StandardTrajectory, scope: str) -> dict[str, Any]:
        async with semaphore:
            return await score_trajectory(trajectory, scope=scope, monitor=monitor)

    return await asyncio.gather(
        *(one(trajectory, scope) for trajectory in trajectories for scope in scopes)
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="One StandardTrajectory JSON object per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Inspect model ID for the monitor")
    parser.add_argument(
        "--scope",
        action="append",
        choices=sorted(SCOPES),
        dest="scopes",
        help="Repeat to select scopes; defaults to all three",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check scope compatibility without making model calls",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N trajectories")
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()
    scopes = args.scopes or list(SCOPES)
    trajectories = read_trajectories(args.input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        trajectories = trajectories[: args.limit]
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")

    if args.validate_only:
        rows = compatibility_rows(trajectories, scopes)
    else:
        if not args.model:
            parser.error("--model is required unless --validate-only is used")
        load_dotenv(PROJECT_ROOT / ".env")
        rows = asyncio.run(
            score_all(trajectories, scopes, args.model, args.max_concurrency)
        )

    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"trajectories": len(trajectories), "rows": len(rows), "status": counts}))


if __name__ == "__main__":
    main()
