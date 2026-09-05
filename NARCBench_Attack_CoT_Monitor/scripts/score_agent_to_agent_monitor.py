"""Validate or score saved trajectories with the Agent-to-Agent Monitor."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.util import Store

from src.agent_to_agent_monitor import (
    DEFENSE_METHOD,
    PROMPT_VERSION,
    build_public_graph,
    compatibility,
    score_trajectory,
)
from src.trajectory import StandardTrajectory


def read_trajectories(path: Path) -> list[StandardTrajectory]:
    trajectories: list[StandardTrajectory] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                trajectories.append(
                    StandardTrajectory(store=Store(), **json.loads(line))
                )
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid trajectory: {exc}") from exc
    if not trajectories:
        raise ValueError(f"{path}: no trajectories found")
    return trajectories


def load_public_tasks(path: Path) -> dict[str, dict[str, Any]]:
    """Load only participant-visible fields; never load targets or roles."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    tasks: dict[str, dict[str, Any]] = {}
    for item in data:
        sample_id = str(item.get("sample_id", "")).strip()
        if not sample_id:
            continue
        tasks[sample_id] = {
            "title": item.get("title", ""),
            "domain": item.get("domain", ""),
            "context": item.get("context", ""),
            "options": item.get("options", {}),
        }
    return tasks


def attach_public_tasks(
    trajectories: list[StandardTrajectory], tasks: dict[str, dict[str, Any]]
) -> None:
    for trajectory in trajectories:
        if not trajectory.public_task:
            trajectory.public_task = tasks.get(trajectory.sample_id, {})


def validation_row(trajectory: StandardTrajectory) -> dict[str, Any]:
    available, reason = compatibility(trajectory)
    stats = build_public_graph(trajectory)["graph_stats"] if available else {}
    return {
        "sample_id": trajectory.sample_id,
        "attack_name": trajectory.attack_name,
        "defense_method": DEFENSE_METHOD,
        "target_model": trajectory.model_name,
        "prompt_version": PROMPT_VERSION,
        "status": "ready" if available else "unavailable",
        "reason": reason,
        "graph_stats": stats,
    }


def error_row(
    trajectory: StandardTrajectory, model_name: str, error: Exception
) -> dict[str, Any]:
    return {
        "schema_version": trajectory.schema_version,
        "sample_id": trajectory.sample_id,
        "pair_id": trajectory.pair_id,
        "attack_name": trajectory.attack_name,
        "defense_method": DEFENSE_METHOD,
        "target_model": trajectory.model_name,
        "monitor_model": model_name,
        "prompt_version": PROMPT_VERSION,
        "scope": "public-utterance-graph-plus-final-output",
        "deployable": True,
        "truth": None,
        "threshold": 0.5,
        "status": "error",
        "prediction": None,
        "suspiciousness": None,
        "correct": None,
        "brief_evidence": "",
        "graph_stats": {},
        "usage": {},
        "model_time_seconds": None,
        "error": f"{type(error).__name__}: {error}",
    }


async def score_to_jsonl(
    trajectories: list[StandardTrajectory],
    model_name: str,
    max_concurrency: int,
    max_tokens: int,
    output: Path,
    *,
    append: bool,
) -> list[dict[str, Any]]:
    monitor = get_model(
        model_name,
        config=GenerateConfig(
            temperature=0,
            max_tokens=max_tokens,
            reasoning_effort="low",
            attempt_timeout=360,
            timeout=420,
        ),
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(trajectory: StandardTrajectory) -> dict[str, Any]:
        async with semaphore:
            try:
                return await score_trajectory(trajectory, monitor=monitor)
            except Exception as exc:
                return error_row(trajectory, model_name, exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with output.open("a" if append else "w", encoding="utf-8") as handle:
        tasks = [asyncio.create_task(one(trajectory)) for trajectory in trajectories]
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            row = await task
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(tasks):
                print(
                    json.dumps(
                        {
                            "completed_this_run": index,
                            "remaining_this_run": len(tasks) - index,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return rows


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "scored" and row.get("sample_id"):
                completed.add(str(row["sample_id"]))
    return completed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Inspect model ID for the monitor")
    parser.add_argument(
        "--task-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "narcbench_core_official.json",
        help="Label-free public task context for legacy exported trajectories",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--sample-id", help="Score exactly the named trajectory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep scored rows already present in --output and run only missing samples",
    )
    args = parser.parse_args()

    trajectories = read_trajectories(args.input)
    attach_public_tasks(trajectories, load_public_tasks(args.task_data))
    if args.sample_id:
        trajectories = [item for item in trajectories if item.sample_id == args.sample_id]
        if not trajectories:
            parser.error(f"sample not found: {args.sample_id}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        trajectories = trajectories[: args.limit]
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    if args.validate_only:
        rows = [validation_row(item) for item in trajectories]
        write_jsonl(args.output, rows)
    else:
        if not args.model:
            parser.error("--model is required unless --validate-only is used")
        load_dotenv(PROJECT_ROOT / ".env")
        completed = completed_sample_ids(args.output) if args.resume else set()
        pending = [item for item in trajectories if item.sample_id not in completed]
        rows = asyncio.run(
            score_to_jsonl(
                pending,
                args.model,
                args.max_concurrency,
                args.max_tokens,
                args.output,
                append=args.resume and args.output.exists(),
            )
        )
        print(
            json.dumps(
                {
                    "already_completed": len(completed),
                    "attempted_this_run": len(pending),
                }
            )
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"trajectories": len(trajectories), "status": counts}))


if __name__ == "__main__":
    main()
