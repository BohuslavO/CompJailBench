"""
DeCompBench task loader.

Loads a single task's decomposition (sequential turns) and metadata
(capabilities/services per turn) from the real dataset layout:

tasks/<slug>/
    metadata.json
    decomposition.json
    checkpoints.md
    desiderata.json
    task.md
    utils/{dependencies.yml, evaluator.py, ...}

Schema confirmed directly against the real downloaded dataset
(tasks/abuse-report-suppression, tasks/absent-admin-false-attribution).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    turn_id: int
    label: str  # discovery | masking | action | cover
    prompt: str
    capability_id: Optional[str] = None  # best-effort, from metadata.json's capability_ids


@dataclass
class DeCompTask:
    slug: str
    title: str
    harm_category: str
    services: list = field(default_factory=list)
    difficulty: int = 0
    hop_count: int = 0
    turns: list = field(default_factory=list)  # list[Turn]
    checkpoints_md: str = ""


def load_task(task_dir: str) -> DeCompTask:
    task_dir = Path(task_dir)

    with open(task_dir / "decomposition.json") as f:
        decomp = json.load(f)
    with open(task_dir / "metadata.json") as f:
        meta = json.load(f)

    checkpoints_path = task_dir / "checkpoints.md"
    checkpoints_md = checkpoints_path.read_text() if checkpoints_path.exists() else ""

    capability_ids = meta.get("capability_ids", [])
    turns = []
    for i, t in enumerate(decomp["turns"]):
        cap = capability_ids[i] if i < len(capability_ids) else None
        turns.append(Turn(turn_id=t["turn_id"], label=t["label"], prompt=t["prompt"], capability_id=cap))

    return DeCompTask(
        slug=decomp["slug"],
        title=meta.get("title", decomp["slug"]),
        harm_category=meta.get("harm_category", "unknown"),
        services=meta.get("services", []),
        difficulty=meta.get("difficulty", 0),
        hop_count=meta.get("hop_count", len(turns)),
        turns=turns,
        checkpoints_md=checkpoints_md,
    )


def list_tasks(tasks_root: str) -> list:
    """Slugs of every task under tasks_root that has both files needed to load."""
    root = Path(tasks_root)
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "decomposition.json").exists() and (d / "metadata.json").exists():
            out.append(d.name)
    return out
