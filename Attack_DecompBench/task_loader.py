"""
DeCompBench task loader.

Loads a single task's decomposition (sequential turns) and metadata
(services/capabilities per turn) from the real dataset layout:

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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Keyword hints per service, used to guess which service a turn's prompt
# is actually acting on. NOT from metadata.json -- metadata's
# capability_ids/step_outline only cover a subset of turns (the ones it
# considers "core steps"), so it can't be relied on for 1:1 per-turn
# matching. This is a best-effort heuristic, intersected with the task's
# own declared `services` list to cut down false positives.
SERVICE_KEYWORDS = {
    "postgres": ["postgres", "sql", "select ", "update ", "insert ", " table", "database", "query", "row"],
    "gitlab": ["gitlab", "merge request", "repository", "commit", "branch", "pull request"],
    "owncloud": ["owncloud", "documents/"],
    "plane": ["plane", "project management", "ticket", "issue tracker"],
    "redis": ["redis", "session key", "cache"],
    "rocketchat": ["rocketchat", "channel", "chat message"],
    "filesystem": ["/workspace", "directory", "file listing", "save the result", "save it to"],
    "terminal": ["terminal", "shell command", "bash", "run the following script"],
    "voip-gateway": ["voip", "phone call", "gateway"],
}


def _infer_service_hint(prompt: str, task_services: list) -> Optional[str]:
    """Best-effort: which of this task's declared services does this
    turn's prompt text look like it's acting on. Returns None if no
    keyword hits -- caller should fall back to a default role."""
    text = prompt.lower()
    for service in task_services:
        keywords = SERVICE_KEYWORDS.get(service, [])
        if any(kw in text for kw in keywords):
            return service
    return None


@dataclass
class Turn:
    turn_id: int
    label: str  # discovery | masking | action | cover
    prompt: str
    capability_id: Optional[str] = None  # best-effort, from metadata.json's capability_ids (subset only)
    service_hint: Optional[str] = None   # best-effort, keyword-inferred from prompt text


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
    task_services = meta.get("services", [])

    turns = []
    for i, t in enumerate(decomp["turns"]):
        cap = capability_ids[i] if i < len(capability_ids) else None
        hint = _infer_service_hint(t["prompt"], task_services)
        turns.append(Turn(turn_id=t["turn_id"], label=t["label"], prompt=t["prompt"],
                           capability_id=cap, service_hint=hint))

    return DeCompTask(
        slug=decomp["slug"],
        title=meta.get("title", decomp["slug"]),
        harm_category=meta.get("harm_category", "unknown"),
        services=task_services,
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
