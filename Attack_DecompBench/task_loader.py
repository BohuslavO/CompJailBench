import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    text = prompt.lower()
    for service in task_services:
        keywords = SERVICE_KEYWORDS.get(service, [])
        if any(kw in text for kw in keywords):
            return service
    return None


@dataclass
class Turn:
    turn_id: int
    label: str
    prompt: str
    capability_id: Optional[str] = None
    service_hint: Optional[str] = None


@dataclass
class DeCompTask:
    slug: str
    title: str
    harm_category: str
    services: list = field(default_factory=list)
    difficulty: int = 0
    hop_count: int = 0
    turns: list = field(default_factory=list)
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
    root = Path(tasks_root)
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "decomposition.json").exists() and (d / "metadata.json").exists():
            out.append(d.name)
    return out
