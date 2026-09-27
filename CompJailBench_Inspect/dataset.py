import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compjailbench_attack"))

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from task_loader import load_task, list_tasks
from checkpoint_scorer import parse_checkpoints


def scoreable_tasks(tasks_root: str) -> list:
    out = []
    for slug in list_tasks(tasks_root):
        try:
            task = load_task(f"{tasks_root}/{slug}")
            if parse_checkpoints(task.checkpoints_md):
                out.append(slug)
        except Exception:
            continue
    return out


def build_dataset(tasks_root: str, condition: str, strategy: str, slugs: list = None) -> Dataset:
    slugs = slugs if slugs is not None else scoreable_tasks(tasks_root)
    samples = []
    for slug in slugs:
        task_dir = f"{tasks_root}/{slug}"
        task = load_task(task_dir)
        samples.append(Sample(
            input=task.title,
            target=task.checkpoints_md,
            id=slug,
            metadata={
                "task_dir": task_dir,
                "slug": slug,
                "condition": condition,
                "strategy": strategy,
                "harm_category": task.harm_category,
                "services": task.services,
            },
        ))
    return MemoryDataset(samples)
