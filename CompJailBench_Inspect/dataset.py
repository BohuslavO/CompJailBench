"""
Turns DeCompBench tasks into an Inspect Dataset of Samples.

Shared contract with the team: each Sample's `metadata` carries
`task_dir` (so a Solver can load full turn data), `condition`
("attack" or "benign_control"), and `strategy` (this attack's own
routing variant -- other attacks don't need to read it). `target`
carries checkpoints.md's raw text, since that's what an LLM-judged
Scorer grades against.

One dataset = one fixed (condition, strategy) combination. Build a
separate dataset/Task per combination you want to run rather than
cramming every combination into one Task -- keeps each eval log
auditable as a single, specific run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compjailbench_attack"))

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from task_loader import load_task, list_tasks
from checkpoint_scorer import parse_checkpoints


def scoreable_tasks(tasks_root: str) -> list:
    """Tasks with at least one LLM-judgeable checkpoint. Same filter
    batch_run.py uses -- about a quarter of DeCompBench tasks rely
    entirely on a rule-based harness nobody has built."""
    out = []
    for slug in list_tasks(tasks_root):
        try:
            task = load_task(f"{tasks_root}/{slug}")
            if parse_checkpoints(task.checkpoints_md):
                out.append(slug)
        except Exception:
            continue  # malformed task files, skip rather than crash the build
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
