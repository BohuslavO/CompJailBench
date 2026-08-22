from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from config import AGENTHARM_CONFIG

_HERE = Path(__file__).resolve().parent
_PLACEHOLDER_PATH = _HERE / "sample_agentharm.json"


def load_agentharm() -> Dataset:
    """Returns the AgentHarm dataset (harmful or benign split per config)."""
    try:
        # Official Inspect Evals integration. See:
        # https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm
        from inspect_evals.agentharm.dataset import load_dataset as _load_agentharm_dataset

        dataset = _load_agentharm_dataset(
            split=AGENTHARM_CONFIG.split,
        )
        if AGENTHARM_CONFIG.limit is not None:
            samples = list(dataset)[: AGENTHARM_CONFIG.limit]
            dataset = MemoryDataset(samples=samples, name=f"agentharm_{AGENTHARM_CONFIG.split}")
        return dataset

    except Exception as e:  # ImportError, missing data files, etc.
        print(
            f"[agentharm_loader] Falling back to placeholder dataset -- "
            f"could not load real AgentHarm ({type(e).__name__}: {e}). "
            f"Install `inspect_evals` (see requirements.txt) to use the real benchmark."
        )
        return _load_placeholder()


def _load_placeholder() -> Dataset:
    with _PLACEHOLDER_PATH.open() as f:
        raw_samples = json.load(f)

    samples = [
        Sample(
            id=item["id"],
            input=item["prompt"],
            target=item.get("target", ""),
            metadata={
                "category": item.get("category", "unknown"),
                "is_placeholder": True,
            },
        )
        for item in raw_samples
    ]
    limit = AGENTHARM_CONFIG.limit
    if limit is not None:
        samples = samples[:limit]
    return MemoryDataset(samples=samples, name="agentharm_placeholder")