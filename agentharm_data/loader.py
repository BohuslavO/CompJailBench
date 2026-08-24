from __future__ import annotations

import copy
from collections.abc import Callable

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from config import AGENTHARM_CONFIG

OfficialLoader = Callable[[str, str, bool], Dataset]


def _official_loader(condition: str, split: str, force_redownload: bool) -> Dataset:
    # Importing the Hugging Face package `datasets` must work here. The old
    # local package named `datasets/` shadowed it and broke Inspect Evals.
    from inspect_evals.agentharm.utils import load_dataset

    return load_dataset(condition, split, force_redownload)


def _normalise_sample(sample: Sample, condition: str, split: str) -> Sample:
    normalised = copy.deepcopy(sample)
    metadata = dict(normalised.metadata or {})
    original_category = metadata.get("category", "unknown")
    pair_id = str(metadata.get("id", normalised.id))
    condition_label = "attack" if condition == "harmful" else "benign_control"
    metadata.update(
        {
            "agentharm_condition": condition,
            "agentharm_split": split,
            "condition": condition_label,
            "condition_label": condition_label,
            "pair_id": pair_id,
            "category_original": original_category,
            # Existing defense helpers use category == benign for controls.
            "category": original_category if condition == "harmful" else "benign",
            "is_placeholder": False,
        }
    )
    normalised.metadata = metadata
    return normalised


def load_agentharm(
    condition: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    force_redownload: bool | None = None,
    allow_placeholder: bool | None = None,
    loader: OfficialLoader | None = None,
) -> Dataset:
    """Load a real AgentHarm condition and fail closed on any data error."""
    condition = condition or AGENTHARM_CONFIG.condition
    split = split or AGENTHARM_CONFIG.split
    limit = AGENTHARM_CONFIG.limit if limit is None else limit
    force_redownload = (
        AGENTHARM_CONFIG.force_redownload
        if force_redownload is None
        else force_redownload
    )
    allow_placeholder = (
        AGENTHARM_CONFIG.allow_placeholder
        if allow_placeholder is None
        else allow_placeholder
    )

    if condition not in {"harmful", "benign"}:
        raise ValueError("condition must be 'harmful' or 'benign'")
    if split not in {"val", "test_public", "test_private"}:
        raise ValueError("split must be 'val', 'test_public', or 'test_private'")

    selected_loader = loader or _official_loader
    try:
        source = selected_loader(condition, split, force_redownload)
    except Exception as exc:
        if allow_placeholder:
            return _placeholder_dataset(condition, split, limit)
        raise RuntimeError(
            "Real AgentHarm data could not be loaded. The run was stopped before "
            "any model call; placeholder fallback is disabled for experiments. "
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc

    samples = [_normalise_sample(sample, condition, split) for sample in source]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"AgentHarm returned no {condition} samples for split {split}.")
    return MemoryDataset(samples=samples, name=f"agentharm_{condition}_{split}")


def load_paired_agentharm(
    split: str | None = None,
    limit: int | None = None,
    loader: OfficialLoader | None = None,
) -> tuple[Dataset, Dataset]:
    """Load harmful/benign arms and require identical ordered pair IDs."""
    split = split or AGENTHARM_CONFIG.split
    harmful = load_agentharm("harmful", split, limit, loader=loader)
    benign = load_agentharm("benign", split, limit, loader=loader)
    harmful_ids = [str((s.metadata or {}).get("pair_id")) for s in harmful]
    benign_ids = [str((s.metadata or {}).get("pair_id")) for s in benign]
    if harmful_ids != benign_ids:
        raise RuntimeError(
            "AgentHarm harmful and benign controls are not identically paired. "
            "Stop the experiment and inspect the selected dataset revision."
        )
    return harmful, benign


def _placeholder_dataset(condition: str, split: str, limit: int | None) -> Dataset:
    """Explicit demo-only placeholder; never reached in default configuration."""
    label = "attack" if condition == "harmful" else "benign_control"
    sample = Sample(
        id=f"placeholder_{condition}",
        input=f"PLACEHOLDER {condition} AgentHarm sample for UI testing only.",
        target="",
        metadata={
            "agentharm_condition": condition,
            "agentharm_split": split,
            "condition": label,
            "condition_label": label,
            "pair_id": "placeholder",
            "category": "placeholder" if condition == "harmful" else "benign",
            "category_original": "placeholder",
            "is_placeholder": True,
        },
    )
    samples = [sample] if limit is None or limit > 0 else []
    return MemoryDataset(samples=samples, name="agentharm_placeholder_demo_only")
