import json
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample


# This file lives inside:
# compjailbenc/src/narcbench_dataset.py
#
# .parent       -> src
# .parent.parent -> compjailbenc (our project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build an absolute path to the dataset.
# This works regardless of which folder Inspect runs the task from.
DATA_PATH = PROJECT_ROOT / "data" / "narcbench_core_small.json"


def load_narcbench_dataset() -> MemoryDataset:
    """
    Convert our NARCBench JSON records into Inspect Samples.

    Each Sample:
    - has text that identifies the committee task;
    - keeps all NARCBench-specific information in metadata.
    """

    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    samples = []

    for record in records:

        # This is the human-readable task that the solver will work with.
        input_text = (
            f"{record['title']}\n\n"
            f"{record['context']}\n\n"
            f"Option A: {record['options']['A']}\n"
            f"Option B: {record['options']['B']}"
        )

        samples.append(
            Sample(
                id=record["sample_id"],
                input=input_text,

                # target_option is useful for checking whether
                # colluders managed to push the committee toward their target.
                target=record["target_option"],

                # Keep the original benchmark information attached
                # to the Inspect sample.
                metadata=record,
            )
        )

    return MemoryDataset(
        samples=samples,
        name="narcbench_core_small",
    )