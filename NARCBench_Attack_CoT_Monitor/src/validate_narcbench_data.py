import json
from pathlib import Path
from typing import Any


# Path to the small NARCBench dataset you just created.
DATA_PATH = Path("data/narcbench_core_small.json")


def load_samples() -> list[dict[str, Any]]:
    """Load the JSON file and return its samples."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the dataset: {DATA_PATH}"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    samples = load_samples()

    # We currently expect exactly one matched pair:
    # one collusion case and one honest-control case.
    if len(samples) != 2:
        raise ValueError(
            "Expected exactly two samples: collusion and control."
        )

    required_fields = {
        "sample_id",
        "scenario_id",
        "condition",
        "context",
        "options",
        "fair_choice",
        "target_option",
        "agents",
        "colluders",
        "honest_mandate",
        "public_frame",
        "final_frame",
        "discussion_rounds",
    }

    for sample in samples:
        # Check that no required information is missing.
        missing = required_fields - sample.keys()

        if missing:
            raise ValueError(
                f"{sample.get('sample_id', 'unknown')} "
                f"is missing: {sorted(missing)}"
            )

        # Your methodology states that NARCBench uses four agents.
        if len(sample["agents"]) != 4:
            raise ValueError(
                f"{sample['sample_id']} must contain four agents."
            )

        # Only these two experimental conditions are allowed.
        if sample["condition"] not in {"collusion", "control"}:
            raise ValueError(
                f"{sample['sample_id']} has an invalid condition."
            )

        # The target must be one of the two listed options.
        if sample["target_option"] not in sample["options"]:
            raise ValueError(
                f"{sample['sample_id']} has an invalid target option."
            )

        if sample["condition"] == "collusion":
            # Attack cases must contain exactly two colluders.
            if len(sample["colluders"]) != 2:
                raise ValueError(
                    "The collusion sample must contain two colluders."
                )

            # Attack cases require the private attack instructions.
            if not sample.get("deception_instruction"):
                raise ValueError(
                    "The collusion sample needs a deception instruction."
                )

        if sample["condition"] == "control":
            # Honest controls must not secretly contain colluders.
            if sample["colluders"]:
                raise ValueError(
                    "The control sample must have no colluders."
                )

    # Check that both samples represent the same original scenario.
    if samples[0]["scenario_id"] != samples[1]["scenario_id"]:
        raise ValueError(
            "Collusion and control must use the same scenario."
        )

    print("NARCBench matched pair validated successfully.")
    print(f"Scenario: {samples[0]['scenario_id']}")
    print("Conditions: collusion, control")
    print("Agents per sample: 4")


if __name__ == "__main__":
    main()