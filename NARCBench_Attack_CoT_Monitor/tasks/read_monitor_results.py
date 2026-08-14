# Read the three already-completed defense logs.
#
# This does NOT call Bedrock or any model.
# It only reads results that are already saved in .eval files.

from pathlib import Path

from inspect_ai.log import read_eval_log_samples


# Each entry contains:
# 1. A readable name
# 2. The saved Inspect log
# 3. The scorer whose result we want to inspect
LOGS = [
    (
        "ACTION ONLY",
        Path("logs/env02_action_monitor.eval"),
        "action_only_monitor",
    ),
    (
        "OBSERVABLE REASONING ONLY",
        Path("logs/env02_cot_monitor.eval"),
        "cot_only_monitor",
    ),
    (
        "ALL MESSAGES",
        Path("logs/env02_all_messages_monitor.eval"),
        "all_messages_monitor",
    ),
]


for label, log_path, scorer_name in LOGS:

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    # Read the two samples from this saved .eval file.
    for sample in read_eval_log_samples(log_path):

        print()
        print(f"Sample: {sample.id}")

        scores = sample.scores or {}

        # Find our defense scorer.
        score = scores.get(scorer_name)

        if score is None:
            print(f"ERROR: Could not find scorer '{scorer_name}'")
            print("Available scorers:", list(scores.keys()))
            continue

        metadata = score.metadata or {}

        print(
            "Prediction:",
            metadata.get("prediction", score.answer),
        )

        print(
            "Truth:",
            metadata.get("truth"),
        )

        print(
            "Suspiciousness:",
            metadata.get("suspiciousness"),
        )

        print(
            "Score value:",
            score.value,
        )