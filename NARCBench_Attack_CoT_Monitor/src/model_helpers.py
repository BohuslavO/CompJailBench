import json
from typing import Any


def parse_json_response(text: str) -> dict[str, Any]:
    # Try to extract a JSON object from the model response.
    #
    # Important:
    # A safety refusal is NOT a program failure.
    # If the model refuses instead of returning JSON,
    # we preserve the raw response and mark it as a refusal.

    cleaned = text.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = cleaned[start:end + 1]

        try:
            result = json.loads(json_text)

            if isinstance(result, dict):
                result["_refusal"] = False
                return result

        except json.JSONDecodeError:
            pass

    # No usable JSON was returned.
    # Preserve the model's real output rather than crashing.
    return {
        "_refusal": True,
        "_raw_response": cleaned,
    }


def validate_vote(vote: str) -> str:
    # Normalize normal A/B votes.

    cleaned = vote.strip().upper()

    if cleaned in {"A", "OPTION A"}:
        return "A"

    if cleaned in {"B", "OPTION B"}:
        return "B"

    # Never invent a vote when the model did not provide one.
    return "ABSTAIN"