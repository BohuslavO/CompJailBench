"""Defensive parsing helpers for model-generated structured output."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse the first JSON object while preserving malformed raw output."""

    cleaned = text.strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            result["_parse_ok"] = True
            result["_raw_response"] = cleaned
            return result
    return {"_parse_ok": False, "_raw_response": cleaned}


def validate_choice(value: str, allowed: Iterable[str]) -> str:
    """Normalize a recommendation without inventing a missing choice."""

    allowed_labels = [str(item).strip().upper() for item in allowed]
    cleaned = str(value).strip().upper()
    cleaned = re.sub(r"^OPTION\s+", "", cleaned)
    cleaned = cleaned.strip("[](){}*.:; ")
    return cleaned if cleaned in allowed_labels else "ABSTAIN"


def extract_recommendation(text: str, allowed: Iterable[str]) -> str:
    """Recover a documented recommendation from structured or plain output."""

    labels = [str(item).strip().upper() for item in allowed]
    parsed = parse_json_response(text)
    if parsed.get("_parse_ok"):
        for key in ("recommendation", "vote", "choice"):
            if key in parsed:
                choice = validate_choice(str(parsed[key]), labels)
                if choice != "ABSTAIN":
                    return choice
        response = str(parsed.get("response", ""))
        if response:
            recovered = extract_recommendation(response, labels)
            if recovered != "ABSTAIN":
                return recovered

    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    patterns = [
        rf"RECOMMENDATION\s*:\s*(?:\*{{0,2}}|\[)?\s*(?:OPTION\s+)?({label_pattern})\b",
        rf"\bOPTION\s+({label_pattern})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return validate_choice(match.group(1), labels)
    return "ABSTAIN"


def validate_vote(vote: str) -> str:
    """Backward-compatible A/B validator."""

    return validate_choice(vote, ("A", "B"))
