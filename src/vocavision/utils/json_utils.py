"""Helpers for parsing and validating structured model responses."""

from __future__ import annotations

import json
from typing import Any

from vocavision.exceptions import JsonResponseError


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonResponseError(f"Failed to parse JSON response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JsonResponseError("Expected a JSON object from the model.")
    return parsed
