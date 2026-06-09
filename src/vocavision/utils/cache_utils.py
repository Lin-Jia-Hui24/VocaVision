"""Helpers for request signatures and sidecar metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_request_signature(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def load_metadata(output_path: Path) -> dict[str, Any] | None:
    metadata_path = metadata_path_for(output_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_metadata(output_path: Path, payload: dict[str, Any]) -> Path:
    metadata_path = metadata_path_for(output_path)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path
