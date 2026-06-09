"""Structured workspace logging helpers for pipeline runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class PipelineRunLogger:
    events_path: Path
    story_iterations_path: Path
    story_summary_path: Path
    visual_iterations_path: Path
    global_visual_iterations_path: Path
    _lock: Lock = field(default_factory=Lock)

    def log_event(self, stage: str, message: str, **details: Any) -> None:
        payload = {
            "timestamp": _utc_now(),
            "stage": stage,
            "message": message,
            "details": details,
        }
        self._append_jsonl(self.events_path, payload)

    def log_story_iteration(
        self,
        *,
        iteration: int,
        feedback_used: str,
        playwright_output: list[dict[str, Any]],
        validation_error: str | None,
        review: dict[str, Any] | None,
        accepted: bool,
    ) -> None:
        payload = {
            "timestamp": _utc_now(),
            "iteration": iteration,
            "feedback_used": feedback_used,
            "playwright_output": playwright_output,
            "validation_error": validation_error,
            "review": review,
            "accepted": accepted,
        }
        self._append_jsonl(self.story_iterations_path, payload)

    def write_story_summary(
        self,
        *,
        target_words: list[str],
        accepted_iteration: int | None,
        fallback_iteration: int | None,
        max_iterations: int,
        score_threshold: float,
        iterations: list[dict[str, Any]],
    ) -> None:
        lines = [
            "# Story Iteration Summary",
            "",
            f"- target_words: {', '.join(target_words)}",
            f"- score_threshold: {score_threshold}",
            f"- max_iterations: {max_iterations}",
            f"- accepted_iteration: {accepted_iteration if accepted_iteration is not None else 'none'}",
            f"- fallback_iteration: {fallback_iteration if fallback_iteration is not None else 'none'}",
            "",
            "## Iterations",
            "",
        ]
        if not iterations:
            lines.append("- No iterations recorded.")
        for record in iterations:
            review = record.get("review") or {}
            lines.extend(
                [
                    f"### Round {record['iteration']}",
                    "",
                    f"- accepted: {record['accepted']}",
                    f"- score: {review.get('score', 'n/a')}",
                    f"- passed: {review.get('passed', 'n/a')}",
                    f"- feedback_used: {record['feedback_used'] or 'none'}",
                    f"- validation_error: {record['validation_error'] or 'none'}",
                    f"- educator_feedback: {review.get('feedback', 'n/a')}",
                    "",
                ]
            )
        self.story_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log_visual_iteration(
        self,
        *,
        scene_index: int,
        iteration: int,
        director_prompt: str,
        image_url: str,
        review: dict[str, Any],
        approved: bool,
    ) -> None:
        payload = {
            "timestamp": _utc_now(),
            "scene_index": scene_index,
            "iteration": iteration,
            "director_prompt": director_prompt,
            "image_url": image_url,
            "review": review,
            "approved": approved,
        }
        self._append_jsonl(self.visual_iterations_path, payload)

    def log_global_visual_iteration(
        self,
        *,
        iteration: int,
        review: dict[str, Any],
        targeted_scene_indexes: list[int],
    ) -> None:
        payload = {
            "timestamp": _utc_now(),
            "iteration": iteration,
            "review": review,
            "targeted_scene_indexes": targeted_scene_indexes,
        }
        self._append_jsonl(self.global_visual_iterations_path, payload)

    def _append_jsonl(self, file_path: Path, payload: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
