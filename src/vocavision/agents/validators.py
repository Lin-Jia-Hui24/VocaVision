"""Local validation utilities for generated scenes."""

from __future__ import annotations

import re

from vocavision.exceptions import VocaVisionError
from vocavision.state import LearningPlan, Scene


def ensure_scene_count(learning_plan: LearningPlan, scenes: list[Scene]) -> None:
    expected_count = learning_plan.recommended_scene_count
    if expected_count != len(scenes):
        raise VocaVisionError(
            f"Expected {expected_count} scenes for learning mode '{learning_plan.mode}', got {len(scenes)}."
        )


def ensure_story_text_coverage(target_words: list[str], scenes: list[Scene]) -> None:
    unique_words = {word.strip().lower() for word in target_words if word and word.strip()}
    combined_text = " ".join(
        f"{scene.script.plot_description} {scene.script.voiceover_and_dialogue}" for scene in scenes
    )
    missing_words = [
        word
        for word in sorted(unique_words)
        if not re.search(rf"\b{re.escape(word)}\b", combined_text, flags=re.IGNORECASE)
    ]
    if missing_words:
        raise VocaVisionError(
            "The story text does not mention these target words anywhere: " + ", ".join(missing_words) + "."
        )


def ensure_target_word_coverage(target_words: list[str], scenes: list[Scene]) -> None:
    remaining_words = {word.lower() for word in target_words}
    for scene in scenes:
        remaining_words.discard(scene.target_word_in_scene.lower())
    if remaining_words:
        missing_words = ", ".join(sorted(remaining_words))
        raise VocaVisionError(f"The story does not assign any scene focus to: {missing_words}.")
