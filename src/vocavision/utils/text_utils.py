"""Text-related helpers for scene validation and subtitle highlighting."""

from __future__ import annotations

import re


HIGHLIGHT_START = r"{\c&H507FFF&}"
HIGHLIGHT_END = r"{\c&HFFFFFF&}"


def _build_target_word_pattern(target_words: list[str]) -> re.Pattern[str] | None:
    normalized_words = [word.strip() for word in target_words if word.strip()]
    if not normalized_words:
        return None
    deduplicated_words = sorted({word.lower(): word for word in normalized_words}.values(), key=len, reverse=True)
    return re.compile(
        rf"\b({'|'.join(re.escape(word) for word in deduplicated_words)})\b",
        flags=re.IGNORECASE,
    )


def highlight_target_words(text: str, target_words: list[str]) -> str:
    pattern = _build_target_word_pattern(target_words)
    if pattern is None:
        return text
    return pattern.sub(lambda match: f"{HIGHLIGHT_START}{match.group(1)}{HIGHLIGHT_END}", text)


def mask_target_words(
    text: str,
    target_words: list[str],
    *,
    replacement: str = "_____",
    highlight_mask: bool = False,
) -> str:
    pattern = _build_target_word_pattern(target_words)
    if pattern is None:
        return text
    replacement_text = (
        f"{HIGHLIGHT_START}{replacement}{HIGHLIGHT_END}" if highlight_mask else replacement
    )
    return pattern.sub(lambda _match: replacement_text, text)


def normalize_spoken_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    return normalized


def normalize_words(words: list[str]) -> list[str]:
    return [word.strip() for word in words if word.strip()]


def normalize_visible_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text).upper()
