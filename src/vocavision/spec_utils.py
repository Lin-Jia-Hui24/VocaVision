"""Helpers for normalizing user-provided target word specs."""

from __future__ import annotations

from typing import Any

from vocavision.state import TargetWordSenseCandidate, TargetWordSpec


def _normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _build_candidate(
    *,
    sense_id: str,
    label: str,
    part_of_speech: str,
    gloss_en: str,
    gloss_zh: str,
    visual_anchors: list[str],
    negative_anchors: list[str],
    example_sentence: str,
) -> TargetWordSenseCandidate:
    return TargetWordSenseCandidate(
        sense_id=sense_id,
        label=label,
        part_of_speech=part_of_speech,
        gloss_en=gloss_en,
        gloss_zh=gloss_zh,
        visual_anchors=list(visual_anchors),
        negative_anchors=list(negative_anchors),
        example_sentence=example_sentence,
    )


def normalize_selected_spec(
    spec: TargetWordSpec,
    *,
    fallback_word: str | None = None,
    fallback_index: int | None = None,
    confirmed_by_user: bool | None = None,
) -> TargetWordSpec:
    updated = spec.model_copy(deep=True)
    if fallback_word and not updated.word.strip():
        updated.word = fallback_word.strip()
    updated.word = updated.word.strip()
    if not updated.word:
        raise ValueError("Each target word spec must include a word.")

    selected_candidate = next(
        (candidate for candidate in updated.candidates if candidate.sense_id == updated.selected_sense_id),
        None,
    )
    if selected_candidate is None and updated.recommended_sense_id:
        selected_candidate = next(
            (candidate for candidate in updated.candidates if candidate.sense_id == updated.recommended_sense_id),
            None,
        )
    if selected_candidate is None and updated.candidates:
        selected_candidate = updated.candidates[0]

    if selected_candidate is None:
        fallback_suffix = fallback_index if fallback_index is not None else 1
        sense_id = (
            updated.selected_sense_id.strip()
            or updated.recommended_sense_id.strip()
            or f"{updated.word.lower()}_manual_{fallback_suffix}"
        )
        label = updated.selected_sense_label.strip() or updated.word
        selected_candidate = _build_candidate(
            sense_id=sense_id,
            label=label,
            part_of_speech=updated.part_of_speech.strip(),
            gloss_en=updated.gloss_en.strip(),
            gloss_zh=updated.gloss_zh.strip(),
            visual_anchors=_normalize_list(updated.visual_anchors),
            negative_anchors=_normalize_list(updated.negative_anchors),
            example_sentence=updated.example_sentence.strip(),
        )
        updated.candidates = [selected_candidate]

    updated.recommended_sense_id = selected_candidate.sense_id
    updated.selected_sense_id = selected_candidate.sense_id
    updated.selected_sense_label = selected_candidate.label
    updated.part_of_speech = selected_candidate.part_of_speech
    updated.gloss_en = selected_candidate.gloss_en
    updated.gloss_zh = selected_candidate.gloss_zh
    updated.visual_anchors = list(selected_candidate.visual_anchors)
    updated.negative_anchors = list(selected_candidate.negative_anchors)
    updated.example_sentence = selected_candidate.example_sentence
    updated.needs_user_confirmation = False
    if confirmed_by_user is not None:
        updated.confirmed_by_user = confirmed_by_user
    return updated


def coerce_target_word_specs(
    raw_specs: list[TargetWordSpec | dict[str, Any]],
    *,
    fallback_words: list[str] | None = None,
    confirmed_by_user: bool | None = None,
) -> list[TargetWordSpec]:
    normalized_specs: list[TargetWordSpec] = []
    fallback_words = fallback_words or []
    for index, raw_spec in enumerate(raw_specs, start=1):
        fallback_word = fallback_words[index - 1] if index - 1 < len(fallback_words) else None
        if isinstance(raw_spec, TargetWordSpec):
            spec = raw_spec.model_copy(deep=True)
        else:
            payload = dict(raw_spec)
            word = str(payload.get("word") or fallback_word or "").strip()
            if not word:
                raise ValueError("Each target word spec must include a word.")
            if not payload.get("candidates") and not payload.get("selected_sense_id") and payload.get("sense_id"):
                payload["selected_sense_id"] = str(payload.get("sense_id", "")).strip()
            if not payload.get("selected_sense_label") and payload.get("label"):
                payload["selected_sense_label"] = str(payload.get("label", "")).strip()
            payload["word"] = word
            spec = TargetWordSpec.model_validate(payload)
        normalized_specs.append(
            normalize_selected_spec(
                spec,
                fallback_word=fallback_word,
                fallback_index=index,
                confirmed_by_user=confirmed_by_user,
            )
        )
    return normalized_specs
