"""Lightweight web console for VocaVision."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vocavision.config import VocavisionSettings, default_scene_cap_for_words
from vocavision.pipeline import VocaVisionPipeline
from vocavision.runtime_report import validate_environment
from vocavision.spec_utils import coerce_target_word_specs
from vocavision.state import ProjectRunSettings, TargetWordSenseCandidate, TargetWordSpec, VideoProjectState
from vocavision.utils.text_utils import mask_target_words
from vocavision.utils.text_utils import normalize_words
from vocavision.workspace import ProjectWorkspace


WEB_ROOT = Path(__file__).resolve().parent / "web"
ADMIN_COOKIE_NAME = "vocavision_researcher"
STAGE_SEQUENCE = [
    "pipeline",
    "sense_disambiguation",
    "story",
    "character_design",
    "visual",
    "global_visual",
    "media",
    "finalize",
]
ARTIFACT_NAMES = {
    "final_video": ("final", "final_video.mp4"),
    "final_cloze_video": ("final", "final_cloze_video.mp4"),
    "state": ("state", "project_state.json"),
    "events_log": ("logs", "events.jsonl"),
    "story_iterations": ("logs", "story_iterations.jsonl"),
    "visual_iterations": ("logs", "visual_iterations.jsonl"),
    "global_visual_iterations": ("logs", "global_visual_iterations.jsonl"),
    "story_summary": ("logs", "story_iteration_summary.md"),
}
USER_ARTIFACT_NAMES = {
    "final_video": ("final", "final_video.mp4"),
    "final_cloze_video": ("final", "final_cloze_video.mp4"),
    "state": ("state", "project_state.json"),
    "events_log": ("logs", "events.jsonl"),
    "story_iterations": ("logs", "story_iterations.jsonl"),
    "visual_iterations": ("logs", "visual_iterations.jsonl"),
    "global_visual_iterations": ("logs", "global_visual_iterations.jsonl"),
    "story_summary": ("logs", "story_iteration_summary.md"),
}
STUDY_EVENT_TYPES = {
    "session_started",
    "step_changed",
    "video_play",
    "video_pause",
    "video_ended",
    "video_progress",
    "exercise_answered",
    "survey_submitted",
    "pairwise_started",
    "pairwise_rating_submitted",
}


def _researcher_password() -> str:
    return os.getenv("VOCAVISION_RESEARCHER_PASSWORD", "").strip()


def _admin_auth_enabled() -> bool:
    return bool(_researcher_password())


def _admin_cookie_value() -> str:
    password = _researcher_password()
    return hashlib.sha256(f"vocavision-researcher:{password}".encode("utf-8")).hexdigest()


def _has_researcher_access(request: Request) -> bool:
    password = _researcher_password()
    if not password:
        return True
    header_value = request.headers.get("x-vocavision-admin-key", "")
    if header_value and hmac.compare_digest(header_value, password):
        return True
    cookie_value = request.cookies.get(ADMIN_COOKIE_NAME, "")
    return bool(cookie_value and hmac.compare_digest(cookie_value, _admin_cookie_value()))


def _require_researcher_access(request: Request) -> None:
    if not _has_researcher_access(request):
        raise HTTPException(status_code=401, detail="Researcher access required.")


def _login_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VocaVision 研究者登录</title>
    <link rel="stylesheet" href="/assets/study.css" />
  </head>
  <body>
    <main class="study-shell">
      <section class="setup-panel">
        <div class="setup-copy">
          <p class="eyebrow">研究者入口</p>
          <h1>登录后使用生成控制台</h1>
          <p>学生体验入口仍然公开在 /study。生成面板和生成 API 需要研究者密码。</p>
        </div>
        <form id="admin-login-form" class="study-form">
          <label>
            <span>研究者密码</span>
            <input id="admin-password" type="password" autocomplete="current-password" autofocus />
          </label>
          <p id="admin-login-error" class="error-text" role="alert"></p>
          <button class="primary-button" type="submit">进入控制台</button>
          <a class="secondary-button" href="/study">去学生体验入口</a>
        </form>
      </section>
    </main>
    <script src="/assets/admin_login.js" defer></script>
  </body>
</html>"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _format_timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify_project_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _build_project_word_slug(words: list[str], *, max_words: int = 3, max_chars: int = 48) -> str:
    parts: list[str] = []
    for word in words[:max_words]:
        token = _slugify_project_token(word)
        if token:
            parts.append(token)
    if not parts:
        return "project"
    slug = "-".join(parts)
    if len(words) > max_words:
        slug = f"{slug}-more"
    return slug[:max_chars].rstrip("-") or "project"


def _generate_project_id(words: list[str]) -> str:
    return f"{_build_project_word_slug(words)}-{_format_timestamp_slug()}"


def _resolve_project_id(raw_project_id: str, words: list[str]) -> str:
    return raw_project_id.strip() or _generate_project_id(words)


def _split_words(words_text: str) -> list[str]:
    raw_words = [item for item in re.split(r"[\s,，、;；\n\r\t]+", words_text.strip()) if item]
    return [word.lower() for word in normalize_words(raw_words)]


def _select_candidate(
    spec: TargetWordSpec,
    candidate: TargetWordSenseCandidate,
    *,
    confirmed_by_user: bool,
) -> TargetWordSpec:
    updated = spec.model_copy(deep=True)
    updated.selected_sense_id = candidate.sense_id
    updated.selected_sense_label = candidate.label
    updated.part_of_speech = candidate.part_of_speech
    updated.gloss_en = candidate.gloss_en
    updated.gloss_zh = candidate.gloss_zh
    updated.visual_anchors = list(candidate.visual_anchors)
    updated.negative_anchors = list(candidate.negative_anchors)
    updated.example_sentence = candidate.example_sentence
    updated.confirmed_by_user = confirmed_by_user
    updated.needs_user_confirmation = False
    return updated


def _resolve_selected_specs(
    specs: list[TargetWordSpec],
    *,
    auto_accept: bool,
) -> list[TargetWordSpec]:
    resolved_specs: list[TargetWordSpec] = []
    for spec in specs:
        if spec.candidates:
            selected_candidate = next(
                (candidate for candidate in spec.candidates if candidate.sense_id == spec.selected_sense_id),
                None,
            )
            if selected_candidate is None:
                selected_candidate = next(
                    (candidate for candidate in spec.candidates if candidate.sense_id == spec.recommended_sense_id),
                    spec.candidates[0],
                )
            resolved_specs.append(_select_candidate(spec, selected_candidate, confirmed_by_user=not auto_accept))
            continue
        resolved_specs.append(spec.model_copy(deep=True))
    return resolved_specs


def _build_settings_for_request(
    *,
    words: list[str],
    test_mode: bool,
    max_scenes: int | None,
    media_workers: int | None,
    storyboard_only: bool = False,
    story_score_threshold: float | None = None,
    global_visual_score_threshold: float | None = None,
) -> VocavisionSettings:
    settings = VocavisionSettings.from_env()
    settings.storyboard_only = storyboard_only
    if story_score_threshold is not None:
        settings.story_score_threshold = max(0.0, story_score_threshold)
    if global_visual_score_threshold is not None:
        settings.global_visual_score_threshold = max(0.0, global_visual_score_threshold)
    if test_mode:
        settings.max_scenes_per_run = 2 if max_scenes is None else min(max_scenes, 2)
        requested_workers = 2 if media_workers is None else media_workers
        settings.media_max_workers = max(1, min(requested_workers, settings.max_scenes_per_run))
        return settings
    settings.max_scenes_per_run = max(1, max_scenes) if max_scenes is not None else default_scene_cap_for_words(words)
    if media_workers is not None:
        settings.media_max_workers = max(1, media_workers)
    elif settings.max_scenes_per_run is not None:
        settings.media_max_workers = max(1, min(settings.max_scenes_per_run, 5))
    else:
        settings.media_max_workers = 5
    return settings


def _read_jsonl(file_path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if limit <= 0:
        return entries
    return entries[-limit:]


def _load_state(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _artifact_url(project_id: str, artifact_name: str, file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    return f"/api/projects/{project_id}/artifacts/{artifact_name}"


def _scene_keyframe_url(project_id: str, scene_index: int, *, iteration: int | None = None) -> str:
    if iteration is None:
        return f"/api/projects/{project_id}/scenes/{scene_index}/keyframe"
    return f"/api/projects/{project_id}/scenes/{scene_index}/keyframes/{iteration}"


def _resolve_local_keyframe_path(project_root: Path, scene_index: int, *, iteration: int | None = None) -> Path:
    image_dir = project_root / "images"
    if iteration is None:
        return image_dir / f"scene_{scene_index:02d}_keyframe.jpeg"
    return image_dir / f"scene_{scene_index:02d}_keyframe_iter_{iteration:02d}.jpeg"


def _resolve_scene_image_url(
    project_root: Path,
    project_id: str,
    *,
    scene_index: Any,
    iteration: Any | None = None,
    preferred_iterations: list[Any] | None = None,
    fallback_url: str | None = None,
) -> str | None:
    try:
        normalized_scene_index = int(scene_index)
    except (TypeError, ValueError):
        return fallback_url
    candidate_iterations: list[int | None] = []
    if iteration is not None:
        try:
            candidate_iterations.append(int(iteration))
        except (TypeError, ValueError):
            pass
    for preferred_iteration in preferred_iterations or []:
        try:
            normalized_iteration = int(preferred_iteration)
        except (TypeError, ValueError):
            continue
        if normalized_iteration not in candidate_iterations:
            candidate_iterations.append(normalized_iteration)

    for normalized_iteration in candidate_iterations:
        local_path = _resolve_local_keyframe_path(
            project_root,
            normalized_scene_index,
            iteration=normalized_iteration,
        )
        if local_path.exists() and local_path.stat().st_size > 0:
            return _scene_keyframe_url(
                project_id,
                normalized_scene_index,
                iteration=normalized_iteration,
            )

    local_path = _resolve_local_keyframe_path(
        project_root,
        normalized_scene_index,
        iteration=None,
    )
    if local_path.exists() and local_path.stat().st_size > 0:
        return _scene_keyframe_url(
            project_id,
            normalized_scene_index,
            iteration=None,
        )

    fallback_candidates = sorted(
        (project_root / "images").glob(f"scene_{normalized_scene_index:02d}_keyframe_iter_*.jpeg"),
        reverse=True,
    )
    for candidate_path in fallback_candidates:
        if candidate_path.exists() and candidate_path.stat().st_size > 0:
            suffix = candidate_path.stem.rsplit("_iter_", 1)[-1]
            try:
                detected_iteration = int(suffix)
            except ValueError:
                continue
            return _scene_keyframe_url(
                project_id,
                normalized_scene_index,
                iteration=detected_iteration,
            )
    return fallback_url


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _limit_text(value: str, *, max_chars: int = 220) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}..."


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_text_list(values: Any, *, limit: int | None = None, max_chars: int = 160) -> list[str]:
    cleaned: list[str] = []
    ignored_markers = {"none", "n/a", "na", "null", "无", "暂无", "没有", "无问题"}
    for entry in values or []:
        text = _clean_text(entry)
        if not text:
            continue
        if text.casefold() in ignored_markers:
            continue
        cleaned.append(_limit_text(text, max_chars=max_chars))
    if limit is None:
        return cleaned
    return cleaned[:limit]


def _safe_int_list(values: Any, *, limit: int | None = None) -> list[int]:
    cleaned: list[int] = []
    for entry in values or []:
        try:
            cleaned.append(int(entry))
        except (TypeError, ValueError):
            continue
    if limit is None:
        return cleaned
    return cleaned[:limit]


def _build_story_draft_scenes(playwright_output: Any) -> list[dict[str, Any]]:
    draft_scenes: list[dict[str, Any]] = []
    for fallback_index, scene_payload in enumerate(playwright_output or [], start=1):
        if not isinstance(scene_payload, dict):
            continue
        script_payload = scene_payload.get("script") or {}
        continuity_items = []
        for item in script_payload.get("continuity_items", [])[:4]:
            if not isinstance(item, dict):
                continue
            continuity_items.append(
                {
                    "label": _clean_text(item.get("label") or item.get("item_key")),
                    "description": _limit_text(_clean_text(item.get("description")), max_chars=120),
                    "carry_state": _clean_text(item.get("carry_state")),
                }
            )
        draft_scenes.append(
            {
                "scene_index": int(scene_payload.get("scene_index") or fallback_index),
                "target_word_in_scene": _clean_text(scene_payload.get("target_word_in_scene")),
                "plot_description": _clean_text(script_payload.get("plot_description")),
                "voiceover_and_dialogue": _clean_text(script_payload.get("voiceover_and_dialogue")),
                "continuity_items": continuity_items,
            }
        )
    return draft_scenes


def _sanitize_target_word_spec(spec: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for candidate in spec.get("candidates", []):
        candidates.append(
            {
                "sense_id": str(candidate.get("sense_id", "")),
                "label": str(candidate.get("label", "")),
                "part_of_speech": str(candidate.get("part_of_speech", "")),
                "gloss_en": str(candidate.get("gloss_en", "")),
                "gloss_zh": str(candidate.get("gloss_zh", "")),
                "visual_anchors": [str(item) for item in candidate.get("visual_anchors", [])[:4]],
                "negative_anchors": [str(item) for item in candidate.get("negative_anchors", [])[:4]],
            }
        )
    return {
        "word": str(spec.get("word", "")),
        "source_word": str(spec.get("source_word", "")),
        "relation_to_source": str(spec.get("relation_to_source", "")),
        "recommended_sense_id": str(spec.get("recommended_sense_id", "")),
        "selected_sense_id": str(spec.get("selected_sense_id", "")),
        "selected_sense_label": str(spec.get("selected_sense_label", "")),
        "part_of_speech": str(spec.get("part_of_speech", "")),
        "gloss_en": str(spec.get("gloss_en", "")),
        "gloss_zh": str(spec.get("gloss_zh", "")),
        "confidence": _safe_float(spec.get("confidence")) or 0.0,
        "confirmed_by_user": bool(spec.get("confirmed_by_user")),
        "needs_user_confirmation": bool(spec.get("needs_user_confirmation")),
        "visual_anchors": [str(item) for item in spec.get("visual_anchors", [])[:4]],
        "negative_anchors": [str(item) for item in spec.get("negative_anchors", [])[:4]],
        "candidates": candidates,
    }


def _build_related_word_family(target_word_specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not target_word_specs:
        return None
    seed_spec = next(
        (
            spec
            for spec in target_word_specs
            if str(spec.get("relation_to_source", "")).strip().lower() == "seed_word"
        ),
        None,
    )
    if seed_spec is None:
        return None
    seed_word = str(seed_spec.get("word", "")).strip()
    if not seed_word:
        return None
    related_words: list[dict[str, Any]] = []
    for spec in target_word_specs:
        word = str(spec.get("word", "")).strip()
        if not word or word.casefold() == seed_word.casefold():
            continue
        related_words.append(
            {
                "word": word,
                "source_word": str(spec.get("source_word", "")).strip(),
                "relation_to_source": str(spec.get("relation_to_source", "")).strip(),
                "selected_sense_label": str(spec.get("selected_sense_label", "")).strip(),
                "gloss_en": str(spec.get("gloss_en", "")).strip(),
                "gloss_zh": str(spec.get("gloss_zh", "")).strip(),
                "visual_anchors": [str(item) for item in spec.get("visual_anchors", [])[:4]],
            }
        )
    return {
        "seed_word": seed_word,
        "total_words": 1 + len(related_words),
        "related_words": related_words,
    }


def _build_stage_summary(events: list[dict[str, Any]], job_status: str) -> dict[str, Any]:
    seen_stages: dict[str, dict[str, Any]] = {}
    for event in events:
        stage = str(event.get("stage", "")).strip()
        if not stage:
            continue
        seen_stages[stage] = event
    ordered_stages: list[dict[str, Any]] = []
    current_stage = None
    last_seen_index = -1
    for index, stage in enumerate(STAGE_SEQUENCE):
        event = seen_stages.get(stage)
        if event is not None:
            last_seen_index = index
            current_stage = stage
        if job_status == "succeeded":
            status = "completed"
        elif event is not None:
            status = "completed"
        elif index == last_seen_index + 1 and job_status in {"queued", "running"}:
            status = "current"
        else:
            status = "pending"
        ordered_stages.append(
            {
                "stage": stage,
                "status": status,
                "message": None if event is None else event.get("message"),
                "timestamp": None if event is None else event.get("timestamp"),
            }
        )
    if job_status == "failed":
        current_stage = current_stage or "pipeline"
    return {"current_stage": current_stage, "stages": ordered_stages}


def _build_progress_feed(events: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    feed: list[dict[str, Any]] = []
    for event in events[-limit:]:
        feed.append(
            {
                "stage": str(event.get("stage", "")),
                "message": _limit_text(str(event.get("message", "")), max_chars=120),
                "timestamp": event.get("timestamp"),
            }
        )
    return feed


def _build_scene_summary(scene: dict[str, Any]) -> dict[str, Any]:
    visual = scene.get("visual") or {}
    review = visual.get("review") or {}
    audio = scene.get("audio") or {}
    video = scene.get("video") or {}
    post_processing = scene.get("post_processing") or {}
    target_word_spec = scene.get("target_word_spec") or {}
    return {
        "scene_index": scene.get("scene_index"),
        "target_word_in_scene": scene.get("target_word_in_scene"),
        "selected_sense_label": target_word_spec.get("selected_sense_label"),
        "keyframe_image_url": visual.get("keyframe_image_url"),
        "visual_score": review.get("score"),
        "visual_match_level": review.get("match_level"),
        "selected_iteration": visual.get("selected_iteration"),
        "selected_via_fallback": visual.get("selected_via_fallback"),
        "spoken_text": audio.get("spoken_text"),
        "duration_sec": audio.get("duration_sec"),
        "raw_video_ready": bool(video.get("anim_path")),
        "merged_video_ready": bool(post_processing.get("final_merged_path")),
        "cloze_video_ready": bool(post_processing.get("cloze_merged_path")),
        "scene_video_path": post_processing.get("final_merged_path"),
        "scene_cloze_video_path": post_processing.get("cloze_merged_path"),
    }


def _build_run_settings_payload(run_settings: dict[str, Any] | None) -> dict[str, Any]:
    payload = run_settings or {}
    return {
        "learning_mode": str(payload.get("learning_mode", "auto")),
        "storyboard_only": bool(payload.get("storyboard_only", False)),
        "test_mode": bool(payload.get("test_mode", False)),
        "max_scenes": payload.get("max_scenes"),
        "media_workers": payload.get("media_workers"),
        "story_score_threshold": payload.get("story_score_threshold"),
        "global_visual_score_threshold": payload.get("global_visual_score_threshold"),
        "auto_accept_senses": bool(payload.get("auto_accept_senses", True)),
    }


def _build_project_run_settings(
    *,
    learning_mode: str,
    storyboard_only: bool,
    test_mode: bool,
    max_scenes: int | None,
    media_workers: int | None,
    story_score_threshold: float | None,
    global_visual_score_threshold: float | None,
    auto_accept_senses: bool,
) -> ProjectRunSettings:
    return ProjectRunSettings(
        learning_mode=learning_mode,
        storyboard_only=storyboard_only,
        test_mode=test_mode,
        max_scenes=max_scenes,
        media_workers=media_workers,
        story_score_threshold=story_score_threshold,
        global_visual_score_threshold=global_visual_score_threshold,
        auto_accept_senses=auto_accept_senses,
    )


def _project_has_all_keyframes(state_payload: dict[str, Any] | None) -> bool:
    if state_payload is None:
        return False
    scenes = state_payload.get("scenes", [])
    if not scenes:
        return False
    return all(bool((scene.get("visual") or {}).get("keyframe_image_url")) for scene in scenes)


def _infer_render_profile(
    state_payload: dict[str, Any] | None,
    *,
    final_video_ready: bool,
) -> str | None:
    if state_payload is None:
        return None
    explicit_profile = state_payload.get("render_profile")
    if explicit_profile:
        return str(explicit_profile)
    if final_video_ready:
        return "full_video"
    if _project_has_all_keyframes(state_payload):
        return "storybook_only"
    return "full_video"


def _build_storybook_review_cards(
    scenes: list[dict[str, Any]],
    *,
    project_id: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for scene in scenes:
        spoken_text = str((scene.get("audio") or {}).get("spoken_text", "")).strip() or str(
            (scene.get("script") or {}).get("voiceover_and_dialogue", "")
        ).strip()
        target_word = str(scene.get("target_word_in_scene", "")).strip()
        target_word_spec = scene.get("target_word_spec") or {}
        post_processing = scene.get("post_processing") or {}
        cards.append(
            {
                "scene_index": scene.get("scene_index"),
                "target_word": target_word,
                "selected_sense_label": target_word_spec.get("selected_sense_label"),
                "image_url": _resolve_scene_image_url(
                    project_root,
                    project_id,
                    scene_index=scene.get("scene_index"),
                    preferred_iterations=[
                        (scene.get("visual") or {}).get("selected_iteration"),
                        (scene.get("visual") or {}).get("approved_iteration"),
                        ((scene.get("visual") or {}).get("review") or {}).get("iteration"),
                    ],
                    fallback_url=(scene.get("visual") or {}).get("keyframe_image_url"),
                ),
                "spoken_text": spoken_text,
                "cloze_text": mask_target_words(spoken_text, [target_word]) if spoken_text and target_word else spoken_text,
                "scene_video_url": (
                    f"/api/projects/{project_id}/scenes/{scene.get('scene_index')}/video"
                    if post_processing.get("final_merged_path")
                    else None
                ),
                "scene_cloze_video_url": (
                    f"/api/projects/{project_id}/scenes/{scene.get('scene_index')}/cloze-video"
                    if post_processing.get("cloze_merged_path")
                    else None
                ),
                "render_profile": scene.get("render_profile"),
            }
        )
    return cards


def _build_learning_exercises_panel(state_payload: dict[str, Any] | None) -> dict[str, Any]:
    if state_payload is None:
        return {"recommended_interaction_mode": "multiple_choice", "cloze_challenges": [], "practice_questions": []}
    learning_exercises = state_payload.get("learning_exercises") or {}
    return {
        "recommended_interaction_mode": str(
            learning_exercises.get("recommended_interaction_mode", "multiple_choice")
        ),
        "cloze_challenges": list(learning_exercises.get("cloze_challenges", [])),
        "practice_questions": list(learning_exercises.get("practice_questions", [])),
    }


def _safe_storage_token(value: str, *, field_name: str) -> str:
    token = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", token):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}.")
    return token


def _study_sessions_root(settings: VocavisionSettings) -> Path:
    return Path(settings.workspace_root) / "study_sessions"


def _study_session_dir(settings: VocavisionSettings, session_id: str) -> Path:
    safe_session_id = _safe_storage_token(session_id, field_name="session_id")
    return _study_sessions_root(settings) / safe_session_id


def _load_study_session(settings: VocavisionSettings, session_id: str) -> dict[str, Any]:
    session_path = _study_session_dir(settings, session_id) / "session.json"
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Study session not found.")
    return json.loads(session_path.read_text(encoding="utf-8"))


def _generate_study_session_id() -> str:
    return f"study-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


def _generate_anonymous_label() -> str:
    return f"匿名学习者 {secrets.randbelow(9000) + 1000}"


def _load_project_state_payload(settings: VocavisionSettings, project_id: str) -> dict[str, Any]:
    safe_project_id = _safe_storage_token(project_id, field_name="project_id")
    state_path = Path(settings.workspace_root) / safe_project_id / "state" / "project_state.json"
    payload = _load_state(state_path)
    if payload is None:
        raise HTTPException(status_code=404, detail="Study package not found.")
    return payload


def _build_study_question(question: dict[str, Any], *, question_id: str, group: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "group": group,
        "question_type": str(question.get("question_type", "multiple_choice")),
        "question_category": str(question.get("question_category", "")),
        "prompt": str(question.get("prompt", "")),
        "options": [str(option) for option in question.get("options", [])],
        "related_words": [str(word) for word in question.get("related_words", [])],
        "recommended_scene_indices": [int(index) for index in question.get("recommended_scene_indices", [])],
        "target_word": str(question.get("target_word", "")),
        "scene_index": question.get("scene_index"),
    }


def _build_study_exercises(state_payload: dict[str, Any]) -> dict[str, Any]:
    exercises = _build_learning_exercises_panel(state_payload)
    cloze_questions = []
    for index, question in enumerate(exercises["cloze_challenges"], start=1):
        question_id = f"cloze-{question.get('scene_index') or index}-{index}"
        cloze_questions.append(_build_study_question(question, question_id=question_id, group="cloze"))
    practice_questions = []
    for index, question in enumerate(exercises["practice_questions"], start=1):
        question_id = str(question.get("question_id") or f"practice-{index}")
        practice_questions.append(_build_study_question(question, question_id=question_id, group="practice"))
    return {
        "recommended_interaction_mode": exercises["recommended_interaction_mode"],
        "cloze_challenges": cloze_questions,
        "practice_questions": practice_questions,
    }


def _find_study_question_answer(
    state_payload: dict[str, Any],
    *,
    question_group: str,
    question_id: str,
) -> dict[str, Any] | None:
    exercises = _build_learning_exercises_panel(state_payload)
    if question_group == "cloze":
        for index, question in enumerate(exercises["cloze_challenges"], start=1):
            candidate_id = f"cloze-{question.get('scene_index') or index}-{index}"
            if candidate_id == question_id:
                return question
        return None
    if question_group == "practice":
        for index, question in enumerate(exercises["practice_questions"], start=1):
            candidate_id = str(question.get("question_id") or f"practice-{index}")
            if candidate_id == question_id:
                return question
    return None


def _build_study_package_payload(settings: VocavisionSettings, project_id: str) -> dict[str, Any]:
    safe_project_id = _safe_storage_token(project_id, field_name="project_id")
    project_root = Path(settings.workspace_root) / safe_project_id
    final_video_path = project_root / "final" / "final_video.mp4"
    if not final_video_path.exists():
        raise HTTPException(status_code=404, detail="Study package has no final video.")
    state_payload = _load_project_state_payload(settings, safe_project_id)
    learning_plan = state_payload.get("learning_plan") or {}
    run_settings = _build_run_settings_payload(state_payload.get("run_settings"))
    storybook_review = _build_storybook_review_cards(
        state_payload.get("scenes", []),
        project_id=safe_project_id,
        project_root=project_root,
    )
    return {
        "package_id": safe_project_id,
        "title": " / ".join([str(word) for word in state_payload.get("target_words", [])]) or safe_project_id,
        "target_words": [str(word) for word in state_payload.get("target_words", [])],
        "learning_mode": str(learning_plan.get("mode") or run_settings.get("learning_mode") or "auto"),
        "requested_learning_mode": str(run_settings.get("learning_mode") or "auto"),
        "scene_count": len(state_payload.get("scenes", [])),
        "final_video_url": f"/api/projects/{safe_project_id}/artifacts/final_video",
        "final_cloze_video_url": _artifact_url(
            safe_project_id,
            "final_cloze_video",
            project_root / "final" / "final_cloze_video.mp4",
        ),
        "storybook_review": storybook_review,
        "learning_exercises": _build_study_exercises(state_payload),
    }


def _load_pairwise_manifest(settings: VocavisionSettings) -> list[dict[str, Any]]:
    manifest_path = Path(settings.workspace_root) / "study_pairwise_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    raw_items = payload.get("pairs", payload if isinstance(payload, list) else [])
    pairs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id") or f"pair-{index}")
        full_project_id = str(item.get("full_project_id", "")).strip()
        ablation_project_id = str(item.get("ablation_project_id", "")).strip()
        if not full_project_id or not ablation_project_id:
            continue
        try:
            full_package = _build_study_package_payload(settings, full_project_id)
            ablation_package = _build_study_package_payload(settings, ablation_project_id)
        except HTTPException:
            continue
        pairs.append(
            {
                "pair_id": pair_id,
                "title": str(item.get("title") or full_package["title"]),
                "target_words": full_package["target_words"],
                "comparison_focus": str(item.get("comparison_focus") or "overall_design"),
                "ablation_label": str(item.get("ablation_label") or "ablation"),
                "full_project_id": full_project_id,
                "ablation_project_id": ablation_project_id,
                "full_video_url": full_package["final_video_url"],
                "ablation_video_url": ablation_package["final_video_url"],
                "learning_mode": full_package["learning_mode"],
            }
        )
    return pairs


def _pairwise_orders_root(settings: VocavisionSettings) -> Path:
    return Path(settings.workspace_root) / "study_pairwise_orders"


def _save_pairwise_order(settings: VocavisionSettings, answer_key: dict[str, Any]) -> str:
    order_token = f"order-{secrets.token_hex(12)}"
    order_root = _pairwise_orders_root(settings)
    order_root.mkdir(parents=True, exist_ok=True)
    (order_root / f"{order_token}.json").write_text(
        json.dumps({"created_at": _utc_now(), "answer_key": answer_key}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return order_token


def _load_pairwise_order(settings: VocavisionSettings, order_token: str) -> dict[str, Any] | None:
    if not re.fullmatch(r"order-[a-f0-9]{24}", order_token):
        return None
    order_path = _pairwise_orders_root(settings) / f"{order_token}.json"
    if not order_path.exists():
        return None
    try:
        return json.loads(order_path.read_text(encoding="utf-8")).get("answer_key")
    except json.JSONDecodeError:
        return None


def _mask_pairwise_pair(settings: VocavisionSettings, pair: dict[str, Any]) -> dict[str, Any]:
    swap = bool(secrets.randbits(1))
    left_role = "ablation" if swap else "full"
    right_role = "full" if swap else "ablation"
    full_package = _build_study_package_payload(settings, pair["full_project_id"])
    ablation_package = _build_study_package_payload(settings, pair["ablation_project_id"])
    left_package = ablation_package if swap else full_package
    right_package = full_package if swap else ablation_package
    answer_key = {
        "left_role": left_role,
        "right_role": right_role,
        "full_project_id": pair["full_project_id"],
        "ablation_project_id": pair["ablation_project_id"],
        "ablation_label": pair["ablation_label"],
    }
    return {
        "pair_id": pair["pair_id"],
        "title": pair["title"],
        "target_words": pair["target_words"],
        "comparison_focus": pair["comparison_focus"],
        "learning_mode": pair["learning_mode"],
        "order_token": _save_pairwise_order(settings, answer_key),
        "session_package_id": pair["full_project_id"],
        "left": {
            "label": "版本 A",
            "package": left_package,
        },
        "right": {
            "label": "版本 B",
            "package": right_package,
        },
    }


def _list_study_packages(settings: VocavisionSettings) -> list[dict[str, Any]]:
    packages = []
    for project in _list_recent_projects(settings, limit=100):
        if not project.get("has_final_video"):
            continue
        try:
            package = _build_study_package_payload(settings, str(project["project_id"]))
        except HTTPException:
            continue
        packages.append(
            {
                "package_id": package["package_id"],
                "title": package["title"],
                "target_words": package["target_words"],
                "learning_mode": package["learning_mode"],
                "requested_learning_mode": package["requested_learning_mode"],
                "scene_count": package["scene_count"],
                "updated_at": project.get("updated_at"),
            }
        )
    return packages


def _write_study_event(
    settings: VocavisionSettings,
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if event_type not in STUDY_EVENT_TYPES:
        raise HTTPException(status_code=400, detail="Unknown study event type.")
    session_dir = _study_session_dir(settings, session_id)
    session_path = session_dir / "session.json"
    session = _load_study_session(settings, session_id)
    event = {
        "timestamp": _utc_now(),
        "event_type": event_type,
        "payload": payload,
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    summary = session.setdefault("summary", {})
    summary["last_event_at"] = event["timestamp"]
    summary["event_count"] = int(summary.get("event_count", 0)) + 1
    if event_type == "video_play":
        summary["video_play_count"] = int(summary.get("video_play_count", 0)) + 1
    elif event_type == "video_progress":
        summary["last_video_time_sec"] = payload.get("current_time_sec")
        summary["last_video_duration_sec"] = payload.get("duration_sec")
    elif event_type == "exercise_answered":
        exercise_summary = summary.setdefault("exercise", {"answered": 0, "correct": 0})
        exercise_summary["answered"] = int(exercise_summary.get("answered", 0)) + 1
        if payload.get("is_correct") is True:
            exercise_summary["correct"] = int(exercise_summary.get("correct", 0)) + 1
    elif event_type == "survey_submitted":
        summary["survey_submitted"] = True
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    return event


def _build_story_review_panel(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    accepted_iteration: int | None = None
    for item in iterations:
        review = item.get("review") or {}
        validation_issue = str(item.get("validation_error") or "").strip()
        accepted = bool(item.get("accepted"))
        draft_scenes = _build_story_draft_scenes(item.get("playwright_output"))
        if accepted:
            accepted_iteration = int(item.get("iteration", 0) or 0)
        rounds.append(
            {
                "timestamp": item.get("timestamp"),
                "iteration": int(item.get("iteration", 0) or 0),
                "accepted": accepted,
                "passed": bool(review.get("passed")),
                "score": _safe_float(review.get("score")),
                "summary": str(
                    review.get("feedback") or validation_issue or "The story is being revised to make the teaching clearer."
                ).strip(),
                "feedback_used": _limit_text(_clean_text(item.get("feedback_used") or "none"), max_chars=220),
                "strengths": _clean_text_list(review.get("strengths"), limit=3),
                "improvement_focus": _clean_text_list(review.get("improvement_focus"), limit=3),
                "validation_issue": validation_issue.strip() if validation_issue else "",
                "scene_count": len(draft_scenes),
                "draft_scenes": draft_scenes,
            }
        )
    return {
        "accepted_iteration": accepted_iteration,
        "round_count": len(rounds),
        "rounds": rounds,
    }


def _build_visual_review_panel(
    visual_iterations: list[dict[str, Any]],
    scene_summaries: list[dict[str, Any]],
    *,
    project_root: Path,
    project_id: str,
) -> list[dict[str, Any]]:
    scene_lookup = {int(scene["scene_index"]): scene for scene in scene_summaries if scene.get("scene_index") is not None}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in visual_iterations:
        scene_index = int(item.get("scene_index", 0) or 0)
        grouped.setdefault(scene_index, []).append(item)

    panel_items: list[dict[str, Any]] = []
    for scene_index in sorted(grouped):
        scene_summary = scene_lookup.get(scene_index, {})
        rounds: list[dict[str, Any]] = []
        for item in grouped[scene_index]:
            review = item.get("review") or {}
            director_feedback = review.get("director_feedback") or {}
            rounds.append(
                {
                    "timestamp": item.get("timestamp"),
                    "iteration": int(item.get("iteration", 0) or 0),
                    "approved": bool(item.get("approved")),
                    "score": _safe_float(review.get("score")),
                    "match_level": str(review.get("match_level", "")),
                    "summary": str(
                        review.get("reason")
                        or director_feedback.get("summary")
                        or "The image is being adjusted to match the teaching scene."
                    ).strip(),
                    "image_url": _resolve_scene_image_url(
                        project_root,
                        project_id,
                        scene_index=scene_index,
                        iteration=item.get("iteration"),
                        fallback_url=_clean_text(item.get("image_url")) or None,
                    ),
                    "visual_issues": _clean_text_list(director_feedback.get("visual_issues"), limit=3),
                    "suggestions": _clean_text_list(director_feedback.get("optimization_suggestions"), limit=3),
                    "prompt_adjustments": _clean_text_list(
                        director_feedback.get("recommended_prompt_adjustments"),
                        limit=3,
                    ),
                    "repair_instruction": _clean_text(director_feedback.get("repair_instruction")),
                    "regeneration_mode": _clean_text(review.get("regeneration_mode")),
                    "revised_plot_description": _clean_text(review.get("revised_plot_description")),
                    "revised_voiceover_and_dialogue": _clean_text(review.get("revised_voiceover_and_dialogue")),
                    "has_visible_target_word_text": bool(review.get("has_visible_target_word_text")),
                    "observed_text": _clean_text(review.get("observed_text")),
                    "text_legibility_passed": review.get("text_legibility_passed"),
                    "text_legibility_reason": _clean_text(review.get("text_legibility_reason")),
                }
            )
        panel_items.append(
            {
                "scene_index": scene_index,
                "target_word_in_scene": scene_summary.get("target_word_in_scene", ""),
                "selected_sense_label": scene_summary.get("selected_sense_label", ""),
                "selected_iteration": scene_summary.get("selected_iteration"),
                "selected_via_fallback": bool(scene_summary.get("selected_via_fallback")),
                "visual_score": scene_summary.get("visual_score"),
                "final_image_url": scene_summary.get("keyframe_image_url"),
                "rounds": rounds,
            }
        )
    return panel_items


def _build_global_review_panel(global_visual_iterations: list[dict[str, Any]]) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    for item in global_visual_iterations:
        review = item.get("review") or {}
        scene_feedback_payload = review.get("scene_feedback") or {}
        scene_script_feedback_payload = review.get("scene_script_feedback") or {}
        scene_feedback = []
        for raw_scene_index, feedback_payload in scene_feedback_payload.items():
            try:
                scene_index = int(raw_scene_index)
            except (TypeError, ValueError):
                continue
            if not isinstance(feedback_payload, dict):
                continue
            scene_feedback.append(
                {
                    "scene_index": scene_index,
                    "summary": _clean_text(feedback_payload.get("summary")),
                    "visual_issues": _clean_text_list(feedback_payload.get("visual_issues"), limit=3),
                    "suggestions": _clean_text_list(feedback_payload.get("optimization_suggestions"), limit=3),
                    "prompt_adjustments": _clean_text_list(
                        feedback_payload.get("recommended_prompt_adjustments"),
                        limit=3,
                    ),
                    "repair_instruction": _clean_text(feedback_payload.get("repair_instruction")),
                }
            )
        scene_feedback.sort(key=lambda entry: entry["scene_index"])
        scene_script_feedback = []
        for raw_scene_index, feedback_payload in scene_script_feedback_payload.items():
            try:
                scene_index = int(raw_scene_index)
            except (TypeError, ValueError):
                continue
            if not isinstance(feedback_payload, dict):
                continue
            scene_script_feedback.append(
                {
                    "scene_index": scene_index,
                    "summary": _clean_text(feedback_payload.get("summary")),
                    "script_issues": _clean_text_list(feedback_payload.get("script_issues"), limit=3),
                    "revised_plot_description": _clean_text(feedback_payload.get("revised_plot_description")),
                    "revised_voiceover_and_dialogue": _clean_text(
                        feedback_payload.get("revised_voiceover_and_dialogue")
                    ),
                }
            )
        scene_script_feedback.sort(key=lambda entry: entry["scene_index"])
        rounds.append(
            {
                "timestamp": item.get("timestamp"),
                "iteration": int(item.get("iteration", 0) or 0),
                "passed": bool(review.get("passed")),
                "score": _safe_float(review.get("score")),
                "summary": str(
                    review.get("feedback") or "The system is checking whether the full set of scenes feels consistent."
                ).strip(),
                "problem_scenes": _safe_int_list(review.get("problem_scenes"), limit=6),
                "style_adjustments": _clean_text_list(review.get("global_style_adjustments"), limit=4),
                "blocking_issues": _clean_text_list(review.get("blocking_issues"), limit=4),
                "targeted_scene_indexes": _safe_int_list(item.get("targeted_scene_indexes"), limit=6),
                "scene_feedback": scene_feedback,
                "scene_script_feedback": scene_script_feedback,
            }
        )
    return {
        "round_count": len(rounds),
        "rounds": rounds,
    }


def _collect_project_snapshot(settings: VocavisionSettings, project_id: str, job_status: str) -> dict[str, Any]:
    workspace_root = Path(settings.workspace_root) / project_id
    state_path = workspace_root / "state" / "project_state.json"
    events_path = workspace_root / "logs" / "events.jsonl"
    story_iterations_path = workspace_root / "logs" / "story_iterations.jsonl"
    visual_iterations_path = workspace_root / "logs" / "visual_iterations.jsonl"
    global_visual_iterations_path = workspace_root / "logs" / "global_visual_iterations.jsonl"
    final_video_path = workspace_root / "final" / "final_video.mp4"
    final_cloze_video_path = workspace_root / "final" / "final_cloze_video.mp4"
    state_payload = _load_state(state_path)
    final_video_ready = final_video_path.exists()
    events = _read_jsonl(events_path, limit=60)
    story_iterations = _read_jsonl(story_iterations_path, limit=8)
    visual_iterations = _read_jsonl(visual_iterations_path, limit=12)
    global_visual_iterations = _read_jsonl(global_visual_iterations_path, limit=6)
    scene_summaries = []
    target_word_specs: list[dict[str, Any]] = []
    if state_payload is not None:
        scene_summaries = [_build_scene_summary(scene) for scene in state_payload.get("scenes", [])]
        for scene_summary in scene_summaries:
            scene_summary["keyframe_image_url"] = _resolve_scene_image_url(
                workspace_root,
                project_id,
                scene_index=scene_summary.get("scene_index"),
                preferred_iterations=[
                    scene_summary.get("selected_iteration"),
                ],
                fallback_url=scene_summary.get("keyframe_image_url"),
            )
        target_word_specs = [_sanitize_target_word_spec(spec) for spec in state_payload.get("target_word_specs", [])]
    storybook_review = (
        _build_storybook_review_cards(
            state_payload.get("scenes", []),
            project_id=project_id,
            project_root=workspace_root,
        )
        if state_payload is not None
        else []
    )
    run_settings = (
        _build_run_settings_payload(state_payload.get("run_settings"))
        if state_payload is not None
        else _build_run_settings_payload(None)
    )
    render_profile = _infer_render_profile(state_payload, final_video_ready=final_video_ready)
    can_promote_storyboard = bool(state_payload is not None and not final_video_ready and _project_has_all_keyframes(state_payload))
    artifacts = {
        artifact_name: _artifact_url(project_id, artifact_name, workspace_root / folder / filename)
        for artifact_name, (folder, filename) in USER_ARTIFACT_NAMES.items()
    }
    return {
        "final_video_url": artifacts["final_video"],
        "final_cloze_video_url": artifacts["final_cloze_video"],
        "artifacts": artifacts,
        "progress_feed": _build_progress_feed(events),
        "stage_summary": _build_stage_summary(events, job_status),
        "state": state_payload,
        "render_profile": render_profile,
        "can_promote_storyboard": can_promote_storyboard,
        "run_settings": run_settings,
        "scene_summaries": scene_summaries,
        "storybook_review": storybook_review,
        "learning_exercises": _build_learning_exercises_panel(state_payload),
        "target_word_specs": target_word_specs,
        "related_word_family": _build_related_word_family(target_word_specs),
        "story_review_panel": _build_story_review_panel(story_iterations),
        "visual_review_panel": _build_visual_review_panel(
            visual_iterations,
            scene_summaries,
            project_root=workspace_root,
            project_id=project_id,
        ),
        "global_review_panel": _build_global_review_panel(global_visual_iterations),
    }


def _list_recent_projects(settings: VocavisionSettings, *, limit: int | None = 10) -> list[dict[str, Any]]:
    workspace_root = Path(settings.workspace_root)
    if not workspace_root.exists():
        return []
    project_entries: list[tuple[float, dict[str, Any]]] = []
    for directory in workspace_root.iterdir():
        if not directory.is_dir():
            continue
        state_path = directory / "state" / "project_state.json"
        final_video_path = directory / "final" / "final_video.mp4"
        final_cloze_video_path = directory / "final" / "final_cloze_video.mp4"
        state_payload = _load_state(state_path) or {}
        project_entries.append(
            (
                directory.stat().st_mtime,
                {
                    "project_id": directory.name,
                    "updated_at": datetime.fromtimestamp(directory.stat().st_mtime, tz=UTC).isoformat(),
                    "target_words": list(state_payload.get("target_words", [])),
                    "scene_count": len(state_payload.get("scenes", [])),
                    "render_profile": _infer_render_profile(
                        state_payload,
                        final_video_ready=final_video_path.exists(),
                    ),
                    "can_promote_storyboard": bool(
                        not final_video_path.exists() and _project_has_all_keyframes(state_payload)
                    ),
                    "has_storyboard_review": any(
                        bool((scene.get("visual") or {}).get("keyframe_image_url"))
                        for scene in state_payload.get("scenes", [])
                    ),
                    "has_final_video": final_video_path.exists(),
                    "has_final_cloze_video": final_cloze_video_path.exists(),
                },
            )
        )
    project_entries.sort(key=lambda item: item[0], reverse=True)
    payloads = [payload for _, payload in project_entries]
    if limit is None:
        return payloads
    return payloads[:limit]


class SenseSuggestionRequest(BaseModel):
    project_id: str = ""
    words_text: str
    target_word_specs: list[dict[str, Any]] = Field(default_factory=list)
    max_scenes: int | None = Field(default=None, ge=1)
    media_workers: int | None = Field(default=None, ge=1)
    storyboard_only: bool = False
    story_score_threshold: float | None = Field(default=None, ge=0)
    global_visual_score_threshold: float | None = Field(default=None, ge=0)
    test_mode: bool = False
    learning_mode: str = "auto"
    auto_accept_senses: bool = True


class RunJobRequest(BaseModel):
    project_id: str = ""
    words_text: str
    target_word_specs: list[dict[str, Any]] = Field(default_factory=list)
    learning_mode: str = "auto"
    max_scenes: int | None = Field(default=None, ge=1)
    media_workers: int | None = Field(default=None, ge=1)
    storyboard_only: bool = False
    story_score_threshold: float | None = Field(default=None, ge=0)
    global_visual_score_threshold: float | None = Field(default=None, ge=0)
    test_mode: bool = False
    auto_accept_senses: bool = True


class PromoteStoryboardRequest(BaseModel):
    project_id: str
    media_workers: int | None = Field(default=None, ge=1)


class StudySessionRequest(BaseModel):
    package_id: str
    class_code: str = Field(default="", max_length=80)
    grade_band: str = Field(default="", max_length=40)
    english_level: str = Field(default="", max_length=40)
    word_familiarity: str = Field(default="", max_length=40)


class StudyEventRequest(BaseModel):
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class JobRecord:
    project_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    final_video_path: str | None = None
    state_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "final_video_path": self.final_video_path,
            "state_path": self.state_path,
        }


class WebJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def submit(self, request: RunJobRequest) -> JobRecord:
        words = _split_words(request.words_text)
        project_id = _resolve_project_id(request.project_id, words)
        return self._reserve_job(project_id, lambda: self._run_job(request, project_id))

    def submit_storyboard_render(self, request: PromoteStoryboardRequest) -> JobRecord:
        project_id = request.project_id.strip()
        if not project_id:
            raise ValueError("A project_id is required.")
        return self._reserve_job(
            project_id,
            lambda: self._render_storyboard_job(request, project_id),
        )

    def _reserve_job(self, project_id: str, runner: Any) -> JobRecord:
        with self._lock:
            existing = self._jobs.get(project_id)
            if existing is not None and existing.status in {"queued", "running"}:
                raise ValueError(f"Project '{project_id}' is already running.")
            record = JobRecord(project_id=project_id)
            self._jobs[project_id] = record
        self._executor.submit(runner)
        return record

    def get(self, project_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(project_id)
            return None if record is None else JobRecord(**record.to_dict())

    def _run_job(self, request: RunJobRequest, project_id: str) -> None:
        with self._lock:
            record = self._jobs[project_id]
            record.status = "running"
            record.started_at = _utc_now()
            record.error = None
        try:
            request.project_id = project_id
            words = _split_words(request.words_text)
            settings = _build_settings_for_request(
                words=words,
                test_mode=request.test_mode,
                max_scenes=request.max_scenes,
                media_workers=request.media_workers,
                storyboard_only=request.storyboard_only,
                story_score_threshold=request.story_score_threshold,
                global_visual_score_threshold=request.global_visual_score_threshold,
            )
            pipeline = VocaVisionPipeline.from_settings(settings)
            if request.target_word_specs:
                resolved_specs = coerce_target_word_specs(
                    request.target_word_specs,
                    fallback_words=words,
                    confirmed_by_user=None,
                )
            else:
                suggested_specs = pipeline.suggest_target_word_specs(words)
                resolved_specs = _resolve_selected_specs(suggested_specs, auto_accept=request.auto_accept_senses)
            state, final_video_path = pipeline.run(
                project_id=request.project_id,
                target_words=words,
                target_word_specs=resolved_specs,
                learning_mode=request.learning_mode,
            )
            state.run_settings = _build_project_run_settings(
                learning_mode=request.learning_mode,
                storyboard_only=request.storyboard_only,
                test_mode=request.test_mode,
                max_scenes=request.max_scenes,
                media_workers=request.media_workers,
                story_score_threshold=request.story_score_threshold,
                global_visual_score_threshold=request.global_visual_score_threshold,
                auto_accept_senses=request.auto_accept_senses,
            )
            workspace = ProjectWorkspace.create(settings.workspace_root, state.project_id)
            state.save(workspace.state_path())
        except Exception as exc:  # pragma: no cover - defensive for background task reporting.
            with self._lock:
                record = self._jobs[project_id]
                record.status = "failed"
                record.finished_at = _utc_now()
                record.error = str(exc)
            return
        with self._lock:
            record = self._jobs[project_id]
            record.status = "succeeded"
            record.finished_at = _utc_now()
            record.final_video_path = None if final_video_path is None else str(final_video_path.resolve())
            record.state_path = str((settings.workspace_root / state.project_id / "state" / "project_state.json").resolve())

    def _render_storyboard_job(self, request: PromoteStoryboardRequest, project_id: str) -> None:
        with self._lock:
            record = self._jobs[project_id]
            record.status = "running"
            record.started_at = _utc_now()
            record.error = None
        try:
            settings = VocavisionSettings.from_env()
            if request.media_workers is not None:
                settings.media_max_workers = max(1, request.media_workers)
            settings.storyboard_only = False
            pipeline = VocaVisionPipeline.from_settings(settings)
            state, final_video_path = pipeline.render_storyboard_project_to_video(project_id=project_id)
        except Exception as exc:  # pragma: no cover - defensive for background task reporting.
            with self._lock:
                record = self._jobs[project_id]
                record.status = "failed"
                record.finished_at = _utc_now()
                record.error = str(exc)
            return
        with self._lock:
            record = self._jobs[project_id]
            record.status = "succeeded"
            record.finished_at = _utc_now()
            record.final_video_path = str(final_video_path.resolve())
            record.state_path = str((settings.workspace_root / state.project_id / "state" / "project_state.json").resolve())


def create_app() -> FastAPI:
    app = FastAPI(title="VocaVision Web Console")
    app.mount("/assets", StaticFiles(directory=str(WEB_ROOT)), name="assets")
    job_manager = WebJobManager()

    @app.get("/", response_class=HTMLResponse, response_model=None)
    def index(request: Request) -> Response:
        if not _has_researcher_access(request):
            return RedirectResponse(url="/admin-login", status_code=303)
        return HTMLResponse((WEB_ROOT / "index.html").read_text(encoding="utf-8"))

    @app.get("/admin-login", response_class=HTMLResponse, response_model=None)
    def admin_login(request: Request) -> Response:
        if _has_researcher_access(request):
            return RedirectResponse(url="/", status_code=303)
        return HTMLResponse(_login_page_html())

    @app.post("/api/admin/session")
    def create_admin_session(payload: dict[str, str], response: Response) -> dict[str, Any]:
        password = _researcher_password()
        if not password:
            return {"ok": True, "auth_enabled": False}
        submitted_password = str(payload.get("password", ""))
        if not hmac.compare_digest(submitted_password, password):
            raise HTTPException(status_code=401, detail="研究者密码不正确。")
        response.set_cookie(
            ADMIN_COOKIE_NAME,
            _admin_cookie_value(),
            httponly=True,
            samesite="lax",
            secure=os.getenv("VOCAVISION_SECURE_COOKIES", "false").strip().lower() in {"1", "true", "yes"},
        )
        return {"ok": True, "auth_enabled": True}

    @app.post("/api/admin/logout")
    def logout_admin(response: Response) -> dict[str, Any]:
        response.delete_cookie(ADMIN_COOKIE_NAME)
        return {"ok": True}

    @app.get("/study", response_class=HTMLResponse)
    def study_index() -> HTMLResponse:
        return HTMLResponse((WEB_ROOT / "study.html").read_text(encoding="utf-8"))

    @app.get("/api/overview")
    def get_overview(request: Request) -> dict[str, Any]:
        _require_researcher_access(request)
        settings = VocavisionSettings.from_env()
        return {
            "environment": validate_environment(settings),
            "recent_projects": _list_recent_projects(settings, limit=None),
        }

    @app.get("/api/study/packages")
    def list_study_package_api() -> dict[str, Any]:
        settings = VocavisionSettings.from_env()
        return {"packages": _list_study_packages(settings)}

    @app.get("/api/study/packages/{package_id}")
    def get_study_package_api(package_id: str) -> dict[str, Any]:
        settings = VocavisionSettings.from_env()
        return {"package": _build_study_package_payload(settings, package_id)}

    @app.get("/api/study/pairwise")
    def list_pairwise_pairs_api() -> dict[str, Any]:
        settings = VocavisionSettings.from_env()
        raw_pairs = _load_pairwise_manifest(settings)
        secrets.SystemRandom().shuffle(raw_pairs)
        pairs = [_mask_pairwise_pair(settings, pair) for pair in raw_pairs]
        return {"pairs": pairs}

    @app.post("/api/study/sessions")
    def create_study_session_api(request: StudySessionRequest) -> dict[str, Any]:
        settings = VocavisionSettings.from_env()
        package = _build_study_package_payload(settings, request.package_id)
        session_id = _generate_study_session_id()
        anonymous_label = _generate_anonymous_label()
        session_dir = _study_session_dir(settings, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)
        session = {
            "session_id": session_id,
            "anonymous_label": anonymous_label,
            "created_at": _utc_now(),
            "package_id": package["package_id"],
            "target_words": package["target_words"],
            "learning_mode": package["learning_mode"],
            "requested_learning_mode": package["requested_learning_mode"],
            "precheck": {
                "class_code": request.class_code.strip(),
                "grade_band": request.grade_band.strip(),
                "english_level": request.english_level.strip(),
                "word_familiarity": request.word_familiarity.strip(),
            },
            "summary": {
                "event_count": 0,
                "video_play_count": 0,
                "survey_submitted": False,
            },
        }
        (session_dir / "session.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_study_event(
            settings,
            session_id,
            "session_started",
            {
                "package_id": package["package_id"],
                "learning_mode": package["learning_mode"],
                "requested_learning_mode": package["requested_learning_mode"],
                "word_count": len(package["target_words"]),
            },
        )
        return {
            "session_id": session_id,
            "anonymous_label": anonymous_label,
            "package": package,
        }

    @app.post("/api/study/sessions/{session_id}/events")
    def record_study_event_api(session_id: str, request: StudyEventRequest) -> dict[str, Any]:
        settings = VocavisionSettings.from_env()
        session = _load_study_session(settings, session_id)
        payload = dict(request.payload)
        payload["package_id"] = str(session.get("package_id", ""))
        payload.setdefault("learning_mode", session.get("learning_mode"))
        payload.setdefault("requested_learning_mode", session.get("requested_learning_mode"))
        payload.setdefault("word_count", len(session.get("target_words", [])))
        if request.event_type in {"pairwise_started", "pairwise_rating_submitted"}:
            answer_key = _load_pairwise_order(settings, str(payload.get("order_token", "")))
            if answer_key is not None:
                payload["answer_key"] = answer_key
        result: dict[str, Any] = {"ok": True}
        if request.event_type == "exercise_answered":
            question_group = str(payload.get("question_group", ""))
            question_id = str(payload.get("question_id", ""))
            selected_answer = str(payload.get("selected_answer", ""))
            state_payload = _load_project_state_payload(settings, str(session.get("package_id", "")))
            question = _find_study_question_answer(
                state_payload,
                question_group=question_group,
                question_id=question_id,
            )
            if question is None:
                raise HTTPException(status_code=404, detail="Question not found.")
            correct_answer = str(question.get("correct_answer", ""))
            is_correct = selected_answer.strip().casefold() == correct_answer.strip().casefold()
            payload.update(
                {
                    "is_correct": is_correct,
                    "correct_answer": correct_answer,
                    "explanation": str(question.get("explanation", "")),
                }
            )
            result.update(
                {
                    "is_correct": is_correct,
                    "correct_answer": correct_answer,
                    "explanation": payload["explanation"],
                }
            )
        event = _write_study_event(settings, session_id, request.event_type, payload)
        result["event"] = event
        return result

    @app.post("/api/senses")
    def suggest_senses(request_body: SenseSuggestionRequest, request: Request) -> dict[str, Any]:
        _require_researcher_access(request)
        words = _split_words(request_body.words_text)
        if not words:
            raise HTTPException(status_code=400, detail="Please provide at least one target word.")
        settings = VocavisionSettings.from_env()
        project_id = _resolve_project_id(request_body.project_id, words)
        pipeline = VocaVisionPipeline.from_settings(settings)
        if request_body.target_word_specs:
            resolved_specs = coerce_target_word_specs(
                request_body.target_word_specs,
                fallback_words=words,
                confirmed_by_user=True,
            )
        else:
            specs = pipeline.suggest_target_word_specs(words, apply_scene_limit=False)
            resolved_specs = _resolve_selected_specs(specs, auto_accept=True)
        effective_specs, learning_plan = pipeline.suggest_learning_plan_preview(
            target_word_specs=resolved_specs,
            learning_mode=request_body.learning_mode,
            apply_scene_limit=False,
            use_model_planner=False,
        )
        workspace = ProjectWorkspace.create(settings.workspace_root, project_id)
        state = VideoProjectState(
            project_id=project_id,
            target_words=words,
            target_word_specs=[spec.model_copy(deep=True) for spec in resolved_specs],
            run_settings=_build_project_run_settings(
                learning_mode=request_body.learning_mode,
                storyboard_only=request_body.storyboard_only,
                test_mode=request_body.test_mode,
                max_scenes=request_body.max_scenes,
                media_workers=request_body.media_workers,
                story_score_threshold=request_body.story_score_threshold,
                global_visual_score_threshold=request_body.global_visual_score_threshold,
                auto_accept_senses=request_body.auto_accept_senses,
            ),
            learning_plan=learning_plan.model_copy(deep=True),
        )
        state.save(workspace.state_path())
        return {
            "project_id": project_id,
            "words": words,
            "learning_mode": request_body.learning_mode,
            "learning_plan": learning_plan.model_dump(),
            "target_word_specs": [_sanitize_target_word_spec(spec.model_dump()) for spec in resolved_specs],
            "effective_target_word_specs": [_sanitize_target_word_spec(spec.model_dump()) for spec in effective_specs],
            "related_word_family": _build_related_word_family(
                [_sanitize_target_word_spec(spec.model_dump()) for spec in effective_specs]
            ),
        }

    @app.post("/api/jobs")
    def start_job(request_body: RunJobRequest, request: Request) -> dict[str, Any]:
        _require_researcher_access(request)
        words = _split_words(request_body.words_text)
        if not words:
            raise HTTPException(status_code=400, detail="Please provide at least one target word.")
        try:
            record = job_manager.submit(request_body)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        settings = _build_settings_for_request(
            words=words,
            test_mode=request_body.test_mode,
            max_scenes=request_body.max_scenes,
            media_workers=request_body.media_workers,
        )
        snapshot = _collect_project_snapshot(settings, record.project_id, record.status)
        return {"job": record.to_dict(), "snapshot": snapshot}

    @app.post("/api/projects/{project_id}/render-video")
    def render_storyboard_video(
        project_id: str,
        request_body: PromoteStoryboardRequest,
        request: Request,
    ) -> dict[str, Any]:
        _require_researcher_access(request)
        normalized_project_id = project_id.strip()
        if not normalized_project_id:
            raise HTTPException(status_code=400, detail="A project_id is required.")
        if request_body.project_id.strip() != normalized_project_id:
            raise HTTPException(status_code=400, detail="Path project_id does not match request body.")
        settings = VocavisionSettings.from_env()
        snapshot = _collect_project_snapshot(settings, normalized_project_id, "idle")
        if not snapshot.get("state"):
            raise HTTPException(status_code=404, detail=f"Project '{normalized_project_id}' was not found.")
        if snapshot.get("final_video_url"):
            raise HTTPException(status_code=409, detail="This project already has a final video.")
        if not snapshot.get("can_promote_storyboard"):
            raise HTTPException(
                status_code=409,
                detail="This project does not have a complete storyboard ready for video promotion.",
            )
        try:
            record = job_manager.submit_storyboard_render(request_body)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        snapshot = _collect_project_snapshot(settings, normalized_project_id, record.status)
        return {"job": record.to_dict(), "snapshot": snapshot}

    @app.get("/api/jobs/{project_id}")
    def get_job(project_id: str, request: Request) -> dict[str, Any]:
        _require_researcher_access(request)
        settings = VocavisionSettings.from_env()
        record = job_manager.get(project_id) or JobRecord(project_id=project_id, status="idle")
        snapshot = _collect_project_snapshot(settings, project_id, record.status)
        return {"job": record.to_dict(), "snapshot": snapshot}

    @app.get("/api/projects")
    def list_projects(request: Request) -> dict[str, Any]:
        _require_researcher_access(request)
        settings = VocavisionSettings.from_env()
        return {"projects": _list_recent_projects(settings, limit=None)}

    @app.get("/api/projects/{project_id}/artifacts/{artifact_name}")
    def get_artifact(project_id: str, artifact_name: str, request: Request) -> FileResponse:
        if artifact_name not in ARTIFACT_NAMES:
            raise HTTPException(status_code=404, detail="Unknown artifact.")
        if artifact_name not in {"final_video", "final_cloze_video"}:
            _require_researcher_access(request)
        settings = VocavisionSettings.from_env()
        folder, filename = ARTIFACT_NAMES[artifact_name]
        file_path = Path(settings.workspace_root) / project_id / folder / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(file_path)

    @app.get("/api/projects/{project_id}/scenes/{scene_index}/keyframe")
    def get_scene_keyframe(project_id: str, scene_index: int) -> FileResponse:
        settings = VocavisionSettings.from_env()
        file_path = _resolve_local_keyframe_path(
            Path(settings.workspace_root) / project_id,
            scene_index,
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Scene keyframe not found.")
        return FileResponse(file_path)

    @app.get("/api/projects/{project_id}/scenes/{scene_index}/keyframes/{iteration}")
    def get_scene_keyframe_iteration(project_id: str, scene_index: int, iteration: int) -> FileResponse:
        settings = VocavisionSettings.from_env()
        file_path = _resolve_local_keyframe_path(
            Path(settings.workspace_root) / project_id,
            scene_index,
            iteration=iteration,
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Scene keyframe iteration not found.")
        return FileResponse(file_path)

    @app.get("/api/projects/{project_id}/scenes/{scene_index}/{variant}")
    def get_scene_media(project_id: str, scene_index: int, variant: str) -> FileResponse:
        settings = VocavisionSettings.from_env()
        workspace_root = Path(settings.workspace_root) / project_id
        if variant == "video":
            file_path = workspace_root / "video" / f"scene_{scene_index:02d}_merged.mp4"
        elif variant == "cloze-video":
            file_path = workspace_root / "video" / f"scene_{scene_index:02d}_cloze_merged.mp4"
        else:
            raise HTTPException(status_code=404, detail="Unknown scene media variant.")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Scene media not found.")
        return FileResponse(file_path)

    return app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
