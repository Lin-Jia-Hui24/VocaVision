"""Shared runtime inspection helpers for CLI and web."""

from __future__ import annotations

from pathlib import Path
import shutil

from vocavision.config import VocavisionSettings
from vocavision.video_stack import inspect_video_stack


def has_real_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip()
    if not normalized:
        return False
    return normalized not in {
        "your_dashscope_api_key",
        "your_volcengine_ark_api_key",
    }


def validate_environment(settings: VocavisionSettings) -> dict[str, object]:
    return {
        "has_dashscope_api_key": has_real_secret(settings.dashscope_api_key),
        "has_ark_api_key": has_real_secret(settings.ark_api_key),
        "ffmpeg_found": shutil.which(settings.ffmpeg_bin) is not None,
        "ffprobe_found": shutil.which(settings.ffprobe_bin) is not None,
        "workspace_root": str(Path(settings.workspace_root).resolve()),
        "story_max_iterations": settings.story_max_iterations,
        "story_score_threshold": settings.story_score_threshold,
        "visual_max_retries": settings.visual_max_retries,
        "global_visual_max_rounds": settings.global_visual_max_rounds,
        "global_visual_score_threshold": settings.global_visual_score_threshold,
        "max_scenes_per_run": settings.max_scenes_per_run,
        "media_max_workers": settings.media_max_workers,
        "reuse_cached_assets": settings.reuse_cached_assets,
        "video_stack": inspect_video_stack(settings),
    }
