"""Runtime settings for VocaVision."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from vocavision.utils.text_utils import normalize_words


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_dotenv(dotenv_path: Path | None = None) -> Path:
    resolved_path = dotenv_path or PROJECT_ROOT / ".env"
    if not resolved_path.exists():
        return resolved_path

    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    return resolved_path


def _read_optional_int(env_name: str) -> int | None:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def _read_bool(env_name: str, default: bool) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def default_scene_cap_for_words(words: list[str]) -> int:
    normalized_words = normalize_words(words)
    return max(1, len(normalized_words) + 1)


def _read_story_max_iterations() -> int:
    explicit_value = _read_optional_int("VOCAVISION_STORY_MAX_ITERATIONS")
    if explicit_value is not None:
        return explicit_value
    retries = _read_optional_int("VOCAVISION_PLAYWRIGHT_MAX_RETRIES")
    if retries is not None:
        return retries + 1
    return 5


@dataclass(slots=True)
class VocavisionSettings:
    dashscope_api_key: str | None
    ark_api_key: str | None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_http_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_model: str = "qwen3.5-plus"
    vlm_model: str = "qwen3.5-plus"
    tts_model: str = "qwen3-tts-flash"
    tts_voice: str = "Cherry"
    tts_language: str = "English"
    image_model: str = "doubao-seedream-4-5-251128"
    image_size: str = "1312x736"
    video_model: str = "doubao-seedance-1-5-pro-251215"
    video_ratio: str = "16:9"
    video_duration_sec: int = 5
    video_resolution: str = "480p"
    request_timeout_sec: int = 120
    polling_interval_sec: int = 10
    polling_timeout_sec: int = 900
    video_polling_timeout_retries: int = 1
    video_polling_recovery_timeout_sec: int = 1800
    tts_missing_audio_url_retries: int = 2
    tts_retry_backoff_sec: float = 1.5
    playwright_max_retries: int = 2
    story_max_iterations: int = 5
    story_score_threshold: float = 8.0
    visual_max_retries: int = 3
    global_visual_max_rounds: int = 2
    global_visual_score_threshold: float = 8.0
    max_scenes_per_run: int | None = None
    media_max_workers: int = 5
    storyboard_only: bool = False
    reuse_cached_assets: bool = True
    auto_generate_comparison_variants: bool = False
    tts_playback_rate: float = 0.94
    tts_mix_volume: float = 1.0
    video_audio_mix_volume: float = 0.35
    workspace_root: Path = Path("workspace")
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    @classmethod
    def from_env(cls) -> "VocavisionSettings":
        load_project_dotenv()
        workspace_root = Path(os.getenv("VOCAVISION_WORKSPACE_ROOT", "workspace"))
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            ark_api_key=os.getenv("ARK_API_KEY"),
            image_model=os.getenv("VOCAVISION_IMAGE_MODEL", "doubao-seedream-4-5-251128"),
            image_size=os.getenv("VOCAVISION_IMAGE_SIZE", "2560x1440"),
            video_model=os.getenv("VOCAVISION_VIDEO_MODEL", "doubao-seedance-1-5-pro-251215"),
            story_max_iterations=_read_story_max_iterations(),
            story_score_threshold=float(os.getenv("VOCAVISION_STORY_SCORE_THRESHOLD", "8.0")),
            visual_max_retries=int(os.getenv("VOCAVISION_VISUAL_MAX_RETRIES", "3")),
            global_visual_max_rounds=int(os.getenv("VOCAVISION_GLOBAL_VISUAL_MAX_ROUNDS", "2")),
            global_visual_score_threshold=float(os.getenv("VOCAVISION_GLOBAL_VISUAL_SCORE_THRESHOLD", "8.0")),
            video_polling_timeout_retries=int(os.getenv("VOCAVISION_VIDEO_POLLING_TIMEOUT_RETRIES", "1")),
            video_polling_recovery_timeout_sec=int(
                os.getenv("VOCAVISION_VIDEO_POLLING_RECOVERY_TIMEOUT_SEC", "1800")
            ),
            tts_missing_audio_url_retries=int(os.getenv("VOCAVISION_TTS_MISSING_AUDIO_URL_RETRIES", "2")),
            tts_retry_backoff_sec=float(os.getenv("VOCAVISION_TTS_RETRY_BACKOFF_SEC", "1.5")),
            max_scenes_per_run=_read_optional_int("VOCAVISION_MAX_SCENES"),
            media_max_workers=int(os.getenv("VOCAVISION_MEDIA_MAX_WORKERS", "5")),
            reuse_cached_assets=_read_bool("VOCAVISION_REUSE_CACHED_ASSETS", True),
            auto_generate_comparison_variants=_read_bool("VOCAVISION_AUTO_GENERATE_COMPARISON_VARIANTS", True),
            tts_playback_rate=float(os.getenv("VOCAVISION_TTS_PLAYBACK_RATE", "0.94")),
            tts_mix_volume=float(os.getenv("VOCAVISION_TTS_MIX_VOLUME", "1.0")),
            video_audio_mix_volume=float(os.getenv("VOCAVISION_VIDEO_AUDIO_MIX_VOLUME", "0.35")),
            workspace_root=workspace_root,
            ffmpeg_bin=os.getenv("VOCAVISION_FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("VOCAVISION_FFPROBE_BIN", "ffprobe"),
        )
