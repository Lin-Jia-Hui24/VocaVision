"""DashScope TTS wrapper."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from vocavision.config import VocavisionSettings
from vocavision.exceptions import ConfigurationError, ExternalServiceError
from vocavision.services.download_service import DownloadService
from vocavision.utils.cache_utils import build_request_signature, load_metadata, save_metadata


def _build_atempo_filter(playback_rate: float) -> str:
    if playback_rate <= 0:
        raise ValueError("playback_rate must be positive.")
    remaining = playback_rate
    filters: list[str] = []
    while remaining < 0.5:
        filters.append("atempo=0.500000")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.000000")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


class DashScopeTTSService:
    def __init__(self, settings: VocavisionSettings, downloader: DownloadService | None = None) -> None:
        if not settings.dashscope_api_key:
            raise ConfigurationError("DASHSCOPE_API_KEY is required to call DashScope TTS.")
        import dashscope

        self.settings = settings
        self.downloader = downloader or DownloadService(timeout_sec=settings.request_timeout_sec)
        dashscope.base_http_api_url = settings.dashscope_http_base_url
        self._dashscope = dashscope

    def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        request_payload = {
            "model": self.settings.tts_model,
            "text": text,
            "voice": self.settings.tts_voice,
            "language_type": self.settings.tts_language,
            "stream": False,
        }
        playback_rate = max(0.1, float(self.settings.tts_playback_rate))
        request_signature = build_request_signature(
            {
                "tts_request": request_payload,
                "tts_playback_rate": round(playback_rate, 4),
            }
        )
        metadata = load_metadata(output_path)
        if (
            self.settings.reuse_cached_assets
            and output_path.exists()
            and output_path.stat().st_size > 0
            and (
                (metadata is not None and metadata.get("request_signature") == request_signature)
                or (metadata is None and abs(playback_rate - 1.0) < 1e-6)
            )
        ):
            return output_path

        url, attempt_count = self._request_audio_url_with_retries(
            request_payload=request_payload,
            request_signature=request_signature,
            output_path=output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if abs(playback_rate - 1.0) < 1e-6:
            downloaded_path = self.downloader.download_to_file(url, output_path, overwrite=True)
        else:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=output_path.suffix,
                dir=output_path.parent,
            ) as temp_file:
                temp_download_path = Path(temp_file.name)
            try:
                self.downloader.download_to_file(url, temp_download_path, overwrite=True)
                downloaded_path = self._apply_playback_rate(temp_download_path, output_path, playback_rate)
            finally:
                if temp_download_path.exists():
                    temp_download_path.unlink()
        save_metadata(
            output_path,
            {
                "request_signature": request_signature,
                "request_payload": request_payload,
                "tts_playback_rate": playback_rate,
                "tts_attempt_count": attempt_count,
                "remote_url": url,
                "local_path": str(output_path.resolve()),
            },
        )
        return downloaded_path

    def _request_audio_url_with_retries(
        self,
        *,
        request_payload: dict[str, object],
        request_signature: str,
        output_path: Path,
    ) -> tuple[str, int]:
        max_attempts = max(1, int(self.settings.tts_missing_audio_url_retries) + 1)
        last_error_message = "TTS response did not include an audio URL."
        for attempt in range(1, max_attempts + 1):
            response = self._dashscope.MultiModalConversation.call(
                api_key=self.settings.dashscope_api_key,
                **request_payload,
            )
            audio_url = getattr(getattr(response, "output", None), "audio", None)
            url = getattr(audio_url, "url", None)
            if url:
                if attempt > 1:
                    save_metadata(
                        output_path,
                        {
                            "request_signature": request_signature,
                            "request_payload": request_payload,
                            "status": "audio_url_recovered",
                            "tts_attempt_count": attempt,
                            "local_path": str(output_path.resolve()),
                        },
                    )
                return str(url), attempt

            last_error_message = "TTS response did not include an audio URL."
            save_metadata(
                output_path,
                {
                    "request_signature": request_signature,
                    "request_payload": request_payload,
                    "status": "missing_audio_url_retry_pending"
                    if attempt < max_attempts
                    else "missing_audio_url_failed",
                    "tts_attempt_count": attempt,
                    "local_path": str(output_path.resolve()),
                },
            )
            if attempt < max_attempts:
                backoff_sec = max(0.0, float(self.settings.tts_retry_backoff_sec)) * attempt
                time.sleep(backoff_sec)

        raise ExternalServiceError(last_error_message)

    def _apply_playback_rate(self, source_path: Path, output_path: Path, playback_rate: float) -> Path:
        filter_chain = _build_atempo_filter(playback_rate)
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(source_path),
            "-filter:a",
            filter_chain,
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ExternalServiceError(f"TTS post-processing failed: {result.stderr.strip()}")
        return output_path
