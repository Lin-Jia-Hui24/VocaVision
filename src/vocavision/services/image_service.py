"""Volcengine image generation wrapper."""

from __future__ import annotations

from pathlib import Path

from vocavision.config import VocavisionSettings
from vocavision.exceptions import ConfigurationError, ExternalServiceError
from vocavision.services.download_service import DownloadService
from vocavision.utils.cache_utils import build_request_signature, load_metadata, save_metadata


class VolcengineImageService:
    def __init__(self, settings: VocavisionSettings, downloader: DownloadService | None = None) -> None:
        if not settings.ark_api_key:
            raise ConfigurationError("ARK_API_KEY is required to call Volcengine image generation.")
        from volcenginesdkarkruntime import Ark

        self.settings = settings
        self.downloader = downloader or DownloadService(timeout_sec=settings.request_timeout_sec)
        self.client = Ark(
            api_key=settings.ark_api_key,
            base_url=settings.ark_base_url,
            timeout=settings.request_timeout_sec,
        )

    def generate_image(self, prompt: str, output_path: Path, source_image: str | list[str] | None = None) -> str:
        request_payload = {
            "model": self.settings.image_model,
            "prompt": prompt,
            "size": self.settings.image_size,
            "response_format": "url",
            "watermark": False,
        }
        if source_image:
            request_payload["image"] = source_image
        request_signature = build_request_signature(request_payload)
        metadata = load_metadata(output_path)
        if (
            self.settings.reuse_cached_assets
            and output_path.exists()
            and output_path.stat().st_size > 0
            and metadata
            and metadata.get("request_signature") == request_signature
            and metadata.get("remote_url")
        ):
            return str(metadata["remote_url"])

        response = self.client.images.generate(
            **request_payload,
        )
        if not response.data or not response.data[0].url:
            raise ExternalServiceError("Image generation returned no downloadable URL.")
        url = response.data[0].url
        self.downloader.download_to_file(url, output_path)
        save_metadata(
            output_path,
            {
                "request_signature": request_signature,
                "request_payload": request_payload,
                "remote_url": url,
                "local_path": str(output_path.resolve()),
            },
        )
        return url
