"""Volcengine video generation wrapper with polling."""

from __future__ import annotations

import time
from pathlib import Path

from vocavision.config import VocavisionSettings
from vocavision.exceptions import ConfigurationError, ExternalServiceError
from vocavision.services.download_service import DownloadService
from vocavision.utils.cache_utils import build_request_signature, load_metadata, save_metadata


class VolcengineVideoService:
    def __init__(self, settings: VocavisionSettings, downloader: DownloadService | None = None) -> None:
        if not settings.ark_api_key:
            raise ConfigurationError("ARK_API_KEY is required to call Volcengine video generation.")
        from volcenginesdkarkruntime import Ark

        self.settings = settings
        self.downloader = downloader or DownloadService(timeout_sec=settings.request_timeout_sec)
        self.client = Ark(
            api_key=settings.ark_api_key,
            base_url=settings.ark_base_url,
            timeout=settings.request_timeout_sec,
        )

    def generate_video_from_image(self, image_url: str, prompt: str, output_path: Path) -> Path:
        request_payload = {
            "model": self.settings.video_model,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            "resolution": self.settings.video_resolution,
            "ratio": self.settings.video_ratio,
            "duration": self.settings.video_duration_sec,
            "watermark": False,
        }
        request_signature = build_request_signature(request_payload)
        metadata = load_metadata(output_path)

        if (
            self.settings.reuse_cached_assets
            and output_path.exists()
            and output_path.stat().st_size > 0
            and metadata
            and metadata.get("request_signature") == request_signature
        ):
            return output_path

        if (
            metadata
            and metadata.get("request_signature") == request_signature
            and metadata.get("task_id")
            and metadata.get("status") != "failed"
        ):
            return self._poll_existing_task(
                task_id=str(metadata["task_id"]),
                output_path=output_path,
                request_signature=request_signature,
                request_payload=request_payload,
            )

        create_result = self.client.content_generation.tasks.create(**request_payload)
        task_id = getattr(create_result, "id", None)
        if not task_id:
            raise ExternalServiceError("Video generation task creation returned no task id.")
        save_metadata(
            output_path,
            {
                "request_signature": request_signature,
                "request_payload": request_payload,
                "task_id": task_id,
                "status": "submitted",
                "local_path": str(output_path.resolve()),
            },
        )
        return self._poll_existing_task(
            task_id=str(task_id),
            output_path=output_path,
            request_signature=request_signature,
            request_payload=request_payload,
        )

    def _poll_existing_task(
        self,
        *,
        task_id: str,
        output_path: Path,
        request_signature: str,
        request_payload: dict[str, object],
    ) -> Path:
        recovery_round = 0
        current_timeout_sec = max(1, int(self.settings.polling_timeout_sec))
        last_known_status = "submitted"

        while True:
            deadline = time.time() + current_timeout_sec
            while time.time() < deadline:
                result = self.client.content_generation.tasks.get(task_id=task_id)
                result_status = str(getattr(result, "status", "unknown") or "unknown")
                normalized_status = result_status.strip().lower()
                last_known_status = normalized_status
                if normalized_status == "succeeded":
                    video_url = getattr(getattr(result, "content", None), "video_url", None)
                    if not video_url:
                        raise ExternalServiceError("Video generation succeeded but returned no downloadable video URL.")
                    downloaded_path = self.downloader.download_to_file(video_url, output_path)
                    save_metadata(
                        output_path,
                        {
                            "request_signature": request_signature,
                            "request_payload": request_payload,
                            "task_id": task_id,
                            "status": "succeeded",
                            "last_known_status": normalized_status,
                            "timeout_recovery_count": recovery_round,
                            "remote_url": video_url,
                            "local_path": str(output_path.resolve()),
                        },
                    )
                    return downloaded_path
                if normalized_status == "failed":
                    save_metadata(
                        output_path,
                        {
                            "request_signature": request_signature,
                            "request_payload": request_payload,
                            "task_id": task_id,
                            "status": "failed",
                            "last_known_status": normalized_status,
                            "timeout_recovery_count": recovery_round,
                            "error": str(result.error),
                            "local_path": str(output_path.resolve()),
                        },
                    )
                    raise ExternalServiceError(f"Video generation failed: {result.error}")
                time.sleep(self.settings.polling_interval_sec)

            if recovery_round >= max(0, int(self.settings.video_polling_timeout_retries)):
                save_metadata(
                    output_path,
                    {
                        "request_signature": request_signature,
                        "request_payload": request_payload,
                        "task_id": task_id,
                        "status": "polling_timed_out",
                        "last_known_status": last_known_status,
                        "timeout_recovery_count": recovery_round,
                        "local_path": str(output_path.resolve()),
                    },
                )
                raise ExternalServiceError(
                    f"Video generation polling timed out for task {task_id}. Last known status: {last_known_status}."
                )

            recovery_round += 1
            current_timeout_sec = max(1, int(self.settings.video_polling_recovery_timeout_sec))
            save_metadata(
                output_path,
                {
                    "request_signature": request_signature,
                    "request_payload": request_payload,
                    "task_id": task_id,
                    "status": "polling_recovery_pending",
                    "last_known_status": last_known_status,
                    "timeout_recovery_count": recovery_round,
                    "local_path": str(output_path.resolve()),
                },
            )
