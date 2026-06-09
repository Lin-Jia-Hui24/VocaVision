"""Inspection helpers for the local video-processing runtime."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess

from vocavision.config import VocavisionSettings


def _resolve_binary(binary_name: str) -> dict[str, object]:
    resolved_path = shutil.which(binary_name)
    info: dict[str, object] = {
        "configured_name": binary_name,
        "found": resolved_path is not None,
        "path": resolved_path,
        "version": None,
    }
    if not resolved_path:
        return info

    try:
        result = subprocess.run(
            [binary_name, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            info["version"] = result.stdout.splitlines()[0].strip()
    except Exception:
        info["version"] = None
    return info


def _resolve_python_package(package_name: str) -> dict[str, object]:
    try:
        version = importlib.metadata.version(package_name)
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def inspect_video_stack(settings: VocavisionSettings) -> dict[str, object]:
    return {
        "strategy": "ffmpeg_subprocess_plus_pysubs2",
        "why_ffprobe": "Measure exact TTS duration before aligning each 5-second generated video.",
        "why_ffmpeg": "Trim or slow down video, burn highlighted ASS subtitles, and concat final scenes deterministically.",
        "python_packages": {
            "pysubs2": _resolve_python_package("pysubs2"),
        },
        "binaries": {
            "ffmpeg": _resolve_binary(settings.ffmpeg_bin),
            "ffprobe": _resolve_binary(settings.ffprobe_bin),
        },
    }
