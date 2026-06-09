"""FFmpeg and FFprobe helpers for alignment and final merge."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from vocavision.config import VocavisionSettings
from vocavision.exceptions import CommandExecutionError


def _escape_for_ass_filter(file_path: Path) -> str:
    return file_path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


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


def _make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _parse_ratio(ratio: str) -> float:
    raw_ratio = ratio.strip()
    if ":" in raw_ratio:
        width_part, height_part = raw_ratio.split(":", 1)
        return float(width_part) / float(height_part)
    return float(raw_ratio)


class FFmpegPostProcessor:
    def __init__(self, settings: VocavisionSettings) -> None:
        self.settings = settings

    def _resolve_video_dimensions(self) -> tuple[int, int]:
        resolution = self.settings.video_resolution.strip().lower()
        explicit_match = re.fullmatch(r"(\d+)x(\d+)", resolution)
        if explicit_match:
            return _make_even(int(explicit_match.group(1))), _make_even(int(explicit_match.group(2)))

        height_match = re.fullmatch(r"(\d+)p", resolution)
        if not height_match:
            raise ValueError(f"Unsupported video resolution format: {self.settings.video_resolution}")

        height = _make_even(int(height_match.group(1)))
        width = _make_even(int(round(height * _parse_ratio(self.settings.video_ratio))))
        return width, height

    def probe_duration_seconds(self, audio_path: Path) -> float:
        command = [
            self.settings.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandExecutionError(f"ffprobe failed: {result.stderr.strip()}")
        return float(result.stdout.strip())

    def has_audio_stream(self, media_path: Path) -> bool:
        command = [
            self.settings.ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandExecutionError(f"ffprobe audio stream probe failed: {result.stderr.strip()}")
        return bool(result.stdout.strip())

    def build_alignment_command(
        self,
        *,
        raw_video_path: Path,
        audio_path: Path,
        subtitle_path: Path,
        duration_sec: float,
        output_path: Path,
        has_video_audio: bool,
    ) -> list[str]:
        ass_filter = f"ass='{_escape_for_ass_filter(subtitle_path)}'"
        if duration_sec <= 5.0:
            video_filter = ass_filter
            extra_args = ["-t", f"{duration_sec:.3f}"]
            video_audio_chain = ""
        else:
            slowdown_factor = duration_sec / 5.0
            video_filter = f"setpts={slowdown_factor:.6f}*PTS,{ass_filter}"
            extra_args = []
            slowed_audio_rate = 1.0 / slowdown_factor
            video_audio_chain = f"{_build_atempo_filter(slowed_audio_rate)},"

        if has_video_audio:
            filter_complex = (
                f"[0:v]{video_filter}[vout];"
                f"[0:a]{video_audio_chain}volume={self.settings.video_audio_mix_volume:.3f}[video_bed];"
                f"[1:a]volume={self.settings.tts_mix_volume:.3f}[tts_track];"
                "[video_bed][tts_track]amix=inputs=2:duration=longest:normalize=0[aout]"
            )
            return [
                self.settings.ffmpeg_bin,
                "-y",
                "-i",
                str(raw_video_path),
                "-i",
                str(audio_path),
                *extra_args,
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]

        return [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(raw_video_path),
            "-i",
            str(audio_path),
            *extra_args,
            "-vf",
            video_filter,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

    def merge_scene(
        self,
        *,
        raw_video_path: Path,
        audio_path: Path,
        subtitle_path: Path,
        duration_sec: float,
        output_path: Path,
    ) -> Path:
        has_video_audio = self.has_audio_stream(raw_video_path)
        command = self.build_alignment_command(
            raw_video_path=raw_video_path,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            duration_sec=duration_sec,
            output_path=output_path,
            has_video_audio=has_video_audio,
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandExecutionError(f"ffmpeg scene merge failed: {result.stderr.strip()}")
        return output_path

    def render_still_image_clip(
        self,
        *,
        image_path: Path,
        duration_sec: float,
        output_path: Path,
    ) -> Path:
        width, height = self._resolve_video_dimensions()
        safe_duration = max(0.1, float(duration_sec))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "format=yuv420p"
        )
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            video_filter,
            "-t",
            f"{safe_duration:.3f}",
            "-r",
            "24",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandExecutionError(f"ffmpeg still-image render failed: {result.stderr.strip()}")
        return output_path

    def concat_videos(self, video_paths: list[Path], concat_file_path: Path, output_path: Path) -> Path:
        concat_file_path.parent.mkdir(parents=True, exist_ok=True)
        concat_lines = [f"file '{video_path.resolve().as_posix()}'" for video_path in video_paths]
        concat_file_path.write_text("\n".join(concat_lines), encoding="utf-8")

        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file_path),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandExecutionError(f"ffmpeg final concat failed: {result.stderr.strip()}")
        return output_path
