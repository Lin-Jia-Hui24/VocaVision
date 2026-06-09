"""Workspace and file layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectWorkspace:
    root: Path

    @classmethod
    def create(cls, workspace_root: Path, project_id: str) -> "ProjectWorkspace":
        root = workspace_root / project_id
        for directory in (
            root,
            root / "audio",
            root / "images",
            root / "logs",
            root / "video",
            root / "subtitles",
            root / "state",
            root / "final",
            root / "temp",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def state_path(self) -> Path:
        return self.root / "state" / "project_state.json"

    def logs_dir(self) -> Path:
        return self.root / "logs"

    def events_log_path(self) -> Path:
        return self.logs_dir() / "events.jsonl"

    def story_iterations_log_path(self) -> Path:
        return self.logs_dir() / "story_iterations.jsonl"

    def visual_iterations_log_path(self) -> Path:
        return self.logs_dir() / "visual_iterations.jsonl"

    def global_visual_iterations_log_path(self) -> Path:
        return self.logs_dir() / "global_visual_iterations.jsonl"

    def latest_story_summary_path(self) -> Path:
        return self.logs_dir() / "story_iteration_summary.md"

    def character_reference_path(self) -> Path:
        return self.root / "images" / "character_reference.jpeg"

    def keyframe_path(self, scene_index: int, iteration: int | None = None) -> Path:
        suffix = "" if iteration is None else f"_iter_{iteration:02d}"
        return self.root / "images" / f"scene_{scene_index:02d}_keyframe{suffix}.jpeg"

    def tts_path(self, scene_index: int) -> Path:
        return self.root / "audio" / f"scene_{scene_index:02d}.wav"

    def raw_video_path(self, scene_index: int) -> Path:
        return self.root / "video" / f"scene_{scene_index:02d}_raw.mp4"

    def subtitle_path(self, scene_index: int) -> Path:
        return self.root / "subtitles" / f"scene_{scene_index:02d}.ass"

    def cloze_subtitle_path(self, scene_index: int) -> Path:
        return self.root / "subtitles" / f"scene_{scene_index:02d}_cloze.ass"

    def merged_video_path(self, scene_index: int) -> Path:
        return self.root / "video" / f"scene_{scene_index:02d}_merged.mp4"

    def cloze_merged_video_path(self, scene_index: int) -> Path:
        return self.root / "video" / f"scene_{scene_index:02d}_cloze_merged.mp4"

    def concat_file_path(self) -> Path:
        return self.root / "temp" / "concat.txt"

    def cloze_concat_file_path(self) -> Path:
        return self.root / "temp" / "cloze_concat.txt"

    def final_video_path(self) -> Path:
        return self.root / "final" / "final_video.mp4"

    def final_cloze_video_path(self) -> Path:
        return self.root / "final" / "final_cloze_video.mp4"
