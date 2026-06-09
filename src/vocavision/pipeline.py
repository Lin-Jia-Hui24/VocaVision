"""Main orchestration pipeline for VocaVision."""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from vocavision.agents.story_agents import StoryAgents
from vocavision.agents.validators import ensure_scene_count, ensure_story_text_coverage, ensure_target_word_coverage
from vocavision.config import VocavisionSettings
from vocavision.exceptions import ConfigurationError, VocaVisionError
from vocavision.logging_utils import PipelineRunLogger
from vocavision.post_processing.ffmpeg_tools import FFmpegPostProcessor
from vocavision.post_processing.subtitles import SubtitleRenderer
from vocavision.services.download_service import DownloadService
from vocavision.services.image_service import VolcengineImageService
from vocavision.services.llm_service import DashScopeLLMService
from vocavision.services.tts_service import DashScopeTTSService
from vocavision.services.video_service import VolcengineVideoService
from vocavision.state import (
    DirectorFeedback,
    GlobalSceneScriptFeedback,
    LearningMemoryTarget,
    LearningExerciseBundle,
    LearningExerciseQuestion,
    LearningPlan,
    LearningSceneBlueprint,
    GlobalVisualConsistencyReview,
    Scene,
    SceneClozeChallenge,
    SceneScript,
    SceneVisualReview,
    TargetWordSenseCandidate,
    TargetWordSpec,
    VideoProjectState,
)
from vocavision.utils.cache_utils import load_metadata, save_metadata
from vocavision.utils.text_utils import normalize_spoken_text, normalize_visible_text, normalize_words
from vocavision.workspace import ProjectWorkspace

COMPARISON_VARIANT_PRIMARY = "primary"
COMPARISON_VARIANT_REVIEWED = "reviewed"
COMPARISON_VARIANT_NO_LOCAL = "no_local"
COMPARISON_VARIANT_NO_GLOBAL = "no_global"
COMPARISON_VARIANT_SEQUENCE = (
    COMPARISON_VARIANT_REVIEWED,
    COMPARISON_VARIANT_NO_LOCAL,
    COMPARISON_VARIANT_NO_GLOBAL,
)


class VocaVisionPipeline:
    def __init__(
        self,
        *,
        settings: VocavisionSettings,
        story_agents: StoryAgents,
        image_service: VolcengineImageService,
        tts_service: DashScopeTTSService,
        video_service: VolcengineVideoService,
        subtitle_renderer: SubtitleRenderer,
        ffmpeg_post_processor: FFmpegPostProcessor,
    ) -> None:
        self.settings = settings
        self.story_agents = story_agents
        self.image_service = image_service
        self.tts_service = tts_service
        self.video_service = video_service
        self.subtitle_renderer = subtitle_renderer
        self.ffmpeg_post_processor = ffmpeg_post_processor

    @classmethod
    def from_settings(cls, settings: VocavisionSettings) -> "VocaVisionPipeline":
        if not settings.dashscope_api_key or not settings.ark_api_key:
            raise ConfigurationError("DASHSCOPE_API_KEY and ARK_API_KEY are required for the full pipeline.")
        downloader = DownloadService(timeout_sec=settings.request_timeout_sec)
        return cls(
            settings=settings,
            story_agents=StoryAgents(DashScopeLLMService(settings)),
            image_service=VolcengineImageService(settings, downloader=downloader),
            tts_service=DashScopeTTSService(settings, downloader=downloader),
            video_service=VolcengineVideoService(settings, downloader=downloader),
            subtitle_renderer=SubtitleRenderer(),
            ffmpeg_post_processor=FFmpegPostProcessor(settings),
        )

    def run(
        self,
        *,
        project_id: str,
        target_words: list[str],
        target_word_specs: list[TargetWordSpec] | None = None,
        learning_mode: str = "auto",
    ) -> tuple[VideoProjectState, Path | None]:
        normalized_words = normalize_words(target_words)
        if not normalized_words and not target_word_specs:
            raise VocaVisionError("At least one target word is required.")
        resolved_target_specs = self._resolve_target_word_specs(normalized_words, target_word_specs)
        if self.settings.max_scenes_per_run is not None:
            resolved_target_specs = resolved_target_specs[: self.settings.max_scenes_per_run]
        if not resolved_target_specs:
            raise VocaVisionError("No target words remain after applying the scene limit.")
        effective_target_specs, learning_plan = self._prepare_learning_inputs(
            resolved_target_specs,
            learning_mode=learning_mode,
        )
        normalized_words = [spec.word for spec in effective_target_specs]

        workspace = ProjectWorkspace.create(self.settings.workspace_root, project_id)
        logger = PipelineRunLogger(
            events_path=workspace.events_log_path(),
            story_iterations_path=workspace.story_iterations_log_path(),
            story_summary_path=workspace.latest_story_summary_path(),
            visual_iterations_path=workspace.visual_iterations_log_path(),
            global_visual_iterations_path=workspace.global_visual_iterations_log_path(),
        )
        state = VideoProjectState(
            project_id=project_id,
            target_words=normalized_words,
            target_word_specs=[spec.model_copy(deep=True) for spec in effective_target_specs],
            render_profile="storybook_only" if self.settings.storyboard_only else "full_video",
            learning_plan=learning_plan.model_copy(deep=True),
        )
        logger.log_event(
            "pipeline",
            "Pipeline started",
            project_id=project_id,
            target_words=normalized_words,
            learning_mode=learning_plan.mode,
        )
        logger.log_event(
            "sense_disambiguation",
            "Target word senses resolved",
            target_word_specs=[spec.model_dump() for spec in effective_target_specs],
        )
        logger.log_event(
            "story",
            "Learning plan resolved",
            learning_mode=learning_plan.mode,
            recommended_scene_count=learning_plan.recommended_scene_count,
            rationale=learning_plan.rationale,
        )
        self._save_state(state, workspace)

        if len(effective_target_specs) > len(resolved_target_specs):
            logger.log_event(
                "learning_mode",
                "Expanded single seed word into related target words",
                seed_word=resolved_target_specs[0].word,
                expanded_words=normalized_words,
            )

        state.scenes = self._draft_validated_scenes(effective_target_specs, learning_plan, logger)
        self._save_state(state, workspace)

        logger.log_event("character_design", "Generating character design", scene_count=len(state.scenes))
        state.character_design = self.story_agents.director_character(state.scenes)
        reference_url = self.image_service.generate_image(
            state.character_design.visual_prompt,
            workspace.character_reference_path(),
        )
        state.character_design.reference_image_url = reference_url
        logger.log_event("character_design", "Character design ready", reference_image_url=reference_url)
        self._save_state(state, workspace)

        self._resolve_scene_visuals(state, workspace, logger)
        self._save_state(state, workspace)
        self._ensure_global_visual_consistency(state, workspace, logger)
        self._save_state(state, workspace)

        if self.settings.storyboard_only:
            for scene in state.scenes:
                scene.audio.spoken_text = normalize_spoken_text(scene.script.voiceover_and_dialogue)
            logger.log_event(
                "media",
                "Storyboard-only mode: skipped TTS and video generation",
                scene_count=len(state.scenes),
            )
            logger.log_event("teaching", "Building learning review package", scene_count=len(state.scenes))
            state.learning_exercises = self._build_learning_exercises(state)
            self._save_state(state, workspace)
            logger.log_event(
                "finalize",
                "Storyboard-only mode completed with keyframes and captions",
                scene_count=len(state.scenes),
            )
            logger.log_event(
                "pipeline",
                "Pipeline completed in storyboard-only mode",
                final_video_path=None,
            )
            return state, None

        logger.log_event("media", "Generating scene media", scene_count=len(state.scenes))
        self._generate_media_assets(state, workspace, logger)
        self._save_state(state, workspace)

        logger.log_event("teaching", "Building learning review package", scene_count=len(state.scenes))
        state.learning_exercises = self._build_learning_exercises(state)
        self._save_state(state, workspace)

        merged_scene_paths = [Path(scene.post_processing.final_merged_path) for scene in state.scenes]
        cloze_scene_paths = [Path(scene.post_processing.cloze_merged_path) for scene in state.scenes]
        logger.log_event("finalize", "Concatenating final video", merged_scene_count=len(merged_scene_paths))
        final_video_path = self.ffmpeg_post_processor.concat_videos(
            video_paths=merged_scene_paths,
            concat_file_path=workspace.concat_file_path(),
            output_path=workspace.final_video_path(),
        )
        self.ffmpeg_post_processor.concat_videos(
            video_paths=cloze_scene_paths,
            concat_file_path=workspace.cloze_concat_file_path(),
            output_path=workspace.final_cloze_video_path(),
        )
        logger.log_event("pipeline", "Pipeline completed", final_video_path=str(final_video_path))
        self._save_state(state, workspace)
        self._maybe_generate_comparison_variants(
            state=state,
            workspace=workspace,
            logger=logger,
            final_video_path=final_video_path,
        )
        return state, final_video_path

    def suggest_target_word_specs(
        self,
        target_words: list[str],
        *,
        apply_scene_limit: bool = True,
    ) -> list[TargetWordSpec]:
        normalized_words = normalize_words(target_words)
        if not normalized_words:
            raise VocaVisionError("At least one target word is required.")
        if apply_scene_limit and self.settings.max_scenes_per_run is not None:
            normalized_words = normalized_words[: self.settings.max_scenes_per_run]
        if not normalized_words:
            raise VocaVisionError("No target words remain after applying the scene limit.")
        return self._resolve_target_word_specs(normalized_words, None)

    def suggest_learning_plan(
        self,
        *,
        target_words: list[str] | None = None,
        target_word_specs: list[TargetWordSpec] | None = None,
        learning_mode: str = "auto",
        apply_scene_limit: bool = True,
        use_model_planner: bool = True,
    ) -> LearningPlan:
        resolved_specs = self._resolve_target_word_specs(
            normalize_words(target_words or []),
            target_word_specs,
        )
        if apply_scene_limit and self.settings.max_scenes_per_run is not None:
            resolved_specs = resolved_specs[: self.settings.max_scenes_per_run]
        if not resolved_specs:
            raise VocaVisionError("No target words remain after applying the scene limit.")
        _, learning_plan = self._prepare_learning_inputs(
            resolved_specs,
            learning_mode=learning_mode,
            use_model_planner=use_model_planner,
        )
        return learning_plan

    def suggest_learning_plan_preview(
        self,
        *,
        target_words: list[str] | None = None,
        target_word_specs: list[TargetWordSpec] | None = None,
        learning_mode: str = "auto",
        apply_scene_limit: bool = True,
        use_model_planner: bool = True,
    ) -> tuple[list[TargetWordSpec], LearningPlan]:
        resolved_specs = self._resolve_target_word_specs(
            normalize_words(target_words or []),
            target_word_specs,
        )
        if apply_scene_limit and self.settings.max_scenes_per_run is not None:
            resolved_specs = resolved_specs[: self.settings.max_scenes_per_run]
        if not resolved_specs:
            raise VocaVisionError("No target words remain after applying the scene limit.")
        return self._prepare_learning_inputs(
            resolved_specs,
            learning_mode=learning_mode,
            use_model_planner=use_model_planner,
        )

    def run_visual_recheck_experiment(
        self,
        *,
        source_project_id: str,
        experiment_project_id: str,
        scene_indexes: list[int] | None = None,
    ) -> tuple[VideoProjectState, Path]:
        if source_project_id == experiment_project_id:
            raise VocaVisionError("Experiment project_id must be different from the source project_id.")

        workspace = self._prepare_visual_recheck_workspace(
            source_project_id=source_project_id,
            experiment_project_id=experiment_project_id,
        )
        logger = PipelineRunLogger(
            events_path=workspace.events_log_path(),
            story_iterations_path=workspace.story_iterations_log_path(),
            story_summary_path=workspace.latest_story_summary_path(),
            visual_iterations_path=workspace.visual_iterations_log_path(),
            global_visual_iterations_path=workspace.global_visual_iterations_log_path(),
        )
        state = VideoProjectState.load(workspace.state_path())
        state.project_id = experiment_project_id
        self._hydrate_missing_target_word_specs(state)
        self._clear_non_visual_outputs_for_recheck(state)

        all_scene_indexes = [scene.scene_index for scene in state.scenes]
        selected_scene_indexes = scene_indexes or all_scene_indexes
        invalid_indexes = [scene_index for scene_index in selected_scene_indexes if scene_index not in all_scene_indexes]
        if invalid_indexes:
            raise VocaVisionError(f"Unknown scene indexes for visual recheck: {invalid_indexes}")

        logger.log_event(
            "visual_recheck",
            "Starting visual recheck experiment",
            source_project_id=source_project_id,
            experiment_project_id=experiment_project_id,
            scene_indexes=selected_scene_indexes,
        )

        summary_records: list[dict[str, Any]] = []
        for scene in state.scenes:
            if scene.scene_index not in set(selected_scene_indexes):
                continue
            if not scene.visual.keyframe_image_url:
                raise VocaVisionError(f"Scene {scene.scene_index} has no existing keyframe to recheck.")

            existing_payload = self.story_agents.review_local_visual_consistency(scene, scene.visual.keyframe_image_url)
            existing_review = self._build_visual_review(
                max(1, scene.visual.selected_iteration or scene.visual.review.iteration or 1),
                existing_payload,
                scene,
            )
            scene.visual.review = existing_review
            scene.visual.selected_score = existing_review.score
            initial_result = {
                "scene_index": scene.scene_index,
                "target_word": scene.target_word_in_scene,
                "initial_match_level": existing_review.match_level,
                "initial_score": existing_review.score,
                "initial_regeneration_mode": existing_review.regeneration_mode,
                "initial_observed_text": existing_review.observed_text,
                "initial_text_legibility_passed": existing_review.text_legibility_passed,
                "initial_reason": existing_review.reason,
            }
            logger.log_event(
                "visual_recheck",
                "Existing keyframe reviewed",
                **initial_result,
            )

            if existing_review.approved:
                scene.visual.approved_iteration = scene.visual.selected_iteration
                scene.visual.selected_via_fallback = False
                summary_records.append(
                    {
                        **initial_result,
                        "final_match_level": existing_review.match_level,
                        "final_score": existing_review.score,
                        "final_selected_iteration": scene.visual.selected_iteration,
                        "final_regeneration_mode": "none",
                        "final_observed_text": existing_review.observed_text,
                        "final_text_legibility_passed": existing_review.text_legibility_passed,
                    }
                )
                continue

            regenerated_scene = self._resolve_scene_visual(
                scene,
                state.character_design,
                workspace,
                logger,
                self._compose_visual_feedback(existing_review),
            )
            scene.script = regenerated_scene.script
            scene.visual = regenerated_scene.visual
            summary_records.append(
                {
                    **initial_result,
                    "final_match_level": regenerated_scene.visual.review.match_level,
                    "final_score": regenerated_scene.visual.review.score,
                    "final_selected_iteration": regenerated_scene.visual.selected_iteration,
                    "final_regeneration_mode": regenerated_scene.visual.review.regeneration_mode,
                        "final_observed_text": regenerated_scene.visual.review.observed_text,
                        "final_text_legibility_passed": regenerated_scene.visual.review.text_legibility_passed,
                }
            )
            self._save_state(state, workspace)

        self._save_state(state, workspace)
        summary_path = self._write_visual_recheck_summary(
            workspace=workspace,
            source_project_id=source_project_id,
            experiment_project_id=experiment_project_id,
            summary_records=summary_records,
        )
        logger.log_event(
            "visual_recheck",
            "Visual recheck experiment completed",
            summary_path=str(summary_path),
        )
        return state, summary_path

    def render_storyboard_project_to_video(
        self,
        *,
        project_id: str,
    ) -> tuple[VideoProjectState, Path]:
        workspace = ProjectWorkspace.create(self.settings.workspace_root, project_id)
        state_path = workspace.state_path()
        if not state_path.exists():
            raise VocaVisionError(f"Project state not found: {state_path}")

        logger = PipelineRunLogger(
            events_path=workspace.events_log_path(),
            story_iterations_path=workspace.story_iterations_log_path(),
            story_summary_path=workspace.latest_story_summary_path(),
            visual_iterations_path=workspace.visual_iterations_log_path(),
            global_visual_iterations_path=workspace.global_visual_iterations_log_path(),
        )
        state = VideoProjectState.load(state_path)
        self._hydrate_missing_target_word_specs(state)
        if not state.scenes:
            raise VocaVisionError("Storyboard project has no scenes to render.")

        missing_keyframes = [scene.scene_index for scene in state.scenes if not scene.visual.keyframe_image_url]
        if missing_keyframes:
            raise VocaVisionError(
                f"Storyboard project is missing approved keyframes for scenes: {missing_keyframes}"
            )

        self._hydrate_missing_spoken_text(state)
        state.render_profile = "full_video"
        if hasattr(state, "run_settings"):
            state.run_settings.storyboard_only = False

        logger.log_event(
            "pipeline",
            "Promoting storyboard project to full video",
            project_id=project_id,
            scene_count=len(state.scenes),
        )
        self._save_state(state, workspace)

        logger.log_event("media", "Generating scene media from approved storyboard", scene_count=len(state.scenes))
        try:
            self._generate_media_assets(state, workspace, logger)
        except Exception:
            self._save_state(state, workspace)
            raise
        self._save_state(state, workspace)

        logger.log_event("teaching", "Refreshing learning review package", scene_count=len(state.scenes))
        state.learning_exercises = self._build_learning_exercises(state)
        self._save_state(state, workspace)

        merged_scene_paths = [Path(scene.post_processing.final_merged_path) for scene in state.scenes]
        cloze_scene_paths = [Path(scene.post_processing.cloze_merged_path) for scene in state.scenes]
        logger.log_event("finalize", "Concatenating final video from storyboard project", merged_scene_count=len(merged_scene_paths))
        final_video_path = self.ffmpeg_post_processor.concat_videos(
            video_paths=merged_scene_paths,
            concat_file_path=workspace.concat_file_path(),
            output_path=workspace.final_video_path(),
        )
        self.ffmpeg_post_processor.concat_videos(
            video_paths=cloze_scene_paths,
            concat_file_path=workspace.cloze_concat_file_path(),
            output_path=workspace.final_cloze_video_path(),
        )
        logger.log_event("pipeline", "Storyboard project promoted to full video", final_video_path=str(final_video_path))
        self._save_state(state, workspace)
        self._maybe_generate_comparison_variants(
            state=state,
            workspace=workspace,
            logger=logger,
            final_video_path=final_video_path,
        )
        return state, final_video_path

    def generate_comparison_variants_for_project(
        self,
        *,
        source_project_id: str,
        rerender_existing: bool = False,
        variants: list[str] | None = None,
    ) -> dict[str, Path]:
        source_workspace = ProjectWorkspace.create(self.settings.workspace_root, source_project_id)
        source_state_path = source_workspace.state_path()
        if not source_state_path.exists():
            raise VocaVisionError(f"Project state not found: {source_state_path}")

        source_state = VideoProjectState.load(source_state_path)
        if self._is_comparison_variant_project(source_state):
            raise VocaVisionError(
                f"Project '{source_project_id}' is already a comparison variant and will not generate nested variants."
            )
        if not source_workspace.final_video_path().exists():
            raise VocaVisionError(
                f"Project '{source_project_id}' has no final video yet. Generate the main video before comparison variants."
            )

        requested_variants = variants or list(COMPARISON_VARIANT_SEQUENCE)
        variant_results: dict[str, Path] = {}
        for variant in requested_variants:
            normalized_variant = self._normalize_comparison_variant(variant)
            variant_project_id = self._build_comparison_variant_project_id(source_project_id, normalized_variant)
            variant_results[normalized_variant] = self._materialize_comparison_variant(
                source_project_id=source_project_id,
                source_workspace=source_workspace,
                source_state=source_state,
                variant_project_id=variant_project_id,
                variant=normalized_variant,
                rerender_existing=rerender_existing,
            )
        return variant_results

    def generate_comparison_variants_for_all_completed_projects(
        self,
        *,
        rerender_existing: bool = False,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for workspace_root in sorted(self.settings.workspace_root.iterdir()):
            if not workspace_root.is_dir():
                continue
            workspace = ProjectWorkspace.create(self.settings.workspace_root, workspace_root.name)
            state_path = workspace.state_path()
            if not state_path.exists() or not workspace.final_video_path().exists():
                continue
            state = VideoProjectState.load(state_path)
            if self._is_comparison_variant_project(state):
                continue
            try:
                variant_paths = self.generate_comparison_variants_for_project(
                    source_project_id=workspace_root.name,
                    rerender_existing=rerender_existing,
                )
            except Exception as exc:
                results.append(
                    {
                        "project_id": workspace_root.name,
                        "error": str(exc),
                    }
                )
                continue
            results.append(
                {
                    "project_id": workspace_root.name,
                    "variants": {name: str(path.resolve()) for name, path in variant_paths.items()},
                }
            )
        return results

    def _draft_validated_scenes(
        self,
        target_word_specs: list[TargetWordSpec],
        learning_plan: LearningPlan,
        logger: PipelineRunLogger,
    ) -> list[Scene]:
        target_words = [spec.word for spec in target_word_specs]
        max_iterations = max(1, self.settings.story_max_iterations)
        score_threshold = self.settings.story_score_threshold
        feedback = ""
        last_feedback = "No feedback yet."
        previous_review: dict[str, Any] | None = None
        accepted_iteration: int | None = None
        fallback_iteration: int | None = None
        iteration_records: list[dict[str, Any]] = []
        best_story_candidate: dict[str, Any] | None = None

        for iteration in range(1, max_iterations + 1):
            feedback_used = feedback or "none"
            logger.log_event(
                "story",
                "Starting story iteration",
                iteration=iteration,
                score_threshold=score_threshold,
                feedback_used=feedback_used,
                learning_mode=learning_plan.mode,
            )
            drafted = self.story_agents.playwright(
                target_word_specs,
                learning_plan,
                feedback,
                educator_review=previous_review,
            )
            validation_error: str | None = None
            review: dict[str, Any] | None = None
            accepted = False

            try:
                specs_by_word = {spec.word.lower(): spec for spec in target_word_specs}
                scenes = [
                    Scene(
                        scene_index=index,
                        target_word_in_scene=item["target_word_in_scene"],
                        target_word_spec=specs_by_word.get(
                            str(item["target_word_in_scene"]).strip().lower(),
                            TargetWordSpec(word=str(item["target_word_in_scene"]).strip()),
                        ).model_copy(deep=True),
                        script=SceneScript(**item["script"]),
                    )
                    for index, item in enumerate(drafted, start=1)
                ]
                ensure_scene_count(learning_plan, scenes)
                ensure_target_word_coverage(target_words, scenes)
                ensure_story_text_coverage(target_words, scenes)
            except (KeyError, TypeError, ValueError, VocaVisionError) as exc:
                validation_error = str(exc)
                feedback = f"Local validation failed. Fix this exactly: {validation_error}"
                last_feedback = feedback
                logger.log_event(
                    "story",
                    "Story iteration failed local validation",
                    iteration=iteration,
                    validation_error=validation_error,
                )
            else:
                review = self.story_agents.educator(
                    target_word_specs,
                    learning_plan,
                    [scene.model_dump() for scene in scenes],
                )
                previous_review = dict(review)
                review["score"] = self._coerce_review_score(review.get("score"))
                must_fix = bool(review.get("must_fix", False))
                blocking_issues = [str(item) for item in review.get("blocking_issues", []) if str(item).strip()]
                scene_script_feedback = review.get("scene_script_feedback") or {}
                hard_blocking_issues = self._story_review_has_hard_blockers(review)
                effective_must_fix = bool(must_fix and hard_blocking_issues)
                accepted = bool(
                    review["score"] >= score_threshold
                    and not effective_must_fix
                    and not hard_blocking_issues
                )
                review["effective_must_fix"] = effective_must_fix
                review["accepted_via_relaxed_story_gate"] = bool(
                    accepted and (review.get("passed") is not True or must_fix or blocking_issues or scene_script_feedback)
                )
                last_feedback = str(review.get("feedback", "No feedback provided."))
                feedback = self._compose_story_feedback(review, score_threshold)
                logger.log_event(
                    "story",
                    "Story iteration reviewed",
                    iteration=iteration,
                    score=review["score"],
                    passed=review.get("passed"),
                    must_fix=must_fix,
                    effective_must_fix=effective_must_fix,
                    hard_blocking_issues=hard_blocking_issues,
                    blocking_issues=blocking_issues,
                    accepted=accepted,
                )
                candidate = {
                    "iteration": iteration,
                    "scenes": [scene.model_copy(deep=True) for scene in scenes],
                    "review": dict(review),
                }
                if best_story_candidate is None or self._is_better_story_candidate(candidate, best_story_candidate):
                    best_story_candidate = candidate
                if accepted:
                    accepted_iteration = iteration
                    iteration_record = {
                        "iteration": iteration,
                        "feedback_used": feedback_used,
                        "playwright_output": drafted,
                        "validation_error": None,
                        "review": review,
                        "accepted": True,
                    }
                    iteration_records.append(iteration_record)
                    logger.log_story_iteration(**iteration_record)
                    logger.write_story_summary(
                        target_words=target_words,
                        accepted_iteration=accepted_iteration,
                        fallback_iteration=None,
                        max_iterations=max_iterations,
                        score_threshold=score_threshold,
                        iterations=iteration_records,
                    )
                    return scenes

            iteration_record = {
                "iteration": iteration,
                "feedback_used": feedback_used,
                "playwright_output": drafted,
                "validation_error": validation_error,
                "review": review,
                "accepted": accepted,
            }
            iteration_records.append(iteration_record)
            logger.log_story_iteration(**iteration_record)

        logger.write_story_summary(
            target_words=target_words,
            accepted_iteration=accepted_iteration,
            fallback_iteration=fallback_iteration,
            max_iterations=max_iterations,
            score_threshold=score_threshold,
            iterations=iteration_records,
        )
        if best_story_candidate is not None:
            fallback_iteration = int(best_story_candidate["iteration"])
            fallback_review = dict(best_story_candidate["review"])
            logger.log_event(
                "story",
                "Story drafting fell back to best scored reviewed iteration",
                iteration=fallback_iteration,
                score=fallback_review.get("score"),
                passed=fallback_review.get("passed"),
                must_fix=fallback_review.get("must_fix", False),
            )
            logger.write_story_summary(
                target_words=target_words,
                accepted_iteration=accepted_iteration,
                fallback_iteration=fallback_iteration,
                max_iterations=max_iterations,
                score_threshold=score_threshold,
                iterations=iteration_records,
            )
            return [scene.model_copy(deep=True) for scene in best_story_candidate["scenes"]]
        raise VocaVisionError(
            "Story drafting stopped after reaching the maximum iterations. "
            f"Threshold: {score_threshold}. Last feedback: {last_feedback}"
        )

    def _maybe_generate_comparison_variants(
        self,
        *,
        state: VideoProjectState,
        workspace: ProjectWorkspace,
        logger: PipelineRunLogger,
        final_video_path: Path,
    ) -> None:
        if not self.settings.auto_generate_comparison_variants:
            return
        if self._is_comparison_variant_project(state):
            return
        try:
            variant_paths = self.generate_comparison_variants_for_project(source_project_id=state.project_id)
        except Exception as exc:
            logger.log_event(
                "comparison_variants",
                "Automatic comparison variant generation failed",
                project_id=state.project_id,
                final_video_path=str(final_video_path),
                error=str(exc),
            )
            return

        logger.log_event(
            "comparison_variants",
            "Automatic comparison variants generated",
            project_id=state.project_id,
            variants={name: str(path) for name, path in variant_paths.items()},
        )
        self._save_state(state, workspace)

    def _materialize_comparison_variant(
        self,
        *,
        source_project_id: str,
        source_workspace: ProjectWorkspace,
        source_state: VideoProjectState,
        variant_project_id: str,
        variant: str,
        rerender_existing: bool,
    ) -> Path:
        existing_workspace = ProjectWorkspace.create(self.settings.workspace_root, variant_project_id)
        existing_final_video_path = existing_workspace.final_video_path()
        if (
            not rerender_existing
            and existing_final_video_path.exists()
            and existing_final_video_path.stat().st_size > 0
        ):
            return existing_final_video_path

        workspace = self._prepare_comparison_variant_workspace(
            source_project_id=source_project_id,
            variant_project_id=variant_project_id,
        )
        logger = PipelineRunLogger(
            events_path=workspace.events_log_path(),
            story_iterations_path=workspace.story_iterations_log_path(),
            story_summary_path=workspace.latest_story_summary_path(),
            visual_iterations_path=workspace.visual_iterations_log_path(),
            global_visual_iterations_path=workspace.global_visual_iterations_log_path(),
        )
        state = VideoProjectState.load(workspace.state_path())
        state.project_id = variant_project_id
        state.source_project_id = source_project_id
        state.comparison_variant = variant
        state.render_profile = "full_video"
        self._hydrate_missing_target_word_specs(state)
        self._hydrate_missing_spoken_text(state)
        self._clear_video_outputs_for_comparison_variant(state)
        self._apply_comparison_variant_visuals(
            state=state,
            source_workspace=source_workspace,
            source_state=source_state,
            variant=variant,
        )
        self._save_state(state, workspace)

        logger.log_event(
            "comparison_variants",
            "Starting local comparison variant render",
            source_project_id=source_project_id,
            variant_project_id=variant_project_id,
            variant=variant,
            rerender_existing=rerender_existing,
        )
        self._render_local_comparison_media(
            state=state,
            source_workspace=source_workspace,
            workspace=workspace,
            rerender_existing=rerender_existing,
            logger=logger,
        )
        logger.log_event(
            "comparison_variants",
            "Local comparison variant render completed",
            source_project_id=source_project_id,
            variant_project_id=variant_project_id,
            variant=variant,
            final_video_path=str(workspace.final_video_path()),
        )
        self._save_state(state, workspace)
        return workspace.final_video_path()

    def _render_local_comparison_media(
        self,
        *,
        state: VideoProjectState,
        source_workspace: ProjectWorkspace,
        workspace: ProjectWorkspace,
        rerender_existing: bool,
        logger: PipelineRunLogger,
    ) -> None:
        for scene in state.scenes:
            spoken_text = normalize_spoken_text(scene.script.voiceover_and_dialogue)
            tts_path = self._copy_or_reuse_scene_audio(
                scene=scene,
                spoken_text=spoken_text,
                source_workspace=source_workspace,
                workspace=workspace,
            )
            duration_sec = self.ffmpeg_post_processor.probe_duration_seconds(tts_path)
            ass_path, cloze_ass_path = self._copy_or_reuse_scene_subtitles(
                scene=scene,
                spoken_text=spoken_text,
                duration_sec=duration_sec,
                target_words=state.target_words,
                source_workspace=source_workspace,
                workspace=workspace,
            )
            raw_video_path = workspace.raw_video_path(scene.scene_index)
            if rerender_existing or not raw_video_path.exists() or raw_video_path.stat().st_size == 0:
                keyframe_path = self._resolve_local_keyframe_path(scene, workspace)
                if keyframe_path is None:
                    raise VocaVisionError(
                        f"Comparison variant '{state.project_id}' is missing a local keyframe for scene {scene.scene_index}."
                    )
                self.ffmpeg_post_processor.render_still_image_clip(
                    image_path=keyframe_path,
                    duration_sec=duration_sec,
                    output_path=raw_video_path,
                )
                save_metadata(
                    raw_video_path,
                    {
                        "status": "local_comparison_variant",
                        "source_project_id": state.source_project_id or state.project_id,
                        "comparison_variant": state.comparison_variant,
                        "source_image_path": str(keyframe_path.resolve()),
                        "duration_sec": round(float(duration_sec), 3),
                        "local_path": str(raw_video_path.resolve()),
                    },
                )
            merged_video_path = self.ffmpeg_post_processor.merge_scene(
                raw_video_path=raw_video_path,
                audio_path=tts_path,
                subtitle_path=ass_path,
                duration_sec=duration_sec,
                output_path=workspace.merged_video_path(scene.scene_index),
            )
            cloze_merged_video_path = self.ffmpeg_post_processor.merge_scene(
                raw_video_path=raw_video_path,
                audio_path=tts_path,
                subtitle_path=cloze_ass_path,
                duration_sec=duration_sec,
                output_path=workspace.cloze_merged_video_path(scene.scene_index),
            )
            scene.audio.spoken_text = spoken_text
            scene.audio.tts_path = str(tts_path)
            scene.audio.duration_sec = duration_sec
            scene.video.anim_path = str(raw_video_path)
            scene.post_processing.ass_path = str(ass_path)
            scene.post_processing.final_merged_path = str(merged_video_path)
            scene.post_processing.cloze_ass_path = str(cloze_ass_path)
            scene.post_processing.cloze_merged_path = str(cloze_merged_video_path)
            logger.log_event(
                "comparison_variants",
                "Scene comparison media generated",
                scene_index=scene.scene_index,
                tts_path=scene.audio.tts_path,
                raw_video_path=scene.video.anim_path,
                merged_video_path=scene.post_processing.final_merged_path,
            )

        final_video_path = self.ffmpeg_post_processor.concat_videos(
            video_paths=[Path(scene.post_processing.final_merged_path) for scene in state.scenes],
            concat_file_path=workspace.concat_file_path(),
            output_path=workspace.final_video_path(),
        )
        self.ffmpeg_post_processor.concat_videos(
            video_paths=[Path(scene.post_processing.cloze_merged_path) for scene in state.scenes],
            concat_file_path=workspace.cloze_concat_file_path(),
            output_path=workspace.final_cloze_video_path(),
        )
        logger.log_event(
            "comparison_variants",
            "Comparison final videos concatenated",
            final_video_path=str(final_video_path),
            cloze_final_video_path=str(workspace.final_cloze_video_path()),
        )

    def _copy_or_reuse_scene_audio(
        self,
        *,
        scene: Scene,
        spoken_text: str,
        source_workspace: ProjectWorkspace,
        workspace: ProjectWorkspace,
    ) -> Path:
        source_tts_path = source_workspace.tts_path(scene.scene_index)
        target_tts_path = workspace.tts_path(scene.scene_index)
        if source_tts_path.exists() and source_tts_path.stat().st_size > 0:
            shutil.copy2(source_tts_path, target_tts_path)
            source_meta_path = source_tts_path.with_suffix(".wav.meta.json")
            target_meta_path = target_tts_path.with_suffix(".wav.meta.json")
            if source_meta_path.exists():
                shutil.copy2(source_meta_path, target_meta_path)
            return target_tts_path

        synthesized_path = self.tts_service.synthesize_to_file(spoken_text, target_tts_path)
        save_metadata(
            synthesized_path,
            {
                "status": "comparison_variant_regenerated",
                "source_project_id": "",
                "comparison_variant_fallback": True,
                "local_path": str(synthesized_path.resolve()),
            },
        )
        return synthesized_path

    def _copy_or_reuse_scene_subtitles(
        self,
        *,
        scene: Scene,
        spoken_text: str,
        duration_sec: float,
        target_words: list[str],
        source_workspace: ProjectWorkspace,
        workspace: ProjectWorkspace,
    ) -> tuple[Path, Path]:
        source_ass_path = source_workspace.subtitle_path(scene.scene_index)
        target_ass_path = workspace.subtitle_path(scene.scene_index)
        if source_ass_path.exists() and source_ass_path.stat().st_size > 0:
            shutil.copy2(source_ass_path, target_ass_path)
        else:
            self.subtitle_renderer.render(
                text=spoken_text,
                target_words=target_words,
                duration_sec=duration_sec,
                output_path=target_ass_path,
            )

        source_cloze_ass_path = source_workspace.cloze_subtitle_path(scene.scene_index)
        target_cloze_ass_path = workspace.cloze_subtitle_path(scene.scene_index)
        if source_cloze_ass_path.exists() and source_cloze_ass_path.stat().st_size > 0:
            shutil.copy2(source_cloze_ass_path, target_cloze_ass_path)
        else:
            self.subtitle_renderer.render(
                text=spoken_text,
                target_words=[scene.target_word_in_scene],
                masked_words=[scene.target_word_in_scene],
                duration_sec=duration_sec,
                output_path=target_cloze_ass_path,
            )
        return target_ass_path, target_cloze_ass_path

    def _apply_comparison_variant_visuals(
        self,
        *,
        state: VideoProjectState,
        source_workspace: ProjectWorkspace,
        source_state: VideoProjectState,
        variant: str,
    ) -> None:
        if variant == COMPARISON_VARIANT_REVIEWED:
            state.scenes = [scene.model_copy(deep=True) for scene in source_state.scenes]
            state.global_visual_review = source_state.global_visual_review.model_copy(deep=True)
            return

        visual_records = self._read_jsonl(source_workspace.visual_iterations_log_path())
        if not visual_records:
            state.scenes = [scene.model_copy(deep=True) for scene in source_state.scenes]
            if variant == COMPARISON_VARIANT_NO_GLOBAL:
                state.global_visual_review = GlobalVisualConsistencyReview(
                    iteration=0,
                    passed=True,
                    must_fix=False,
                    score=0.0,
                    feedback=(
                        "Historical visual iteration logs are missing, so the no-global comparison variant "
                        "falls back to the currently approved keyframes."
                    ),
                    selected_via_fallback=True,
                )
            return
        if variant == COMPARISON_VARIANT_NO_LOCAL:
            snapshots = self._select_no_local_comparison_snapshots(visual_records)
        elif variant == COMPARISON_VARIANT_NO_GLOBAL:
            global_records = self._read_jsonl(source_workspace.global_visual_iterations_log_path())
            first_global_ts = (
                self._parse_logged_timestamp(str(global_records[0].get("timestamp") or "")) if global_records else None
            )
            snapshots = self._select_no_global_comparison_snapshots(visual_records, first_global_ts)
        else:
            raise VocaVisionError(f"Unsupported comparison variant: {variant}")

        for scene in state.scenes:
            snapshot = snapshots.get(scene.scene_index)
            if snapshot is None:
                source_scene = next((item for item in source_state.scenes if item.scene_index == scene.scene_index), None)
                snapshot = None if source_scene is None else self._build_scene_visual_snapshot(source_scene)
            if snapshot is None:
                raise VocaVisionError(
                    f"Comparison variant '{variant}' is missing a visual snapshot for scene {scene.scene_index}."
                )
            self._apply_visual_snapshot_to_scene(scene, snapshot)

        if variant == COMPARISON_VARIANT_NO_GLOBAL:
            state.global_visual_review = GlobalVisualConsistencyReview(
                iteration=0,
                passed=True,
                must_fix=False,
                score=0.0,
                feedback="Global visual review intentionally skipped for comparison rendering.",
            )

    @staticmethod
    def _apply_visual_snapshot_to_scene(scene: Scene, snapshot: dict[str, Any]) -> None:
        review_payload = snapshot.get("review") or {}
        scene.visual.director_prompt = str(snapshot.get("director_prompt") or scene.visual.director_prompt)
        scene.visual.keyframe_image_url = str(snapshot.get("image_url") or scene.visual.keyframe_image_url)
        scene.visual.selected_iteration = int(snapshot.get("iteration") or 1)
        scene.visual.approved_iteration = scene.visual.selected_iteration
        scene.visual.selected_score = float(review_payload.get("score") or 0.0)
        scene.visual.selected_via_fallback = False
        scene.visual.review = SceneVisualReview.model_validate(review_payload)

    @staticmethod
    def _build_scene_visual_snapshot(scene: Scene) -> dict[str, Any] | None:
        if not scene.visual.keyframe_image_url:
            return None
        return {
            "director_prompt": scene.visual.director_prompt,
            "image_url": scene.visual.keyframe_image_url,
            "iteration": scene.visual.selected_iteration or scene.visual.approved_iteration or scene.visual.review.iteration or 1,
            "review": scene.visual.review.model_dump(),
        }

    def _prepare_comparison_variant_workspace(
        self,
        *,
        source_project_id: str,
        variant_project_id: str,
    ) -> ProjectWorkspace:
        workspace = self._prepare_visual_recheck_workspace(
            source_project_id=source_project_id,
            experiment_project_id=variant_project_id,
        )
        source_workspace = ProjectWorkspace.create(self.settings.workspace_root, source_project_id)
        for relative_glob in (
            "audio/*.wav",
            "audio/*.wav.meta.json",
            "subtitles/*.ass",
            "logs/visual_iterations.jsonl",
            "logs/global_visual_iterations.jsonl",
            "logs/story_iterations.jsonl",
            "logs/story_iteration_summary.md",
        ):
            for source_path in source_workspace.root.glob(relative_glob):
                target_path = workspace.root / source_path.relative_to(source_workspace.root)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        return workspace

    @staticmethod
    def _clear_video_outputs_for_comparison_variant(state: VideoProjectState) -> None:
        for scene in state.scenes:
            scene.video.anim_path = ""
            scene.post_processing.final_merged_path = ""
            scene.post_processing.cloze_merged_path = ""

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _parse_logged_timestamp(value: str) -> str:
        return value.replace("Z", "+00:00")

    def _select_no_local_comparison_snapshots(self, visual_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        snapshots: dict[int, dict[str, Any]] = {}
        for row in visual_records:
            scene_index = int(row.get("scene_index") or 0)
            if scene_index and scene_index not in snapshots:
                snapshots[scene_index] = row
        return snapshots

    def _select_no_global_comparison_snapshots(
        self,
        visual_records: list[dict[str, Any]],
        first_global_timestamp: str | None,
    ) -> dict[int, dict[str, Any]]:
        snapshots: dict[int, dict[str, Any]] = {}
        for row in visual_records:
            scene_index = int(row.get("scene_index") or 0)
            if not scene_index:
                continue
            row_timestamp = self._parse_logged_timestamp(str(row.get("timestamp") or "1970-01-01T00:00:00+00:00"))
            if first_global_timestamp is not None and row_timestamp >= first_global_timestamp:
                continue
            snapshots[scene_index] = row
        return snapshots

    @staticmethod
    def _normalize_comparison_variant(raw_variant: str) -> str:
        normalized = str(raw_variant).strip().lower()
        alias_map = {
            "reviewed": COMPARISON_VARIANT_REVIEWED,
            "reviewed_local": COMPARISON_VARIANT_REVIEWED,
            "with_review": COMPARISON_VARIANT_REVIEWED,
            "no_local": COMPARISON_VARIANT_NO_LOCAL,
            "nolocal": COMPARISON_VARIANT_NO_LOCAL,
            "no_global": COMPARISON_VARIANT_NO_GLOBAL,
            "noglobal": COMPARISON_VARIANT_NO_GLOBAL,
        }
        if normalized not in alias_map:
            raise VocaVisionError(f"Unsupported comparison variant: {raw_variant}")
        return alias_map[normalized]

    @staticmethod
    def _build_comparison_variant_project_id(source_project_id: str, variant: str) -> str:
        suffix_map = {
            COMPARISON_VARIANT_REVIEWED: "ablation-reviewed",
            COMPARISON_VARIANT_NO_LOCAL: "ablation-nolocal",
            COMPARISON_VARIANT_NO_GLOBAL: "ablation-noglobal",
        }
        return f"{source_project_id}-{suffix_map[variant]}"

    @staticmethod
    def _is_comparison_variant_project(state: VideoProjectState) -> bool:
        return (
            bool(state.source_project_id)
            or state.comparison_variant != COMPARISON_VARIANT_PRIMARY
            or VocaVisionPipeline._looks_like_comparison_project_id(state.project_id)
        )

    @staticmethod
    def _looks_like_comparison_project_id(project_id: str) -> bool:
        normalized = str(project_id).strip().lower()
        markers = (
            "-ablation-",
            "-nolocal",
            "-noglobal",
            "_nolocal",
            "_noglobal",
        )
        return any(marker in normalized for marker in markers)

    def _resolve_scene_visuals(
        self,
        state: VideoProjectState,
        workspace: ProjectWorkspace,
        logger: PipelineRunLogger,
        scene_indexes: list[int] | None = None,
        initial_feedback_map: dict[int, str] | None = None,
    ) -> None:
        selected_indexes = scene_indexes or [scene.scene_index for scene in state.scenes]
        selected_scenes = [scene for scene in state.scenes if scene.scene_index in set(selected_indexes)]
        max_workers = max(1, min(self.settings.media_max_workers, len(selected_scenes)))
        logger.log_event("visual", "Resolving scene keyframes", scene_count=len(selected_scenes), max_workers=max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._resolve_scene_visual,
                    scene,
                    state.character_design,
                    workspace,
                    logger,
                    "" if initial_feedback_map is None else initial_feedback_map.get(scene.scene_index, ""),
                    [item.model_copy(deep=True) for item in state.scenes],
                ): scene.scene_index
                for scene in selected_scenes
            }
            for future in as_completed(future_map):
                resolved_scene = future.result()
                scene_index = future_map[future]
                scene = next(item for item in state.scenes if item.scene_index == scene_index)
                scene.visual = resolved_scene.visual
                scene.script = resolved_scene.script
                self._save_state(state, workspace)

    def _resolve_scene_visual(
        self,
        scene: Scene,
        character_design,
        workspace: ProjectWorkspace,
        logger: PipelineRunLogger,
        initial_feedback: str = "",
        all_scenes: list[Scene] | None = None,
    ) -> Scene:
        max_attempts = max(1, self.settings.visual_max_retries + 1)
        director_feedback = initial_feedback.strip()
        best_candidate: dict[str, Any] | None = self._build_existing_visual_candidate(scene)
        working_scene = scene.model_copy(deep=True)
        attempt_offset = max(scene.visual.selected_iteration, scene.visual.review.iteration)
        reference_scene_images = self._collect_reference_scene_images(working_scene, all_scenes or [])
        continuity_context = self._build_scene_continuity_context(working_scene, all_scenes or [])

        for iteration in range(1, max_attempts + 1):
            absolute_iteration = attempt_offset + iteration
            feedback_used = director_feedback or "none"
            logger.log_event(
                "visual",
                "Starting scene visual iteration",
                scene_index=scene.scene_index,
                iteration=absolute_iteration,
                feedback_used=feedback_used,
            )
            director_prompt = self.story_agents.director_scene(
                working_scene,
                character_design,
                director_feedback,
                continuity_context=continuity_context,
            )
            source_image = self._select_source_image_for_generation(
                scene=working_scene,
                initial_feedback=initial_feedback,
                current_iteration=iteration,
                reference_scene_images=reference_scene_images,
            )
            keyframe_image_url = self.image_service.generate_image(
                director_prompt,
                workspace.keyframe_path(scene.scene_index, absolute_iteration),
                source_image=source_image,
            )
            review_payload = self.story_agents.review_local_visual_consistency(working_scene, keyframe_image_url)
            review = self._build_visual_review(absolute_iteration, review_payload, working_scene)
            proposed_script = self.story_agents.build_scene_script_from_review(
                review_payload,
                existing_script=working_scene.script,
            )
            trial_scene = working_scene.model_copy(deep=True)
            trial_scene.script = proposed_script

            candidate = {
                "scene": trial_scene,
                "director_prompt": director_prompt,
                "image_url": keyframe_image_url,
                "review": review,
            }
            if best_candidate is None or self._is_better_visual_candidate(candidate, best_candidate):
                best_candidate = candidate

            working_scene.visual.director_prompt = director_prompt
            working_scene.visual.keyframe_image_url = keyframe_image_url
            working_scene.visual.review = review
            working_scene.script = proposed_script
            approved = review.approved

            logger.log_visual_iteration(
                scene_index=scene.scene_index,
                iteration=absolute_iteration,
                director_prompt=director_prompt,
                image_url=keyframe_image_url,
                review=review.model_dump(),
                approved=approved,
            )
            logger.log_event(
                "visual",
                "Scene image reviewed",
                scene_index=scene.scene_index,
                iteration=absolute_iteration,
                match_level=review.match_level,
                score=review.score,
                approved=approved,
                keyframe_image_url=keyframe_image_url,
            )

            if approved:
                working_scene.visual.approved_iteration = absolute_iteration
                working_scene.visual.selected_iteration = absolute_iteration
                working_scene.visual.selected_score = review.score
                working_scene.visual.selected_via_fallback = False
                logger.log_event(
                    "scene",
                    "Scene approved for media generation",
                    scene_index=scene.scene_index,
                    target_word=scene.target_word_in_scene,
                    approved_iteration=absolute_iteration,
                    selected_score=review.score,
                )
                return working_scene

            director_feedback = self._compose_visual_feedback(review)

        if best_candidate is None:
            raise VocaVisionError(f"Scene {scene.scene_index} produced no visual candidates.")

        selected_scene = best_candidate["scene"]
        selected_review = best_candidate["review"]
        selected_scene.visual.director_prompt = str(best_candidate["director_prompt"])
        selected_scene.visual.keyframe_image_url = str(best_candidate["image_url"])
        selected_scene.visual.review = selected_review
        selected_scene.visual.approved_iteration = 0
        selected_scene.visual.selected_iteration = selected_review.iteration
        selected_scene.visual.selected_score = selected_review.score
        selected_scene.visual.selected_via_fallback = True
        logger.log_event(
            "scene",
            "Scene fell back to best scored keyframe for media generation",
            scene_index=scene.scene_index,
            target_word=scene.target_word_in_scene,
            selected_iteration=selected_review.iteration,
            selected_score=selected_review.score,
            selected_match_level=selected_review.match_level,
        )
        return selected_scene

    def _ensure_global_visual_consistency(
        self,
        state: VideoProjectState,
        workspace: ProjectWorkspace,
        logger: PipelineRunLogger,
    ) -> None:
        max_rounds = max(1, self.settings.global_visual_max_rounds)
        score_threshold = self.settings.global_visual_score_threshold
        best_snapshot: dict[str, Any] | None = None

        for iteration in range(1, max_rounds + 1):
            logger.log_event(
                "global_visual",
                "Starting global visual consistency review",
                iteration=iteration,
                score_threshold=score_threshold,
            )
            review_payload = self.story_agents.review_global_visual_consistency(state.scenes, state.character_design)
            review = self._build_global_visual_review(iteration, review_payload)
            state.global_visual_review = review
            target_scene_indexes = self._resolve_problem_scene_indexes(review, state.scenes)
            logger.log_global_visual_iteration(
                iteration=iteration,
                review=review.model_dump(),
                targeted_scene_indexes=target_scene_indexes,
            )
            logger.log_event(
                "global_visual",
                "Global visual consistency reviewed",
                iteration=iteration,
                score=review.score,
                passed=review.passed,
                targeted_scene_indexes=target_scene_indexes,
            )
            if best_snapshot is None or self._is_better_global_snapshot(review, best_snapshot["review"]):
                best_snapshot = {
                    "review": review.model_copy(deep=True),
                    "scenes": [scene.model_copy(deep=True) for scene in state.scenes],
                }
            self._save_state(state, workspace)

            target_scene_indexes = self._expand_global_target_scene_indexes(review, state.scenes)

            if (
                review.passed
                and review.score >= score_threshold
                and not review.must_fix
                and not review.blocking_issues
                and not target_scene_indexes
            ):
                return

            self._apply_global_scene_script_feedback(state, review, logger)
            feedback_map = {
                scene_index: self._compose_global_scene_feedback(review, scene_index)
                for scene_index in target_scene_indexes
            }
            self._resolve_scene_visuals(
                state,
                workspace,
                logger,
                scene_indexes=target_scene_indexes,
                initial_feedback_map=feedback_map,
            )
            self._save_state(state, workspace)

        if best_snapshot is None:
            raise VocaVisionError("Global visual consistency review produced no snapshots.")
        state.scenes = [scene.model_copy(deep=True) for scene in best_snapshot["scenes"]]
        best_review = best_snapshot["review"].model_copy(deep=True)
        best_review.selected_via_fallback = True
        state.global_visual_review = best_review
        logger.log_event(
            "global_visual",
            "Global visual consistency fell back to best scored snapshot",
            iteration=best_review.iteration,
            score=best_review.score,
            problem_scenes=best_review.problem_scenes,
        )

    def _generate_media_assets(
        self,
        state: VideoProjectState,
        workspace: ProjectWorkspace,
        logger: PipelineRunLogger,
    ) -> None:
        max_workers = max(1, min(self.settings.media_max_workers, len(state.scenes)))
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._process_scene_media, scene, state.target_words, workspace): scene.scene_index
                for scene in state.scenes
            }
            for future in as_completed(future_map):
                scene_index = future_map[future]
                scene = next(item for item in state.scenes if item.scene_index == scene_index)
                try:
                    result = future.result()
                except Exception as exc:
                    self._recover_partial_scene_media(scene, workspace)
                    errors.append(f"scene {scene_index}: {exc}")
                    logger.log_event(
                        "media",
                        "Scene media generation failed",
                        scene_index=scene_index,
                        error=str(exc),
                    )
                    continue
                scene.audio.spoken_text = str(result["spoken_text"])
                scene.audio.tts_path = str(result["tts_path"])
                scene.audio.duration_sec = result["duration_sec"]
                scene.video.anim_path = str(result["raw_video_path"])
                scene.post_processing.ass_path = str(result["ass_path"])
                scene.post_processing.final_merged_path = str(result["merged_video_path"])
                scene.post_processing.cloze_ass_path = str(result["cloze_ass_path"])
                scene.post_processing.cloze_merged_path = str(result["cloze_merged_video_path"])
                logger.log_event(
                    "media",
                    "Scene media generated",
                    scene_index=scene_index,
                    tts_path=scene.audio.tts_path,
                    raw_video_path=scene.video.anim_path,
                    merged_video_path=scene.post_processing.final_merged_path,
                )
        if errors:
            raise VocaVisionError("Media generation failed for " + "; ".join(errors))

    def _process_scene_media(
        self,
        scene: Scene,
        target_words: list[str],
        workspace: ProjectWorkspace,
    ) -> dict[str, object]:
        spoken_text = normalize_spoken_text(scene.script.voiceover_and_dialogue)
        tts_path = self._ensure_cached_asset(
            output_path=workspace.tts_path(scene.scene_index),
            download_source=getattr(self.tts_service, "downloader", None),
        )
        if tts_path is None:
            tts_path = self.tts_service.synthesize_to_file(
                spoken_text,
                workspace.tts_path(scene.scene_index),
            )
        duration_sec = self.ffmpeg_post_processor.probe_duration_seconds(tts_path)
        raw_video_path = self._ensure_cached_asset(
            output_path=workspace.raw_video_path(scene.scene_index),
            download_source=getattr(self.video_service, "downloader", None),
        )
        if raw_video_path is None:
            raw_video_path = self._generate_scene_raw_video(
                scene=scene,
                workspace=workspace,
                duration_sec=duration_sec,
            )
        ass_path = self.subtitle_renderer.render(
            text=spoken_text,
            target_words=target_words,
            duration_sec=duration_sec,
            output_path=workspace.subtitle_path(scene.scene_index),
        )
        cloze_ass_path = self.subtitle_renderer.render(
            text=spoken_text,
            target_words=[scene.target_word_in_scene],
            masked_words=[scene.target_word_in_scene],
            duration_sec=duration_sec,
            output_path=workspace.cloze_subtitle_path(scene.scene_index),
        )
        merged_video_path = self.ffmpeg_post_processor.merge_scene(
            raw_video_path=raw_video_path,
            audio_path=tts_path,
            subtitle_path=ass_path,
            duration_sec=duration_sec,
            output_path=workspace.merged_video_path(scene.scene_index),
        )
        cloze_merged_video_path = self.ffmpeg_post_processor.merge_scene(
            raw_video_path=raw_video_path,
            audio_path=tts_path,
            subtitle_path=cloze_ass_path,
            duration_sec=duration_sec,
            output_path=workspace.cloze_merged_video_path(scene.scene_index),
        )
        return {
            "spoken_text": spoken_text,
            "tts_path": tts_path,
            "duration_sec": duration_sec,
            "raw_video_path": raw_video_path,
            "ass_path": ass_path,
            "merged_video_path": merged_video_path,
            "cloze_ass_path": cloze_ass_path,
            "cloze_merged_video_path": cloze_merged_video_path,
        }

    def _generate_scene_raw_video(
        self,
        *,
        scene: Scene,
        workspace: ProjectWorkspace,
        duration_sec: float,
    ) -> Path:
        output_path = workspace.raw_video_path(scene.scene_index)
        motion_prompt = (
            "subtle cinematic motion, slow breathing, gentle parallax, motion-comic consistency, "
            "natural ambience only, no visible speaking mouth movements, no lip-sync dialogue, "
            "express meaning through gesture, expression, props, and camera motion"
        )
        try:
            return self.video_service.generate_video_from_image(
                scene.visual.keyframe_image_url,
                motion_prompt,
                output_path,
            )
        except Exception as exc:
            local_keyframe_path = self._resolve_local_keyframe_path(scene, workspace)
            if local_keyframe_path is None or not self._should_use_local_still_video_fallback(exc):
                raise

            rendered_path = self.ffmpeg_post_processor.render_still_image_clip(
                image_path=local_keyframe_path,
                duration_sec=duration_sec,
                output_path=output_path,
            )
            save_metadata(
                rendered_path,
                {
                    "status": "local_still_image_fallback",
                    "fallback_reason": str(exc),
                    "source_image_path": str(local_keyframe_path.resolve()),
                    "duration_sec": round(float(duration_sec), 3),
                    "local_path": str(rendered_path.resolve()),
                },
            )
            return rendered_path

    @staticmethod
    def _should_use_local_still_video_fallback(exc: Exception) -> bool:
        normalized_error = str(exc).strip().lower()
        fallback_markers = (
            "resource download failed",
            "invalidparameter",
            "image_url",
            "downloadable video url",
        )
        return any(marker in normalized_error for marker in fallback_markers)

    @staticmethod
    def _resolve_local_keyframe_path(scene: Scene, workspace: ProjectWorkspace) -> Path | None:
        seen_candidates: set[Path] = set()
        candidate_paths: list[Path] = []

        for iteration in (
            scene.visual.selected_iteration,
            scene.visual.approved_iteration,
            scene.visual.review.iteration,
        ):
            if iteration and iteration > 0:
                candidate_path = workspace.keyframe_path(scene.scene_index, iteration)
                if candidate_path not in seen_candidates:
                    candidate_paths.append(candidate_path)
                    seen_candidates.add(candidate_path)

        base_keyframe_path = workspace.keyframe_path(scene.scene_index)
        if base_keyframe_path not in seen_candidates:
            candidate_paths.append(base_keyframe_path)
            seen_candidates.add(base_keyframe_path)

        for candidate_path in candidate_paths:
            if candidate_path.exists() and candidate_path.stat().st_size > 0:
                return candidate_path

        fallback_candidates = sorted(
            (workspace.root / "images").glob(f"scene_{scene.scene_index:02d}_keyframe_iter_*.jpeg"),
            reverse=True,
        )
        for candidate_path in fallback_candidates:
            if candidate_path.exists() and candidate_path.stat().st_size > 0:
                return candidate_path
        return None

    @staticmethod
    def _ensure_cached_asset(output_path: Path, download_source: DownloadService | None) -> Path | None:
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        metadata = load_metadata(output_path)
        remote_url = None if metadata is None else metadata.get("remote_url")
        if not remote_url or download_source is None:
            return None
        downloaded_path = download_source.download_to_file(str(remote_url), output_path)
        if downloaded_path.exists() and downloaded_path.stat().st_size > 0:
            return downloaded_path
        return None

    def _recover_partial_scene_media(self, scene: Scene, workspace: ProjectWorkspace) -> None:
        spoken_text = normalize_spoken_text(scene.script.voiceover_and_dialogue)
        tts_path = workspace.tts_path(scene.scene_index)
        raw_video_path = workspace.raw_video_path(scene.scene_index)
        ass_path = workspace.subtitle_path(scene.scene_index)
        merged_path = workspace.merged_video_path(scene.scene_index)
        cloze_ass_path = workspace.cloze_subtitle_path(scene.scene_index)
        cloze_merged_path = workspace.cloze_merged_video_path(scene.scene_index)

        if tts_path.exists() and tts_path.stat().st_size > 0:
            scene.audio.spoken_text = spoken_text
            scene.audio.tts_path = str(tts_path)
            try:
                scene.audio.duration_sec = self.ffmpeg_post_processor.probe_duration_seconds(tts_path)
            except Exception:
                pass
        if raw_video_path.exists() and raw_video_path.stat().st_size > 0:
            scene.video.anim_path = str(raw_video_path)
        if ass_path.exists() and ass_path.stat().st_size > 0:
            scene.post_processing.ass_path = str(ass_path)
        if merged_path.exists() and merged_path.stat().st_size > 0:
            scene.post_processing.final_merged_path = str(merged_path)
        if cloze_ass_path.exists() and cloze_ass_path.stat().st_size > 0:
            scene.post_processing.cloze_ass_path = str(cloze_ass_path)
        if cloze_merged_path.exists() and cloze_merged_path.stat().st_size > 0:
            scene.post_processing.cloze_merged_path = str(cloze_merged_path)

    def _build_learning_exercises(self, state: VideoProjectState):
        if hasattr(self.story_agents, "teaching_agent"):
            payload = self.story_agents.teaching_agent(
                state.target_word_specs,
                state.learning_plan,
                state.scenes,
            )
            if isinstance(payload, LearningExerciseBundle):
                return self._ensure_collocation_extension_questions(payload, state)
            if isinstance(payload, dict):
                return self._ensure_collocation_extension_questions(
                    LearningExerciseBundle.model_validate(payload),
                    state,
                )
        return self._ensure_collocation_extension_questions(self._build_fallback_learning_exercises(state), state)

    @staticmethod
    def _build_fallback_learning_exercises(state: VideoProjectState):
        distractor_pool = [spec.word for spec in state.target_word_specs]
        cloze_challenges: list[SceneClozeChallenge] = []
        practice_questions: list[LearningExerciseQuestion] = []
        for scene in state.scenes:
            options = VocaVisionPipeline._build_choice_options(scene.target_word_in_scene, distractor_pool)
            cloze_prompt = normalize_spoken_text(scene.audio.spoken_text or scene.script.voiceover_and_dialogue)
            if re.search(rf"\b{re.escape(scene.target_word_in_scene)}\b", cloze_prompt, flags=re.IGNORECASE):
                cloze_prompt = re.sub(
                    rf"\b{re.escape(scene.target_word_in_scene)}\b",
                    "_____",
                    cloze_prompt,
                    count=1,
                    flags=re.IGNORECASE,
                )
            else:
                cloze_prompt = f"In this scene, the missing focus word is _____."
            cloze_challenges.append(
                SceneClozeChallenge(
                    scene_index=scene.scene_index,
                    target_word=scene.target_word_in_scene,
                    question_category="cloze_recall",
                    prompt=cloze_prompt,
                    options=options,
                    correct_answer=scene.target_word_in_scene,
                    explanation=f"The missing word is '{scene.target_word_in_scene}' because it is the focus of this scene.",
                )
            )
        for index, spec in enumerate(state.target_word_specs, start=1):
            question_category = ["sense_discrimination", "context_transfer", "usage_correction"][
                (index - 1) % 3
            ]
            options = VocaVisionPipeline._build_choice_options(spec.word, distractor_pool)
            if question_category == "sense_discrimination":
                error_reason_tag = "sense_confusion"
                prompt = f"Which word best matches this lesson focus: {spec.gloss_en}"
                explanation = f"'{spec.word}' is the lesson word that matches this meaning."
            elif question_category == "context_transfer":
                error_reason_tag = "transfer_failure"
                prompt = (
                    f"Which word best completes this new situation naturally: "
                    f"'{spec.example_sentence or f'This story teaches {spec.word}.'}'"
                )
                explanation = f"'{spec.word}' transfers naturally to this new sentence."
            else:
                error_reason_tag = "unnatural_collocation"
                prompt = f"Which word would fix the wrong sentence and make it natural: {spec.gloss_en}"
                explanation = f"'{spec.word}' is the correct word to repair the usage."
            practice_questions.append(
                LearningExerciseQuestion(
                    question_id=f"fallback-q{index}",
                    question_type="multiple_choice",
                    question_category=question_category,
                    error_reason_tag=error_reason_tag,
                    prompt=prompt,
                    options=options,
                    correct_answer=spec.word,
                    explanation=explanation,
                    related_words=[spec.word],
                    recommended_scene_indices=[
                        scene.scene_index
                        for scene in state.scenes
                        if scene.target_word_in_scene == spec.word
                    ],
                )
            )
        return LearningExerciseBundle(
            recommended_interaction_mode="multiple_choice",
            cloze_challenges=cloze_challenges,
            practice_questions=practice_questions,
        )

    @staticmethod
    def _ensure_collocation_extension_questions(
        exercises: LearningExerciseBundle,
        state: VideoProjectState,
    ) -> LearningExerciseBundle:
        enriched = exercises.model_copy(deep=True)
        existing_collocation_questions = [
            question
            for question in enriched.practice_questions
            if question.question_category == "collocation_extension"
        ]
        target_count = min(2, len(state.target_word_specs))
        if len(existing_collocation_questions) >= target_count or target_count == 0:
            return enriched
        existing_words = {
            word.lower()
            for question in existing_collocation_questions
            for word in question.related_words
            if word and word.strip()
        }
        next_index = len(enriched.practice_questions) + 1
        for spec in state.target_word_specs:
            if len(existing_collocation_questions) >= target_count:
                break
            if spec.word.strip().lower() in existing_words:
                continue
            question = VocaVisionPipeline._build_collocation_extension_question(
                spec=spec,
                question_index=next_index,
                state=state,
            )
            enriched.practice_questions.append(question)
            existing_collocation_questions.append(question)
            existing_words.add(spec.word.strip().lower())
            next_index += 1
        return enriched

    @staticmethod
    def _build_collocation_extension_question(
        *,
        spec: TargetWordSpec,
        question_index: int,
        state: VideoProjectState,
    ) -> LearningExerciseQuestion:
        word = spec.word.strip()
        lower_word = word.lower()
        if lower_word == "care":
            prompt = "Which phrase is the most natural fixed expression with 'care'?"
            options = ["take care", "make care", "do carely", "very care"]
            correct_answer = "take care"
            explanation = "'Take care' is a common and natural fixed phrase in English."
        elif lower_word.endswith("ing"):
            prompt = f"Which phrase sounds most natural with '{word}'?"
            options = [f"a {word} friend", f"do {word}", f"{word} very", f"{word} withly"]
            correct_answer = f"a {word} friend"
            explanation = f"'A {word} friend' is a natural collocation. The other options are not standard English phrases."
        elif lower_word.endswith("ful") or lower_word.endswith("less"):
            prompt = f"Which phrase sounds most natural with '{word}'?"
            options = [f"be {word}", f"do {word}", f"make {word}", f"{word}ly act"]
            correct_answer = f"be {word}"
            explanation = f"'Be {word}' is a natural everyday phrase. The other options are awkward or incorrect."
        else:
            prompt = f"Which short phrase sounds most natural with '{word}'?"
            options = [f"show {word}", f"make {word}ly", f"{word} withly", f"very {word}ly"]
            correct_answer = f"show {word}"
            explanation = f"'Show {word}' is the most natural short partnership among these options."
        scene_indices = [
            scene.scene_index
            for scene in state.scenes
            if scene.target_word_in_scene.strip().lower() == lower_word
        ]
        return LearningExerciseQuestion(
            question_id=f"collocation-q{question_index}",
            question_type="multiple_choice",
            question_category="collocation_extension",
            error_reason_tag="collocation_gap",
            prompt=prompt,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            related_words=[word],
            recommended_scene_indices=scene_indices,
        )

    @staticmethod
    def _build_choice_options(correct_word: str, distractor_pool: list[str]) -> list[str]:
        options = [correct_word]
        for candidate in distractor_pool:
            if candidate.lower() == correct_word.lower() or candidate in options:
                continue
            options.append(candidate)
            if len(options) == 4:
                break
        generic_distractors = ["quiet", "glow", "rescue", "bridge", "light", "careful"]
        for candidate in generic_distractors:
            if candidate.lower() == correct_word.lower() or candidate in options:
                continue
            options.append(candidate)
            if len(options) == 4:
                break
        return options

    @staticmethod
    def _save_state(state: VideoProjectState, workspace: ProjectWorkspace) -> None:
        state.save(workspace.state_path())

    @staticmethod
    def _coerce_review_score(score: Any) -> float:
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_visual_review(iteration: int, review_payload: dict[str, Any], scene: Scene) -> SceneVisualReview:
        normalized_payload = VocaVisionPipeline._enforce_target_text_gate(scene, review_payload)
        match_level = str(normalized_payload.get("match_level", "major")).strip().lower()
        if match_level not in {"minor", "major"}:
            match_level = "major"
        regeneration_mode = VocaVisionPipeline._normalize_regeneration_mode(
            normalized_payload.get("regeneration_mode"),
            match_level=match_level,
        )
        director_feedback_payload = normalized_payload.get("director_feedback") or {}
        return SceneVisualReview(
            iteration=iteration,
            match_level=match_level,
            score=VocaVisionPipeline._coerce_review_score(normalized_payload.get("score")),
            reason=str(normalized_payload.get("reason", "No reason provided.")),
            approved=match_level == "minor",
            regeneration_mode=regeneration_mode,
            has_visible_target_word_text=bool(normalized_payload.get("has_visible_target_word_text", False)),
            observed_text=str(normalized_payload.get("observed_text", "")),
            text_legibility_passed=VocaVisionPipeline._coerce_optional_bool(
                normalized_payload.get("text_legibility_passed")
            ),
            text_legibility_reason=str(normalized_payload.get("text_legibility_reason", "")),
            revised_plot_description=str(normalized_payload.get("revised_plot_description", "")),
            revised_voiceover_and_dialogue=str(normalized_payload.get("revised_voiceover_and_dialogue", "")),
            director_feedback=DirectorFeedback(
                summary=str(director_feedback_payload.get("summary", "No director feedback provided.")),
                visual_issues=[str(item) for item in director_feedback_payload.get("visual_issues", [])],
                optimization_suggestions=[
                    str(item) for item in director_feedback_payload.get("optimization_suggestions", [])
                ],
                recommended_prompt_adjustments=[
                    str(item) for item in director_feedback_payload.get("recommended_prompt_adjustments", [])
                ],
                repair_instruction=str(director_feedback_payload.get("repair_instruction", "")),
            ),
        )

    @staticmethod
    def _build_global_visual_review(
        iteration: int,
        review_payload: dict[str, Any],
    ) -> GlobalVisualConsistencyReview:
        scene_feedback_payload = review_payload.get("scene_feedback") or {}
        normalized_scene_feedback: dict[str, DirectorFeedback] = {}
        for scene_key, feedback_payload in scene_feedback_payload.items():
            normalized_scene_feedback[str(scene_key)] = DirectorFeedback(
                summary=str(feedback_payload.get("summary", "No scene feedback provided.")),
                visual_issues=[str(item) for item in feedback_payload.get("visual_issues", [])],
                optimization_suggestions=[str(item) for item in feedback_payload.get("optimization_suggestions", [])],
                recommended_prompt_adjustments=[
                    str(item) for item in feedback_payload.get("recommended_prompt_adjustments", [])
                ],
            )
        scene_script_feedback_payload = review_payload.get("scene_script_feedback") or {}
        normalized_scene_script_feedback: dict[str, GlobalSceneScriptFeedback] = {}
        for scene_key, feedback_payload in scene_script_feedback_payload.items():
            normalized_scene_script_feedback[str(scene_key)] = GlobalSceneScriptFeedback(
                summary=str(feedback_payload.get("summary", "No script feedback provided.")),
                script_issues=[str(item) for item in feedback_payload.get("script_issues", [])],
                revised_plot_description=str(feedback_payload.get("revised_plot_description", "")),
                revised_voiceover_and_dialogue=str(feedback_payload.get("revised_voiceover_and_dialogue", "")),
            )
        return GlobalVisualConsistencyReview(
            iteration=iteration,
            passed=bool(review_payload.get("passed", False)),
            must_fix=bool(review_payload.get("must_fix", False)),
            score=VocaVisionPipeline._coerce_review_score(review_payload.get("score")),
            feedback=str(review_payload.get("feedback", "No global consistency feedback provided.")),
            blocking_issues=[str(item) for item in review_payload.get("blocking_issues", [])],
            problem_scenes=[int(item) for item in review_payload.get("problem_scenes", [])],
            global_style_adjustments=[str(item) for item in review_payload.get("global_style_adjustments", [])],
            scene_feedback=normalized_scene_feedback,
            scene_script_feedback=normalized_scene_script_feedback,
        )

    @staticmethod
    def _compose_story_feedback(review: dict[str, Any], score_threshold: float) -> str:
        strengths = review.get("strengths") or []
        improvements = review.get("improvement_focus") or []
        blocking_issues = [str(item) for item in review.get("blocking_issues", []) if str(item).strip()]
        scene_script_feedback = review.get("scene_script_feedback") or {}
        parts = [
            f"Current score: {review.get('score', 0.0)}. Required score: {score_threshold}.",
            f"Reviewer feedback: {review.get('feedback', 'No feedback provided.')}",
        ]
        if review.get("effective_must_fix", review.get("must_fix")):
            parts.append("This draft is not allowed to enter image generation yet because it still has must-fix story issues.")
        if blocking_issues:
            parts.append("Blocking issues: " + ", ".join(blocking_issues) + ".")
        if strengths:
            parts.append(f"Keep these strengths: {', '.join(str(item) for item in strengths)}.")
        if improvements:
            parts.append(f"Focus on these improvements: {', '.join(str(item) for item in improvements)}.")
        for scene_key, raw_feedback in scene_script_feedback.items():
            feedback = GlobalSceneScriptFeedback(
                summary=str(raw_feedback.get("summary", "No scene script feedback provided.")),
                script_issues=[str(item) for item in raw_feedback.get("script_issues", [])],
                revised_plot_description=str(raw_feedback.get("revised_plot_description", "")),
                revised_voiceover_and_dialogue=str(raw_feedback.get("revised_voiceover_and_dialogue", "")),
            )
            parts.append(f"Scene {scene_key} summary: {feedback.summary}")
            if feedback.script_issues:
                parts.append("Scene " + str(scene_key) + " issues: " + ", ".join(feedback.script_issues) + ".")
            if feedback.revised_plot_description:
                parts.append(
                    "Scene " + str(scene_key) + " revised plot description: " + feedback.revised_plot_description + "."
                )
            if feedback.revised_voiceover_and_dialogue:
                parts.append(
                    "Scene "
                    + str(scene_key)
                    + " revised narrator line: "
                    + feedback.revised_voiceover_and_dialogue
                    + "."
                )
        return " ".join(parts)

    @classmethod
    def _story_review_has_hard_blockers(cls, review: dict[str, Any]) -> bool:
        if (
            review.get("passed") is True
            and not bool(review.get("must_fix", False))
            and not [str(item) for item in review.get("blocking_issues", []) if str(item).strip()]
        ):
            return False
        issue_texts: list[str] = [str(review.get("feedback", ""))]
        issue_texts.extend(str(item) for item in review.get("blocking_issues", []) if str(item).strip())
        scene_script_feedback = review.get("scene_script_feedback") or {}
        for raw_feedback in scene_script_feedback.values():
            if not isinstance(raw_feedback, dict):
                continue
            issue_texts.append(str(raw_feedback.get("summary", "")))
            issue_texts.extend(str(item) for item in raw_feedback.get("script_issues", []) if str(item).strip())
        normalized_texts = [text.strip().lower() for text in issue_texts if text and text.strip()]
        if not normalized_texts:
            return bool(review.get("must_fix", False))
        return any(cls._story_issue_is_hard_blocker(text) for text in normalized_texts)

    @staticmethod
    def _story_issue_is_hard_blocker(text: str) -> bool:
        hard_markers = (
            "wrong sense",
            "selected sense",
            "sense drift",
            "drifts into the wrong sense",
            "unsafe",
            "child safety",
            "dangerous",
            "reckless",
            "harmful",
            "causal continuity",
            "continuity gap",
            "prop continuity",
            "state continuity",
            "timeline contradiction",
            "narrative regression",
            "logical loop",
            "without explanation",
            "without showing why",
            "visual ambiguity",
            "visual executability",
            "not ready for image generation",
            "before image generation",
            "not drawable",
            "descriptor precision",
            "under-specified",
            "under specified",
            "continuity item",
            "recurring visual",
            "recurring object",
            "carried over",
            "carry over",
            "object reappearing",
            "object disappearing",
        )
        soft_markers = (
            "teaching goal",
            "teaching arc",
            "arc role",
            "contrast_or_misuse",
            "contrast or misuse",
            "transfer sentence",
            "transfer line",
            "language memory target",
            "reusable sentence",
            "curiosity",
            "natural introduction",
            "scene progression",
            "glossary",
            "definitional",
            "definition rather than",
            "wording is slightly stiff",
            "more natural for young learners",
            "reuse the word",
            "could flow better",
            "slightly choppy",
            "vary sentence structure",
            "more vivid phrasing",
            "closer alignment to the planned arc",
            "planned reinforcement beat",
            "story turn or conflict",
        )
        if any(marker in text for marker in hard_markers):
            return True
        if any(marker in text for marker in soft_markers):
            return False
        return False

    @staticmethod
    def _compose_visual_feedback(review: SceneVisualReview) -> str:
        parts = [
            f"VLM review marked this scene as {review.match_level}.",
            f"Current visual score: {review.score}.",
            f"Required regeneration mode: {review.regeneration_mode}.",
            f"Reason: {review.reason}",
            f"Director summary: {review.director_feedback.summary}",
        ]
        if review.has_visible_target_word_text:
            parts.append(
                "Observed target-word text: "
                + (review.observed_text if review.observed_text else "[unreadable or missing]")
                + "."
            )
            parts.append(f"Text legibility passed: {review.text_legibility_passed}.")
            if review.text_legibility_reason:
                parts.append(f"Text legibility reason: {review.text_legibility_reason}")
        if review.director_feedback.visual_issues:
            parts.append(
                "Visual issues: " + ", ".join(str(item) for item in review.director_feedback.visual_issues) + "."
            )
        if review.director_feedback.optimization_suggestions:
            parts.append(
                "Optimization suggestions: "
                + ", ".join(str(item) for item in review.director_feedback.optimization_suggestions)
                + "."
            )
        if review.director_feedback.recommended_prompt_adjustments:
            parts.append(
                "Recommended prompt adjustments: "
                + ", ".join(str(item) for item in review.director_feedback.recommended_prompt_adjustments)
                + "."
            )
        if review.director_feedback.repair_instruction:
            parts.append(f"Repair instruction: {review.director_feedback.repair_instruction}")
        parts.append(
            "Keep the target word visually clear, preserve the intended teaching moment, retain character consistency, and ensure any visible target-word text is spelled exactly right and fully legible."
        )
        return " ".join(parts)

    @staticmethod
    def _compose_global_scene_feedback(review: GlobalVisualConsistencyReview, scene_index: int) -> str:
        parts = [
            f"Global visual review score: {review.score}.",
            f"Project-level feedback: {review.feedback}",
        ]
        if review.must_fix:
            parts.append("This round contains must-fix issues and cannot be shipped yet.")
        if review.blocking_issues:
            parts.append("Blocking issues: " + ", ".join(str(item) for item in review.blocking_issues) + ".")
        if review.global_style_adjustments:
            parts.append(
                "Global style adjustments: " + ", ".join(str(item) for item in review.global_style_adjustments) + "."
            )
        scene_feedback = review.scene_feedback.get(str(scene_index))
        if scene_feedback is not None:
            parts.append(f"Scene-specific summary: {scene_feedback.summary}")
            if scene_feedback.visual_issues:
                parts.append("Scene visual issues: " + ", ".join(str(item) for item in scene_feedback.visual_issues) + ".")
            if scene_feedback.optimization_suggestions:
                parts.append(
                    "Scene optimization suggestions: "
                    + ", ".join(str(item) for item in scene_feedback.optimization_suggestions)
                    + "."
                )
            if scene_feedback.recommended_prompt_adjustments:
                parts.append(
                    "Scene prompt adjustments: "
                    + ", ".join(str(item) for item in scene_feedback.recommended_prompt_adjustments)
                    + "."
                )
        script_feedback = review.scene_script_feedback.get(str(scene_index))
        if script_feedback is not None:
            parts.append(f"Scene script summary: {script_feedback.summary}")
            if script_feedback.script_issues:
                parts.append("Scene script issues: " + ", ".join(str(item) for item in script_feedback.script_issues) + ".")
            if script_feedback.revised_plot_description:
                parts.append("Use this revised plot description: " + script_feedback.revised_plot_description + ".")
            if script_feedback.revised_voiceover_and_dialogue:
                parts.append(
                    "Use this revised narrator line: " + script_feedback.revised_voiceover_and_dialogue + "."
                )
        parts.append(
            "Treat cross-scene continuity as a hard constraint. Keep recurring props, clothing, safety gear, and visual logic aligned with the already selected keyframes unless the story clearly motivates a change."
        )
        return " ".join(parts)

    @staticmethod
    def _expand_global_target_scene_indexes(review: GlobalVisualConsistencyReview, scenes: list[Scene]) -> list[int]:
        all_scene_indexes = {scene.scene_index for scene in scenes}
        scene_indexes = set(review.problem_scenes) if review.problem_scenes else set()
        scene_indexes.update(
            int(scene_key)
            for scene_key in review.scene_script_feedback
            if str(scene_key).strip().isdigit()
        )
        if not scene_indexes and (review.must_fix or not review.passed):
            scene_indexes = set(all_scene_indexes)
        return sorted(scene_index for scene_index in scene_indexes if scene_index in all_scene_indexes)

    @staticmethod
    def _apply_global_scene_script_feedback(
        state: VideoProjectState,
        review: GlobalVisualConsistencyReview,
        logger: PipelineRunLogger,
    ) -> None:
        for scene in state.scenes:
            feedback = review.scene_script_feedback.get(str(scene.scene_index))
            if feedback is None:
                continue
            applied = False
            if feedback.revised_plot_description.strip():
                scene.script.plot_description = feedback.revised_plot_description.strip()
                applied = True
            if feedback.revised_voiceover_and_dialogue.strip():
                scene.script.voiceover_and_dialogue = feedback.revised_voiceover_and_dialogue.strip()
                applied = True
            if applied:
                logger.log_event(
                    "global_visual",
                    "Applied global script revision before scene rerender",
                    scene_index=scene.scene_index,
                    summary=feedback.summary,
                    script_issues=feedback.script_issues,
                )

    @staticmethod
    def _collect_reference_scene_images(scene: Scene, all_scenes: list[Scene], *, limit: int = 2) -> list[str]:
        references: list[tuple[int, str]] = []
        for other_scene in all_scenes:
            if other_scene.scene_index == scene.scene_index:
                continue
            image_url = other_scene.visual.keyframe_image_url.strip()
            if not image_url:
                continue
            references.append((abs(other_scene.scene_index - scene.scene_index), image_url))
        references.sort(key=lambda item: item[0])
        return [image_url for _, image_url in references[:limit]]

    @staticmethod
    def _build_scene_continuity_context(scene: Scene, all_scenes: list[Scene], *, limit: int = 3) -> str:
        references: list[tuple[int, str]] = []
        for other_scene in all_scenes:
            if other_scene.scene_index == scene.scene_index:
                continue
            if not other_scene.script.plot_description.strip():
                continue
            references.append(
                (
                    abs(other_scene.scene_index - scene.scene_index),
                    (
                        f"Reference scene {other_scene.scene_index}: "
                        f"{other_scene.script.plot_description.strip()} "
                        f"Continuity items: {VocaVisionPipeline._format_scene_continuity_items(other_scene)} "
                        f"Focus word: {other_scene.target_word_in_scene}."
                    ),
                )
            )
        references.sort(key=lambda item: item[0])
        if not references:
            return ""
        return " ".join(text for _, text in references[:limit])

    @staticmethod
    def _format_scene_continuity_items(scene: Scene) -> str:
        if not scene.script.continuity_items:
            return "none."
        parts: list[str] = []
        for item in scene.script.continuity_items:
            label = item.label.strip() or item.item_key.strip() or "unnamed item"
            description = item.description.strip() or "no description"
            carry_state = item.carry_state.strip() or "unspecified"
            parts.append(f"{label} [{carry_state}]: {description}")
        return "; ".join(parts) + "."

    @staticmethod
    def _is_better_visual_candidate(candidate: dict[str, Any], current_best: dict[str, Any]) -> bool:
        candidate_review = candidate["review"]
        best_review = current_best["review"]
        candidate_rank = (
            1 if candidate_review.approved else 0,
            candidate_review.score,
            candidate_review.iteration,
        )
        best_rank = (
            1 if best_review.approved else 0,
            best_review.score,
            best_review.iteration,
        )
        return candidate_rank > best_rank

    @staticmethod
    def _resolve_problem_scene_indexes(review: GlobalVisualConsistencyReview, scenes: list[Scene]) -> list[int]:
        all_scene_indexes = [scene.scene_index for scene in scenes]
        if review.problem_scenes:
            return [scene_index for scene_index in review.problem_scenes if scene_index in all_scene_indexes]
        return all_scene_indexes

    @staticmethod
    def _build_existing_visual_candidate(scene: Scene) -> dict[str, Any] | None:
        if not scene.visual.keyframe_image_url:
            return None
        return {
            "scene": scene.model_copy(deep=True),
            "director_prompt": scene.visual.director_prompt,
            "image_url": scene.visual.keyframe_image_url,
            "review": scene.visual.review.model_copy(deep=True),
        }

    @staticmethod
    def _is_better_global_snapshot(
        candidate: GlobalVisualConsistencyReview,
        current_best: GlobalVisualConsistencyReview,
    ) -> bool:
        candidate_rank = (
            1 if candidate.passed else 0,
            candidate.score,
            -len(candidate.problem_scenes),
            candidate.iteration,
        )
        best_rank = (
            1 if current_best.passed else 0,
            current_best.score,
            -len(current_best.problem_scenes),
            current_best.iteration,
        )
        return candidate_rank > best_rank

    @staticmethod
    def _normalize_regeneration_mode(raw_value: Any, *, match_level: str) -> str:
        normalized = str(raw_value or "").strip().lower()
        if normalized in {"none", "image_to_image", "text_to_image"}:
            return normalized
        if match_level == "minor":
            return "none"
        return "text_to_image"

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None

    @classmethod
    def _enforce_target_text_gate(cls, scene: Scene, review_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(review_payload)
        director_feedback = dict(payload.get("director_feedback") or {})
        observed_text = str(payload.get("observed_text", "")).strip()
        normalized_observed_text = normalize_visible_text(observed_text)
        normalized_target_word = normalize_visible_text(scene.target_word_in_scene)
        has_visible_target_word_text = bool(payload.get("has_visible_target_word_text", False)) or bool(
            normalized_observed_text
        )
        text_legibility_passed = cls._coerce_optional_bool(payload.get("text_legibility_passed"))
        text_legibility_reason = str(payload.get("text_legibility_reason", "")).strip()
        gate_reasons: list[str] = []

        if has_visible_target_word_text:
            if not normalized_observed_text:
                gate_reasons.append("Visible target-word text is present but the letters are not readable enough to transcribe.")
            elif normalized_observed_text != normalized_target_word:
                gate_reasons.append(
                    f"Observed text '{observed_text}' does not exactly match the target word '{scene.target_word_in_scene}'."
                )
            if text_legibility_passed is not True:
                gate_reasons.append(
                    text_legibility_reason
                    or "The visible target-word text does not pass the legibility check."
                )

        payload["has_visible_target_word_text"] = has_visible_target_word_text
        payload["observed_text"] = observed_text
        payload["text_legibility_passed"] = text_legibility_passed
        payload["text_legibility_reason"] = text_legibility_reason

        if not gate_reasons:
            payload["director_feedback"] = director_feedback
            return payload

        payload["match_level"] = "major"
        payload["score"] = min(cls._coerce_review_score(payload.get("score")), 4.5)
        if str(payload.get("regeneration_mode", "")).strip().lower() in {"", "none"}:
            payload["regeneration_mode"] = "image_to_image"

        gate_summary = "Text gate failed: " + " ".join(gate_reasons)
        existing_reason = str(payload.get("reason", "")).strip()
        payload["reason"] = gate_summary if not existing_reason else f"{existing_reason} {gate_summary}".strip()

        summary = str(director_feedback.get("summary", "")).strip()
        director_feedback["summary"] = (
            f"{summary} Text rendering must be repaired before approval.".strip()
            if summary
            else "Text rendering must be repaired before approval."
        )
        visual_issues = [str(item) for item in director_feedback.get("visual_issues", [])]
        optimization_suggestions = [str(item) for item in director_feedback.get("optimization_suggestions", [])]
        prompt_adjustments = [str(item) for item in director_feedback.get("recommended_prompt_adjustments", [])]
        cls._append_unique_feedback_item(visual_issues, gate_summary)
        cls._append_unique_feedback_item(
            optimization_suggestions,
            "Repair the visible target-word text so every letter is crisp, high-contrast, and easy to read.",
        )
        cls._append_unique_feedback_item(
            prompt_adjustments,
            f"Render the target word '{scene.target_word_in_scene}' exactly, with clean sans-serif lettering and no stylized distortions.",
        )
        director_feedback["visual_issues"] = visual_issues
        director_feedback["optimization_suggestions"] = optimization_suggestions
        director_feedback["recommended_prompt_adjustments"] = prompt_adjustments
        director_feedback["repair_instruction"] = (
            f"Keep the scene composition, but repair the visible text so it spells '{scene.target_word_in_scene}' exactly and remains fully legible."
        )
        payload["director_feedback"] = director_feedback
        return payload

    @staticmethod
    def _append_unique_feedback_item(items: list[str], item: str) -> None:
        normalized_item = item.strip()
        if normalized_item and normalized_item not in items:
            items.append(normalized_item)

    @classmethod
    def _select_source_image_for_generation(
        cls,
        *,
        scene: Scene,
        initial_feedback: str,
        current_iteration: int,
        reference_scene_images: list[str] | None = None,
    ) -> str | list[str] | None:
        if not scene.visual.keyframe_image_url:
            return None
        if current_iteration == 1 and initial_feedback.strip():
            sources = [scene.visual.keyframe_image_url]
            for reference_image in reference_scene_images or []:
                if reference_image and reference_image not in sources:
                    sources.append(reference_image)
            return sources if len(sources) > 1 else sources[0]
        if scene.visual.review.regeneration_mode == "image_to_image":
            return scene.visual.keyframe_image_url
        return None

    def _prepare_visual_recheck_workspace(
        self,
        *,
        source_project_id: str,
        experiment_project_id: str,
    ) -> ProjectWorkspace:
        source_workspace = ProjectWorkspace(self.settings.workspace_root / source_project_id)
        source_state_path = source_workspace.state_path()
        if not source_state_path.exists():
            raise VocaVisionError(f"Source project state not found: {source_state_path}")

        workspace = ProjectWorkspace.create(self.settings.workspace_root, experiment_project_id)
        for relative_path in ("state/project_state.json",):
            source_path = source_workspace.root / relative_path
            target_path = workspace.root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        if source_workspace.character_reference_path().exists():
            shutil.copy2(source_workspace.character_reference_path(), workspace.character_reference_path())
        source_reference_meta = source_workspace.character_reference_path().with_suffix(".jpeg.meta.json")
        if source_reference_meta.exists():
            shutil.copy2(source_reference_meta, workspace.character_reference_path().with_suffix(".jpeg.meta.json"))

        for source_image_path in sorted((source_workspace.root / "images").glob("scene_*_keyframe*.jpeg")):
            shutil.copy2(source_image_path, workspace.root / "images" / source_image_path.name)
            metadata_path = source_image_path.with_suffix(".jpeg.meta.json")
            if metadata_path.exists():
                shutil.copy2(metadata_path, workspace.root / "images" / metadata_path.name)
        return workspace

    @staticmethod
    def _clear_non_visual_outputs_for_recheck(state: VideoProjectState) -> None:
        for scene in state.scenes:
            scene.audio.tts_path = ""
            scene.audio.duration_sec = 0.0
            scene.audio.spoken_text = ""
            scene.video.anim_path = ""
            scene.post_processing.ass_path = ""
            scene.post_processing.final_merged_path = ""
            scene.post_processing.cloze_ass_path = ""
            scene.post_processing.cloze_merged_path = ""

    @staticmethod
    def _hydrate_missing_spoken_text(state: VideoProjectState) -> None:
        for scene in state.scenes:
            if scene.audio.spoken_text.strip():
                continue
            scene.audio.spoken_text = normalize_spoken_text(scene.script.voiceover_and_dialogue)

    def _hydrate_missing_target_word_specs(self, state: VideoProjectState) -> None:
        specs_by_word = {
            spec.word.lower(): spec.model_copy(deep=True)
            for spec in state.target_word_specs
            if spec.word.strip()
        }
        hydrated_specs: list[TargetWordSpec] = []
        for target_word in state.target_words:
            normalized_word = target_word.strip().lower()
            spec = specs_by_word.get(normalized_word)
            if spec is None:
                spec = self._build_default_target_word_spec(target_word)
            hydrated_specs.append(spec.model_copy(deep=True))
        state.target_word_specs = hydrated_specs

        for scene in state.scenes:
            if scene.target_word_spec.word.strip():
                continue
            fallback_spec = specs_by_word.get(scene.target_word_in_scene.strip().lower())
            if fallback_spec is None:
                fallback_spec = self._build_default_target_word_spec(scene.target_word_in_scene)
            scene.target_word_spec = fallback_spec.model_copy(deep=True)

    @staticmethod
    def _write_visual_recheck_summary(
        *,
        workspace: ProjectWorkspace,
        source_project_id: str,
        experiment_project_id: str,
        summary_records: list[dict[str, Any]],
    ) -> Path:
        summary_path = workspace.logs_dir() / "visual_recheck_summary.md"
        lines = [
            "# Visual Recheck Summary",
            "",
            f"- source_project_id: {source_project_id}",
            f"- experiment_project_id: {experiment_project_id}",
            "",
            "## Scenes",
            "",
        ]
        if not summary_records:
            lines.append("- No scenes were rechecked.")
        for record in summary_records:
            lines.extend(
                [
                    f"### Scene {record['scene_index']} - {record['target_word']}",
                    "",
                    f"- initial_match_level: {record['initial_match_level']}",
                    f"- initial_score: {record['initial_score']}",
                    f"- initial_regeneration_mode: {record['initial_regeneration_mode']}",
                    f"- initial_observed_text: {record['initial_observed_text']}",
                    f"- initial_text_legibility_passed: {record['initial_text_legibility_passed']}",
                    f"- initial_reason: {record['initial_reason']}",
                    f"- final_match_level: {record['final_match_level']}",
                    f"- final_score: {record['final_score']}",
                    f"- final_selected_iteration: {record['final_selected_iteration']}",
                    f"- final_regeneration_mode: {record['final_regeneration_mode']}",
                    f"- final_observed_text: {record['final_observed_text']}",
                    f"- final_text_legibility_passed: {record['final_text_legibility_passed']}",
                    "",
                ]
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path

    def _resolve_target_word_specs(
        self,
        target_words: list[str],
        target_word_specs: list[TargetWordSpec] | None,
    ) -> list[TargetWordSpec]:
        provided_specs = [spec.model_copy(deep=True) for spec in target_word_specs or []]
        if provided_specs and not target_words:
            return provided_specs
        if provided_specs and len(provided_specs) == len(target_words):
            hydrated_specs: list[TargetWordSpec] = []
            for word, spec in zip(target_words, provided_specs, strict=False):
                hydrated_spec = spec.model_copy(deep=True)
                hydrated_spec.word = word
                hydrated_specs.append(hydrated_spec)
            return hydrated_specs

        specs_by_word: dict[str, TargetWordSpec] = {}
        for spec in provided_specs:
            normalized_word = spec.word.strip().lower()
            if normalized_word and normalized_word not in specs_by_word:
                specs_by_word[normalized_word] = spec.model_copy(deep=True)

        if not specs_by_word and hasattr(self.story_agents, "disambiguate_target_words"):
            resolved_specs = self.story_agents.disambiguate_target_words(target_words)
            for spec in resolved_specs:
                normalized_word = spec.word.strip().lower()
                if normalized_word and normalized_word not in specs_by_word:
                    specs_by_word[normalized_word] = spec.model_copy(deep=True)

        hydrated_specs: list[TargetWordSpec] = []
        for word in target_words:
            normalized_word = word.strip().lower()
            spec = specs_by_word.get(normalized_word)
            if spec is None:
                spec = self._build_default_target_word_spec(word)
            else:
                spec = spec.model_copy(deep=True)
                spec.word = word
            hydrated_specs.append(spec)
        return hydrated_specs

    @staticmethod
    def _is_better_story_candidate(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
        def _rank(item: dict[str, Any]) -> tuple[float, int, int, int, int, int]:
            review = item.get("review") or {}
            blocking_issues = [str(issue) for issue in review.get("blocking_issues", []) if str(issue).strip()]
            scene_script_feedback = review.get("scene_script_feedback") or {}
            return (
                float(review.get("score") or 0.0),
                1 if bool(review.get("passed")) else 0,
                1 if not bool(review.get("must_fix", False)) else 0,
                -len(blocking_issues),
                -len(scene_script_feedback),
                int(item.get("iteration") or 0),
            )

        return _rank(candidate) > _rank(incumbent)

    @staticmethod
    def _build_default_target_word_spec(word: str) -> TargetWordSpec:
        sense_id = f"{word.lower()}_default"
        default_candidate = TargetWordSenseCandidate(
            sense_id=sense_id,
            label=word,
            part_of_speech="",
            gloss_en=f"The instructional sense of '{word}'.",
            gloss_zh="",
            visual_anchors=[word],
            negative_anchors=[],
            example_sentence=f"This lesson teaches the word {word}.",
        )
        return TargetWordSpec(
            word=word,
            source_word="",
            relation_to_source="",
            recommended_sense_id=sense_id,
            selected_sense_id=sense_id,
            selected_sense_label=word,
            part_of_speech="",
            gloss_en=default_candidate.gloss_en,
            gloss_zh=default_candidate.gloss_zh,
            visual_anchors=default_candidate.visual_anchors,
            negative_anchors=default_candidate.negative_anchors,
            example_sentence=default_candidate.example_sentence,
            confidence=1.0,
            needs_user_confirmation=False,
            confirmed_by_user=False,
            candidates=[default_candidate],
        )

    def _prepare_learning_inputs(
        self,
        target_word_specs: list[TargetWordSpec],
        *,
        learning_mode: str,
        use_model_planner: bool = True,
    ) -> tuple[list[TargetWordSpec], LearningPlan]:
        requested_mode = (learning_mode or "auto").strip().lower()
        planner_decision = self._resolve_learning_mode_decision(
            target_word_specs,
            requested_mode,
            use_model_planner=use_model_planner,
        )
        resolved_mode = planner_decision["mode"]
        if resolved_mode == "deep_single_word":
            expanded_specs = self._expand_deep_single_word_target_specs(target_word_specs)
            return expanded_specs, self._build_deep_single_word_plan(expanded_specs, planner_decision)
        if resolved_mode == "theme_story":
            return target_word_specs, self._build_theme_story_plan(target_word_specs, planner_decision)
        if resolved_mode == "vocab_sprint":
            return target_word_specs, self._build_vocab_sprint_plan(target_word_specs, planner_decision)
        raise VocaVisionError(f"Unsupported learning mode: {learning_mode}")

    def _expand_deep_single_word_target_specs(
        self,
        target_word_specs: list[TargetWordSpec],
    ) -> list[TargetWordSpec]:
        if len(target_word_specs) != 1:
            raise VocaVisionError("learning_mode 'deep_single_word' currently requires exactly 1 target word.")

        seed_spec = target_word_specs[0].model_copy(deep=True)
        seed_spec.source_word = seed_spec.word
        seed_spec.relation_to_source = "seed_word"
        expanded_specs = [seed_spec]
        seen_words = {seed_spec.word.strip().lower()}

        raw_related_words: list[dict[str, Any]] = []
        if hasattr(self.story_agents, "expand_related_words"):
            try:
                raw_related_words = list(self.story_agents.expand_related_words(seed_spec))
            except Exception:
                raw_related_words = []

        for item in raw_related_words:
            try:
                related_spec = self._build_related_target_word_spec(seed_spec, item)
            except VocaVisionError:
                continue
            normalized_word = related_spec.word.strip().lower()
            if not normalized_word or normalized_word in seen_words:
                continue
            expanded_specs.append(related_spec)
            seen_words.add(normalized_word)
            if len(expanded_specs) == 5:
                return expanded_specs

        for item in self._build_fallback_related_word_payloads(seed_spec, seen_words):
            related_spec = self._build_related_target_word_spec(seed_spec, item)
            normalized_word = related_spec.word.strip().lower()
            if not normalized_word or normalized_word in seen_words:
                continue
            expanded_specs.append(related_spec)
            seen_words.add(normalized_word)
            if len(expanded_specs) == 5:
                return expanded_specs

        raise VocaVisionError(
            f"Unable to expand '{seed_spec.word}' into 4 related words for deep_single_word mode."
        )

    def _build_related_target_word_spec(
        self,
        seed_spec: TargetWordSpec,
        raw_item: dict[str, Any],
    ) -> TargetWordSpec:
        word = str(raw_item.get("word", "")).strip()
        if not word:
            raise VocaVisionError("Expanded related word is missing the 'word' field.")
        normalized_seed = seed_spec.word.strip().lower()
        normalized_word = word.lower()
        relation = str(raw_item.get("relation_type", "")).strip().lower() or "associated"
        part_of_speech = str(raw_item.get("part_of_speech", "")).strip().lower()
        gloss_en = str(raw_item.get("gloss_en", "")).strip() or f"A related teaching word connected to '{seed_spec.word}'."
        gloss_zh = str(raw_item.get("gloss_zh", "")).strip()
        example_sentence = str(raw_item.get("example_sentence", "")).strip() or (
            f"This lesson links {seed_spec.word} with the related word {word}."
        )
        visual_anchors = [str(item).strip() for item in raw_item.get("visual_anchors", []) if str(item).strip()]
        negative_anchors = [str(item).strip() for item in raw_item.get("negative_anchors", []) if str(item).strip()]
        if not visual_anchors:
            visual_anchors = [word]
        sense_id = f"{normalized_word}_related_to_{normalized_seed}"
        candidate = TargetWordSenseCandidate(
            sense_id=sense_id,
            label=word,
            part_of_speech=part_of_speech,
            gloss_en=gloss_en,
            gloss_zh=gloss_zh,
            visual_anchors=visual_anchors,
            negative_anchors=negative_anchors,
            example_sentence=example_sentence,
        )
        return TargetWordSpec(
            word=word,
            source_word=seed_spec.word,
            relation_to_source=relation,
            recommended_sense_id=sense_id,
            selected_sense_id=sense_id,
            selected_sense_label=word,
            part_of_speech=part_of_speech,
            gloss_en=gloss_en,
            gloss_zh=gloss_zh,
            visual_anchors=visual_anchors,
            negative_anchors=negative_anchors,
            example_sentence=example_sentence,
            confidence=1.0,
            needs_user_confirmation=False,
            confirmed_by_user=False,
            candidates=[candidate],
        )

    @staticmethod
    def _build_fallback_related_word_payloads(
        seed_spec: TargetWordSpec,
        seen_words: set[str],
    ) -> list[dict[str, Any]]:
        seed_word = seed_spec.word.strip()
        lower_word = seed_word.lower()

        def add_candidate(
            sink: list[dict[str, Any]],
            word: str,
            relation_type: str,
            gloss_en: str,
        ) -> None:
            normalized_word = word.strip().lower()
            if not normalized_word or normalized_word in seen_words:
                return
            if normalized_word == lower_word:
                return
            sink.append(
                {
                    "word": word.strip(),
                    "relation_type": relation_type,
                    "part_of_speech": "",
                    "gloss_en": gloss_en,
                    "visual_anchors": [word.strip()],
                    "negative_anchors": [],
                    "example_sentence": f"This lesson connects {seed_word} with {word.strip()}.",
                }
            )

        candidates: list[dict[str, Any]] = []
        if lower_word.endswith("e") and len(lower_word) > 2:
            add_candidate(candidates, lower_word[:-1] + "ing", "derivative", f"The ongoing form of '{seed_word}'.")
        else:
            add_candidate(candidates, lower_word + "ing", "derivative", f"The ongoing form of '{seed_word}'.")
        add_candidate(candidates, lower_word + "ful", "derivative", f"An adjective form related to '{seed_word}'.")
        add_candidate(candidates, lower_word + "less", "antonym_like", f"A contrast form that lacks '{seed_word}'.")
        add_candidate(candidates, lower_word + "ly", "derivative", f"An adverb form related to '{seed_word}'.")
        add_candidate(candidates, lower_word + "ness", "derivative", f"A noun form built from '{seed_word}'.")
        add_candidate(candidates, lower_word + "ment", "derivative", f"A noun form related to '{seed_word}'.")
        add_candidate(candidates, lower_word + "er", "derivative", f"A person or thing associated with '{seed_word}'.")
        if not lower_word.startswith(("un", "in", "im", "ir", "il", "non", "dis")):
            add_candidate(candidates, "un" + lower_word, "antonym_like", f"A simple contrast word for '{seed_word}'.")
        return candidates

    def _resolve_learning_mode_decision(
        self,
        target_word_specs: list[TargetWordSpec],
        requested_mode: str,
        *,
        use_model_planner: bool = True,
    ) -> dict[str, Any]:
        if requested_mode != "auto":
            self._validate_learning_mode_request(target_word_specs, requested_mode)
            return {
                "mode": requested_mode,
                "requested_mode": requested_mode,
                "planner_source": "manual",
                "planner_confidence": 1.0,
                "rationale": f"Use the user-requested learning mode '{requested_mode}'.",
                "recommended_scene_count": 0,
            }
        planner_payload = (
            self._plan_learning_mode(target_word_specs)
            if use_model_planner
            else self._fallback_learning_mode_decision(target_word_specs)
        )
        resolved_mode = self._normalize_planned_mode(
            target_word_specs,
            str(planner_payload.get("mode", "")).strip().lower(),
        )
        return {
            "mode": resolved_mode,
            "requested_mode": requested_mode,
            "planner_source": "learning_mode_planner",
            "planner_confidence": self._coerce_review_score(planner_payload.get("confidence")),
            "rationale": str(planner_payload.get("rationale", "")).strip()
            or f"Planner selected '{resolved_mode}'.",
            "recommended_scene_count": int(planner_payload.get("recommended_scene_count", 0) or 0),
        }

    @staticmethod
    def _validate_learning_mode_request(target_word_specs: list[TargetWordSpec], requested_mode: str) -> None:
        word_count = len(target_word_specs)
        if requested_mode == "deep_single_word" and word_count != 1:
            raise VocaVisionError("learning_mode 'deep_single_word' currently requires exactly 1 target word.")
        if requested_mode == "theme_story" and not 2 <= word_count <= 5:
            raise VocaVisionError("learning_mode 'theme_story' currently requires 2 to 5 target words.")
        if requested_mode == "vocab_sprint" and word_count < 2:
            raise VocaVisionError("learning_mode 'vocab_sprint' currently requires at least 2 target words.")
        if requested_mode not in {"deep_single_word", "theme_story", "vocab_sprint"}:
            raise VocaVisionError(
                "learning_mode must be one of: auto, deep_single_word, theme_story, vocab_sprint."
            )

    def _plan_learning_mode(self, target_word_specs: list[TargetWordSpec]) -> dict[str, Any]:
        if hasattr(self.story_agents, "plan_learning_mode"):
            try:
                payload = self.story_agents.plan_learning_mode(target_word_specs)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        return self._fallback_learning_mode_decision(target_word_specs)

    def _normalize_planned_mode(self, target_word_specs: list[TargetWordSpec], planned_mode: str) -> str:
        word_count = len(target_word_specs)
        if planned_mode == "deep_single_word" and word_count == 1:
            return planned_mode
        if planned_mode == "theme_story" and 2 <= word_count <= 5:
            return planned_mode
        if planned_mode == "vocab_sprint" and word_count >= 2:
            return planned_mode
        fallback = self._fallback_learning_mode_decision(target_word_specs)
        return str(fallback["mode"])

    @staticmethod
    def _fallback_learning_mode_decision(target_word_specs: list[TargetWordSpec]) -> dict[str, Any]:
        word_count = len(target_word_specs)
        if word_count == 1:
            return {
                "mode": "deep_single_word",
                "confidence": 0.72,
                "rationale": "A single seed word is best expanded into a related five-word family and then taught through one coherent theme story.",
                "recommended_scene_count": 7,
            }
        if 2 <= word_count <= 5:
            return {
                "mode": "theme_story",
                "confidence": 0.7,
                "rationale": "A small word set can stay memorable inside one shared mini-story.",
                "recommended_scene_count": word_count + 2,
            }
        if word_count >= 6:
            return {
                "mode": "vocab_sprint",
                "confidence": 0.78,
                "rationale": "A larger word list needs brisk coverage and compact memorable scenes.",
                "recommended_scene_count": word_count + 1,
            }
        raise VocaVisionError("At least one target word is required to build a learning plan.")

    def _build_deep_single_word_plan(
        self,
        target_word_specs: list[TargetWordSpec],
        planner_decision: dict[str, Any],
    ) -> LearningPlan:
        seed_spec = target_word_specs[0]
        theme_story_plan = self._build_theme_story_plan(target_word_specs, planner_decision)
        related_words = [spec.word for spec in target_word_specs[1:]]
        seed_rationale = (
            f"Expand the seed word '{seed_spec.word}' into a five-word related family"
            if related_words
            else f"Teach the seed word '{seed_spec.word}' through a coherent related-word story"
        )
        relation_summary = ", ".join(
            f"{spec.word} ({spec.relation_to_source or 'related'})" for spec in target_word_specs[1:]
        )
        rationale = str(planner_decision.get("rationale", "")).strip()
        if relation_summary:
            rationale = (
                f"{rationale} {seed_rationale}: {relation_summary}."
                if rationale
                else f"{seed_rationale}: {relation_summary}."
            )
        elif not rationale:
            rationale = seed_rationale + "."
        return theme_story_plan.model_copy(
            update={
                "requested_mode": str(planner_decision.get("requested_mode", "auto")),
                "rationale": rationale,
            }
        )

    def _build_theme_story_plan(
        self,
        target_word_specs: list[TargetWordSpec],
        planner_decision: dict[str, Any],
    ) -> LearningPlan:
        memory_targets = [self._build_memory_target(spec) for spec in target_word_specs]
        bridge_focus_spec = target_word_specs[len(target_word_specs) // 2]
        focus_words = [spec.word for spec in target_word_specs]
        focus_word_list = ", ".join(focus_words[:-1]) + f", and {focus_words[-1]}" if len(focus_words) > 2 else " and ".join(focus_words)
        story_setting = (
            "One child-safe mini-adventure unfolds across a single continuous outing, so each scene should feel like the next step "
            "in the same mission rather than a reset into a new vignette."
        )
        main_character = (
            "One recurring child learner-protagonist stays at the center of every scene, keeps the same goal, and carries the same core visual identity."
        )
        main_goal = (
            f"Complete one simple mission that naturally requires the learner to experience {focus_word_list} in sequence."
        )
        central_conflict = (
            f"The goal cannot be completed immediately; each scene should introduce a new clue, obstacle, or decision that makes {bridge_focus_spec.word} "
            "and the other focus words matter to the same unfolding problem."
        )
        resolution_condition = (
            "The final scene resolves the mission, shows how the earlier scene outcomes led to the ending, and ties the full word set together in one natural narrator moment."
        )
        continuity_requirements = [
            "Every scene must begin from the changed story state created by the previous scene.",
            "Adjacent scenes must be causally linked, not just thematically related.",
            "The same protagonist, mission, and emotional through-line should remain visible from start to finish.",
            "Scenes should not be reorderable without obviously breaking the story logic.",
            "Each scene must introduce one new development, clue, obstacle, or consequence that pushes the plot forward.",
        ]
        blueprint: list[LearningSceneBlueprint] = [
            LearningSceneBlueprint(
                scene_index=1,
                arc_role="shared_setup",
                focus_word=target_word_specs[0].word,
                teaching_goal=(
                    "Establish the protagonist, the mission, the setting, and the first focus word in a way that makes the next scene necessary."
                ),
            )
        ]
        for index, spec in enumerate(target_word_specs[1:], start=2):
            blueprint.append(
                LearningSceneBlueprint(
                    scene_index=index,
                    arc_role="word_spotlight",
                    focus_word=spec.word,
                    teaching_goal=(
                        f"Make '{spec.word}' the primary teaching focus while continuing the same mission, directly reacting to what happened in scene {index - 1}, "
                        "and ending with a new problem, clue, or decision that leads into the next scene."
                    ),
                )
            )
        blueprint.append(
            LearningSceneBlueprint(
                scene_index=len(target_word_specs) + 1,
                arc_role="story_turn_or_conflict",
                focus_word=bridge_focus_spec.word,
                teaching_goal=(
                    f"Create the main turning point of the mission, making '{bridge_focus_spec.word}' central to the strongest obstacle or choice in the story."
                ),
            )
        )
        blueprint.append(
            LearningSceneBlueprint(
                scene_index=len(target_word_specs) + 2,
                arc_role="story_resolution_and_transfer",
                focus_word=target_word_specs[-1].word,
                teaching_goal=(
                    "Resolve the mission through the accumulated results of the earlier scenes, echo the full target-word set, and end on reusable language."
                ),
            )
        )
        return LearningPlan(
            mode="theme_story",
            requested_mode=str(planner_decision.get("requested_mode", "auto")),
            planner_source=str(planner_decision.get("planner_source", "manual")),
            planner_confidence=self._coerce_review_score(planner_decision.get("planner_confidence")),
            rationale=str(planner_decision.get("rationale", "")).strip()
            or "Use a shared theme-story arc because multiple words can reinforce each other more naturally in one coherent mini-story.",
            recommended_scene_count=len(target_word_specs) + 2,
            story_setting=story_setting,
            main_character=main_character,
            main_goal=main_goal,
            central_conflict=central_conflict,
            resolution_condition=resolution_condition,
            continuity_requirements=continuity_requirements,
            story_arc=[
                "mission setup and learner buy-in",
                "causally linked word spotlight beats",
                "major turning point or conflict",
                "earned story resolution and transfer",
            ],
            memory_targets=memory_targets,
            scene_blueprint=blueprint,
        )

    def _build_vocab_sprint_plan(
        self,
        target_word_specs: list[TargetWordSpec],
        planner_decision: dict[str, Any],
    ) -> LearningPlan:
        memory_targets = [self._build_memory_target(spec) for spec in target_word_specs]
        blueprint: list[LearningSceneBlueprint] = []
        for index, spec in enumerate(target_word_specs, start=1):
            blueprint.append(
                LearningSceneBlueprint(
                    scene_index=index,
                    arc_role="rapid_word_spotlight",
                    focus_word=spec.word,
                    teaching_goal=(
                        f"Teach '{spec.word}' in one compact, visually punchy beat with a memorable situational anchor."
                    ),
                )
            )
        blueprint.append(
            LearningSceneBlueprint(
                scene_index=len(target_word_specs) + 1,
                arc_role="recap_transfer",
                focus_word=target_word_specs[-1].word,
                teaching_goal="Finish with a brisk recap that reinforces the word set and leaves the learner with reusable language.",
            )
        )
        return LearningPlan(
            mode="vocab_sprint",
            requested_mode=str(planner_decision.get("requested_mode", "auto")),
            planner_source=str(planner_decision.get("planner_source", "manual")),
            planner_confidence=self._coerce_review_score(planner_decision.get("planner_confidence")),
            rationale=str(planner_decision.get("rationale", "")).strip()
            or "Use a fast-paced coverage mode because the word list is better served by compact memorable beats than one dense story arc.",
            recommended_scene_count=len(target_word_specs) + 1,
            story_arc=[
                "rapid word spotlights",
                "final recap and transfer",
            ],
            memory_targets=memory_targets,
            scene_blueprint=blueprint,
        )

    @staticmethod
    def _build_memory_target(spec: TargetWordSpec) -> LearningMemoryTarget:
        visual_anchor = spec.visual_anchors[0] if spec.visual_anchors else spec.word
        language_memory = spec.example_sentence.strip() or f"The learner can naturally say a sentence with '{spec.word}'."
        return LearningMemoryTarget(
            word=spec.word,
            situational_memory=(
                f"The learner should remember a clear event that demonstrates '{spec.word}' as '{spec.selected_sense_label or spec.gloss_en}'."
            ),
            visual_memory=(
                f"The learner should remember the distinctive visual anchor '{visual_anchor}' whenever '{spec.word}' appears."
            ),
            language_memory=language_memory,
        )
