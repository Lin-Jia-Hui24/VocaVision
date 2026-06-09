"""Pydantic state objects used across the full pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class CharacterDesign(BaseModel):
    visual_prompt: str = ""
    reference_image_url: str = ""


class TargetWordSenseCandidate(BaseModel):
    sense_id: str = ""
    label: str = ""
    part_of_speech: str = ""
    gloss_en: str = ""
    gloss_zh: str = ""
    visual_anchors: List[str] = Field(default_factory=list)
    negative_anchors: List[str] = Field(default_factory=list)
    example_sentence: str = ""


class TargetWordSpec(BaseModel):
    word: str = ""
    source_word: str = ""
    relation_to_source: str = ""
    recommended_sense_id: str = ""
    selected_sense_id: str = ""
    selected_sense_label: str = ""
    part_of_speech: str = ""
    gloss_en: str = ""
    gloss_zh: str = ""
    visual_anchors: List[str] = Field(default_factory=list)
    negative_anchors: List[str] = Field(default_factory=list)
    example_sentence: str = ""
    confidence: float = 0.0
    needs_user_confirmation: bool = False
    confirmed_by_user: bool = False
    candidates: List[TargetWordSenseCandidate] = Field(default_factory=list)


class SceneContinuityItem(BaseModel):
    item_key: str = ""
    label: str = ""
    description: str = ""
    carry_state: str = ""


class SceneScript(BaseModel):
    plot_description: str = ""
    voiceover_and_dialogue: str = ""
    continuity_items: List[SceneContinuityItem] = Field(default_factory=list)


class DirectorFeedback(BaseModel):
    summary: str = ""
    visual_issues: List[str] = Field(default_factory=list)
    optimization_suggestions: List[str] = Field(default_factory=list)
    recommended_prompt_adjustments: List[str] = Field(default_factory=list)
    repair_instruction: str = ""


class GlobalSceneScriptFeedback(BaseModel):
    summary: str = ""
    script_issues: List[str] = Field(default_factory=list)
    revised_plot_description: str = ""
    revised_voiceover_and_dialogue: str = ""


class SceneVisualReview(BaseModel):
    iteration: int = 0
    match_level: str = ""
    score: float = 0.0
    reason: str = ""
    approved: bool = False
    regeneration_mode: str = "none"
    has_visible_target_word_text: bool = False
    observed_text: str = ""
    text_legibility_passed: bool | None = None
    text_legibility_reason: str = ""
    revised_plot_description: str = ""
    revised_voiceover_and_dialogue: str = ""
    director_feedback: DirectorFeedback = Field(default_factory=DirectorFeedback)


class SceneVisual(BaseModel):
    director_prompt: str = ""
    keyframe_image_url: str = ""
    approved_iteration: int = 0
    selected_iteration: int = 0
    selected_score: float = 0.0
    selected_via_fallback: bool = False
    review: SceneVisualReview = Field(default_factory=SceneVisualReview)


class GlobalVisualConsistencyReview(BaseModel):
    iteration: int = 0
    passed: bool = False
    must_fix: bool = False
    score: float = 0.0
    feedback: str = ""
    blocking_issues: List[str] = Field(default_factory=list)
    problem_scenes: List[int] = Field(default_factory=list)
    global_style_adjustments: List[str] = Field(default_factory=list)
    scene_feedback: dict[str, DirectorFeedback] = Field(default_factory=dict)
    scene_script_feedback: dict[str, GlobalSceneScriptFeedback] = Field(default_factory=dict)
    selected_via_fallback: bool = False


class SceneAudio(BaseModel):
    spoken_text: str = ""
    tts_path: str = ""
    duration_sec: float = 0.0


class SceneVideo(BaseModel):
    anim_path: str = ""


class ScenePostProcessing(BaseModel):
    ass_path: str = ""
    final_merged_path: str = ""
    cloze_ass_path: str = ""
    cloze_merged_path: str = ""


class LearningExerciseQuestion(BaseModel):
    question_id: str = ""
    question_type: str = "multiple_choice"
    question_category: str = "sense_discrimination"
    error_reason_tag: str = ""
    prompt: str = ""
    options: List[str] = Field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    related_words: List[str] = Field(default_factory=list)
    recommended_scene_indices: List[int] = Field(default_factory=list)


class SceneClozeChallenge(BaseModel):
    scene_index: int = 0
    target_word: str = ""
    question_category: str = "cloze_recall"
    prompt: str = ""
    options: List[str] = Field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""


class LearningExerciseBundle(BaseModel):
    recommended_interaction_mode: str = "multiple_choice"
    cloze_challenges: List[SceneClozeChallenge] = Field(default_factory=list)
    practice_questions: List[LearningExerciseQuestion] = Field(default_factory=list)


class Scene(BaseModel):
    scene_index: int
    target_word_in_scene: str
    target_word_spec: TargetWordSpec = Field(default_factory=TargetWordSpec)
    script: SceneScript = Field(default_factory=SceneScript)
    visual: SceneVisual = Field(default_factory=SceneVisual)
    audio: SceneAudio = Field(default_factory=SceneAudio)
    video: SceneVideo = Field(default_factory=SceneVideo)
    post_processing: ScenePostProcessing = Field(default_factory=ScenePostProcessing)


class LearningMemoryTarget(BaseModel):
    word: str = ""
    situational_memory: str = ""
    visual_memory: str = ""
    language_memory: str = ""


class LearningSceneBlueprint(BaseModel):
    scene_index: int = 0
    arc_role: str = ""
    focus_word: str = ""
    teaching_goal: str = ""


class LearningPlan(BaseModel):
    mode: str = "auto"
    requested_mode: str = "auto"
    planner_source: str = "manual"
    planner_confidence: float = 0.0
    rationale: str = ""
    recommended_scene_count: int = 0
    story_setting: str = ""
    main_character: str = ""
    main_goal: str = ""
    central_conflict: str = ""
    resolution_condition: str = ""
    continuity_requirements: List[str] = Field(default_factory=list)
    story_arc: List[str] = Field(default_factory=list)
    memory_targets: List[LearningMemoryTarget] = Field(default_factory=list)
    scene_blueprint: List[LearningSceneBlueprint] = Field(default_factory=list)


class ProjectRunSettings(BaseModel):
    learning_mode: str = "auto"
    storyboard_only: bool = False
    test_mode: bool = False
    max_scenes: int | None = None
    media_workers: int | None = None
    story_score_threshold: float | None = None
    global_visual_score_threshold: float | None = None
    auto_accept_senses: bool = True


class VideoProjectState(BaseModel):
    project_id: str
    source_project_id: str = ""
    comparison_variant: str = "primary"
    target_words: List[str]
    target_word_specs: List[TargetWordSpec] = Field(default_factory=list)
    render_profile: str = "full_video"
    run_settings: ProjectRunSettings = Field(default_factory=ProjectRunSettings)
    learning_plan: LearningPlan = Field(default_factory=LearningPlan)
    learning_exercises: LearningExerciseBundle = Field(default_factory=LearningExerciseBundle)
    character_design: CharacterDesign = Field(default_factory=CharacterDesign)
    scenes: List[Scene] = Field(default_factory=list)
    global_visual_review: GlobalVisualConsistencyReview = Field(default_factory=GlobalVisualConsistencyReview)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Path) -> "VideoProjectState":
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))
