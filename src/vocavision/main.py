"""CLI entrypoint for VocaVision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from vocavision.config import VocavisionSettings, default_scene_cap_for_words
from vocavision.pipeline import VocaVisionPipeline
from vocavision.spec_utils import coerce_target_word_specs
from vocavision.runtime_report import validate_environment
from vocavision.state import TargetWordSenseCandidate, TargetWordSpec
from vocavision.web_app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VocaVision backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full video generation pipeline")
    run_parser.add_argument("--project-id", required=True, help="Unique project identifier")
    run_parser.add_argument("--words", nargs="+", required=True, help="English target words")
    run_parser.add_argument(
        "--learning-mode",
        choices=["auto", "deep_single_word", "theme_story", "vocab_sprint"],
        default="auto",
        help="Teaching structure mode for the story: auto, deep_single_word, theme_story, or vocab_sprint.",
    )
    run_parser.add_argument("--max-scenes", type=int, default=None, help="Optional upper limit for generated scenes")
    run_parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Cost-safe test mode: limits scenes to 2 and serializes media generation by default",
    )
    run_parser.add_argument(
        "--media-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent media workers used in the expensive media stage",
    )
    run_parser.add_argument(
        "--storyboard-only",
        action="store_true",
        help="Stop after approved keyframes and caption-ready story text. Skip TTS and video generation.",
    )
    run_parser.add_argument(
        "--story-score-threshold",
        type=float,
        default=None,
        help="Optional override for the story review acceptance threshold.",
    )
    run_parser.add_argument(
        "--global-visual-score-threshold",
        type=float,
        default=None,
        help="Optional override for the global visual review acceptance threshold.",
    )
    run_parser.add_argument(
        "--sense",
        action="append",
        default=[],
        help="Optional sense override in the form word=sense_id. Can be used multiple times.",
    )
    run_parser.add_argument(
        "--auto-accept-senses",
        action="store_true",
        help="Accept the system's recommended senses without interactive confirmation.",
    )
    run_parser.add_argument(
        "--target-specs-file",
        default="",
        help="Optional JSON file containing user-specified target word specs. When provided, sense suggestion is skipped.",
    )

    recheck_parser = subparsers.add_parser(
        "recheck-visuals",
        help="Clone an existing project into a new workspace and rerun the scene visual review loop",
    )
    recheck_parser.add_argument("--source-project-id", required=True, help="Existing project to copy from")
    recheck_parser.add_argument("--project-id", required=True, help="New experiment project identifier")
    recheck_parser.add_argument(
        "--scenes",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of scene indexes to recheck. Default: all scenes.",
    )

    comparison_parser = subparsers.add_parser(
        "generate-comparison-videos",
        help="Generate reviewed / no-local / no-global comparison videos from existing completed projects",
    )
    comparison_target_group = comparison_parser.add_mutually_exclusive_group(required=True)
    comparison_target_group.add_argument("--project-id", help="Existing completed project to materialize comparison videos for")
    comparison_target_group.add_argument(
        "--all-completed",
        action="store_true",
        help="Scan all completed primary projects in the workspace and generate comparison videos for each one",
    )
    comparison_parser.add_argument(
        "--rerender-existing",
        action="store_true",
        help="Rebuild comparison videos even when the target variant workspaces already contain final videos",
    )

    subparsers.add_parser("validate-env", help="Validate runtime prerequisites")
    web_parser = subparsers.add_parser("web", help="Start the web console")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host for the web console")
    web_parser.add_argument("--port", type=int, default=8000, help="Port for the web console")
    return parser
def _parse_sense_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_override in raw_overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --sense value: {raw_override}. Expected word=sense_id.")
        word, sense_id = raw_override.split("=", 1)
        normalized_word = word.strip().lower()
        normalized_sense_id = sense_id.strip()
        if not normalized_word or not normalized_sense_id:
            raise ValueError(f"Invalid --sense value: {raw_override}. Expected word=sense_id.")
        overrides[normalized_word] = normalized_sense_id
    return overrides


def _select_candidate(spec: TargetWordSpec, candidate: TargetWordSenseCandidate, *, confirmed_by_user: bool) -> TargetWordSpec:
    updated = spec.model_copy(deep=True)
    updated.selected_sense_id = candidate.sense_id
    updated.selected_sense_label = candidate.label
    updated.part_of_speech = candidate.part_of_speech
    updated.gloss_en = candidate.gloss_en
    updated.gloss_zh = candidate.gloss_zh
    updated.visual_anchors = list(candidate.visual_anchors)
    updated.negative_anchors = list(candidate.negative_anchors)
    updated.example_sentence = candidate.example_sentence
    updated.confirmed_by_user = confirmed_by_user
    updated.needs_user_confirmation = False
    return updated


def _confirm_target_word_specs(
    specs: list[TargetWordSpec],
    *,
    sense_overrides: dict[str, str],
    auto_accept: bool,
) -> list[TargetWordSpec]:
    confirmed_specs: list[TargetWordSpec] = []
    for spec in specs:
        override_sense_id = sense_overrides.get(spec.word.lower())
        if override_sense_id is not None:
            selected_candidate = next((candidate for candidate in spec.candidates if candidate.sense_id == override_sense_id), None)
            if selected_candidate is None:
                valid_senses = ", ".join(candidate.sense_id for candidate in spec.candidates)
                raise ValueError(f"Unknown sense_id '{override_sense_id}' for word '{spec.word}'. Valid values: {valid_senses}")
            confirmed_specs.append(_select_candidate(spec, selected_candidate, confirmed_by_user=True))
            continue

        recommended_candidate = next(
            (candidate for candidate in spec.candidates if candidate.sense_id == spec.recommended_sense_id),
            spec.candidates[0],
        )
        if auto_accept or not spec.needs_user_confirmation or len(spec.candidates) <= 1:
            confirmed_specs.append(_select_candidate(spec, recommended_candidate, confirmed_by_user=auto_accept))
            continue

        print(f"\nConfirm sense for '{spec.word}':")
        for index, candidate in enumerate(spec.candidates, start=1):
            recommended_suffix = " [recommended]" if candidate.sense_id == spec.recommended_sense_id else ""
            print(
                f"  {index}. {candidate.label}{recommended_suffix} | "
                f"{candidate.gloss_en} | anchors: {', '.join(candidate.visual_anchors)}"
            )
        default_index = next(
            (index for index, candidate in enumerate(spec.candidates, start=1) if candidate.sense_id == spec.recommended_sense_id),
            1,
        )
        user_input = input(f"Select 1-{len(spec.candidates)} for '{spec.word}' [default {default_index}]: ").strip()
        selected_index = default_index if not user_input else int(user_input)
        if selected_index < 1 or selected_index > len(spec.candidates):
            raise ValueError(f"Invalid selection for '{spec.word}': {selected_index}")
        confirmed_specs.append(_select_candidate(spec, spec.candidates[selected_index - 1], confirmed_by_user=True))
    return confirmed_specs


def _load_target_word_specs_from_file(file_path: str, *, fallback_words: list[str]) -> list[TargetWordSpec]:
    path = Path(file_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The target specs JSON file must contain a list of specs.")
    return coerce_target_word_specs(payload, fallback_words=fallback_words, confirmed_by_user=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = VocavisionSettings.from_env()

    if args.command == "validate-env":
        print(json.dumps(validate_environment(settings), indent=2))
        return

    if args.command == "web":
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    if args.command == "recheck-visuals":
        pipeline = VocaVisionPipeline.from_settings(settings)
        state, summary_path = pipeline.run_visual_recheck_experiment(
            source_project_id=args.source_project_id,
            experiment_project_id=args.project_id,
            scene_indexes=args.scenes,
        )
        result = {
            "project_id": state.project_id,
            "source_project_id": args.source_project_id,
            "scene_count": len(state.scenes),
            "state_path": str((settings.workspace_root / state.project_id / "state" / "project_state.json").resolve()),
            "summary_path": str(summary_path.resolve()),
        }
        print(json.dumps(result, indent=2))
        return

    if args.command == "generate-comparison-videos":
        pipeline = VocaVisionPipeline.from_settings(settings)
        if args.all_completed:
            results = pipeline.generate_comparison_variants_for_all_completed_projects(
                rerender_existing=args.rerender_existing
            )
            print(json.dumps({"projects": results}, indent=2))
            return

        variant_paths = pipeline.generate_comparison_variants_for_project(
            source_project_id=args.project_id,
            rerender_existing=args.rerender_existing,
        )
        print(
            json.dumps(
                {
                    "project_id": args.project_id,
                    "variants": {name: str(path.resolve()) for name, path in variant_paths.items()},
                },
                indent=2,
            )
        )
        return

    if args.test_mode:
        settings.max_scenes_per_run = 2 if args.max_scenes is None else min(args.max_scenes, 2)
        requested_workers = 2 if args.media_workers is None else args.media_workers
        settings.media_max_workers = max(1, min(requested_workers, settings.max_scenes_per_run))
    else:
        settings.max_scenes_per_run = (
            args.max_scenes if args.max_scenes is not None else default_scene_cap_for_words(args.words)
        )

    if args.media_workers is not None:
        settings.media_max_workers = max(1, args.media_workers)
    else:
        default_scene_cap = settings.max_scenes_per_run
        if default_scene_cap is not None:
            settings.media_max_workers = max(1, min(default_scene_cap, 5))
        else:
            settings.media_max_workers = 5
    if args.story_score_threshold is not None:
        settings.story_score_threshold = max(0.0, args.story_score_threshold)
    if args.global_visual_score_threshold is not None:
        settings.global_visual_score_threshold = max(0.0, args.global_visual_score_threshold)
    settings.storyboard_only = bool(args.storyboard_only)

    pipeline = VocaVisionPipeline.from_settings(settings)
    if args.target_specs_file:
        confirmed_specs = _load_target_word_specs_from_file(args.target_specs_file, fallback_words=args.words)
    else:
        sense_overrides = _parse_sense_overrides(args.sense)
        suggested_specs = pipeline.suggest_target_word_specs(args.words)
        confirmed_specs = _confirm_target_word_specs(
            suggested_specs,
            sense_overrides=sense_overrides,
            auto_accept=args.auto_accept_senses,
        )
    state, final_video_path = pipeline.run(
        project_id=args.project_id,
        target_words=args.words,
        target_word_specs=confirmed_specs,
        learning_mode=args.learning_mode,
    )
    result = {
        "project_id": state.project_id,
        "scene_count": len(state.scenes),
        "learning_mode": state.learning_plan.mode,
        "target_word_specs": [spec.model_dump() for spec in state.target_word_specs],
        "state_path": str((settings.workspace_root / state.project_id / "state" / "project_state.json").resolve()),
        "final_video_path": None if final_video_path is None else str(final_video_path.resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
