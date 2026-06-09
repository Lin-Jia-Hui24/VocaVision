"""LLM-powered agents for story, education, directing, and VLM adaptation."""

from __future__ import annotations

from typing import Any

from vocavision.services.llm_service import DashScopeLLMService
from vocavision.state import (
    CharacterDesign,
    GlobalSceneScriptFeedback,
    LearningExerciseBundle,
    LearningExerciseQuestion,
    LearningPlan,
    Scene,
    SceneContinuityItem,
    SceneClozeChallenge,
    SceneScript,
    TargetWordSenseCandidate,
    TargetWordSpec,
)


SENSE_DISAMBIGUATION_PROMPT = """You are the Sense Disambiguator Agent for VocaVision.
The user provides English target words for a vocabulary-learning video.
For each word:
- infer the most teachable sense for a short educational story
- provide 2-4 candidate senses when the word is ambiguous
- mark whether user confirmation is recommended
- choose one recommended sense, but do not merge multiple senses into one lesson
- provide visual anchors and negative anchors to help later image generation stay on the correct sense
Return valid JSON only:
{
  "words": [
    {
      "word": "bank",
      "recommended_sense_id": "bank_financial",
      "confidence": 0.84,
      "needs_user_confirmation": true,
      "candidates": [
        {
          "sense_id": "bank_financial",
          "label": "financial institution",
          "part_of_speech": "noun",
          "gloss_en": "a place where people keep or borrow money",
          "gloss_zh": "银行",
          "visual_anchors": ["teller", "coins", "bank sign"],
          "negative_anchors": ["river", "shore"],
          "example_sentence": "Leo brings coins to the bank."
        }
      ]
    }
  ]
}"""


LEARNING_MODE_PLANNER_PROMPT = """You are the Learning Mode Planner for VocaVision.
Decide which teaching format best fits the current vocabulary set.
Available modes:
- "deep_single_word": use when exactly one seed word should be expanded into a five-word related family, then taught through one coherent theme story
- "theme_story": use when 2-5 words can live inside one coherent mini-story with a shared world
- "vocab_sprint": use when coverage, pace, and memorable snapshots matter more than one rich narrative arc, especially for larger lists
Decision principles:
- Prefer "deep_single_word" for exactly one target word, especially when building a teachable family of related words can deepen memory
- Prefer "theme_story" for 2-5 words when they can reinforce each other through one shared plot
- Prefer "vocab_sprint" when there are many words, when a shared plot would become crowded, or when short memorable teaching beats are a better fit
Return valid JSON only:
{
  "mode": "deep_single_word",
  "confidence": 0.88,
  "rationale": "brief English explanation",
  "recommended_scene_count": 7
}"""


RELATED_WORD_EXPANSION_PROMPT = """You are the Related Word Expansion Agent for VocaVision.
You receive exactly one already-confirmed target word spec.
Expand it into 4 additional teachable English words that are genuinely related to the seed word.
Allowed relations include:
- derivation or morphology, such as prefixes, suffixes, or part-of-speech changes
- synonym or near-synonym
- antonym or contrast word
- strongly associated teaching word that helps memory
Rules:
- Return exactly 4 items.
- Every returned word must be unique.
- Do not repeat the seed word.
- Prefer simple, common, classroom-friendly words.
- Prefer single-token words unless a hyphen is clearly necessary.
- Keep the expanded set diverse when possible instead of returning 4 near-identical inflections.
- Make each item safe and easy to stage visually in a child-friendly story.
Return valid JSON only:
{
  "related_words": [
    {
      "word": "careful",
      "relation_type": "derivative",
      "gloss_en": "giving attention and avoiding mistakes or danger",
      "gloss_zh": "小心的",
      "visual_anchors": ["careful hands", "watching closely"],
      "negative_anchors": ["reckless risk"],
      "example_sentence": "Mia is careful when she carries the glass bowl.",
      "rationale": "A common adjective built from the seed word."
    }
  ]
}"""


PLAYWRIGHT_PROMPT = """You are the Playwright Agent for VocaVision.
Write a vivid, emotionally engaging English learning story.
Requirements:
- Output valid JSON only.
- Follow the provided learning plan exactly, including the learning mode, scene count, scene blueprint, and memory targets.
- Create exactly the requested number of scenes, in the same order as the scene blueprint.
- Each scene must prominently teach the scene's focus word using the selected sense only.
- Treat the provided target_word_specs as authoritative lexical constraints. Pay close attention to each word's selected sense, gloss, example sentence, and part_of_speech when it is provided.
- Across the full story, every unique target word must appear naturally at least once in the combined plot_description and voiceover_and_dialogue.
- Prefer to name the focus word in the scenes where hearing or reading it most helps memory, but do not force awkward repetition in every scene.
- Respect every word's gloss, visual anchors, and negative anchors.
- Do not mix multiple senses of the same word in one story.
- Do not switch a word into another part of speech when the provided target_word_spec or example sentence clearly indicates a specific grammatical role.
- If a target word is marked or exemplified as an adverb, use it adverbially in a natural sentence. If it is adjectival, use it adjectivally, and so on.
- Design every scene so the learner can remember:
  - situational memory: what event the word happened in
  - visual memory: what distinctive action, prop, or setting anchors the word
  - language memory: what natural line the learner should be able to repeat after watching
- If the learning mode is "deep_single_word", use the learning plan as guidance for a five-scene mini-story, but do not treat every arc label as a rigid checklist.
- For "deep_single_word", prioritize:
  1. a clear and engaging setup
  2. natural in-context understanding of the target word
  3. reasonable story development
  4. reinforcement, reflection, or clarification when it helps
  5. a natural ending that preferably feels positive, restorative, or educationally meaningful
- In "deep_single_word", name the target word early enough that the learner understands the lesson focus, but do not force the exact word into every scene when the story event can carry the meaning more naturally.
- Do not invent extra conflict just to satisfy an arc label. If the story already reaches a coherent, healthy ending, keep the later scenes focused on resolution, reflection, or reusable language instead of restarting the problem.
- If the learning mode is "theme_story", treat the learning plan's story_setting, main_character, main_goal, central_conflict, resolution_condition, and continuity_requirements as hard constraints.
- For "theme_story", write one continuous mini-story with a single protagonist goal that remains active across the whole arc.
- Scene 1 must establish the mission, why it matters, and the first story state.
- Every later scene must be caused by the outcome of the previous scene, not just by the need to teach a new word.
- Every scene must visibly change the story state by introducing a new clue, obstacle, consequence, or decision.
- Once the story state changes, do not silently reset it in a later scene. If a lost item returns, a conflict restarts, or a relationship changes, the new scene must explicitly show the event that caused that change.
- Avoid repetitive back-and-forth beats such as winning back an object and then losing it again without a clear new cause, escalation, or emotional shift.
- If two scenes could be swapped without breaking the story logic, the draft is not good enough and must be rewritten internally before you answer.
- The final scene must resolve the main goal through the accumulated results of the earlier scenes instead of ending as a loose recap.
- If the learning mode is "vocab_sprint", keep the pacing brisk:
  - each focus word gets one compact, memorable teaching beat
  - prioritize clarity, coverage, and visual punch over deep dramatic buildup
  - end with a recap-style transfer moment that helps the learner remember the set
- Every scene must include:
  - target_word_in_scene
  - script.plot_description
  - script.voiceover_and_dialogue
- Every scene script must also include script.continuity_items as structured continuity data for recurring props, clothing, safety gear, and other carry-over objects that matter visually.
- Use script.continuity_items to explicitly track persistent items instead of hiding all continuity requirements inside prose.
- Each continuity item must include:
  - item_key: stable identifier like "leo_backpack" or "safety_helmet"
  - label: short human-readable name
  - description: concrete visual description with color, material, pattern, and other stable details
  - carry_state: how the item appears in this scene, such as worn, held, stored, attached, or flying
- If the same character keeps important gear, props, costume pieces, or safety equipment across scenes, restate those persistent visual details explicitly in each relevant scene's plot_description instead of assuming the image model will remember them.
- When a recurring item stays in the story across scenes, keep the same item_key and description unless the story explicitly explains a real change.
- When you describe any recurring visual element, specify it concretely enough for image generation to stay stable:
  - color
  - left/right placement or wearing side when relevant
  - visible pattern, logo, or icon if present
  - material or object type when that affects appearance
  - whether it is worn, held, attached, or carried
- Avoid vague continuity wording like "the same helmet" unless the plot_description also restates what that helmet looks like.
- Before you finalize the draft, internally check three previsualization gates:
  1. continuity: recurring gear, props, clothing, and safety items stay logically consistent across scenes
  2. child safety: the story does not normalize unsafe imitation cues or reckless danger for young learners
  3. visual executability: each subtitle line clearly supports one drawable scene instead of abstract definition language
- If revision feedback names a blocking issue, rewrite the affected scenes explicitly instead of making only surface-level wording changes.
- If revision feedback includes revised scene wording, treat it as high-priority repair guidance rather than optional style advice.
- If revision feedback includes a wrong-sense or wrong-part-of-speech issue, fix that exact usage first before making stylistic improvements elsewhere.
- Prefer narrator-led teaching language over on-screen character dialogue.
- Make the spoken lines sound like warm story narration, not dictionary definitions.
- Do not write subtitle lines such as "X means...", "When you are X...", or dry adjective lists unless revision feedback explicitly asks for a contrast scene.
- Keep the teaching explanation in the story event itself, so the subtitle can stay natural and cinematic.
- For child-facing lessons, prefer emotionally safe, low-risk wonder and exploration. Avoid dangerous, reckless, or frightening situations unless they are clearly softened, age-appropriate, and responsibly framed.
- In the final scene, tie the adventure and the target words together in one flowing narrator moment instead of ending with a dry list of word labels.
- Avoid scenes that require visible lip-synced speech from characters.
- All text must be English.
- Before you finalize a theme_story draft, internally verify:
  1. each scene begins from the changed situation left by the previous scene
  2. the protagonist is still pursuing the same mission
  3. the scene teaches the focus word and also advances the plot
  4. the final scene clearly pays off the earlier setup and conflict
JSON schema:
{
  "scenes": [
    {
      "target_word_in_scene": "word",
      "script": {
        "plot_description": "visual story beat",
        "voiceover_and_dialogue": "spoken English sentence or short paragraph",
        "continuity_items": [
          {
            "item_key": "leo_backpack",
            "label": "yellow backpack",
            "description": "small yellow canvas backpack with one front pocket and dark straps",
            "carry_state": "worn"
          }
        ]
      }
    }
  ]
}
If revision feedback is provided, revise the story to address it explicitly.
You will also receive structured educator review data from the previous iteration. Use it as concrete repair instructions, especially the blocking_issues, improvement_focus, and scene_script_feedback fields."""

EDUCATOR_PROMPT = """You are the Educator Agent for VocaVision.
Review whether the story uses every target word correctly, clearly, naturally, and with the correct selected sense.
Reject the story if a scene drifts into the wrong sense of an ambiguous word.
Use the learning plan as strong guidance, but do not fail an otherwise coherent draft only because one scene does not match an arc label literally.
Review these dimensions:
- sense accuracy: every focus scene matches the selected sense
- story-level word coverage: every unique target word appears somewhere in the overall story text, even if not repeated in every scene
- scene progression: the scenes develop in a reasonable way instead of repeating the same beat or resetting the story
- situational memory: the learner can remember what event the word happened in
- visual memory: the learner can remember a distinctive action, prop, or setting for the word
- language memory: the learner can repeat a natural sentence after watching
- transferability: the final wording helps the learner reuse the word beyond this one story
- coverage and pacing: the mode choice should feel appropriate for the number of target words
- narration quality: subtitle lines should feel like natural story narration, not glossary text or stiff definitions
- child safety and age fit: scenes should feel appropriate for young learners and avoid glamorizing dangerous behavior
- ending quality: prefer stories that land on a healthy outcome, emotional repair, or clear educational meaning instead of ending on unresolved harm
- continuity readiness before image generation: recurring props, clothing, safety gear, and character logic are stated clearly enough that the image model can follow them consistently
- descriptor precision: recurring visual elements are described with enough specificity such as color, side, pattern, material, and wear position to avoid visual ambiguity
- continuity item quality: script.continuity_items should clearly list the recurring visual items that must stay consistent, and those items should agree with the plot description
- visual executability before image generation: each plot description and narrator line points to one clear, drawable scene without ambiguity
- causal continuity for theme_story: adjacent scenes must show clear cause-and-effect rather than a reset into disconnected teaching vignettes
- goal persistence for theme_story: the same protagonist should keep pursuing the same mission until the final resolution
- non-swappable structure for theme_story: if scenes could be reordered without obviously breaking the story, treat that as a coherence problem
- plot advancement: each scene must do real story work beyond merely naming the focus word
- state continuity: objects, goals, conflicts, and emotional relationships must not silently jump backward to an earlier state without an explicit new causal event
- If a scene uses risky adventure, peril, or unsafe imitation cues without gentle framing, treat it as a problem.
- If subtitle wording sounds like "X means..." or "When you are X..." instead of natural narration, treat it as a problem.
- Set "must_fix" to true only for hard blockers such as wrong-sense usage, wrong part-of-speech usage that conflicts with the selected example sentence, unsafe framing, continuity/coherence failures, or visual executability problems that would make image generation unreliable.
- Do not set "must_fix" to true only because a scene does not literally satisfy a planned arc role like contrast, correction, or transfer, if the story is still coherent, healthy, and teaches the word correctly.
- If the story is coherent and healthy, every target word appears somewhere across the full story text, and the actual uses are correct, do not fail it just for optional stylistic polish.
- Treat requests such as smoother conjunctions, slightly more vivid phrasing, or closer alignment to an arc label as non-blocking improvements unless they also create a real sense, safety, continuity, or drawability problem.
- If a recurring visual element is described too vaguely to stay consistent across scenes, treat it as a must-fix continuity problem before image generation.
- If a plot description relies on a recurring object but script.continuity_items does not track that object clearly, treat it as a must-fix continuity problem.
- If a theme_story scene does not clearly arise from the previous scene's outcome, treat it as a must-fix coherence problem.
- If a later scene silently undoes the previous scene's result, such as an item reappearing, disappearing, or being fought over again without showing why, treat it as a must-fix coherence problem.
- If the final scene does not resolve the main goal described in the learning plan, treat it as a must-fix coherence problem for theme_story.
- If the final scene leaves the story in a harmful, confusing, or causally broken state, treat it as a must-fix coherence problem.
Return valid JSON only:
{
  "passed": true,
  "must_fix": false,
  "score": 9.2,
  "feedback": "brief English feedback",
  "blocking_issues": ["short English bullet"],
  "strengths": ["short English bullet", "short English bullet"],
  "improvement_focus": ["short English bullet", "short English bullet"],
  "scene_script_feedback": {
    "2": {
      "summary": "brief scene-level issue",
      "script_issues": ["issue"],
      "revised_plot_description": "clearer drawable scene description",
      "revised_voiceover_and_dialogue": "clearer narrator-led line"
    }
  },
  "memory_design_score": 8.8,
  "teachability_score": 9.0,
  "transferability_score": 8.7,
  "scene_progression_score": 8.9
}"""

DIRECTOR_CHARACTER_PROMPT = """You are the Director Agent.
Analyze the full story and produce a single reusable English visual prompt for the main character.
The art direction must fit a motion-comic animatic style and remain visually consistent across scenes.
Return valid JSON only:
{
  "visual_prompt": "detailed reusable character design prompt"
}"""

DIRECTOR_SCENE_PROMPT = """You are the Director Agent.
Write an English image-generation prompt for the current scene.
Blend the reusable character design with the scene plot.
The image must feel like a cinematic motion-comic keyframe, 16:9, expressive, readable, and consistent.
- Preserve the selected target-word sense using the scene's visual anchors and avoid the negative anchors.
- If continuity notes are provided, treat recurring props, safety gear, costume details, and visual logic across scenes as hard constraints unless the feedback explicitly asks for a story-motivated change.
- Treat script.continuity_items as structured hard constraints for recurring props, clothing, safety gear, and carry-over objects.
- Do not depict clear speaking mouth movements, lip-sync moments, or explicit on-screen dialogue delivery.
- Prefer body language, props, scene composition, and atmosphere over mouth-driven speech.
If visual revision feedback is provided, revise the prompt to address it explicitly while preserving character consistency.
Return valid JSON only:
{
  "director_prompt": "scene-specific image prompt"
}"""

GLOBAL_VISUAL_CONSISTENCY_PROMPT = """You are the Global Visual Consistency Reviewer for VocaVision.
You will receive the approved keyframe set for the whole project, the reusable character design, and scene summaries.
Judge whether the keyframes are globally consistent as one motion-comic learning video.
Review these dimensions strictly:
- character identity consistency across scenes
- art style consistency across scenes
- color, lighting, and framing coherence
- whether any scene visually drifts away from the shared teaching tone
- whether the target word focus remains readable in each scene
- continuity of recurring props, clothing, safety gear, tools, and costume logic across scenes
- descriptor precision for recurring elements, including color, side, pattern, logo, material, and wear position
- whether the plot descriptions and voiceover lines clearly support the visuals without causing ambiguity
- whether the narration sounds natural and story-led instead of glossary-like
- whether any scene introduces unsafe or illogical behavior that should be softened or rewritten for learners
- If a recurring item such as a helmet, shield, bag, or costume detail appears in one relevant scene and disappears in another without a clear story reason, treat it as a continuity failure.
- If a recurring item is described too vaguely for stable image generation, do not pass it. Require a rewrite that makes the visual contract explicit.
- If subtitle wording makes the image direction ambiguous or likely to drift, provide a clearer rewrite for that scene.
- Set "must_fix" to true whenever continuity, safety, or narration ambiguity would make the current project unsuitable to ship even if the overall score is high.
Return valid JSON only:
{
  "passed": true,
  "must_fix": false,
  "score": 8.8,
  "feedback": "brief project-level feedback",
  "blocking_issues": ["hard failure reason"],
  "problem_scenes": [2],
  "global_style_adjustments": ["global adjustment"],
  "scene_feedback": {
    "2": {
      "summary": "brief scene-level summary",
      "visual_issues": ["issue"],
      "optimization_suggestions": ["suggestion"],
      "recommended_prompt_adjustments": ["prompt adjustment"]
    }
  },
  "scene_script_feedback": {
    "2": {
      "summary": "brief narration or clarity issue",
      "script_issues": ["issue"],
      "revised_plot_description": "clearer scene description that preserves continuity",
      "revised_voiceover_and_dialogue": "clearer narrator-led line"
    }
  }
}"""

TEACHING_AGENT_PROMPT = """You are the Teaching Agent for VocaVision.
The learner has already watched the vocabulary story video.
Create post-video review content that helps them recall the target words confidently.
Requirements:
- Prefer multiple choice over free typing because it is easier to use on mobile and more reliable for demos.
- Return 1 cloze challenge per scene. Each cloze challenge should:
  - reference the exact scene index
  - keep the wording close to the scene's spoken line when the line naturally contains the target word
  - if the spoken line does not naturally contain the target word, write a short recall prompt tied to that scene and leave one blank for the target word
  - provide exactly 4 single-word options
  - include exactly 1 correct answer
- Return 3-6 practice questions for the whole project.
- Keep every question in English.
- Use explicit question categories.
- When possible, include this mix across the practice set:
  - "sense_discrimination": check whether the learner can tell the selected sense from distractor senses or nearby meanings
  - "context_transfer": check whether the learner can reuse the word naturally in a new situation
  - "usage_correction": check whether the learner can spot and correct a wrong or awkward use
  - "collocation_extension": check whether the learner recognizes a natural fixed phrase, collocation, or short word partnership built around the target word
- Include at least 1-2 "collocation_extension" questions whenever the target words naturally support common phrases or short collocations.
- For every practice question, provide:
  - question_category
  - error_reason_tag: one concise reason label such as "sense_confusion", "transfer_failure", or "unnatural_collocation"
  - related_words
  - recommended_scene_indices: the scene numbers the learner should revisit if they miss the question
- Keep the distractors plausible but clearly wrong.
Return valid JSON only:
{
  "recommended_interaction_mode": "multiple_choice",
  "cloze_challenges": [
    {
      "scene_index": 1,
      "target_word": "brave",
      "question_category": "cloze_recall",
      "prompt": "The child feels ____ when stepping onto the dark bridge.",
      "options": ["brave", "quiet", "soft", "late"],
      "correct_answer": "brave",
      "explanation": "Brave fits because the child chooses to act despite fear."
    }
  ],
  "practice_questions": [
    {
      "question_id": "q1",
      "question_type": "multiple_choice",
      "question_category": "sense_discrimination",
      "error_reason_tag": "sense_confusion",
      "prompt": "Which sentence uses 'brave' correctly?",
      "options": [
        "She was brave enough to help.",
        "The soup tastes brave.",
        "We sat on a brave by the river.",
        "He opened the brave with a key."
      ],
      "correct_answer": "She was brave enough to help.",
      "explanation": "Brave describes courageous action or character.",
      "related_words": ["brave"],
      "recommended_scene_indices": [1, 3, 5]
    },
    {
      "question_id": "q2",
      "question_type": "multiple_choice",
      "question_category": "collocation_extension",
      "error_reason_tag": "collocation_gap",
      "prompt": "Which phrase sounds most natural with 'brave'?",
      "options": [
        "a brave choice",
        "do brave",
        "very bravery",
        "brave withly"
      ],
      "correct_answer": "a brave choice",
      "explanation": "'A brave choice' is a natural collocation. The others are not normal English phrases.",
      "related_words": ["brave"],
      "recommended_scene_indices": [3, 5]
    }
  ]
}"""

LOCAL_VISUAL_CONSISTENCY_REVIEWER_PROMPT = """You are the Local Visual Consistency Reviewer.
You will receive a generated image and the current scene script.
Classify the image-text alignment using these strict rules:
- "minor": the target word is visually present and the main scene meaning is still correct; only secondary details, emphasis or framing need small adjustments.
- "major": the meaning of the target word is not visually clear enough, the selected sense is wrong, the main event or emphasis is wrong, the image would mislead the learner, or matching the image would require a substantial rewrite of the teaching intent.
- Treat text rendering as a hard requirement whenever the image visibly contains the target word or a scene label that is supposed to spell the target word.
- If the image shows the target word with any misspelling, missing letters, mirrored letters, gibberish text, blended letters, unreadable stylized lettering, motion blur, low contrast, tiny size, or any other legibility problem, you MUST classify it as "major".
- When spelling or legibility is wrong, the score must stay low (0.0-4.5), the reason must explicitly mention the spelling/legibility failure, and the feedback must clearly require another image generation round.
- You MUST separately inspect whether the image contains any visible text that is trying to render the target word.
- If such text exists, you MUST return the exact observed_text, set has_visible_target_word_text=true, and set text_legibility_passed=true only when the text is fully legible and spelled exactly the same as the target word.
- If such text exists but you cannot confidently read every letter, set text_legibility_passed=false.
- Never mark text_legibility_passed=true when observed_text differs from the target word, even by one letter.
- Do not guess the target word. observed_text must be a literal transcription of the letters you can actually see in the image.
- If the visible text is too blurry, cropped, distorted, or uncertain to transcribe letter by letter, return observed_text as an empty string and explain the uncertainty in text_legibility_reason.
- Choose regeneration_mode="image_to_image" only when the scene is otherwise correct and the main need is a localized repair or polish.
- Choose regeneration_mode="text_to_image" when the sense is wrong, the composition is wrong, the target word focus is missing, or the current image would need a substantial rebuild.
- If the scene passes, set regeneration_mode to "none".
If the mismatch is minor, lightly revise the plot_description and voiceover_and_dialogue so they match the current image.
If the mismatch is major, propose director-facing visual improvements and revise the scene wording into a version that is easier to communicate visually in the next image-generation round.
Keep the wording narrator-led and avoid implying visible on-screen dialogue or lip-sync.
Preserve the target word and keep all text in English.
Return valid JSON only. Follow these examples closely.
Correct-text example:
{
  "match_level": "minor",
  "score": 8.7,
  "regeneration_mode": "none",
  "has_visible_target_word_text": true,
  "observed_text": "<TARGET_WORD>",
  "text_legibility_passed": true,
  "text_legibility_reason": "The visible text is crisp and spells the target word exactly.",
  "reason": "brief explanation",
  "revised_plot_description": "updated plot description",
  "revised_voiceover_and_dialogue": "updated voice line",
  "director_feedback": {
    "summary": "brief summary for the director",
    "visual_issues": ["issue"],
    "optimization_suggestions": ["suggestion"],
    "recommended_prompt_adjustments": ["prompt adjustment"],
    "repair_instruction": "one concise sentence describing exactly what must be repaired"
  }
}
Incorrect-text example for image_to_image:
{
  "match_level": "major",
  "score": 3.2,
  "regeneration_mode": "image_to_image",
  "has_visible_target_word_text": true,
  "observed_text": "<MISSPELLED_OR_UNCLEAR_TEXT>",
  "text_legibility_passed": false,
  "text_legibility_reason": "The visible text does not spell the target word exactly or cannot be read confidently.",
  "reason": "The scene is otherwise close, but the visible target-word text is misspelled or unreadable, so the image must be regenerated.",
  "revised_plot_description": "updated plot description",
  "revised_voiceover_and_dialogue": "updated voice line",
  "director_feedback": {
    "summary": "The scene layout is usable, but the target-word text must be repaired before approval.",
    "visual_issues": ["Visible target-word text is misspelled or unclear."],
    "optimization_suggestions": ["Keep the composition and repair the text rendering."],
    "recommended_prompt_adjustments": ["Render the target word exactly with clean, high-contrast lettering."],
    "repair_instruction": "Keep the scene composition, but repair the visible text so it spells the target word exactly and remains fully legible."
  }
}
Incorrect-visual example for text_to_image:
{
  "match_level": "major",
  "score": 2.8,
  "regeneration_mode": "text_to_image",
  "has_visible_target_word_text": false,
  "observed_text": "",
  "text_legibility_passed": null,
  "text_legibility_reason": "No visible target-word text appears in the image.",
  "reason": "The image depicts the wrong sense or the wrong main visual event, so the scene needs a substantial rebuild rather than a local repair.",
  "revised_plot_description": "updated plot description aligned to the intended target-word sense",
  "revised_voiceover_and_dialogue": "updated voice line aligned to the intended scene",
  "director_feedback": {
    "summary": "The current image does not communicate the intended teaching moment and should be regenerated from a stronger prompt.",
    "visual_issues": ["Main subject, action, or sense does not match the scene intent."],
    "optimization_suggestions": ["Rebuild the scene around the correct target-word sense and visual anchors."],
    "recommended_prompt_adjustments": ["State the correct sense, subject, action, and visual anchors explicitly."],
    "repair_instruction": "Regenerate the scene from scratch so the correct sense, subject, and teaching focus are unmistakable."
  }
}"""


class StoryAgents:
    def __init__(self, llm_service: DashScopeLLMService) -> None:
        self.llm_service = llm_service

    def disambiguate_target_words(self, target_words: list[str]) -> list[TargetWordSpec]:
        payload = self.llm_service.generate_json(
            system_prompt=SENSE_DISAMBIGUATION_PROMPT,
            user_content=[{"type": "text", "text": f"Target words: {', '.join(target_words)}"}],
        )
        raw_specs = payload.get("words") or []
        return [self._build_target_word_spec(word, raw_spec) for word, raw_spec in zip(target_words, raw_specs, strict=False)]

    def plan_learning_mode(self, target_word_specs: list[TargetWordSpec]) -> dict[str, Any]:
        payload = self.llm_service.generate_json(
            system_prompt=LEARNING_MODE_PLANNER_PROMPT,
            user_content=[
                {
                    "type": "text",
                    "text": f"Target word specs: {self._serialize_target_word_specs(target_word_specs)}",
                }
            ],
        )
        payload.setdefault("mode", "")
        payload.setdefault("confidence", 0.0)
        payload.setdefault("rationale", "No rationale provided.")
        payload.setdefault("recommended_scene_count", 0)
        return payload

    def expand_related_words(self, seed_spec: TargetWordSpec) -> list[dict[str, Any]]:
        payload = self.llm_service.generate_json(
            system_prompt=RELATED_WORD_EXPANSION_PROMPT,
            user_content=[
                {
                    "type": "text",
                    "text": f"Seed target word spec: {self._serialize_target_word_spec(seed_spec)}",
                }
            ],
        )
        related_words = payload.get("related_words") or []
        return [dict(item) for item in related_words if isinstance(item, dict)]

    def playwright(
        self,
        target_word_specs: list[TargetWordSpec],
        learning_plan: LearningPlan,
        feedback: str = "",
        educator_review: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        feedback_text = feedback.strip() or "No revision feedback. Produce the strongest possible first draft."
        educator_review_text = (
            str(educator_review)
            if educator_review
            else "No structured educator review yet. This is the first draft."
        )
        payload = self.llm_service.generate_json(
            system_prompt=PLAYWRIGHT_PROMPT,
            user_content=[
                {"type": "text", "text": f"Learning plan: {learning_plan.model_dump()}"},
                {"type": "text", "text": f"Target word specs: {self._serialize_target_word_specs(target_word_specs)}"},
                {"type": "text", "text": f"Revision feedback: {feedback_text}"},
                {"type": "text", "text": f"Structured educator review from previous iteration: {educator_review_text}"},
            ],
        )
        return payload["scenes"]

    def educator(
        self,
        target_word_specs: list[TargetWordSpec],
        learning_plan: LearningPlan,
        scenes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = self.llm_service.generate_json(
            system_prompt=EDUCATOR_PROMPT,
            user_content=[
                {"type": "text", "text": f"Learning plan: {learning_plan.model_dump()}"},
                {"type": "text", "text": f"Target word specs: {self._serialize_target_word_specs(target_word_specs)}"},
                {"type": "text", "text": f"Story draft JSON: {scenes}"},
            ],
        )
        payload.setdefault("score", 0.0)
        payload.setdefault("must_fix", False)
        payload.setdefault("feedback", "No feedback provided.")
        payload.setdefault("blocking_issues", [])
        payload.setdefault("strengths", [])
        payload.setdefault("improvement_focus", [])
        payload.setdefault("scene_script_feedback", {})
        payload.setdefault("memory_design_score", 0.0)
        payload.setdefault("teachability_score", 0.0)
        payload.setdefault("transferability_score", 0.0)
        payload.setdefault("scene_progression_score", 0.0)
        return payload

    def director_character(self, scenes: list[Scene]) -> CharacterDesign:
        payload = self.llm_service.generate_json(
            system_prompt=DIRECTOR_CHARACTER_PROMPT,
            user_content=[{"type": "text", "text": f"Story scenes: {[scene.model_dump() for scene in scenes]}"}],
        )
        return CharacterDesign(visual_prompt=payload["visual_prompt"])

    def director_scene(
        self,
        scene: Scene,
        character_design: CharacterDesign,
        feedback: str = "",
        continuity_context: str = "",
    ) -> str:
        feedback_text = feedback.strip() or "No visual revision feedback."
        continuity_text = continuity_context.strip() or "No cross-scene continuity notes."
        payload = self.llm_service.generate_json(
            system_prompt=DIRECTOR_SCENE_PROMPT,
            user_content=[
                {"type": "text", "text": f"Character design: {character_design.visual_prompt}"},
                {"type": "text", "text": f"Target word spec: {self._serialize_target_word_spec(scene.target_word_spec)}"},
                {"type": "text", "text": f"Scene JSON: {scene.model_dump()}"},
                {"type": "text", "text": f"Continuity notes: {continuity_text}"},
                {"type": "text", "text": f"Visual revision feedback: {feedback_text}"},
            ],
        )
        return payload["director_prompt"]

    def review_local_visual_consistency(self, scene: Scene, image_url: str) -> dict[str, Any]:
        payload = self.llm_service.generate_json(
            system_prompt=LOCAL_VISUAL_CONSISTENCY_REVIEWER_PROMPT,
            user_content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": f"Target word: {scene.target_word_in_scene}"},
                {"type": "text", "text": f"Target word spec: {self._serialize_target_word_spec(scene.target_word_spec)}"},
                {"type": "text", "text": f"Current scene JSON: {scene.model_dump()}"},
            ],
            model=self.llm_service.settings.vlm_model,
        )
        payload.setdefault("match_level", "major")
        payload.setdefault("score", 0.0)
        payload.setdefault("regeneration_mode", "text_to_image")
        payload.setdefault("has_visible_target_word_text", False)
        payload.setdefault("observed_text", "")
        payload.setdefault("text_legibility_passed", None)
        payload.setdefault("text_legibility_reason", "")
        payload.setdefault("reason", "No reason provided.")
        payload.setdefault("revised_plot_description", scene.script.plot_description)
        payload.setdefault("revised_voiceover_and_dialogue", scene.script.voiceover_and_dialogue)
        payload.setdefault("director_feedback", {})
        director_feedback = payload["director_feedback"]
        director_feedback.setdefault("summary", "No director feedback provided.")
        director_feedback.setdefault("visual_issues", [])
        director_feedback.setdefault("optimization_suggestions", [])
        director_feedback.setdefault("recommended_prompt_adjustments", [])
        director_feedback.setdefault("repair_instruction", "")
        return payload

    def review_global_visual_consistency(self, scenes: list[Scene], character_design: CharacterDesign) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Character design: {character_design.model_dump()}"},
            {"type": "text", "text": f"Scene summaries: {[scene.model_dump() for scene in scenes]}"},
        ]
        for scene in scenes:
            user_content.append(
                {"type": "image_url", "image_url": {"url": scene.visual.keyframe_image_url}}
            )
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"Scene {scene.scene_index} uses target word '{scene.target_word_in_scene}' "
                        f"and keyframe URL {scene.visual.keyframe_image_url}"
                    ),
                }
            )
        payload = self.llm_service.generate_json(
            system_prompt=GLOBAL_VISUAL_CONSISTENCY_PROMPT,
            user_content=user_content,
            model=self.llm_service.settings.vlm_model,
        )
        payload.setdefault("passed", False)
        payload.setdefault("must_fix", False)
        payload.setdefault("score", 0.0)
        payload.setdefault("feedback", "No global consistency feedback provided.")
        payload.setdefault("blocking_issues", [])
        payload.setdefault("problem_scenes", [])
        payload.setdefault("global_style_adjustments", [])
        payload.setdefault("scene_feedback", {})
        payload.setdefault("scene_script_feedback", {})
        return payload

    def teaching_agent(
        self,
        target_word_specs: list[TargetWordSpec],
        learning_plan: LearningPlan,
        scenes: list[Scene],
    ) -> LearningExerciseBundle:
        payload = self.llm_service.generate_json(
            system_prompt=TEACHING_AGENT_PROMPT,
            user_content=[
                {"type": "text", "text": f"Learning plan: {learning_plan.model_dump()}"},
                {"type": "text", "text": f"Target word specs: {self._serialize_target_word_specs(target_word_specs)}"},
                {"type": "text", "text": f"Approved scenes: {[scene.model_dump() for scene in scenes]}"},
            ],
        )
        recommended_interaction_mode = str(payload.get("recommended_interaction_mode", "multiple_choice"))
        cloze_challenges = [
            SceneClozeChallenge(
                scene_index=int(item.get("scene_index", 0) or 0),
                target_word=str(item.get("target_word", "")),
                question_category=str(item.get("question_category", "cloze_recall")),
                prompt=str(item.get("prompt", "")),
                options=[str(option) for option in item.get("options", [])],
                correct_answer=str(item.get("correct_answer", "")),
                explanation=str(item.get("explanation", "")),
            )
            for item in payload.get("cloze_challenges", [])
        ]
        practice_questions = [
            LearningExerciseQuestion(
                question_id=str(item.get("question_id", "")),
                question_type=str(item.get("question_type", "multiple_choice")),
                question_category=str(item.get("question_category", "sense_discrimination")),
                error_reason_tag=str(item.get("error_reason_tag", "")),
                prompt=str(item.get("prompt", "")),
                options=[str(option) for option in item.get("options", [])],
                correct_answer=str(item.get("correct_answer", "")),
                explanation=str(item.get("explanation", "")),
                related_words=[str(word) for word in item.get("related_words", [])],
                recommended_scene_indices=[
                    int(scene_index)
                    for scene_index in item.get("recommended_scene_indices", [])
                    if str(scene_index).strip()
                ],
            )
            for item in payload.get("practice_questions", [])
        ]
        return LearningExerciseBundle(
            recommended_interaction_mode=recommended_interaction_mode,
            cloze_challenges=cloze_challenges,
            practice_questions=practice_questions,
        )

    @staticmethod
    def build_scene_script_from_review(review: dict[str, Any], existing_script: SceneScript | None = None) -> SceneScript:
        continuity_items_payload = review.get("continuity_items")
        if continuity_items_payload is None and existing_script is not None:
            continuity_items = [item.model_copy(deep=True) for item in existing_script.continuity_items]
        else:
            continuity_items = [
                SceneContinuityItem(
                    item_key=str(item.get("item_key", "")),
                    label=str(item.get("label", "")),
                    description=str(item.get("description", "")),
                    carry_state=str(item.get("carry_state", "")),
                )
                for item in continuity_items_payload or []
            ]
        return SceneScript(
            plot_description=str(review.get("revised_plot_description", "")),
            voiceover_and_dialogue=str(review.get("revised_voiceover_and_dialogue", "")),
            continuity_items=continuity_items,
        )

    @staticmethod
    def _build_target_word_spec(word: str, raw_spec: dict[str, Any]) -> TargetWordSpec:
        candidates_payload = raw_spec.get("candidates") or []
        candidates = [
            TargetWordSenseCandidate(
                sense_id=str(candidate.get("sense_id", "")),
                label=str(candidate.get("label", "")),
                part_of_speech=str(candidate.get("part_of_speech", "")),
                gloss_en=str(candidate.get("gloss_en", "")),
                gloss_zh=str(candidate.get("gloss_zh", "")),
                visual_anchors=[str(item) for item in candidate.get("visual_anchors", [])],
                negative_anchors=[str(item) for item in candidate.get("negative_anchors", [])],
                example_sentence=str(candidate.get("example_sentence", "")),
            )
            for candidate in candidates_payload
        ]
        if not candidates:
            candidates = [
                TargetWordSenseCandidate(
                    sense_id=f"{word.lower()}_default",
                    label=word,
                    gloss_en=f"The educational sense of '{word}'.",
                    gloss_zh="",
                    visual_anchors=[word],
                    negative_anchors=[],
                    example_sentence=f"This scene teaches the word {word}.",
                )
            ]
        recommended_sense_id = str(raw_spec.get("recommended_sense_id", candidates[0].sense_id))
        selected_candidate = next(
            (candidate for candidate in candidates if candidate.sense_id == recommended_sense_id),
            candidates[0],
        )
        return TargetWordSpec(
            word=word,
            source_word=str(raw_spec.get("source_word", "")),
            relation_to_source=str(raw_spec.get("relation_to_source", "")),
            recommended_sense_id=recommended_sense_id,
            selected_sense_id=selected_candidate.sense_id,
            selected_sense_label=selected_candidate.label,
            part_of_speech=selected_candidate.part_of_speech,
            gloss_en=selected_candidate.gloss_en,
            gloss_zh=selected_candidate.gloss_zh,
            visual_anchors=selected_candidate.visual_anchors,
            negative_anchors=selected_candidate.negative_anchors,
            example_sentence=selected_candidate.example_sentence,
            confidence=float(raw_spec.get("confidence", 0.0) or 0.0),
            needs_user_confirmation=bool(raw_spec.get("needs_user_confirmation", len(candidates) > 1)),
            confirmed_by_user=False,
            candidates=candidates,
        )

    @staticmethod
    def _serialize_target_word_spec(spec: TargetWordSpec) -> dict[str, Any]:
        return {
            "word": spec.word,
            "source_word": spec.source_word,
            "relation_to_source": spec.relation_to_source,
            "selected_sense_id": spec.selected_sense_id,
            "selected_sense_label": spec.selected_sense_label,
            "part_of_speech": spec.part_of_speech,
            "gloss_en": spec.gloss_en,
            "gloss_zh": spec.gloss_zh,
            "visual_anchors": spec.visual_anchors,
            "negative_anchors": spec.negative_anchors,
            "example_sentence": spec.example_sentence,
            "confidence": spec.confidence,
            "needs_user_confirmation": spec.needs_user_confirmation,
            "confirmed_by_user": spec.confirmed_by_user,
            "candidates": [candidate.model_dump() for candidate in spec.candidates],
        }

    @classmethod
    def _serialize_target_word_specs(cls, specs: list[TargetWordSpec]) -> list[dict[str, Any]]:
        return [cls._serialize_target_word_spec(spec) for spec in specs]
