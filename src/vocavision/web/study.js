const state = {
  packages: [],
  selectedPackageId: "",
  sessionId: "",
  anonymousLabel: "",
  package: null,
  entryMode: "learning",
  pairwisePairs: [],
  pairwiseIndex: 0,
  answers: new Map(),
  unlockedStepIndex: 0,
  radioValues: {
    "english-level": "",
    "word-familiarity": "",
  },
  lastProgressSentAt: 0,
};

const stepOrder = ["storybook", "video", "cloze", "practice", "survey", "pairwise"];

const elements = {
  setupPanel: document.getElementById("setup-panel"),
  learningPanel: document.getElementById("learning-panel"),
  packageList: document.getElementById("package-list"),
  studyForm: document.getElementById("study-form"),
  setupError: document.getElementById("setup-error"),
  setupEyebrow: document.getElementById("setup-eyebrow"),
  setupTitle: document.getElementById("setup-title"),
  setupDescription: document.getElementById("setup-description"),
  packageFieldset: document.getElementById("package-fieldset"),
  startButton: document.getElementById("start-button"),
  classCode: document.getElementById("class-code"),
  gradeBand: document.getElementById("grade-band"),
  anonymousLabel: document.getElementById("anonymous-label"),
  packageTitle: document.getElementById("package-title"),
  modeChip: document.getElementById("mode-chip"),
  stepTabs: document.querySelectorAll("[data-step]"),
  storybookList: document.getElementById("storybook-list"),
  storybookPlayerPanel: document.getElementById("storybook-player-panel"),
  storybookSceneVideo: document.getElementById("storybook-scene-video"),
  storybookSceneCaption: document.getElementById("storybook-scene-caption"),
  toVideoButton: document.getElementById("to-video-button"),
  studyVideo: document.getElementById("study-video"),
  clozeVideo: document.getElementById("cloze-video"),
  wordList: document.getElementById("word-list"),
  toClozeButton: document.getElementById("to-cloze-button"),
  clozeQuestionList: document.getElementById("cloze-question-list"),
  clozeCount: document.getElementById("cloze-count"),
  toPracticeButton: document.getElementById("to-practice-button"),
  practiceQuestionList: document.getElementById("practice-question-list"),
  practiceCount: document.getElementById("practice-count"),
  toSurveyButton: document.getElementById("to-survey-button"),
  surveyForm: document.getElementById("survey-form"),
  surveyItems: document.getElementById("survey-items"),
  surveyError: document.getElementById("survey-error"),
  finishPanel: document.getElementById("finish-panel"),
  startPairwiseButton: document.getElementById("start-pairwise-button"),
  pairwiseProgress: document.getElementById("pairwise-progress"),
  pairwiseEmpty: document.getElementById("pairwise-empty"),
  pairwisePanel: document.getElementById("pairwise-panel"),
  pairwiseTitle: document.getElementById("pairwise-title"),
  pairwiseWords: document.getElementById("pairwise-words"),
  pairwiseLeftVideo: document.getElementById("pairwise-left-video"),
  pairwiseRightVideo: document.getElementById("pairwise-right-video"),
  pairwiseLeftArtifacts: document.getElementById("pairwise-left-artifacts"),
  pairwiseRightArtifacts: document.getElementById("pairwise-right-artifacts"),
  pairwiseForm: document.getElementById("pairwise-form"),
  pairwiseItems: document.getElementById("pairwise-items"),
  pairwiseError: document.getElementById("pairwise-error"),
  pairwiseCommentLabel: document.getElementById("pairwise-comment-label"),
  skipPairwiseButton: document.getElementById("skip-pairwise-button"),
};

const surveyQuestions = [
  ["interest", "这个视频让我愿意继续看下去。"],
  ["meaning_help", "视频帮助我理解单词意思。"],
  ["memory_help", "故事或画面帮助我记住单词。"],
  ["storybook_help", "绘本卡片帮助我先理解场景。"],
  ["subtitle_help", "字幕或挖空练习帮助我回想单词。"],
  ["cloze_video_help", "挖空视频让我更想主动回忆单词。"],
  ["question_help", "选择题帮助我检查自己是否真的理解。"],
  ["mode_fit", "这种学习形式适合这组单词的数量。"],
  ["load_fit", "学习量刚刚好，不会太累。"],
  ["confidence", "学完后，我更有信心认出或使用这些单词。"],
  ["review_willingness", "我愿意用这种方式复习英语单词。"],
];

const pairwiseQuestions = [
  ["meaning", "哪个更容易理解单词意思？"],
  ["memory", "哪个更容易记住单词？"],
  ["visual_match", "哪个画面和单词更匹配？"],
  ["story", "哪个故事或场景更清楚？"],
  ["review", "哪个更适合复习？"],
  ["preference", "你总体更喜欢哪个？"],
];

const expertPairwiseQuestions = [
  ["sense_accuracy", "词义教学是否准确，避免了歧义或误导？"],
  ["semantic_grounding", "画面、动作和场景是否能支持儿童建立词义表征？"],
  ["narrative_coherence", "故事/场景之间是否连贯，目标词出现是否自然？"],
  ["visual_consistency", "角色、物体、风格和空间关系是否保持一致？"],
  ["cognitive_load", "信息量、语速、字幕与画面是否避免过载？"],
  ["retrieval_support", "挖空视频、挖空题和练习是否有效促进主动回忆？"],
  ["age_appropriateness", "内容是否适合初中生，安全且有学习动机？"],
  ["classroom_usefulness", "作为课堂或课后复习材料是否可用？"],
  ["module_contribution", "完整系统相较另一版本是否体现出明确模块贡献？"],
  ["overall_quality", "整体教学质量哪个更高？"],
];

const pairwiseOptions = [
  ["left_strong", "A明显更好"],
  ["left_slight", "A稍好"],
  ["tie", "差不多"],
  ["right_slight", "B稍好"],
  ["right_strong", "B明显更好"],
];

function activePairwiseQuestions() {
  return state.entryMode === "pairwise" ? expertPairwiseQuestions : pairwiseQuestions;
}

function modeLabel(mode, wordCount) {
  const countText = wordCount <= 1 ? "1个词" : wordCount <= 5 ? "2-5个词" : "多词";
  const labels = {
    deep_single_word: "单词深学",
    theme_story: "主题故事",
    vocab_sprint: "词汇速学",
    auto: "自适应模式",
  };
  return `${labels[mode] || mode || "学习模式"} · ${countText}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "请求失败，请稍后再试。");
  }
  return payload;
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function setSetupError(message) {
  elements.setupError.textContent = message || "";
}

function renderPackages() {
  clearChildren(elements.packageList);
  if (!state.packages.length) {
    const empty = document.createElement("p");
    empty.className = "error-text";
    empty.textContent = "暂时没有可学习的视频包。请先在教师控制台生成 final video。";
    elements.packageList.appendChild(empty);
    elements.startButton.disabled = true;
    return;
  }
  elements.startButton.disabled = false;
  state.packages.forEach((studyPackage) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "package-option";
    if (studyPackage.package_id === state.selectedPackageId) {
      button.classList.add("active");
    }
    button.dataset.packageId = studyPackage.package_id;

    const title = document.createElement("strong");
    title.textContent = studyPackage.title || studyPackage.package_id;
    const meta = document.createElement("span");
    meta.textContent = modeLabel(studyPackage.learning_mode, studyPackage.target_words.length);

    button.append(title, meta);
    button.addEventListener("click", () => {
      state.selectedPackageId = studyPackage.package_id;
      renderPackages();
      setSetupError("");
    });
    elements.packageList.appendChild(button);
  });
}

async function loadPackages() {
  const { packages } = await fetchJson("/api/study/packages");
  state.packages = packages || [];
  const urlPackage = new URLSearchParams(window.location.search).get("package");
  state.selectedPackageId =
    urlPackage && state.packages.some((item) => item.package_id === urlPackage)
      ? urlPackage
      : state.packages[0]?.package_id || "";
  renderPackages();
}

async function loadPairwisePairs() {
  const { pairs } = await fetchJson("/api/study/pairwise");
  state.pairwisePairs = pairs || [];
  state.pairwiseIndex = 0;
}

function configureEntryMode() {
  const params = new URLSearchParams(window.location.search);
  state.entryMode = params.get("mode") === "pairwise" ? "pairwise" : "learning";
  if (state.entryMode === "pairwise") {
    elements.setupEyebrow.textContent = "匿名评审";
    elements.setupTitle.textContent = "直接进行视频对比评分";
    elements.setupDescription.textContent =
      "适合专家或教师评审。无需完成学生学习流程，只记录匿名编号、可选身份信息和 A/B 对比评分。";
    elements.packageFieldset.classList.add("hidden");
    elements.startButton.textContent = "匿名开始对比评分";
    elements.stepTabs.forEach((button) => {
      button.classList.toggle("hidden", button.dataset.step !== "pairwise");
    });
    elements.pairwiseCommentLabel.textContent =
      "请简要说明主要差异、可能的失败模式或课堂使用建议。可空";
  }
}

function bindRadioGroups() {
  document.querySelectorAll("[data-radio-group]").forEach((group) => {
    const groupName = group.getAttribute("data-radio-group");
    group.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        state.radioValues[groupName] = button.dataset.value || "";
        group.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
      });
    });
  });
}

function showStep(stepName) {
  const requestedIndex = stepOrder.indexOf(stepName);
  if (requestedIndex > state.unlockedStepIndex) {
    return;
  }
  document.querySelectorAll(".study-step").forEach((section) => {
    section.classList.toggle("hidden", section.id !== `${stepName}-step`);
  });
  elements.stepTabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.step === stepName);
    button.disabled = stepOrder.indexOf(button.dataset.step) > state.unlockedStepIndex;
  });
  sendEvent("step_changed", { step: stepName });
}

async function startSession(event) {
  event.preventDefault();
  setSetupError("");
  if (!state.selectedPackageId) {
    setSetupError("请先选择一个学习包。");
    return;
  }
  elements.startButton.disabled = true;
  try {
    const payload = {
      package_id: state.selectedPackageId,
      class_code: elements.classCode.value,
      grade_band: elements.gradeBand.value,
      english_level: state.radioValues["english-level"],
      word_familiarity: state.radioValues["word-familiarity"],
    };
    const result = await fetchJson("/api/study/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.sessionId = result.session_id;
    state.anonymousLabel = result.anonymous_label;
    state.package = result.package;
    state.unlockedStepIndex = 0;
    renderLearningPackage();
    elements.setupPanel.classList.add("hidden");
    elements.learningPanel.classList.remove("hidden");
    showStep("storybook");
  } catch (error) {
    setSetupError(error.message);
  } finally {
    elements.startButton.disabled = false;
  }
}

async function startPairwiseSession(event) {
  event.preventDefault();
  setSetupError("");
  elements.startButton.disabled = true;
  try {
    await loadPairwisePairs();
    if (!state.pairwisePairs.length) {
      setSetupError("暂时没有可对比的视频组。");
      return;
    }
    const result = await fetchJson("/api/study/sessions", {
      method: "POST",
      body: JSON.stringify({
        package_id: state.pairwisePairs[0].session_package_id,
        class_code: elements.classCode.value,
        grade_band: elements.gradeBand.value,
        english_level: state.radioValues["english-level"],
        word_familiarity: state.radioValues["word-familiarity"],
      }),
    });
    state.sessionId = result.session_id;
    state.anonymousLabel = result.anonymous_label;
    state.package = result.package;
    elements.setupPanel.classList.add("hidden");
    elements.learningPanel.classList.remove("hidden");
    elements.anonymousLabel.textContent = state.anonymousLabel;
    elements.packageTitle.textContent = "视频对比评分";
    elements.modeChip.textContent = "专家/教师可用";
    state.unlockedStepIndex = 5;
    showStep("pairwise");
    renderPairwisePair();
  } catch (error) {
    setSetupError(error.message);
  } finally {
    elements.startButton.disabled = false;
  }
}

function renderLearningPackage() {
  const studyPackage = state.package;
  elements.anonymousLabel.textContent = state.anonymousLabel;
  elements.packageTitle.textContent = studyPackage.title;
  elements.modeChip.textContent = modeLabel(studyPackage.learning_mode, studyPackage.target_words.length);
  elements.studyVideo.src = studyPackage.final_video_url;
  if (studyPackage.final_cloze_video_url) {
    elements.clozeVideo.src = studyPackage.final_cloze_video_url;
    elements.clozeVideo.classList.remove("hidden");
  } else {
    elements.clozeVideo.removeAttribute("src");
    elements.clozeVideo.classList.add("hidden");
  }

  clearChildren(elements.wordList);
  studyPackage.target_words.forEach((word) => {
    const pill = document.createElement("span");
    pill.className = "word-pill";
    pill.textContent = word;
    elements.wordList.appendChild(pill);
  });
  renderStorybook();
  renderQuestions();
  renderSurvey();
}

function createImageWithFallback(src, alt, placeholderText, className = "") {
  const image = document.createElement("img");
  if (className) {
    image.className = className;
  }
  image.src = src;
  image.alt = alt;
  image.addEventListener(
    "error",
    () => {
      const placeholder = document.createElement("div");
      placeholder.className = className
        ? `${className} storybook-placeholder`
        : "storybook-placeholder";
      placeholder.textContent = placeholderText;
      image.replaceWith(placeholder);
    },
    { once: true },
  );
  return image;
}

function renderStorybook() {
  const cards = state.package?.storybook_review || [];
  clearChildren(elements.storybookList);
  elements.storybookPlayerPanel.classList.add("hidden");
  elements.storybookSceneVideo.removeAttribute("src");
  if (!cards.length) {
    const empty = document.createElement("p");
    empty.className = "feedback-text";
    empty.textContent = "这个学习包暂时没有绘本卡片，可以直接看完整视频。";
    elements.storybookList.appendChild(empty);
    return;
  }
  cards.forEach((card) => {
    const item = document.createElement("article");
    item.className = "storybook-card";
    if (card.image_url) {
      const image = createImageWithFallback(
        card.image_url,
        `Scene ${card.scene_index}`,
        `Scene ${card.scene_index} 图片暂时不可用`,
      );
      item.appendChild(image);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "storybook-placeholder";
      placeholder.textContent = `Scene ${card.scene_index}`;
      item.appendChild(placeholder);
    }
    const body = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = `${card.scene_index}. ${card.target_word || "scene"}`;
    const text = document.createElement("p");
    text.textContent = card.cloze_text || card.spoken_text || "";
    body.append(title, text);
    if (card.scene_video_url) {
      const playButton = document.createElement("button");
      playButton.type = "button";
      playButton.className = "secondary-button compact-button";
      playButton.textContent = "播放这一幕";
      playButton.addEventListener("click", () => playStorybookScene(card));
      body.appendChild(playButton);
    }
    item.appendChild(body);
    elements.storybookList.appendChild(item);
  });
}

function playStorybookScene(card) {
  elements.storybookPlayerPanel.classList.remove("hidden");
  elements.storybookSceneVideo.src = card.scene_video_url;
  elements.storybookSceneCaption.textContent = card.spoken_text || "";
  sendEvent("step_changed", {
    step: "storybook_scene",
    scene_index: card.scene_index,
    target_word: card.target_word,
  });
  elements.storybookSceneVideo.play().catch(() => {});
}

function collectClozeQuestions() {
  const exercises = state.package?.learning_exercises || {};
  const cloze = exercises.cloze_challenges || [];
  return cloze.slice(0, 5);
}

function collectPracticeQuestions() {
  const exercises = state.package?.learning_exercises || {};
  const practice = exercises.practice_questions || [];
  return practice.slice(0, 8);
}

function renderQuestions() {
  const clozeQuestions = collectClozeQuestions();
  const practiceQuestions = collectPracticeQuestions();
  clearChildren(elements.clozeQuestionList);
  clearChildren(elements.practiceQuestionList);
  state.answers.clear();
  elements.clozeCount.textContent = `${clozeQuestions.length} 题`;
  elements.practiceCount.textContent = `${practiceQuestions.length} 题`;
  if (!clozeQuestions.length) {
    const empty = document.createElement("p");
    empty.className = "feedback-text";
    empty.textContent = "这个学习包暂时没有挖空题，可以继续做理解练习。";
    elements.clozeQuestionList.appendChild(empty);
  }
  clozeQuestions.forEach((question, index) => {
    elements.clozeQuestionList.appendChild(renderQuestionCard(question, index));
  });
  if (!practiceQuestions.length) {
    const empty = document.createElement("p");
    empty.className = "feedback-text";
    empty.textContent = "这个学习包暂时没有理解练习，可以直接填写问卷。";
    elements.practiceQuestionList.appendChild(empty);
  }
  practiceQuestions.forEach((question, index) => {
    elements.practiceQuestionList.appendChild(renderQuestionCard(question, index));
  });
}

function renderQuestionCard(question, index) {
    const card = document.createElement("article");
    card.className = "question-card";
    card.dataset.questionId = question.question_id;

    const title = document.createElement("h3");
    title.textContent = `${index + 1}. ${question.prompt}`;
    const answerGrid = document.createElement("div");
    answerGrid.className = "answer-grid";

    question.options.forEach((option) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = option;
      button.addEventListener("click", () => submitAnswer(question, option, card, button));
      answerGrid.appendChild(button);
    });

    const feedback = document.createElement("p");
    feedback.className = "feedback-text";
    feedback.hidden = true;
    card.append(title, answerGrid, feedback);
    return card;
}

async function submitAnswer(question, selectedAnswer, card, selectedButton) {
  if (state.answers.has(question.question_id)) {
    return;
  }
  const buttons = card.querySelectorAll(".answer-grid button");
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    const result = await sendEvent("exercise_answered", {
      package_id: state.package.package_id,
      question_group: question.group,
      question_id: question.question_id,
      selected_answer: selectedAnswer,
      learning_mode: state.package.learning_mode,
      word_count: state.package.target_words.length,
    });
    state.answers.set(question.question_id, result.is_correct);
    buttons.forEach((button) => {
      if (button.textContent === result.correct_answer) {
        button.classList.add("correct");
      }
    });
    if (!result.is_correct) {
      selectedButton.classList.add("wrong");
    }
    const feedback = card.querySelector(".feedback-text");
    feedback.hidden = false;
    feedback.textContent = result.is_correct
      ? `答对了。${result.explanation || ""}`
      : `正确答案是 ${result.correct_answer}。${result.explanation || ""}`;
  } catch (error) {
    buttons.forEach((button) => {
      button.disabled = false;
    });
    const feedback = card.querySelector(".feedback-text");
    feedback.hidden = false;
    feedback.textContent = error.message;
  }
}

function renderSurvey() {
  clearChildren(elements.surveyItems);
  surveyQuestions.forEach(([id, labelText]) => {
    const item = document.createElement("div");
    item.className = "survey-item";
    const label = document.createElement("span");
    label.textContent = labelText;
    const scale = document.createElement("div");
    scale.className = "scale-row";
    for (let value = 1; value <= 5; value += 1) {
      const optionLabel = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = id;
      input.value = String(value);
      const text = document.createElement("span");
      text.textContent = String(value);
      optionLabel.append(input, text);
      scale.appendChild(optionLabel);
    }
    item.append(label, scale);
    elements.surveyItems.appendChild(item);
  });
}

async function submitSurvey(event) {
  event.preventDefault();
  elements.surveyError.textContent = "";
  const form = new FormData(elements.surveyForm);
  const answers = {};
  for (const [id] of surveyQuestions) {
    const value = form.get(id);
    if (!value) {
      elements.surveyError.textContent = "请完成 1-5 分选择题。";
      return;
    }
    answers[id] = Number(value);
  }
  answers.memorable_word = String(form.get("memorable_word") || "").trim();
  answers.comment = String(form.get("comment") || "").trim();
  try {
    await sendEvent("survey_submitted", {
      package_id: state.package.package_id,
      learning_mode: state.package.learning_mode,
      word_count: state.package.target_words.length,
      answers,
      answered_question_count: state.answers.size,
      correct_question_count: [...state.answers.values()].filter(Boolean).length,
    });
    elements.surveyForm.classList.add("hidden");
    elements.finishPanel.classList.remove("hidden");
    await loadPairwisePairs().catch(() => {
      state.pairwisePairs = [];
    });
  } catch (error) {
    elements.surveyError.textContent = error.message;
  }
}

function renderPairwiseItems() {
  clearChildren(elements.pairwiseItems);
  activePairwiseQuestions().forEach(([id, labelText]) => {
    const item = document.createElement("div");
    item.className = "survey-item";
    const label = document.createElement("span");
    label.textContent = labelText;
    const row = document.createElement("div");
    row.className = "pairwise-choice-row";
    pairwiseOptions.forEach(([value, text]) => {
      const optionLabel = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = id;
      input.value = value;
      const optionText = document.createElement("span");
      optionText.textContent = text;
      optionLabel.append(input, optionText);
      row.appendChild(optionLabel);
    });
    item.append(label, row);
    elements.pairwiseItems.appendChild(item);
  });
}

function currentPairwisePair() {
  return state.pairwisePairs[state.pairwiseIndex] || null;
}

async function startPairwise() {
  if (!state.pairwisePairs.length) {
    await loadPairwisePairs().catch(() => {
      state.pairwisePairs = [];
    });
  }
  state.unlockedStepIndex = Math.max(state.unlockedStepIndex, 5);
  showStep("pairwise");
  renderPairwisePair();
}

function renderPairwisePair() {
  const pair = currentPairwisePair();
  elements.pairwiseError.textContent = "";
  elements.pairwiseProgress.textContent = state.pairwisePairs.length
    ? `第 ${state.pairwiseIndex + 1} / ${state.pairwisePairs.length} 组，可随时停止`
    : "";
  if (!pair) {
    elements.pairwisePanel.classList.add("hidden");
    elements.pairwiseEmpty.classList.remove("hidden");
    return;
  }
  elements.pairwiseEmpty.classList.add("hidden");
  elements.pairwisePanel.classList.remove("hidden");
  elements.pairwiseTitle.textContent = pair.title || pair.pair_id;
  elements.pairwiseWords.textContent = (pair.target_words || []).join(" / ");
  elements.pairwiseLeftVideo.src = pair.left.package.final_video_url;
  elements.pairwiseRightVideo.src = pair.right.package.final_video_url;
  renderExpertArtifacts(elements.pairwiseLeftArtifacts, pair.left.package, "A");
  renderExpertArtifacts(elements.pairwiseRightArtifacts, pair.right.package, "B");
  elements.pairwiseForm.reset();
  renderPairwiseItems();
  sendEvent("pairwise_started", {
    pair_id: pair.pair_id,
    order_token: pair.order_token,
    comparison_focus: pair.comparison_focus,
  });
}

function renderExpertArtifacts(container, studyPackage, label) {
  clearChildren(container);
  const clozeVideoUrl = studyPackage.final_cloze_video_url;
  if (clozeVideoUrl) {
    const block = document.createElement("details");
    block.open = state.entryMode === "pairwise";
    const summary = document.createElement("summary");
    summary.textContent = `${label} 挖空视频`;
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.src = clozeVideoUrl;
    block.append(summary, video);
    container.appendChild(block);
  }

  const storybookCards = studyPackage.storybook_review || [];
  const storybookBlock = document.createElement("details");
  storybookBlock.open = state.entryMode === "pairwise";
  const storybookSummary = document.createElement("summary");
  storybookSummary.textContent = `${label} 绘本卡片`;
  const storybookList = document.createElement("div");
  storybookList.className = "expert-mini-list";
  if (!storybookCards.length) {
    const empty = document.createElement("p");
    empty.textContent = "暂无绘本卡片。";
    storybookList.appendChild(empty);
  }
  storybookCards.slice(0, 6).forEach((card) => {
    const item = document.createElement("article");
    if (card.image_url) {
      const image = createImageWithFallback(
        card.image_url,
        `${label} scene ${card.scene_index}`,
        `Scene ${card.scene_index} 图片暂时不可用`,
      );
      item.appendChild(image);
    }
    const text = document.createElement("p");
    text.textContent = `${card.scene_index}. ${card.target_word}: ${card.cloze_text || card.spoken_text || ""}`;
    item.appendChild(text);
    storybookList.appendChild(item);
  });
  storybookBlock.append(storybookSummary, storybookList);
  container.appendChild(storybookBlock);

  const exercises = studyPackage.learning_exercises || {};
  const clozeQuestions = exercises.cloze_challenges || [];
  const practiceQuestions = exercises.practice_questions || [];
  const questionBlock = document.createElement("details");
  questionBlock.open = state.entryMode === "pairwise";
  const questionSummary = document.createElement("summary");
  questionSummary.textContent = `${label} 挖空题与练习题`;
  const questionList = document.createElement("div");
  questionList.className = "expert-question-list";
  [...clozeQuestions.map((question) => ({ ...question, prefix: "挖空" })),
    ...practiceQuestions.map((question) => ({ ...question, prefix: "练习" }))].slice(0, 10).forEach(
    (question, index) => {
      const item = document.createElement("article");
      const title = document.createElement("strong");
      title.textContent = `${question.prefix} ${index + 1}. ${question.prompt}`;
      const options = document.createElement("p");
      options.textContent = `选项：${(question.options || []).join(" / ")}`;
      item.append(title, options);
      questionList.appendChild(item);
    },
  );
  if (!questionList.childElementCount) {
    const empty = document.createElement("p");
    empty.textContent = "暂无练习题。";
    questionList.appendChild(empty);
  }
  questionBlock.append(questionSummary, questionList);
  container.appendChild(questionBlock);
}

async function submitPairwiseRating(event) {
  event.preventDefault();
  const pair = currentPairwisePair();
  if (!pair) {
    return;
  }
  const form = new FormData(elements.pairwiseForm);
  const ratings = {};
  for (const [id] of activePairwiseQuestions()) {
    const value = form.get(id);
    if (!value) {
      elements.pairwiseError.textContent = "请完成这一组对比评分。";
      return;
    }
    ratings[id] = String(value);
  }
  elements.pairwiseError.textContent = "";
  await sendEvent("pairwise_rating_submitted", {
    pair_id: pair.pair_id,
    order_token: pair.order_token,
    comparison_focus: pair.comparison_focus,
    ratings,
    rater_mode: state.entryMode === "pairwise" ? "expert" : "student_optional",
    comment: String(form.get("pairwise_comment") || "").trim(),
  });
  state.pairwiseIndex += 1;
  renderPairwisePair();
}

function skipPairwisePair() {
  const pair = currentPairwisePair();
  if (pair) {
    sendEvent("pairwise_rating_submitted", {
      pair_id: pair.pair_id,
      order_token: pair.order_token,
      comparison_focus: pair.comparison_focus,
      skipped: true,
      rater_mode: state.entryMode === "pairwise" ? "expert" : "student_optional",
      ratings: {},
    });
  }
  state.pairwiseIndex += 1;
  renderPairwisePair();
}

async function sendEvent(eventType, payload = {}) {
  if (!state.sessionId) {
    return {};
  }
  return fetchJson(`/api/study/sessions/${state.sessionId}/events`, {
    method: "POST",
    body: JSON.stringify({
      event_type: eventType,
      payload,
    }),
  });
}

function bindVideoEvents() {
  elements.studyVideo.addEventListener("play", () => {
    sendEvent("video_play", {
      package_id: state.package?.package_id,
      video_kind: "final",
      learning_mode: state.package?.learning_mode,
      current_time_sec: Math.round(elements.studyVideo.currentTime || 0),
    });
  });
  elements.studyVideo.addEventListener("pause", () => {
    if (elements.studyVideo.ended) {
      return;
    }
    sendEvent("video_pause", {
      package_id: state.package?.package_id,
      video_kind: "final",
      current_time_sec: Math.round(elements.studyVideo.currentTime || 0),
      duration_sec: Math.round(elements.studyVideo.duration || 0),
    });
  });
  elements.studyVideo.addEventListener("ended", () => {
    sendEvent("video_ended", {
      package_id: state.package?.package_id,
      video_kind: "final",
      duration_sec: Math.round(elements.studyVideo.duration || 0),
    });
  });
  elements.studyVideo.addEventListener("timeupdate", () => {
    const now = Date.now();
    if (now - state.lastProgressSentAt < 15000) {
      return;
    }
    state.lastProgressSentAt = now;
    sendEvent("video_progress", {
      package_id: state.package?.package_id,
      video_kind: "final",
      current_time_sec: Math.round(elements.studyVideo.currentTime || 0),
      duration_sec: Math.round(elements.studyVideo.duration || 0),
    });
  });
}

function bindClozeVideoEvents() {
  elements.clozeVideo.addEventListener("play", () => {
    sendEvent("video_play", {
      package_id: state.package?.package_id,
      video_kind: "cloze",
      learning_mode: state.package?.learning_mode,
      current_time_sec: Math.round(elements.clozeVideo.currentTime || 0),
    });
  });
  elements.clozeVideo.addEventListener("pause", () => {
    if (elements.clozeVideo.ended) {
      return;
    }
    sendEvent("video_pause", {
      package_id: state.package?.package_id,
      video_kind: "cloze",
      current_time_sec: Math.round(elements.clozeVideo.currentTime || 0),
      duration_sec: Math.round(elements.clozeVideo.duration || 0),
    });
  });
  elements.clozeVideo.addEventListener("ended", () => {
    sendEvent("video_ended", {
      package_id: state.package?.package_id,
      video_kind: "cloze",
      duration_sec: Math.round(elements.clozeVideo.duration || 0),
    });
  });
}

function bindSteps() {
  elements.stepTabs.forEach((button) => {
    button.addEventListener("click", () => showStep(button.dataset.step));
  });
  elements.toVideoButton.addEventListener("click", () => {
    state.unlockedStepIndex = Math.max(state.unlockedStepIndex, 1);
    showStep("video");
  });
  elements.toClozeButton.addEventListener("click", () => {
    state.unlockedStepIndex = Math.max(state.unlockedStepIndex, 2);
    showStep("cloze");
  });
  elements.toPracticeButton.addEventListener("click", () => {
    state.unlockedStepIndex = Math.max(state.unlockedStepIndex, 3);
    showStep("practice");
  });
  elements.toSurveyButton.addEventListener("click", () => {
    state.unlockedStepIndex = Math.max(state.unlockedStepIndex, 4);
    showStep("survey");
  });
}

async function boot() {
  configureEntryMode();
  bindRadioGroups();
  bindSteps();
  bindVideoEvents();
  bindClozeVideoEvents();
  elements.studyForm.addEventListener(
    "submit",
    state.entryMode === "pairwise" ? startPairwiseSession : startSession,
  );
  elements.surveyForm.addEventListener("submit", submitSurvey);
  elements.pairwiseForm.addEventListener("submit", submitPairwiseRating);
  elements.startPairwiseButton.addEventListener("click", startPairwise);
  elements.skipPairwiseButton.addEventListener("click", skipPairwisePair);
  try {
    if (state.entryMode === "pairwise") {
      await loadPairwisePairs();
    } else {
      await loadPackages();
    }
  } catch (error) {
    setSetupError(error.message);
    elements.startButton.disabled = true;
  }
}

boot();
