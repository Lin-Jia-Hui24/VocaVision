const state = {
  suggestedSpecs: [],
  activeProjectId: "",
  pollTimer: null,
  lastWordsSignature: "",
  analyzedWordsSignature: "",
  projectIdAutoManaged: true,
  environmentReady: false,
  recentProjects: [],
  quizResponses: {},
  storybookCards: [],
  activeStorybookScene: null,
  renderedResultProjectId: "",
  quizAnswerKey: {},
  reviewPanelState: {
    story: true,
    visual: false,
    global: false,
    scenes: {},
  },
};

const elements = {
  analyzeButton: document.getElementById("analyze-button"),
  runButton: document.getElementById("run-button"),
  promoteStoryboardButton: document.getElementById("promote-storyboard-button"),
  refreshButton: document.getElementById("refresh-button"),
  projectId: document.getElementById("project-id"),
  wordsText: document.getElementById("words-text"),
  testMode: document.getElementById("test-mode"),
  storyboardOnly: document.getElementById("storyboard-only"),
  autoAccept: document.getElementById("auto-accept"),
  learningMode: document.getElementById("learning-mode"),
  learningModeHint: document.getElementById("learning-mode-hint"),
  maxScenes: document.getElementById("max-scenes"),
  mediaWorkers: document.getElementById("media-workers"),
  storyScoreThreshold: document.getElementById("story-score-threshold"),
  globalVisualScoreThreshold: document.getElementById(
    "global-visual-score-threshold",
  ),
  wordPreview: document.getElementById("word-preview"),
  envStatus: document.getElementById("env-status"),
  envHint: document.getElementById("env-hint"),
  connectionStatus: document.getElementById("connection-status"),
  systemMessage: document.getElementById("system-message"),
  setupPanel: document.getElementById("setup-panel"),
  sensePanel: document.getElementById("sense-panel"),
  progressPanel: document.getElementById("progress-panel"),
  outputPanel: document.getElementById("output-panel"),
  recentProjects: document.getElementById("recent-projects"),
  senseContainer: document.getElementById("sense-container"),
  relatedWordFamily: document.getElementById("related-word-family"),
  jobBadge: document.getElementById("job-badge"),
  timeline: document.getElementById("timeline"),
  activeProject: document.getElementById("active-project"),
  activeStage: document.getElementById("active-stage"),
  activeSceneCount: document.getElementById("active-scene-count"),
  sceneSummary: document.getElementById("scene-summary"),
  eventLog: document.getElementById("event-log"),
  reviewPanels: document.getElementById("review-panels"),
  resultSummary: document.getElementById("result-summary"),
  videoPreview: document.getElementById("video-preview"),
  learningSummary: document.getElementById("learning-summary"),
  clozeVideoPreview: document.getElementById("cloze-video-preview"),
  clozeChallenge: document.getElementById("cloze-challenge"),
  storybookReview: document.getElementById("storybook-review"),
  storybookPlayerPanel: document.getElementById("storybook-player-panel"),
  storybookScenePlayer: document.getElementById("storybook-scene-player"),
  storybookPlayerTitle: document.getElementById("storybook-player-title"),
  storybookPlayerCaption: document.getElementById("storybook-player-caption"),
  practiceQuiz: document.getElementById("practice-quiz"),
  senseTemplate: document.getElementById("sense-card-template"),
};

const STAGE_LABELS = {
  pipeline: "开始项目 / Start",
  sense_disambiguation: "确认词义 / Sense Check",
  story: "故事打磨 / Story",
  character_design: "角色设计 / Character",
  visual: "关键帧审核 / Visual",
  global_visual: "全局一致性 / Global Review",
  media: "生成媒体 / Media",
  finalize: "输出视频 / Final Video",
};

const ARTIFACT_METADATA = {
  state: {
    title: bi("查看项目状态", "Open Project State"),
    description: bi(
      "检查完整状态快照 JSON",
      "Inspect the full state snapshot JSON",
    ),
  },
  events_log: {
    title: bi("查看阶段日志", "Open Event Log"),
    description: bi(
      "快速定位每个阶段发生了什么",
      "Track what happened at each stage",
    ),
  },
  story_iterations: {
    title: bi("查看故事迭代", "Open Story Iterations"),
    description: bi(
      "查看剧本被如何修改和采纳",
      "See how the story evolved across rounds",
    ),
  },
  visual_iterations: {
    title: bi("查看视觉迭代", "Open Visual Iterations"),
    description: bi(
      "查看关键帧审核与返修记录",
      "Review keyframe critiques and retries",
    ),
  },
  global_visual_iterations: {
    title: bi("查看全局审查", "Open Global Review"),
    description: bi(
      "查看跨场景一致性复核",
      "Inspect cross-scene consistency checks",
    ),
  },
  story_summary: {
    title: bi("查看故事摘要", "Open Story Summary"),
    description: bi(
      "快速浏览故事复盘结论",
      "Read the condensed story review summary",
    ),
  },
};

function bi(zh, en) {
  return `${zh} / ${en}`;
}

function generateProjectId() {
  const words = normalizeWords(elements.wordsText?.value || "");
  const prefixParts = words
    .slice(0, 3)
    .map((word) =>
      word
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, ""),
    )
    .filter(Boolean);
  const now = new Date();
  const datePart = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
  ].join("");
  const timePart = [
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
  ].join("");
  let prefix = prefixParts.join("-") || "project";
  if (words.length > 3) {
    prefix = `${prefix}-more`;
  }
  return `${prefix}-${datePart}-${timePart}`;
}

function syncProjectIdInput({ force = false } = {}) {
  if (!force && !state.projectIdAutoManaged) {
    return;
  }
  elements.projectId.value = generateProjectId();
}

function normalizeWords(wordsText) {
  return wordsText
    .split(/[\s,，、;；\n\r\t]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => item.toLowerCase());
}

function readNumberInput(input) {
  const raw = input.value.trim();
  if (!raw) {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function setBusy(button, busy, busyText, idleText) {
  button.disabled = busy;
  button.textContent = busy ? busyText : idleText;
}

function showSystemMessage(message, tone = "info") {
  if (!message) {
    elements.systemMessage.textContent = "";
    elements.systemMessage.className = "system-message hidden";
    return;
  }
  elements.systemMessage.textContent = message;
  elements.systemMessage.className = `system-message ${tone}`;
}

function showEmpty(container, message, className = "empty-state") {
  container.className = className;
  container.innerHTML = message;
}

function scrollToElement(element) {
  if (!element) {
    return;
  }
  element.scrollIntoView({ behavior: "smooth", block: "center" });
}

function setActiveStorybookCard(sceneIndex) {
  state.activeStorybookScene = Number(sceneIndex);
  elements.storybookReview
    .querySelectorAll("[data-storybook-scene]")
    .forEach((card) => {
      const isActive =
        Number(card.getAttribute("data-storybook-scene")) ===
        Number(sceneIndex);
      card.classList.toggle("active", isActive);
    });
}

function playStorybookScene(sceneIndex, { autoplay = true } = {}) {
  const card = state.storybookCards.find(
    (item) => Number(item.scene_index) === Number(sceneIndex),
  );
  if (!card || !card.scene_video_url) {
    showSystemMessage(
      bi(
        `Scene ${sceneIndex} 暂时还没有可播放的视频片段。`,
        `Scene ${sceneIndex} does not have a playable clip yet.`,
      ),
      "warning",
    );
    scrollToElement(elements.storybookReview);
    return;
  }
  const sceneCard = elements.storybookReview.querySelector(
    `[data-storybook-scene="${sceneIndex}"]`,
  );
  setActiveStorybookCard(sceneIndex);
  scrollToElement(sceneCard || elements.storybookPlayerPanel);
  elements.storybookPlayerPanel.classList.remove("hidden");
  elements.storybookPlayerTitle.textContent = `Scene ${sceneIndex} · ${card.target_word}`;
  elements.storybookPlayerCaption.innerHTML = `${bi("点击卡片即可重新播放这一段短片。", "Click any card to replay its scene clip.")} ${highlightTargetWord(card.spoken_text || "", card.target_word)}`;
  if (
    elements.storybookScenePlayer.getAttribute("src") !== card.scene_video_url
  ) {
    elements.storybookScenePlayer.src = card.scene_video_url;
  }
  if (autoplay) {
    const playPromise = elements.storybookScenePlayer.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
  }
}

function jumpToStorybookScene(sceneIndex, { autoplay = false } = {}) {
  const card = elements.storybookReview.querySelector(
    `[data-storybook-scene="${sceneIndex}"]`,
  );
  if (!card) {
    showSystemMessage(
      bi(
        `还没有 Scene ${sceneIndex} 的绘本卡片。`,
        `Storybook card for Scene ${sceneIndex} is not available yet.`,
      ),
      "warning",
    );
    return;
  }
  scrollToElement(card);
  setActiveStorybookCard(sceneIndex);
  if (autoplay || card?.getAttribute("data-has-video") === "true") {
    playStorybookScene(sceneIndex, { autoplay });
  }
}

function formatScore(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "-"
    : Number(value).toFixed(1);
}

function formatList(items, emptyText) {
  return items && items.length ? items.join(" / ") : emptyText;
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function getReviewPanelOpen(key, defaultOpen = false) {
  return Object.prototype.hasOwnProperty.call(state.reviewPanelState, key)
    ? state.reviewPanelState[key]
    : defaultOpen;
}

function getVisualScenePanelOpen(sceneIndex) {
  const key = String(sceneIndex);
  return Object.prototype.hasOwnProperty.call(
    state.reviewPanelState.scenes,
    key,
  )
    ? state.reviewPanelState.scenes[key]
    : false;
}

function bindReviewPanelState() {
  elements.reviewPanels
    .querySelectorAll("[data-review-panel]")
    .forEach((details) => {
      const panelKey = details.getAttribute("data-review-panel");
      details.addEventListener("toggle", () => {
        state.reviewPanelState[panelKey] = details.open;
      });
    });
  elements.reviewPanels
    .querySelectorAll("[data-visual-scene]")
    .forEach((details) => {
      const sceneKey = details.getAttribute("data-visual-scene");
      details.addEventListener("toggle", () => {
        state.reviewPanelState.scenes[sceneKey] = details.open;
      });
    });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildTargetWordPattern(targetWord) {
  const escaped = String(targetWord || "").replace(
    /[.*+?^${}()|[\]\\]/g,
    "\\$&",
  );
  if (!escaped) {
    return null;
  }
  return new RegExp(`\\b(${escaped})\\b`, "gi");
}

function highlightTargetWord(text, targetWord) {
  const pattern = buildTargetWordPattern(targetWord);
  const safeText = escapeHtml(text);
  if (!pattern) {
    return safeText;
  }
  return safeText.replace(pattern, '<mark class="inline-target">$1</mark>');
}

function maskTargetWord(text, targetWord) {
  const pattern = buildTargetWordPattern(targetWord);
  const safeText = escapeHtml(text);
  if (!pattern) {
    return safeText;
  }
  return safeText.replace(pattern, '<span class="inline-blank">_____</span>');
}

function formatQuestionCategory(category) {
  if (category === "sense_discrimination") {
    return bi("义项辨析题", "Sense Check");
  }
  if (category === "context_transfer") {
    return bi("情境迁移题", "Transfer");
  }
  if (category === "usage_correction") {
    return bi("错误纠正题", "Usage Fix");
  }
  if (category === "cloze_recall") {
    return bi("挖空回忆题", "Cloze Recall");
  }
  return bi("练习题", "Practice");
}

function formatErrorReasonTag(tag) {
  if (tag === "sense_confusion") {
    return bi("义项混淆", "Sense Confusion");
  }
  if (tag === "transfer_failure") {
    return bi("迁移失败", "Transfer Gap");
  }
  if (tag === "unnatural_collocation") {
    return bi("搭配不自然", "Unnatural Collocation");
  }
  if (tag === "cloze_recall_gap") {
    return bi("场景回忆薄弱", "Scene Recall Gap");
  }
  return bi("需要复习", "Needs Review");
}

function parseSceneIndices(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item) && item > 0);
}

function getLearningMode() {
  return elements.learningMode.value || "auto";
}

function updateLearningModeHint() {
  const mode = getLearningMode();
  if (mode === "deep_single_word") {
    elements.learningModeHint.textContent = bi(
      "适合 1 个种子词的深度教学 demo。系统会先扩展 4 个相关词，再把这 5 个词一起送入主题故事。",
      "Best for a 1-word deep-learning demo. The system first expands 4 related words, then teaches the 5-word family through one theme story.",
    );
    return;
  }
  if (mode === "theme_story") {
    elements.learningModeHint.textContent = bi(
      "适合 2-5 个词共享一条主线剧情，每个词都有自己的教学场景，最后会统一回收成完整故事。",
      "Best for 2-5 words in one shared plot. Each word gets a teaching beat, then the story resolves as a whole.",
    );
    return;
  }
  if (mode === "vocab_sprint") {
    elements.learningModeHint.textContent = bi(
      "适合较多目标词的快节奏教学，每个词用一个高记忆度 scene 快速讲清，最后统一 recap。",
      "Best for larger word sets. Each word gets one punchy teaching beat, followed by a recap scene.",
    );
    return;
  }
  elements.learningModeHint.textContent = bi(
    "自动模式下，planner 会结合词数和教学目标，在单词深学、主题故事和词汇速学之间选择。",
    "In auto mode, the planner chooses between deep study, theme story, and vocab sprint.",
  );
}

function renderSensePlaceholder() {
  showEmpty(
    elements.senseContainer,
    bi(
      "先输入词表并点击“分析词义”，这里会展示每个词更适合教学的义项候选。",
      "Analyze your words first to see learner-friendly sense suggestions.",
    ),
  );
  showEmpty(
    elements.relatedWordFamily,
    bi(
      "如果当前模式会自动扩词，这里会显示原词和系统扩展词。",
      "If the current mode expands one seed word, the original word and added related words will appear here.",
    ),
  );
}

function syncActionState() {
  elements.refreshButton.disabled = !state.activeProjectId;
}

function renderWordPreview() {
  const words = normalizeWords(elements.wordsText.value);
  state.lastWordsSignature = words.join("|");
  syncProjectIdInput();
  if (!words.length) {
    elements.wordPreview.innerHTML = "";
  } else {
    elements.wordPreview.innerHTML = words
      .map((word) => `<span class="chip">${word}</span>`)
      .join("");
  }
  if (
    state.suggestedSpecs.length &&
    state.lastWordsSignature !== state.analyzedWordsSignature
  ) {
    state.suggestedSpecs = [];
    renderSensePlaceholder();
    showSystemMessage(
      bi(
        "目标词已修改，请重新点击“分析词义”以获得新的义项建议。",
        "The word list changed. Analyze senses again to refresh the suggestions.",
      ),
      "warning",
    );
  }
}

function fillNumberInput(element, value) {
  element.value =
    value === null || value === undefined || value === "" ? "" : String(value);
}

function applyProjectSnapshotToForm(projectId, snapshot) {
  const projectState = snapshot?.state || {};
  const runSettings = snapshot?.run_settings || {};
  const targetWords = Array.isArray(projectState.target_words)
    ? projectState.target_words
    : [];
  const shouldUpdateWords = targetWords.length > 0;

  if (projectId) {
    elements.projectId.value = projectId;
    state.projectIdAutoManaged = false;
  }
  if (shouldUpdateWords) {
    elements.wordsText.value = targetWords.join("\n");
  }
  if (runSettings.learning_mode) {
    elements.learningMode.value = runSettings.learning_mode;
  }
  elements.storyboardOnly.checked = Boolean(runSettings.storyboard_only);
  elements.testMode.checked = Boolean(runSettings.test_mode);
  elements.autoAccept.checked = Boolean(runSettings.auto_accept_senses);
  fillNumberInput(elements.maxScenes, runSettings.max_scenes);
  fillNumberInput(elements.mediaWorkers, runSettings.media_workers);
  fillNumberInput(
    elements.storyScoreThreshold,
    runSettings.story_score_threshold,
  );
  fillNumberInput(
    elements.globalVisualScoreThreshold,
    runSettings.global_visual_score_threshold,
  );
  renderWordPreview();
  updateLearningModeHint();
}

function renderOverview(data) {
  const environment = data.environment || {};
  state.recentProjects = data.recent_projects || [];
  const envOk =
    environment.has_dashscope_api_key &&
    environment.has_ark_api_key &&
    environment.ffmpeg_found &&
    environment.ffprobe_found;
  state.environmentReady = envOk;
  elements.envStatus.textContent = envOk
    ? bi("基础运行条件已就绪", "Environment Ready")
    : bi("仍有环境项需要补齐", "Some setup items still need attention");
  elements.envStatus.className = envOk ? "success-text" : "warning-text";
  const serviceText =
    environment.has_dashscope_api_key && environment.has_ark_api_key
      ? bi("模型服务已配置", "Model services configured")
      : bi("模型服务未完整配置", "Model services incomplete");
  const ffmpegText =
    environment.ffmpeg_found && environment.ffprobe_found
      ? bi("视频工具已就绪", "Video tools ready")
      : bi("视频工具未就绪", "Video tools missing");
  elements.envHint.textContent = `${serviceText} | ${ffmpegText}`;
  elements.connectionStatus.textContent = envOk
    ? bi(
        "前端已成功读取后端概览接口。",
        "Frontend reached the backend overview API.",
      )
    : bi(
        "前端已连上后端，但后端环境尚未完全就绪。",
        "Frontend reached the backend, but the backend environment is not fully ready.",
      );
  renderRecentProjects();
}

function renderRecentProjects(projects = state.recentProjects) {
  if (!projects.length) {
    showEmpty(
      elements.recentProjects,
      bi("还没有可显示的项目。", "No recent projects yet."),
    );
    return;
  }
  elements.recentProjects.className = "stack-list";
  elements.recentProjects.innerHTML = projects
    .map((project) => {
      const targetWords =
        (project.target_words || []).join(", ") ||
        bi("暂无词表记录", "No words recorded");
      const selectedClass =
        project.project_id === state.activeProjectId ? " active" : "";
      const statusChip = project.has_final_video
        ? `<span class="mini-chip success">${bi("成片已就绪", "Video Ready")}</span>`
        : project.render_profile === "storybook_only" &&
            project.has_storyboard_review
          ? `<span class="mini-chip success">${bi("绘本已就绪", "Storyboard Ready")}</span>`
          : `<span class="mini-chip accent">${bi("继续制作中", "In Progress")}</span>`;
      return `
        <article class="project-card${selectedClass}" data-project-id="${project.project_id}">
          <div class="project-card-head">
            <h3>${project.project_id}</h3>
            ${project.project_id === state.activeProjectId ? `<span class="mini-chip accent">${bi("当前打开", "Open Now")}</span>` : ""}
          </div>
          <p>${targetWords}</p>
          <div class="project-card-meta">
            ${statusChip}
            <span class="mini-chip">${project.scene_count || 0} scenes</span>
            <span class="mini-chip">${bi("更新于", "Updated")} ${formatDateTime(project.updated_at)}</span>
          </div>
        </article>
      `;
    })
    .join("");
  elements.recentProjects
    .querySelectorAll("[data-project-id]")
    .forEach((card) => {
      card.addEventListener("click", () => {
        const projectId = card.getAttribute("data-project-id");
        loadJob(projectId, { shouldStartPolling: false }).catch((error) => {
          showSystemMessage(error.message, "danger");
        });
      });
    });
}

function renderSenseCards(specs) {
  state.suggestedSpecs = specs;
  if (!specs.length) {
    showEmpty(
      elements.senseContainer,
      bi("当前没有可确认的义项。", "No sense suggestions available."),
    );
    return;
  }
  elements.senseContainer.className = "stack-list";
  elements.senseContainer.innerHTML = "";
  specs.forEach((spec, index) => {
    const fragment = elements.senseTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".sense-card");
    card.dataset.word = spec.word;
    fragment.querySelector(".sense-word").textContent =
      `${spec.word}${spec.selected_sense_label ? ` · ${spec.selected_sense_label}` : ""}`;
    fragment.querySelector(".sense-gloss").textContent =
      `${spec.gloss_zh || ""}${spec.gloss_zh && spec.gloss_en ? " | " : ""}${spec.gloss_en || bi("等待义项信息", "Waiting for sense details")}`;
    fragment.querySelector(".sense-meta").innerHTML = `
      ${bi("置信度", "Confidence")}: ${Number(spec.confidence || 0).toFixed(2)}<br />
      ${bi("视觉锚点", "Visual anchors")}: ${formatList(spec.visual_anchors, bi("暂无", "None"))}<br />
      ${bi("避免混淆", "Avoid confusion with")}: ${formatList(spec.negative_anchors, bi("暂无", "None"))}
    `;
    const radioGroup = fragment.querySelector(".radio-group");
    const selectedSenseId = spec.selected_sense_id || spec.recommended_sense_id;
    (spec.candidates || []).forEach((candidate) => {
      const option = document.createElement("label");
      option.className = "radio-option";
      option.innerHTML = `
        <input
          type="radio"
          name="sense-${index}"
          value="${candidate.sense_id}"
          ${candidate.sense_id === selectedSenseId ? "checked" : ""}
        />
        <span>
          <strong>${candidate.label}${candidate.sense_id === spec.recommended_sense_id ? ` · ${bi("推荐", "Recommended")}` : ""}</strong>
          <small>${candidate.gloss_zh || ""}${candidate.gloss_zh && candidate.gloss_en ? " | " : ""}${candidate.gloss_en || bi("暂无释义", "No gloss")} </small>
          <small>${bi("视觉锚点", "Anchors")}: ${formatList(candidate.visual_anchors, bi("无", "none"))}</small>
        </span>
      `;
      radioGroup.appendChild(option);
    });
    elements.senseContainer.appendChild(fragment);
  });
}

function formatRelationLabel(relation) {
  const normalized = String(relation || "")
    .trim()
    .toLowerCase();
  if (normalized === "seed_word") {
    return bi("原始词", "Seed");
  }
  if (normalized === "derivative") {
    return bi("词形变化", "Derivative");
  }
  if (normalized === "associated") {
    return bi("强相关词", "Associated");
  }
  if (normalized === "contrast" || normalized === "antonym_like") {
    return bi("对比词", "Contrast");
  }
  return relation || bi("相关词", "Related");
}

function renderRelatedWordFamily(family) {
  if (!family || !family.seed_word) {
    showEmpty(
      elements.relatedWordFamily,
      bi(
        "当前词表不会额外扩展词家族，或还没有扩展结果可展示。",
        "The current setup does not expand a word family, or no expansion preview is available yet.",
      ),
    );
    return;
  }
  const relatedWords = Array.isArray(family.related_words)
    ? family.related_words
    : [];
  elements.relatedWordFamily.className = "stack-list related-word-family-list";
  elements.relatedWordFamily.innerHTML = `
    <article class="sense-card related-family-card">
      <div class="sense-header">
        <div>
          <h3>${escapeHtml(family.seed_word)}</h3>
          <p>${bi("系统会围绕这个种子词扩展出相关词，再进入 theme story。", "The system expands this seed word into a related family before entering theme story.")}</p>
        </div>
        <span class="recommend-badge">${bi(`共 ${family.total_words || 1} 词`, `${family.total_words || 1} words`)}</span>
      </div>
      <div class="related-chip-row">
        <span class="chip accent-chip">${escapeHtml(family.seed_word)}</span>
        ${relatedWords.map((item) => `<span class="chip">${escapeHtml(item.word || "")}</span>`).join("")}
      </div>
      <div class="radio-group">
        ${relatedWords
          .map(
            (item) => `
              <article class="radio-option related-word-card">
                <span>
                  <strong>${escapeHtml(item.word || "")} · ${escapeHtml(formatRelationLabel(item.relation_to_source))}</strong>
                  <small>${escapeHtml(item.gloss_zh || "")}${item.gloss_zh && item.gloss_en ? " | " : ""}${escapeHtml(item.gloss_en || bi("暂无释义", "No gloss"))}</small>
                  <small>${bi("视觉锚点", "Anchors")}: ${escapeHtml(formatList(item.visual_anchors, bi("无", "none")))}</small>
                </span>
              </article>
            `,
          )
          .join("")}
      </div>
    </article>
  `;
}

function collectSelectedSpecs() {
  return state.suggestedSpecs.map((spec, index) => {
    const selected = document.querySelector(
      `input[name="sense-${index}"]:checked`,
    );
    return {
      ...spec,
      selected_sense_id: selected
        ? selected.value
        : spec.selected_sense_id || spec.recommended_sense_id,
    };
  });
}

function renderTimeline(stageSummary) {
  const stages = (stageSummary && stageSummary.stages) || [];
  if (!stages.length) {
    showEmpty(elements.timeline, bi("尚无阶段信息。", "No stage data yet."));
    return;
  }
  elements.timeline.className = "timeline-grid";
  elements.timeline.innerHTML = stages
    .map((stage) => {
      const className = `timeline-step ${stage.status}`;
      const label =
        STAGE_LABELS[stage.stage] || stage.stage.replaceAll("_", " ");
      const message =
        stage.message ||
        (stage.status === "pending"
          ? bi("等待中", "Waiting")
          : stage.status === "current"
            ? bi("进行中", "In Progress")
            : bi("已完成", "Completed"));
      return `
        <article class="${className}">
          <strong>${label}</strong>
          <span>${message}</span>
        </article>
      `;
    })
    .join("");
}

function renderSceneSummary(sceneSummaries) {
  if (!sceneSummaries.length) {
    showEmpty(
      elements.sceneSummary,
      bi(
        "运行后会在这里显示每个 scene 的状态。",
        "Scene status will appear here after the run starts.",
      ),
    );
    return;
  }
  elements.sceneSummary.className = "scene-grid";
  elements.sceneSummary.innerHTML = sceneSummaries
    .map((scene) => {
      const videoState = scene.merged_video_ready
        ? bi("已完成合成", "Merged")
        : scene.raw_video_ready
          ? bi("已生成视频", "Video ready")
          : bi("处理中", "Processing");
      return `
        <article class="scene-pill">
          <strong>Scene ${scene.scene_index} · ${scene.target_word_in_scene}</strong>
          <span>${bi("义项", "Sense")}: ${scene.selected_sense_label || bi("默认", "Default")}</span>
          <span>${bi("视觉评分", "Visual score")}: ${scene.visual_match_level || "-"} / ${formatScore(scene.visual_score)}</span>
          <span>${bi("采用轮次", "Chosen round")}: ${scene.selected_iteration || 0}${scene.selected_via_fallback ? ` · ${bi("回退采用", "Fallback used")}` : ""}</span>
          <span>${bi("媒体状态", "Media status")}: ${videoState}</span>
        </article>
      `;
    })
    .join("");
}

function renderProgressFeed(feed) {
  if (!feed.length) {
    showEmpty(
      elements.eventLog,
      bi("尚无进展播报。", "No progress updates yet."),
    );
    return;
  }
  elements.eventLog.className = "log-list";
  elements.eventLog.innerHTML = [...feed]
    .reverse()
    .map(
      (item) => `
      <article class="event-entry">
        <strong>${STAGE_LABELS[item.stage] || item.stage}</strong>
        <span>${item.timestamp || ""}</span>
        <p>${item.message || ""}</p>
      </article>
    `,
    )
    .join("");
}

function renderMiniChips(items, tone = "") {
  if (!items || !items.length) {
    return "";
  }
  return items
    .map(
      (item) =>
        `<span class="mini-chip${tone ? ` ${tone}` : ""}">${escapeHtml(item)}</span>`,
    )
    .join("");
}

function renderInsightRow(title, items, tone = "") {
  if (!items || !items.length) {
    return "";
  }
  return `
    <div class="insight-row">
      <span class="detail-label">${escapeHtml(title)}</span>
      <div class="chip-flow">${renderMiniChips(items, tone)}</div>
    </div>
  `;
}

function renderCopyBlock(title, text, emptyText = bi("暂无", "Not available")) {
  return `
    <section class="copy-block">
      <span class="detail-label">${escapeHtml(title)}</span>
      <p>${escapeHtml(text || emptyText)}</p>
    </section>
  `;
}

function renderImageFrame(imageUrl, altText, placeholderText) {
  if (imageUrl) {
    return `<img
      class="iteration-image"
      src="${escapeHtml(imageUrl)}"
      alt="${escapeHtml(altText)}"
      data-image-fallback="${escapeHtml(placeholderText)}"
      data-fallback-class="iteration-image placeholder"
    />`;
  }
  return `<div class="iteration-image placeholder">${escapeHtml(placeholderText)}</div>`;
}

function bindImageFallbacks(root) {
  if (!root) {
    return;
  }
  root.querySelectorAll("img[data-image-fallback]").forEach((image) => {
    if (image.dataset.fallbackBound === "true") {
      return;
    }
    image.dataset.fallbackBound = "true";
    image.addEventListener(
      "error",
      () => {
        const fallback = document.createElement("div");
        fallback.className =
          image.getAttribute("data-fallback-class") || "iteration-image placeholder";
        fallback.textContent =
          image.getAttribute("data-image-fallback") ||
          bi("图片暂时不可用。", "Image is not available.");
        fallback.setAttribute("aria-label", image.getAttribute("alt") || "");
        image.replaceWith(fallback);
      },
      { once: true },
    );
  });
}

function renderStoryDraftScenes(round) {
  if (!round.draft_scenes || !round.draft_scenes.length) {
    return `
      <p class="empty-inline">
        ${bi("这一轮还没有可展示的剧本草稿。", "No story draft is available for this round.")}
      </p>
    `;
  }
  return `
    <div class="draft-scene-grid">
      ${round.draft_scenes
        .map(
          (scene) => `
            <article class="draft-scene-card">
              <div class="draft-scene-head">
                <strong>Scene ${scene.scene_index}</strong>
                <span class="mini-chip accent">${escapeHtml(scene.target_word_in_scene || bi("未标注目标词", "No target word"))}</span>
              </div>
              ${renderCopyBlock(
                bi("画面脚本", "Visual beat"),
                scene.plot_description,
                bi(
                  "这一轮没有填写画面描述。",
                  "No visual beat for this round.",
                ),
              )}
              ${renderCopyBlock(
                bi("旁白文案", "Narration"),
                scene.voiceover_and_dialogue,
                bi("这一轮没有填写旁白。", "No narration for this round."),
              )}
              ${
                scene.continuity_items?.length
                  ? `
                    <div class="insight-row">
                      <span class="detail-label">${bi("连续性约束", "Continuity")}</span>
                      <div class="continuity-list">
                        ${scene.continuity_items
                          .map(
                            (item) => `
                              <article class="continuity-item">
                                <strong>${escapeHtml(item.label || bi("未命名元素", "Unnamed item"))}</strong>
                                <span>${escapeHtml(item.description || bi("未写描述", "No description"))}</span>
                                ${
                                  item.carry_state
                                    ? `<span>${bi("状态", "State")}: ${escapeHtml(item.carry_state)}</span>`
                                    : ""
                                }
                              </article>
                            `,
                          )
                          .join("")}
                      </div>
                    </div>
                  `
                  : ""
              }
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderStoryRound(round) {
  const reviewTone = round.accepted
    ? "success"
    : round.passed
      ? "accent"
      : "warning";
  const feedbackUsed =
    round.feedback_used && round.feedback_used !== "none"
      ? round.feedback_used
      : bi(
          "首轮草稿，没有沿用上一轮反馈。",
          "First draft with no prior feedback.",
        );
  return `
    <article class="iteration-card">
      <div class="iteration-card-head">
        <div>
          <strong>${bi(`第 ${round.iteration} 轮剧本`, `Story Round ${round.iteration}`)}</strong>
          <span>${bi("评分", "Score")}: ${formatScore(round.score)}${round.timestamp ? ` · ${formatDateTime(round.timestamp)}` : ""}</span>
        </div>
        <div class="chip-flow">
          <span class="mini-chip ${reviewTone}">${round.accepted ? bi("最终采用", "Accepted") : round.passed ? bi("通过待选", "Passed") : bi("继续修改", "Revise")}</span>
          <span class="mini-chip">${bi("场景数", "Scenes")}: ${round.scene_count || 0}</span>
        </div>
      </div>
      <div class="comparison-grid">
        <div class="comparison-card">
          <span class="detail-label">${bi("本轮结论", "Round Summary")}</span>
          <p>${escapeHtml(round.summary || bi("暂无总结。", "No summary yet."))}</p>
        </div>
        <div class="comparison-card">
          <span class="detail-label">${bi("本轮依据", "Feedback Used")}</span>
          <p>${escapeHtml(feedbackUsed)}</p>
        </div>
      </div>
      ${renderInsightRow(bi("优点", "Strengths"), round.strengths, "success")}
      ${renderInsightRow(bi("下一轮重点", "Next Focus"), round.improvement_focus, "warning")}
      ${
        round.validation_issue
          ? `<div class="comparison-card warning-surface">${renderCopyBlock(
              bi("本地校验提醒", "Validation Note"),
              round.validation_issue,
            )}</div>`
          : ""
      }
      ${renderStoryDraftScenes(round)}
    </article>
  `;
}

function renderVisualRound(round, sceneIndex) {
  const legibilityStatus =
    round.text_legibility_passed === true
      ? bi("文字通过", "Text passed")
      : round.text_legibility_passed === false
        ? bi("文字需修复", "Text needs repair")
        : "";
  return `
    <article class="iteration-card visual-iteration-card${round.approved ? " selected-iteration-card" : ""}">
      <div class="iteration-card-head">
        <div>
          <strong>${bi(`第 ${round.iteration} 轮画面`, `Visual Round ${round.iteration}`)}</strong>
          <span>${bi("评分", "Score")}: ${formatScore(round.score)} | ${bi("匹配度", "Match")}: ${escapeHtml(round.match_level || "-")}</span>
        </div>
        <div class="chip-flow">
          <span class="mini-chip ${round.approved ? "success" : "warning"}">${round.approved ? bi("本轮通过", "Approved") : bi("继续返修", "Needs fix")}</span>
          ${
            round.regeneration_mode
              ? `<span class="mini-chip">${escapeHtml(round.regeneration_mode)}</span>`
              : ""
          }
          ${
            legibilityStatus
              ? `<span class="mini-chip">${escapeHtml(legibilityStatus)}</span>`
              : ""
          }
        </div>
      </div>
      <div class="visual-round-layout">
        <div class="iteration-media-frame">
          ${renderImageFrame(
            round.image_url,
            `Scene ${sceneIndex} round ${round.iteration}`,
            bi("这一轮没有保存图片。", "No image saved for this round."),
          )}
        </div>
        <div class="iteration-copy">
          ${renderCopyBlock(
            bi("为什么改", "Why it changed"),
            round.summary,
            bi("暂无说明。", "No explanation."),
          )}
          ${
            round.text_legibility_reason
              ? renderCopyBlock(
                  bi("文字检查", "Text check"),
                  round.text_legibility_reason,
                )
              : ""
          }
          ${
            round.observed_text
              ? renderCopyBlock(
                  bi("图中识别到的文字", "Observed text"),
                  round.observed_text,
                )
              : ""
          }
        </div>
      </div>
      ${renderInsightRow(bi("主要问题", "Main issues"), round.visual_issues, "warning")}
      ${renderInsightRow(bi("改进建议", "Suggestions"), round.suggestions, "accent")}
      ${renderInsightRow(
        bi("提示词调整方向", "Prompt adjustments"),
        round.prompt_adjustments,
      )}
      ${
        round.repair_instruction
          ? `<div class="comparison-card">${renderCopyBlock(
              bi("修复指令", "Repair instruction"),
              round.repair_instruction,
            )}</div>`
          : ""
      }
      <div class="comparison-grid">
        ${renderCopyBlock(
          bi("调整后画面描述", "Updated visual beat"),
          round.revised_plot_description,
          bi("暂无更新后的画面描述。", "No updated visual beat."),
        )}
        ${renderCopyBlock(
          bi("调整后旁白", "Updated narration"),
          round.revised_voiceover_and_dialogue,
          bi("暂无更新后的旁白。", "No updated narration."),
        )}
      </div>
    </article>
  `;
}

function renderSceneReferencePreview(sceneIndex, sceneMeta, placeholderText) {
  return `
    <div class="global-scene-preview">
      ${renderImageFrame(
        sceneMeta?.imageUrl,
        `Scene ${sceneIndex} reference image`,
        placeholderText,
      )}
    </div>
  `;
}

function collectFocusedSceneIndexes(round) {
  const focused = new Set();
  (round.problem_scenes || []).forEach((sceneIndex) => focused.add(sceneIndex));
  (round.scene_feedback || []).forEach((item) => focused.add(item.scene_index));
  (round.scene_script_feedback || []).forEach((item) =>
    focused.add(item.scene_index),
  );
  return [...focused].sort((a, b) => a - b);
}

function buildGlobalFocusReasonMap(round) {
  const reasons = new Map();
  const appendReason = (sceneIndex, reason) => {
    const existing = reasons.get(sceneIndex) || [];
    if (!existing.includes(reason)) {
      existing.push(reason);
    }
    reasons.set(sceneIndex, existing);
  };
  (round.problem_scenes || []).forEach((sceneIndex) =>
    appendReason(sceneIndex, bi("全局评分标记", "Flagged by review")),
  );
  (round.scene_feedback || []).forEach((item) =>
    appendReason(item.scene_index, bi("画面问题", "Visual issue")),
  );
  (round.scene_script_feedback || []).forEach((item) =>
    appendReason(item.scene_index, bi("文案修订", "Script revision")),
  );
  return reasons;
}

function buildSceneFeedbackMap(items) {
  return Object.fromEntries((items || []).map((item) => [item.scene_index, item]));
}

function renderGlobalSceneOverviewCard(sceneMeta, reasonLabels = [], visualFeedback = null) {
  if (!sceneMeta) {
    return "";
  }
  return `
    <article class="global-overview-card${reasonLabels.length ? " focused" : ""}">
      <div class="global-overview-media">
        ${renderImageFrame(
          sceneMeta.imageUrl,
          `Scene ${sceneMeta.sceneIndex} overview image`,
          bi("该场景还没有关键帧。", "No keyframe for this scene yet."),
        )}
      </div>
      <div class="global-overview-copy">
        <div class="draft-scene-head">
          <strong>Scene ${sceneMeta.sceneIndex}</strong>
          ${
            sceneMeta.targetWord
              ? `<span class="mini-chip accent">${escapeHtml(sceneMeta.targetWord)}</span>`
              : ""
          }
        </div>
        ${
          reasonLabels.length
            ? `<div class="chip-flow left-align">${renderMiniChips(
                reasonLabels,
                "warning",
              )}</div>`
            : ""
        }
        ${
          visualFeedback
            ? `
              ${renderCopyBlock(
                bi("问题摘要", "Summary"),
                visualFeedback.summary,
                bi("暂无摘要。", "No summary."),
              )}
              ${renderInsightRow(bi("问题", "Issues"), visualFeedback.visual_issues, "warning")}
              ${renderInsightRow(bi("建议", "Suggestions"), visualFeedback.suggestions, "accent")}
              ${renderInsightRow(
                bi("提示词方向", "Prompt adjustments"),
                visualFeedback.prompt_adjustments,
              )}
              ${
                visualFeedback.repair_instruction
                  ? renderCopyBlock(
                      bi("修复指令", "Repair instruction"),
                      visualFeedback.repair_instruction,
                    )
                  : ""
              }
            `
            : ""
        }
      </div>
    </article>
  `;
}

function renderGlobalRound(round, sceneImageLookup) {
  const allScenes = Object.values(sceneImageLookup).sort(
    (left, right) => left.sceneIndex - right.sceneIndex,
  );
  const focusReasonMap = buildGlobalFocusReasonMap(round);
  const sceneFeedbackMap = buildSceneFeedbackMap(round.scene_feedback);
  const focusedSceneIndexes = collectFocusedSceneIndexes(round);
  const focusedScenes = focusedSceneIndexes
    .map((sceneIndex) => sceneImageLookup[sceneIndex])
    .filter(Boolean);
  return `
    <article class="iteration-card">
      <div class="iteration-card-head">
        <div>
          <strong>${bi(`第 ${round.iteration} 轮全局复审`, `Global Round ${round.iteration}`)}</strong>
          <span>${bi("评分", "Score")}: ${formatScore(round.score)}${round.timestamp ? ` · ${formatDateTime(round.timestamp)}` : ""}</span>
        </div>
        <div class="chip-flow">
          <span class="mini-chip ${round.passed ? "success" : "warning"}">${round.passed ? bi("全局通过", "Passed") : bi("还需统一", "Needs alignment")}</span>
          ${
            round.problem_scenes?.length
              ? `<span class="mini-chip warning">${bi("问题场景", "Problem scenes")}: ${round.problem_scenes.join(", ")}</span>`
              : ""
          }
          ${
            round.targeted_scene_indexes?.length
              ? `<span class="mini-chip">${bi("复审范围", "Reviewed scenes")}: ${round.targeted_scene_indexes.join(", ")}</span>`
              : ""
          }
        </div>
      </div>
      <div class="comparison-card">
        <span class="detail-label">${bi("复审结论", "Global Summary")}</span>
        <p>${escapeHtml(round.summary || bi("暂无总结。", "No summary yet."))}</p>
      </div>
      ${
        allScenes.length
          ? `
            <div class="subsection-block">
              <span class="detail-label">${bi("全帧总览", "All Frames Overview")}</span>
              <div class="global-overview-grid">
                ${allScenes
                  .map((sceneMeta) =>
                    renderGlobalSceneOverviewCard(
                      sceneMeta,
                      focusReasonMap.get(sceneMeta.sceneIndex) || [],
                    ),
                  )
                  .join("")}
              </div>
            </div>
          `
          : ""
      }
      ${
        focusedScenes.length
          ? `
            <div class="subsection-block">
              <span class="detail-label">${bi("重点关注帧", "Frames To Inspect")}</span>
              <div class="global-overview-grid focused-grid">
                ${focusedScenes
                  .map((sceneMeta) =>
                    renderGlobalSceneOverviewCard(
                      sceneMeta,
                      focusReasonMap.get(sceneMeta.sceneIndex) || [],
                      sceneFeedbackMap[sceneMeta.sceneIndex] || null,
                    ),
                  )
                  .join("")}
              </div>
            </div>
          `
          : ""
      }
      ${renderInsightRow(bi("整体调整建议", "Style adjustments"), round.style_adjustments, "accent")}
      ${renderInsightRow(bi("阻塞问题", "Blocking issues"), round.blocking_issues, "warning")}
      ${
        round.scene_script_feedback?.length
          ? `
            <div class="subsection-block">
              <span class="detail-label">${bi("逐场景文案修订", "Scene script revisions")}</span>
              <div class="draft-scene-grid">
                ${round.scene_script_feedback
                  .map(
                    (item) => `
                      <article class="draft-scene-card">
                        <div class="draft-scene-head">
                          <strong>Scene ${item.scene_index}</strong>
                        </div>
                        ${renderSceneReferencePreview(
                          item.scene_index,
                          sceneImageLookup[item.scene_index],
                          bi("该场景还没有关键帧。", "No keyframe for this scene yet."),
                        )}
                        ${renderCopyBlock(
                          bi("文案问题", "Script summary"),
                          item.summary,
                          bi("暂无摘要。", "No summary."),
                        )}
                        ${renderInsightRow(bi("问题", "Issues"), item.script_issues, "warning")}
                        ${renderCopyBlock(
                          bi("建议画面描述", "Suggested visual beat"),
                          item.revised_plot_description,
                          bi("暂无建议。", "No suggestion."),
                        )}
                        ${renderCopyBlock(
                          bi("建议旁白", "Suggested narration"),
                          item.revised_voiceover_and_dialogue,
                          bi("暂无建议。", "No suggestion."),
                        )}
                      </article>
                    `,
                  )
                  .join("")}
              </div>
            </div>
          `
          : ""
      }
    </article>
  `;
}

function renderReviewPanels(snapshot) {
  const storyPanel = snapshot.story_review_panel || { rounds: [] };
  const visualPanel = snapshot.visual_review_panel || [];
  const globalPanel = snapshot.global_review_panel || { rounds: [] };
  const hasContent =
    storyPanel.rounds?.length ||
    visualPanel.length ||
    globalPanel.rounds?.length;
  if (!hasContent) {
    showEmpty(
      elements.reviewPanels,
      bi(
        "运行后，这里会生成故事迭代、视觉迭代和全局一致性复盘。",
        "Story, visual, and global review summaries will appear here after the run starts.",
      ),
      "accordion-stack empty-state",
    );
    return;
  }

  const storyRounds = (storyPanel.rounds || [])
    .map((round) => renderStoryRound(round))
    .join("");

  const visualScenes = visualPanel
    .map(
      (scene) => `
      <details
        class="accordion-nested"
        data-visual-scene="${scene.scene_index}"
        ${getVisualScenePanelOpen(scene.scene_index) ? "open" : ""}
      >
        <summary>
          <span>Scene ${scene.scene_index} · ${scene.target_word_in_scene}</span>
          <span>${bi("最终轮次", "Chosen round")}: ${scene.selected_iteration || 0} | ${bi("评分", "Score")}: ${formatScore(scene.visual_score)}</span>
        </summary>
        <div class="nested-content">
          <div class="scene-review-overview">
            <div class="scene-review-copy">
              <span class="detail-label">${bi("对应义项", "Sense")}</span>
              <p>${escapeHtml(scene.selected_sense_label || bi("默认", "Default"))}</p>
              ${
                scene.selected_via_fallback
                  ? `<p>${bi("最终采用的是历史最佳图像回退结果。", "The final image uses the best historical fallback result.")}</p>`
                  : ""
              }
            </div>
            <div class="scene-review-preview">
              ${renderImageFrame(
                scene.final_image_url,
                `Scene ${scene.scene_index} final image`,
                bi("还没有最终关键帧。", "Final keyframe not ready yet."),
              )}
            </div>
          </div>
          <div class="iteration-stack">
            ${(scene.rounds || [])
              .map((round) => renderVisualRound(round, scene.scene_index))
              .join("")}
          </div>
        </div>
      </details>
    `,
    )
    .join("");

  const sceneImageLookup = Object.fromEntries(
    visualPanel.map((scene) => [
      scene.scene_index,
      {
        sceneIndex: scene.scene_index,
        imageUrl: scene.final_image_url,
        targetWord: scene.target_word_in_scene,
      },
    ]),
  );

  const globalRounds = (globalPanel.rounds || [])
    .map((round) => renderGlobalRound(round, sceneImageLookup))
    .join("");

  elements.reviewPanels.className = "accordion-stack";
  elements.reviewPanels.innerHTML = `
    <details
      class="accordion-card"
      data-review-panel="story"
      ${getReviewPanelOpen("story") ? "open" : ""}
    >
      <summary>
        <span>${bi("故事迭代复盘", "Story Iteration Review")}</span>
        <span>${bi("共", "Total")} ${storyPanel.round_count || 0} ${bi("轮", "rounds")}</span>
      </summary>
      <div class="accordion-content">
        ${storyRounds || `<p class="empty-inline">${bi("暂无故事复盘。", "No story review yet.")}</p>`}
      </div>
    </details>
    <details
      class="accordion-card"
      data-review-panel="visual"
      ${getReviewPanelOpen("visual") ? "open" : ""}
    >
      <summary>
        <span>${bi("视觉迭代复盘", "Visual Review")}</span>
        <span>${bi("共", "Total")} ${visualPanel.length || 0} ${bi("个场景", "scenes")}</span>
      </summary>
      <div class="accordion-content">
        ${visualScenes || `<p class="empty-inline">${bi("暂无视觉复盘。", "No visual review yet.")}</p>`}
      </div>
    </details>
    <details
      class="accordion-card"
      data-review-panel="global"
      ${getReviewPanelOpen("global") ? "open" : ""}
    >
      <summary>
        <span>${bi("全局一致性复盘", "Global Consistency Review")}</span>
        <span>${bi("共", "Total")} ${globalPanel.round_count || 0} ${bi("轮", "rounds")}</span>
      </summary>
      <div class="accordion-content">
        ${globalRounds || `<p class="empty-inline">${bi("暂无全局一致性复盘。", "No global review yet.")}</p>`}
      </div>
    </details>
  `;
  bindReviewPanelState();
  bindImageFallbacks(elements.reviewPanels);
}

function renderStorybookReview(cards) {
  state.storybookCards = cards || [];
  if (!cards.length) {
    showEmpty(
      elements.storybookReview,
      bi(
        "关键帧与字幕复习卡会在这里出现。",
        "Keyframe review cards will appear here.",
      ),
      "storybook-grid empty-state",
    );
    elements.storybookPlayerPanel.classList.add("hidden");
    elements.storybookScenePlayer.removeAttribute("src");
    state.activeStorybookScene = null;
    return;
  }
  elements.storybookReview.className = "storybook-grid";
  elements.storybookReview.innerHTML = cards
    .map(
      (card) => `
        <article
          class="storybook-card"
          data-storybook-scene="${card.scene_index}"
          data-has-video="${card.scene_video_url ? "true" : "false"}"
        >
          ${
            card.image_url
              ? `<img
                  class="storybook-image"
                  src="${escapeHtml(card.image_url)}"
                  alt="Scene ${card.scene_index}"
                  data-image-fallback="${escapeHtml(bi("关键帧暂时不可用", "Keyframe unavailable"))}"
                  data-fallback-class="storybook-image placeholder"
                />`
              : `<div class="storybook-image placeholder">${bi("等待关键帧", "Waiting for keyframe")}</div>`
          }
          <div class="storybook-copy">
            <div class="storybook-copy-head">
              <strong>Scene ${card.scene_index} · ${escapeHtml(card.target_word)}</strong>
              ${
                card.scene_video_url
                  ? `<button type="button" class="ghost-button storybook-play-button" data-storybook-play="${card.scene_index}">
                ${bi("播放本段", "Play Clip")}
              </button>`
                  : `<span class="mini-chip">${bi("绘本模式", "Storyboard")}</span>`
              }
            </div>
            <span>${bi("义项", "Sense")}: ${escapeHtml(card.selected_sense_label || bi("默认", "Default"))}</span>
            <div class="storybook-narration">
              <strong>${bi("旁白", "Narration")}</strong>
              <p>${highlightTargetWord(card.spoken_text || bi("旁白尚未生成。", "Narration is not available yet."), card.target_word)}</p>
            </div>
          </div>
        </article>
      `,
    )
    .join("");
  elements.storybookReview
    .querySelectorAll("[data-storybook-play]")
    .forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        playStorybookScene(button.getAttribute("data-storybook-play"), {
          autoplay: true,
        });
      });
    });
  elements.storybookReview
    .querySelectorAll("[data-storybook-scene]")
    .forEach((cardElement) => {
      cardElement.addEventListener("click", () => {
        const sceneIndex = cardElement.getAttribute("data-storybook-scene");
        if (cardElement.getAttribute("data-has-video") === "true") {
          playStorybookScene(sceneIndex, { autoplay: true });
          return;
        }
        jumpToStorybookScene(sceneIndex, { autoplay: false });
      });
    });
  if (state.activeStorybookScene !== null) {
    const stillVisible = cards.some(
      (card) => Number(card.scene_index) === Number(state.activeStorybookScene),
    );
    if (stillVisible) {
      setActiveStorybookCard(state.activeStorybookScene);
    }
  }
  bindImageFallbacks(elements.storybookReview);
}

function renderChoiceQuestion(container, question, index, prefix) {
  const prompt = escapeHtml(question.prompt || "");
  const options = question.options || [];
  const correctAnswer = question.correct_answer || "";
  const explanation = escapeHtml(question.explanation || "");
  const questionCategory = question.question_category || "multiple_choice";
  const errorReasonTag = question.error_reason_tag || "";
  const targetWord = question.target_word || question.related_words?.[0] || "";
  const recommendedScenes =
    question.recommended_scene_indices ||
    (question.scene_index ? [question.scene_index] : []);
  const groupId = `${prefix}-${index}`;
  const questionTitle =
    prefix === "cloze"
      ? `Scene ${question.scene_index || index + 1}`
      : `${bi("练习", "Practice")} ${index + 1}`;
  state.quizAnswerKey[groupId] = {
    correct: correctAnswer,
    explanation: question.explanation || "",
    targetWord,
    questionCategory,
    errorReasonTag,
    recommendedScenes,
  };
  return `
    <article class="quiz-card" data-quiz-card="${groupId}">
      <div class="quiz-card-head">
        <strong>${escapeHtml(questionTitle)}</strong>
        <div class="quiz-card-tags">
          <span class="mini-chip">${escapeHtml(formatQuestionCategory(questionCategory))}</span>
          ${
            errorReasonTag
              ? `<span class="mini-chip warning hidden" data-quiz-error-tag="${groupId}">${escapeHtml(formatErrorReasonTag(errorReasonTag))}</span>`
              : ""
          }
        </div>
      </div>
      <p>${prompt}</p>
      <div class="quiz-options">
        ${options
          .map(
            (option, optionIndex) => `
            <button
              type="button"
              class="quiz-option-button"
              data-quiz-option="${groupId}"
              data-answer="${escapeHtml(option)}"
            >
              ${String.fromCharCode(65 + optionIndex)}. ${escapeHtml(option)}
            </button>
          `,
          )
          .join("")}
      </div>
      <div class="quiz-feedback hidden" data-quiz-feedback="${groupId}"></div>
    </article>
  `;
}

function bindQuizInteractions(root) {
  root.querySelectorAll("[data-quiz-option]").forEach((button) => {
    button.addEventListener("click", () => {
      const groupId = button.getAttribute("data-quiz-option");
      const answer = button.getAttribute("data-answer");
      const answerKey = state.quizAnswerKey[groupId] || {};
      const correct = answerKey.correct || "";
      const explanation = String(answerKey.explanation || "");
      const targetWord = String(answerKey.targetWord || "");
      const questionCategory = String(
        answerKey.questionCategory || "multiple_choice",
      );
      const recommendedScenes = Array.isArray(answerKey.recommendedScenes)
        ? answerKey.recommendedScenes
        : parseSceneIndices(answerKey.recommendedScenes);
      const errorReasonTag = String(answerKey.errorReasonTag || "");
      root
        .querySelectorAll(`[data-quiz-option="${groupId}"]`)
        .forEach((item) => {
          item.disabled = true;
          const itemAnswer = item.getAttribute("data-answer");
          item.classList.toggle("correct", itemAnswer === correct);
          if (item === button && itemAnswer !== correct) {
            item.classList.add("incorrect");
          }
        });
      const feedback = root.querySelector(`[data-quiz-feedback="${groupId}"]`);
      const errorTag = root.querySelector(`[data-quiz-error-tag="${groupId}"]`);
      if (!feedback) {
        return;
      }
      const passed = answer === correct;
      state.quizResponses[groupId] = {
        passed,
        answer,
        correct,
        explanation,
        targetWord,
        questionCategory,
        errorReasonTag,
        recommendedScenes,
      };
      feedback.className = `quiz-feedback ${passed ? "success-text" : "danger-text"}`;
      if (errorTag) {
        errorTag.classList.toggle("hidden", passed);
      }
      feedback.innerHTML = passed
        ? `${bi("答对了", "Correct")} · ${explanation || bi("你已经抓住了这个词。", "You got the word.")}`
        : `${bi("正确答案", "Correct answer")}: <strong>${escapeHtml(correct)}</strong>${explanation ? ` · ${explanation}` : ""}`;
      renderLearningSummary();
    });
  });
}

function renderLearningSummary() {
  const responses = Object.values(state.quizResponses);
  if (!responses.length) {
    showEmpty(
      elements.learningSummary,
      bi(
        "完成几道题后，这里会生成学习结果摘要。",
        "Answer a few questions to unlock the learning summary.",
      ),
      "summary-grid empty-state",
    );
    return;
  }
  const total = responses.length;
  const correctCount = responses.filter((item) => item.passed).length;
  const wrongResponses = responses.filter((item) => !item.passed);
  const accuracy = Math.round((correctCount / total) * 100);
  const weakWords = new Map();
  const weakScenes = new Set();
  wrongResponses.forEach((item) => {
    const key = item.targetWord || bi("未标记词", "Unlabeled");
    const existing = weakWords.get(key) || {
      mistakes: 0,
      categories: new Set(),
      reasons: new Set(),
      scenes: new Set(),
    };
    existing.mistakes += 1;
    existing.categories.add(item.questionCategory);
    if (item.errorReasonTag) {
      existing.reasons.add(item.errorReasonTag);
    }
    item.recommendedScenes.forEach((sceneIndex) => {
      existing.scenes.add(sceneIndex);
      weakScenes.add(sceneIndex);
    });
    weakWords.set(key, existing);
  });
  const weakWordRows = [...weakWords.entries()]
    .map(([word, data]) => {
      const categories = [...data.categories]
        .map(formatQuestionCategory)
        .join(" / ");
      const reasons = [...data.reasons].map(formatErrorReasonTag).join(" / ");
      const scenes = [...data.scenes]
        .sort((a, b) => a - b)
        .map(
          (sceneIndex) => `
            <button type="button" class="summary-scene-link" data-summary-scene="${sceneIndex}">
              Scene ${sceneIndex}
            </button>
          `,
        )
        .join(" / ");
      return `
        <article class="summary-card">
          <strong>${escapeHtml(word)}</strong>
          <span>${bi("错题数", "Misses")}: ${data.mistakes}</span>
          <span>${bi("题型", "Question types")}: ${escapeHtml(categories || "-")}</span>
          <span>${bi("主要原因", "Main reason")}: ${escapeHtml(reasons || bi("需要复习", "Needs review"))}</span>
          <div class="summary-scene-links"><span>${bi("建议重看", "Revisit")}:</span> ${scenes || escapeHtml("-")}</div>
        </article>
      `;
    })
    .join("");
  const overviewTone =
    accuracy >= 85
      ? bi("记忆保持得很好", "Strong retention")
      : accuracy >= 60
        ? bi(
            "整体不错，再复习几帧会更稳",
            "Good overall, but a few scenes need another look",
          )
        : bi(
            "建议先回看重点场景，再做一轮练习",
            "Rewatch the key scenes before another round",
          );
  elements.learningSummary.className = "summary-grid";
  elements.learningSummary.innerHTML = `
    <article class="summary-card strong">
      <strong>${bi("答题表现", "Score Summary")}</strong>
      <span>${bi("已作答", "Answered")}: ${total}</span>
      <span>${bi("答对", "Correct")}: ${correctCount}</span>
      <span>${bi("正确率", "Accuracy")}: ${accuracy}%</span>
      <p>${overviewTone}</p>
    </article>
    <article class="summary-card">
      <strong>${bi("建议回看场景", "Scenes To Rewatch")}</strong>
      <div class="summary-scene-links">${
        weakScenes.size
          ? [...weakScenes]
              .sort((a, b) => a - b)
              .map(
                (sceneIndex) => `
                  <button type="button" class="summary-scene-link" data-summary-scene="${sceneIndex}">
                    Scene ${sceneIndex}
                  </button>
                `,
              )
              .join(" / ")
          : bi("暂时没有明显薄弱场景", "No weak scenes detected yet")
      }</div>
      <p>${
        weakScenes.size
          ? bi(
              "先重看这些关键帧和旁白，再回来做一次题。",
              "Rewatch these keyframes and lines, then try another round.",
            )
          : bi(
              "目前所有已作答题都表现稳定。",
              "Your answered questions look stable so far.",
            )
      }</p>
    </article>
    ${
      weakWordRows ||
      `<article class="summary-card"><strong>${bi("掌握情况", "Mastery")}</strong><p>${bi("目前没有需要额外提示的单词。", "No extra review words yet.")}</p></article>`
    }
  `;
  elements.learningSummary
    .querySelectorAll("[data-summary-scene]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        const sceneIndex = button.getAttribute("data-summary-scene");
        jumpToStorybookScene(sceneIndex, { autoplay: true });
      });
    });
}

function renderClozeChallenge(snapshot) {
  const finalClozeVideoUrl = snapshot.final_cloze_video_url;
  const exercises = snapshot.learning_exercises || {};
  const clozeChallenges = exercises.cloze_challenges || [];
  if (finalClozeVideoUrl) {
    elements.clozeVideoPreview.classList.remove("hidden");
    if (
      elements.clozeVideoPreview.src !==
      window.location.origin + finalClozeVideoUrl
    ) {
      elements.clozeVideoPreview.src = finalClozeVideoUrl;
    }
  } else {
    elements.clozeVideoPreview.classList.add("hidden");
    elements.clozeVideoPreview.removeAttribute("src");
  }
  if (!clozeChallenges.length) {
    showEmpty(
      elements.clozeChallenge,
      bi(
        "挖空视频和互动选择题会在这里出现。",
        "The cloze video and interactive choices will appear here.",
      ),
      "quiz-grid empty-state",
    );
    return;
  }
  elements.clozeChallenge.className = "quiz-grid";
  elements.clozeChallenge.innerHTML = clozeChallenges
    .map((question, index) =>
      renderChoiceQuestion(elements.clozeChallenge, question, index, "cloze"),
    )
    .join("");
  bindQuizInteractions(elements.clozeChallenge);
}

function renderPracticeQuiz(snapshot) {
  const exercises = snapshot.learning_exercises || {};
  const practiceQuestions = exercises.practice_questions || [];
  if (!practiceQuestions.length) {
    showEmpty(
      elements.practiceQuiz,
      bi(
        "教学练习题会在这里出现。",
        "Post-video practice questions will appear here.",
      ),
      "quiz-grid empty-state",
    );
    return;
  }
  elements.practiceQuiz.className = "quiz-grid";
  elements.practiceQuiz.innerHTML = practiceQuestions
    .map((question, index) =>
      renderChoiceQuestion(elements.practiceQuiz, question, index, "practice"),
    )
    .join("");
  bindQuizInteractions(elements.practiceQuiz);
}

function renderResult(job, snapshot) {
  const projectChanged =
    state.renderedResultProjectId &&
    state.renderedResultProjectId !== job.project_id;
  if (projectChanged) {
    state.quizResponses = {};
    state.quizAnswerKey = {};
    state.storybookCards = [];
    state.activeStorybookScene = null;
    elements.storybookPlayerPanel.classList.add("hidden");
    elements.storybookScenePlayer.removeAttribute("src");
  }
  state.renderedResultProjectId = job.project_id;
  state.quizAnswerKey = {};
  const finalVideoReady = Boolean(snapshot.final_video_url);
  const storyboardOnly = snapshot.render_profile === "storybook_only";
  const canPromoteFromSnapshot = Boolean(snapshot.can_promote_storyboard);
  const jobActive = job.status === "running" || job.status === "queued";
  const canPromoteStoryboard =
    !jobActive &&
    !finalVideoReady &&
    Boolean(job.project_id) &&
    (storyboardOnly || canPromoteFromSnapshot);
  if (job.status === "failed") {
    elements.resultSummary.className = "result-summary danger-text";
    const failureDetail = job.error
      ? `${bi("错误信息", "Error")}: ${job.error}`
      : bi("请调整设置后重试。", "Please adjust settings and try again.");
    elements.resultSummary.textContent = `${bi("生成没有完成。", "The run did not complete.")} ${failureDetail}`;
  } else if (finalVideoReady) {
    elements.resultSummary.className = "result-summary";
    elements.resultSummary.innerHTML = `${bi("项目", "Project")} <strong>${job.project_id}</strong> ${bi("已生成完成，可以直接预览或下载最终视频。", "is finished. You can preview or download the final video now.")}`;
  } else if (canPromoteStoryboard) {
    elements.resultSummary.className = "result-summary success-text";
    elements.resultSummary.textContent = bi(
      "绘本测试模式已完成：你现在可以直接查看关键帧、每个 scene 的旁白文本和练习题；如果满意，可以继续把这套绘本补生成视频。",
      "Storyboard-only mode finished. You can now review keyframes, per-scene narration text, and practice questions, then promote this storyboard into a full video if you like it.",
    );
  } else if (jobActive) {
    elements.resultSummary.className = "result-summary warning-text";
    elements.resultSummary.textContent = bi(
      "任务正在运行中，完成后这里会自动切换为视频预览。",
      "The run is in progress. This area will switch to the final preview automatically.",
    );
  } else {
    elements.resultSummary.className = "result-summary empty-state";
    elements.resultSummary.textContent = bi(
      "生成完成后，这里会展示最终视频预览和下载入口。",
      "The final video preview and download link will appear here after completion.",
    );
  }
  elements.promoteStoryboardButton.classList.toggle(
    "hidden",
    !canPromoteStoryboard,
  );
  elements.promoteStoryboardButton.disabled = !canPromoteStoryboard;

  if (finalVideoReady) {
    elements.videoPreview.classList.remove("hidden");
    if (
      elements.videoPreview.src !==
      window.location.origin + snapshot.final_video_url
    ) {
      elements.videoPreview.src = snapshot.final_video_url;
    }
  } else {
    elements.videoPreview.classList.add("hidden");
    elements.videoPreview.removeAttribute("src");
  }
  renderLearningSummary();
  renderClozeChallenge(snapshot);
  renderStorybookReview(snapshot.storybook_review || []);
  renderPracticeQuiz(snapshot);
}

function formatJobStatus(status) {
  if (status === "succeeded") return bi("已完成", "Completed");
  if (status === "failed") return bi("未完成", "Failed");
  if (status === "running") return bi("进行中", "Running");
  if (status === "queued") return bi("排队中", "Queued");
  return bi("尚未开始", "Not Started");
}

function renderSnapshot(job, snapshot) {
  const previousProjectId = state.activeProjectId;
  if (job.project_id && job.project_id !== previousProjectId) {
    applyProjectSnapshotToForm(job.project_id, snapshot);
  }
  state.activeProjectId = job.project_id;
  syncActionState();
  renderRecentProjects();
  elements.jobBadge.textContent = formatJobStatus(job.status);
  elements.jobBadge.className = `section-tag ${job.status === "failed" ? "danger-text" : job.status === "succeeded" ? "success-text" : "muted-tag"}`;
  elements.activeProject.textContent = job.project_id || "-";
  elements.activeStage.textContent =
    STAGE_LABELS[snapshot.stage_summary?.current_stage] || "-";
  elements.activeSceneCount.textContent = String(
    (snapshot.scene_summaries || []).length || "-",
  );
  renderTimeline(snapshot.stage_summary);
  renderSceneSummary(snapshot.scene_summaries || []);
  renderProgressFeed(snapshot.progress_feed || []);
  renderReviewPanels(snapshot);
  renderRelatedWordFamily(snapshot.related_word_family || null);
  renderResult(job, snapshot);
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    throw new Error(
      bi(
        "无法连接后端服务，请确认 Web 控制台后端正在运行。",
        "Cannot reach the backend service. Make sure the web console backend is running.",
      ),
    );
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || bi("请求失败", "Request failed"));
  }
  return payload;
}

async function loadOverview() {
  const data = await requestJson("/api/overview");
  renderOverview(data);
}

async function analyzeSenses() {
  const words = normalizeWords(elements.wordsText.value);
  if (!words.length) {
    showSystemMessage(
      bi("请先输入至少一个目标词。", "Please enter at least one target word."),
      "warning",
    );
    return;
  }
  setBusy(
    elements.analyzeButton,
    true,
    bi("分析中...", "Analyzing..."),
    bi("检查词义方案 / Review Senses", "Review Senses"),
  );
  try {
    const data = await requestJson("/api/senses", {
      method: "POST",
      body: JSON.stringify({
        project_id: elements.projectId.value.trim(),
        words_text: elements.wordsText.value,
        learning_mode: getLearningMode(),
        max_scenes: readNumberInput(elements.maxScenes),
        media_workers: readNumberInput(elements.mediaWorkers),
        storyboard_only: elements.storyboardOnly.checked,
        story_score_threshold: readNumberInput(elements.storyScoreThreshold),
        global_visual_score_threshold: readNumberInput(
          elements.globalVisualScoreThreshold,
        ),
        test_mode: elements.testMode.checked,
        auto_accept_senses: elements.autoAccept.checked,
      }),
    });
    if (data.project_id) {
      elements.projectId.value = data.project_id;
    }
    state.analyzedWordsSignature = (data.words || []).join("|");
    renderSenseCards(data.target_word_specs || []);
    renderRelatedWordFamily(data.related_word_family || null);
    const resolvedPlan = data.learning_plan || {};
    const resolvedMode = resolvedPlan.mode;
    const relatedWordFamily = data.related_word_family || null;
    const modeMessage =
      getLearningMode() === "auto" && resolvedMode
        ? bi(
            `Planner 已选择 ${resolvedMode}，预计 ${resolvedPlan.recommended_scene_count || "-"} 个 scene。`,
            `The planner selected ${resolvedMode} with about ${resolvedPlan.recommended_scene_count || "-"} scenes.`,
          )
        : "";
    const familyMessage =
      relatedWordFamily && relatedWordFamily.seed_word
        ? bi(
            `已为 ${relatedWordFamily.seed_word} 预览 ${relatedWordFamily.total_words || 1} 词词汇家族。`,
            `Previewed a ${relatedWordFamily.total_words || 1}-word family for ${relatedWordFamily.seed_word}.`,
          )
        : "";
    showSystemMessage(
      `${bi(
        "词义建议已更新，你可以直接开始生成，也可以先切换候选义项。",
        "Sense suggestions are ready. You can start the run now or adjust the selected senses first.",
      )}${modeMessage ? ` ${modeMessage}` : ""}${familyMessage ? ` ${familyMessage}` : ""}`,
      "success",
    );
    loadOverview().catch(() => {});
    scrollToElement(elements.sensePanel);
  } catch (error) {
    showSystemMessage(error.message, "danger");
    scrollToElement(elements.systemMessage);
  } finally {
    setBusy(
      elements.analyzeButton,
      false,
      bi("分析中...", "Analyzing..."),
      "检查词义方案 / Review Senses",
    );
  }
}

function startPolling(projectId) {
  stopPolling();
  state.pollTimer = window.setInterval(() => {
    loadJob(projectId).catch((error) => {
      stopPolling();
      showSystemMessage(error.message, "danger");
    });
  }, 3000);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function loadJob(projectId, { shouldStartPolling = false } = {}) {
  if (!projectId) {
    return;
  }
  const data = await requestJson(`/api/jobs/${projectId}`);
  renderSnapshot(data.job, data.snapshot);
  if (
    shouldStartPolling ||
    data.job.status === "running" ||
    data.job.status === "queued"
  ) {
    startPolling(projectId);
  } else {
    stopPolling();
  }
}

async function startGeneration() {
  const words = normalizeWords(elements.wordsText.value);
  if (!words.length) {
    showSystemMessage(
      bi("请先输入至少一个目标词。", "Please enter at least one target word."),
      "warning",
    );
    return;
  }
  const currentSignature = words.join("|");
  const shouldAttachSpecs =
    state.suggestedSpecs.length &&
    currentSignature === state.analyzedWordsSignature;
  setBusy(
    elements.runButton,
    true,
    bi("提交中...", "Submitting..."),
    "生成学习短片 / Create Learning Video",
  );
  try {
    const data = await requestJson("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        project_id: elements.projectId.value.trim(),
        words_text: elements.wordsText.value,
        target_word_specs: shouldAttachSpecs ? collectSelectedSpecs() : [],
        learning_mode: getLearningMode(),
        max_scenes: readNumberInput(elements.maxScenes),
        media_workers: readNumberInput(elements.mediaWorkers),
        storyboard_only: elements.storyboardOnly.checked,
        story_score_threshold: readNumberInput(elements.storyScoreThreshold),
        global_visual_score_threshold: readNumberInput(
          elements.globalVisualScoreThreshold,
        ),
        test_mode: elements.testMode.checked,
        auto_accept_senses: elements.autoAccept.checked,
      }),
    });
    elements.projectId.value = data.job.project_id;
    renderSnapshot(data.job, data.snapshot);
    startPolling(data.job.project_id);
    showSystemMessage(
      bi(
        "任务已提交，前端会每 3 秒自动同步一次后端状态。",
        "The job was submitted. The frontend now syncs backend status every 3 seconds.",
      ),
      "success",
    );
    scrollToElement(elements.progressPanel);
    loadOverview();
  } catch (error) {
    showSystemMessage(error.message, "danger");
    scrollToElement(elements.systemMessage);
  } finally {
    setBusy(
      elements.runButton,
      false,
      bi("提交中...", "Submitting..."),
      "生成学习短片 / Create Learning Video",
    );
  }
}

async function promoteStoryboardToVideo() {
  const projectId = state.activeProjectId || elements.projectId.value.trim();
  if (!projectId) {
    showSystemMessage(
      bi(
        "请先选择一个已完成的绘本项目。",
        "Select a completed storyboard project first.",
      ),
      "warning",
    );
    return;
  }
  setBusy(
    elements.promoteStoryboardButton,
    true,
    bi("转视频中...", "Rendering Video..."),
    bi(
      "把绘本变成视频 / Turn Storyboard Into Video",
      "Turn Storyboard Into Video",
    ),
  );
  try {
    const data = await requestJson(
      `/api/projects/${encodeURIComponent(projectId)}/render-video`,
      {
        method: "POST",
        body: JSON.stringify({
          project_id: projectId,
          media_workers: readNumberInput(elements.mediaWorkers),
        }),
      },
    );
    renderSnapshot(data.job, data.snapshot);
    startPolling(data.job.project_id);
    showSystemMessage(
      bi(
        "已开始基于当前绘本补生成视频，不会重跑故事和关键帧。",
        "Started turning the current storyboard into a video without re-running the story or keyframes.",
      ),
      "success",
    );
    scrollToElement(elements.progressPanel);
    loadOverview();
  } catch (error) {
    showSystemMessage(error.message, "danger");
    scrollToElement(elements.systemMessage);
  } finally {
    setBusy(
      elements.promoteStoryboardButton,
      false,
      bi("转视频中...", "Rendering Video..."),
      bi(
        "把绘本变成视频 / Turn Storyboard Into Video",
        "Turn Storyboard Into Video",
      ),
    );
  }
}

function bindEvents() {
  elements.wordsText.addEventListener("input", renderWordPreview);
  elements.projectId.addEventListener("input", () => {
    state.projectIdAutoManaged = !elements.projectId.value.trim();
  });
  elements.learningMode.addEventListener("change", updateLearningModeHint);
  elements.analyzeButton.addEventListener("click", analyzeSenses);
  elements.runButton.addEventListener("click", startGeneration);
  elements.promoteStoryboardButton.addEventListener(
    "click",
    promoteStoryboardToVideo,
  );
  elements.refreshButton.addEventListener("click", () => {
    if (!state.activeProjectId) {
      showSystemMessage(
        bi(
          "请先启动一个项目，或从右侧最近项目中选择一个项目。",
          "Start a project first, or choose one from recent projects.",
        ),
        "warning",
      );
      scrollToElement(elements.progressPanel);
      return;
    }
    loadJob(state.activeProjectId)
      .then(() => {
        showSystemMessage(
          bi("已手动刷新当前项目状态。", "Project status refreshed."),
          "info",
        );
      })
      .catch((error) => {
        showSystemMessage(error.message, "danger");
      });
  });
}

async function bootstrap() {
  syncProjectIdInput({ force: true });
  updateLearningModeHint();
  renderWordPreview();
  renderSensePlaceholder();
  syncActionState();
  bindEvents();
  await loadOverview();
  renderTimeline({ stages: [] });
}

bootstrap().catch((error) => {
  elements.envStatus.textContent = bi(
    "界面初始化失败",
    "UI failed to initialize",
  );
  elements.envStatus.className = "danger-text";
  elements.envHint.textContent = error.message;
});
