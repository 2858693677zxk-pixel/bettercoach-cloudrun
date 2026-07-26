const fs = require("node:fs");
const path = require("node:path");

const UNCERTAIN_ACTION_SCORE_CAP = 75;
const LOW_WINDOW_SCORE_CAP = 80;
const COUNT_UNSTABLE_SCORE_CAP = 85;
const LOWER_BODY_FAMILIES = new Set(["squat", "hinge", "isolation_knee"]);
const UPPER_BODY_FAMILIES = new Set(["press", "pull", "isolation_shoulder", "isolation_elbow"]);

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampScore(value, maximum) {
  const score = number(value);
  return Math.max(0, Math.min(score, maximum));
}

function familyGroup(family) {
  if (LOWER_BODY_FAMILIES.has(family)) return "lower_body";
  if (UPPER_BODY_FAMILIES.has(family)) return "upper_body";
  return family ? "general" : "";
}

function cleanText(value, maximum = 500) {
  return String(value || "").trim().slice(0, maximum);
}

function calculationLog(stage, title, summary, details = {}, status = "done") {
  return {
    stage,
    status,
    title,
    summary,
    details
  };
}

function appendCalculationLog(analysis, entry) {
  return [
    ...(Array.isArray(analysis.calculationLogs) ? analysis.calculationLogs : []),
    entry
  ].slice(-80);
}

function imageMimeType(filename) {
  const extension = path.extname(String(filename || "")).toLowerCase();
  if (extension === ".png") return "image/png";
  if (extension === ".webp") return "image/webp";
  return "image/jpeg";
}

function summarizeArbitrationImages(images) {
  return images.map((item) => ({
    label: item.label,
    filename: item.filename,
    timeMs: item.timeMs,
    approxBytes: item.dataUrl
      ? Math.round(Math.max(0, item.dataUrl.length - item.dataUrl.indexOf(",") - 1) * 0.75)
      : 0
  }));
}

function readImageDataUrl(filePath) {
  if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
  const bytes = fs.readFileSync(filePath);
  if (!bytes.length || bytes.length > 2.5 * 1024 * 1024) return null;
  return `data:${imageMimeType(filePath)};base64,${bytes.toString("base64")}`;
}

function issueCodes(analysis) {
  return new Set((analysis.issues || []).map((item) => item.code).filter(Boolean));
}

function ensureIssue(analysis, issue) {
  const codes = issueCodes(analysis);
  if (codes.has(issue.code)) return false;
  analysis.issues = [...(analysis.issues || []), issue];
  return true;
}

function issueByCode(analysis, code) {
  return (analysis.issues || []).find((item) => item.code === code) || null;
}

function detectArbitrationReasons(analysis) {
  const reasons = [];
  const diagnostics = analysis.diagnostics || {};
  const codes = issueCodes(analysis);
  const activeWindow = diagnostics.activeTrainingWindow || {};
  const repSegmentation = diagnostics.repSegmentation || {};
  const movementSignature = diagnostics.movementSignature || {};
  const movementMatch = analysis.measurements?.movementMatch || {};
  const selectedFamily = diagnostics.selectedFamily || analysis.family || "";
  const detectedFamily = diagnostics.detectedFamily || "";
  const selectedGroup = movementMatch.expectedGroup || familyGroup(selectedFamily);
  const detectedGroup = movementMatch.detectedGroup || diagnostics.detectedGroup || familyGroup(detectedFamily);
  const signatureConfidence = number(movementSignature.confidence, 0);
  const fullPoseFrames = number(analysis.metadata?.fullPoseFrames, 0);
  const analysisPoseFrames = number(analysis.metadata?.analysisPoseFrames, fullPoseFrames);
  const activeRatio = fullPoseFrames > 0 ? analysisPoseFrames / fullPoseFrames : 1;

  if (analysis.captureQuality === "insufficient" || codes.has("INSUFFICIENT_EVIDENCE")) {
    reasons.push({
      code: "INSUFFICIENT_EVIDENCE",
      severity: "red",
      promptTask: "invalid_or_unusable_video",
      detail: "画面证据不足，不能输出训练计数、评分或进阶建议。"
    });
  }

  if (repSegmentation.countUnstable || codes.has("COUNT_UNSTABLE")) {
    reasons.push({
      code: "COUNT_UNSTABLE",
      severity: "yellow",
      promptTask: "rep_count_review",
      detail: "重复次数分割不稳定，需要限制计数结论的确定性。"
    });
  }

  if (codes.has("TARGET_UNCERTAIN") || number(diagnostics.targetSwitchCount, 0) > 0) {
    reasons.push({
      code: "TARGET_UNCERTAIN",
      severity: "yellow",
      promptTask: "target_selection_review",
      detail: "多人或目标切换导致主训练者锁定不稳定。"
    });
  }

  if (
    analysis.keyframes?.length
    && (
      number(diagnostics.targetLostCount, 0) > 0
      || number(diagnostics.rejectedDistractorCount, 0) > 0
      || number(repSegmentation.rejectedRepCount, 0) > 0
    )
  ) {
    reasons.push({
      code: "STAGE_EVIDENCE_VISUAL_REVIEW",
      severity: "info",
      promptTask: "stage_sequence_visual_confirmation",
      detail: `目标跟踪或 rep 校验曾拒绝可疑片段：targetLost=${number(diagnostics.targetLostCount, 0)}，rejectedDistractor=${number(diagnostics.rejectedDistractorCount, 0)}，rejectedRep=${number(repSegmentation.rejectedRepCount, 0)}；仅让视觉模型复核阶段图是否仍支持当前结论。`
    });
  }

  if (
    selectedFamily
    && detectedFamily
    && selectedFamily !== detectedFamily
    && signatureConfidence < 0.45
  ) {
    reasons.push({
      code: "ACTION_CLASS_UNCERTAIN",
      severity: "yellow",
      promptTask: "action_family_arbitration",
      detail: `用户选择动作族 ${selectedFamily}，算法低置信识别为 ${detectedFamily}。`
    });
  }

  if (
    activeWindow.enabled
    && number(activeWindow.confidence, 1) < 0.22
    && (activeRatio < 0.55 || number(activeWindow.trimmedStartFrames, 0) > 0 || number(activeWindow.trimmedEndFrames, 0) > 0)
  ) {
    reasons.push({
      code: "ACTIVE_WINDOW_UNCERTAIN",
      severity: "yellow",
      promptTask: "active_window_review",
      detail: "有效训练窗口裁剪置信度低，关键帧和计数只应作为低置信结论。"
    });
  }

  return reasons;
}

function detectArbitrationReasonsV2(analysis) {
  const reasons = [];
  const diagnostics = analysis.diagnostics || {};
  const codes = issueCodes(analysis);
  const activeWindow = diagnostics.activeTrainingWindow || {};
  const repSegmentation = diagnostics.repSegmentation || {};
  const movementSignature = diagnostics.movementSignature || {};
  const movementMatch = analysis.measurements?.movementMatch || {};
  const selectedFamily = diagnostics.selectedFamily || analysis.family || "";
  const detectedFamily = diagnostics.detectedFamily || "";
  const selectedGroup = movementMatch.expectedGroup || familyGroup(selectedFamily);
  const detectedGroup = movementMatch.detectedGroup || diagnostics.detectedGroup || familyGroup(detectedFamily);
  const signatureConfidence = number(movementSignature.confidence, 0);
  const fullPoseFrames = number(analysis.metadata?.fullPoseFrames, 0);
  const analysisPoseFrames = number(analysis.metadata?.analysisPoseFrames, fullPoseFrames);
  const activeRatio = fullPoseFrames > 0 ? analysisPoseFrames / fullPoseFrames : 1;

  if (analysis.captureQuality === "insufficient" || codes.has("INSUFFICIENT_EVIDENCE")) {
    reasons.push({
      code: "INSUFFICIENT_EVIDENCE",
      severity: "red",
      promptTask: "invalid_or_unusable_video",
      detail: "画面证据不足，不能输出训练计数、评分或进阶建议。"
    });
  }

  if (repSegmentation.countUnstable || codes.has("COUNT_UNSTABLE")) {
    reasons.push({
      code: "COUNT_UNSTABLE",
      severity: "yellow",
      promptTask: "rep_count_review",
      detail: "重复次数分割不稳定，需要限制计数结论的确定性。"
    });
  }

  if (codes.has("TARGET_UNCERTAIN") || number(diagnostics.targetSwitchCount, 0) > 0) {
    reasons.push({
      code: "TARGET_UNCERTAIN",
      severity: "yellow",
      promptTask: "target_selection_review",
      detail: "多人或目标切换导致主训练者锁定不稳定。"
    });
  }

  if (
    analysis.keyframes?.length
    && (
      number(diagnostics.targetLostCount, 0) > 0
      || number(diagnostics.rejectedDistractorCount, 0) > 0
      || number(repSegmentation.rejectedRepCount, 0) > 0
    )
  ) {
    reasons.push({
      code: "STAGE_EVIDENCE_VISUAL_REVIEW",
      severity: "info",
      promptTask: "stage_sequence_visual_confirmation",
      detail: `目标跟踪或 rep 校验曾拒绝可疑片段：targetLost=${number(diagnostics.targetLostCount, 0)}，rejectedDistractor=${number(diagnostics.rejectedDistractorCount, 0)}，rejectedRep=${number(repSegmentation.rejectedRepCount, 0)}；仅让视觉模型复核阶段图是否仍支持当前结论。`
    });
  }

  if (
    selectedFamily
    && detectedFamily
    && selectedFamily !== detectedFamily
    && signatureConfidence < 0.45
  ) {
    const crossGroupMismatch = Boolean(movementMatch.mismatch)
      || (
        selectedGroup
        && detectedGroup
        && selectedGroup !== detectedGroup
        && selectedGroup !== "general"
        && detectedGroup !== "general"
      );
    if (crossGroupMismatch) {
      reasons.push({
        code: "ACTION_CLASS_UNCERTAIN",
        severity: "yellow",
        promptTask: "action_family_arbitration",
        detail: `用户选择动作族为 ${selectedFamily}，但算法低置信信号更接近 ${detectedFamily}，且身体大类从 ${selectedGroup || "unknown"} 偏向 ${detectedGroup || "unknown"}。`
      });
    } else {
      reasons.push({
        code: "ACTION_SUBTYPE_VISUAL_REVIEW",
        severity: "info",
        promptTask: "same_group_action_visual_confirmation",
        detail: `动作身体大类一致（${selectedGroup || "unknown"}），但关节信号子类从 ${selectedFamily} 偏向 ${detectedFamily}；仅交给视觉模型复核，不直接扣动作类别分。`
      });
    }
  }

  if (
    activeWindow.enabled
    && number(activeWindow.confidence, 1) < 0.22
    && (activeRatio < 0.55 || number(activeWindow.trimmedStartFrames, 0) > 0 || number(activeWindow.trimmedEndFrames, 0) > 0)
  ) {
    reasons.push({
      code: "ACTIVE_WINDOW_UNCERTAIN",
      severity: "yellow",
      promptTask: "active_window_review",
      detail: "有效训练窗口裁剪置信度低，关键帧和计数只应作为低置信结论。"
    });
  }

  return reasons;
}

function compactAnalysisForArbitration(analysis) {
  const diagnostics = analysis.diagnostics || {};
  return {
    actionName: analysis.actionName,
    actionType: analysis.actionType,
    family: analysis.family,
    captureQuality: analysis.captureQuality,
    overallScore: analysis.overallScore,
    repCount: analysis.repCount,
    issues: (analysis.issues || []).map((item) => ({
      code: item.code,
      title: item.title,
      severity: item.severity,
      confidence: item.confidence,
      repIndexes: item.repIndexes,
      stage: item.stage,
      timeRangesMs: item.timeRangesMs
    })),
    diagnostics: {
      selectedFamily: diagnostics.selectedFamily,
      detectedFamily: diagnostics.detectedFamily,
      detectedGroup: diagnostics.detectedGroup,
      movementSignature: diagnostics.movementSignature,
      activeTrainingWindow: diagnostics.activeTrainingWindow,
      repSegmentation: diagnostics.repSegmentation,
      targetSwitchCount: diagnostics.targetSwitchCount,
      targetLockConfidence: diagnostics.targetLockConfidence,
      multiPersonFrames: diagnostics.multiPersonFrames,
      lowConfidenceWindows: diagnostics.lowConfidenceWindows
    },
    metadata: analysis.metadata,
    keyframes: (analysis.keyframes || []).map((item) => ({
      label: item.label,
      timeMs: item.timeMs,
      image: item.image
    })),
    contactSheet: analysis.contactSheet || null
  };
}

function buildArbitrationPrompt({ job, video, analysis, reasons }) {
  return [
    {
      role: "system",
      content: [
        "你是动作分析系统的低置信度仲裁器。",
        "你不能新增动作技术错误，不能直接给动作评分，不能替换算法关键点。",
        "你只能根据结构化诊断判断：是否证据不足、是否需要复核、是否限制分数、是否建议扩大窗口/重拍/使用 ROI。",
        "输出严格 JSON 对象，不要 Markdown 代码块。",
        "JSON 字段：decisions[]。",
        "decisions 每项包含 action, reasonCode, confidence, doNotScoreAbove, requireUserReview, note。",
        "action 只能是 confirm_algorithm, mark_insufficient, mark_uncertain, cap_score, request_rerun。",
        "如果是非训练视频、录屏、关键点全程低置信或证据不足，返回 mark_insufficient，doNotScoreAbove=0，requireUserReview=true。",
        "如果动作族冲突或有效窗口不可靠，返回 mark_uncertain 或 request_rerun，并限制最高分。",
        "如果只能确认算法已经保守处理，返回 confirm_algorithm。"
      ].join("\n")
    },
    {
      role: "user",
      content: JSON.stringify({
        job: {
          mode: job.mode,
          title: job.title,
          trainingDate: job.trainingDate,
          bodyPart: job.bodyPart
        },
        video: {
          id: video.id,
          actionType: video.actionType,
          actionName: video.actionName,
          cameraAngle: video.cameraAngle,
          notes: video.notes
        },
        arbitrationReasons: reasons,
        analysis: compactAnalysisForArbitration(analysis)
      })
    }
  ];
}

function collectArbitrationImages(analysis, artifactDirectory) {
  if (!artifactDirectory) return [];
  const candidates = [];
  if (analysis.contactSheet) {
    candidates.push({
      label: "contact_sheet",
      filename: analysis.contactSheet
    });
  }
  for (const frame of (analysis.keyframes || []).slice(0, 4)) {
    if (frame.image) {
      candidates.push({
        label: frame.label || `keyframe_${candidates.length}`,
        filename: frame.image,
        timeMs: frame.timeMs
      });
    }
  }
  return candidates.map((item) => {
    const resolved = path.resolve(artifactDirectory, item.filename);
    const root = path.resolve(artifactDirectory);
    if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) return null;
    const dataUrl = readImageDataUrl(resolved);
    return dataUrl ? { ...item, dataUrl } : null;
  }).filter(Boolean).slice(0, 5);
}

function buildVisionArbitrationPrompt({ job, video, analysis, reasons, images }) {
  const text = [
    "你是动作分析系统的视觉仲裁器。",
    "请同时看结构化算法结果和图片证据，但不要直接替代算法关键点。",
    "只判断这些问题：是否是有效健身训练画面、主训练者是否明确、用户选择的动作大类是否合理、关键帧/阶段图是否足以支持计数和报告。",
    "输出严格 JSON 对象，不要 Markdown。",
    "JSON 字段：decisions[]。",
    "decisions 每项包含 action, reasonCode, confidence, doNotScoreAbove, requireUserReview, note。",
    "action 只能是 confirm_algorithm, mark_insufficient, mark_uncertain, cap_score, request_rerun。",
    "只有图片明显不是训练动作、或全程无法看到可分析人体动作时，才返回 mark_insufficient。",
    "如果只是动作族冲突、遮挡、窗口裁剪不稳定，返回 mark_uncertain 或 request_rerun，不要直接判为不可分析。",
    "如果 reasonCode 是 ACTION_SUBTYPE_VISUAL_REVIEW，说明算法身体大类一致但关节子类有差异；请重点看图片是否支持用户选择的动作，支持则返回 confirm_algorithm，不要升级为 mark_insufficient。",
    "如果 reasonCode 是 STAGE_EVIDENCE_VISUAL_REVIEW，请只复核四宫格关键帧是否按动作阶段顺序展示；阶段图合理则返回 confirm_algorithm，不要覆盖算法次数、角度和评分。",
    "",
    JSON.stringify({
      job: {
        mode: job.mode,
        title: job.title,
        trainingDate: job.trainingDate,
        bodyPart: job.bodyPart
      },
      video: {
        id: video.id,
        actionType: video.actionType,
        actionName: video.actionName,
        cameraAngle: video.cameraAngle,
        notes: video.notes
      },
      arbitrationReasons: reasons,
      imageLabels: images.map((item) => ({
        label: item.label,
        filename: item.filename,
        timeMs: item.timeMs
      })),
      analysis: compactAnalysisForArbitration(analysis)
    })
  ].join("\n");

  return [{
    role: "user",
    content: [
      { type: "text", text },
      ...images.map((item) => ({
        type: "image_url",
        image_url: { url: item.dataUrl }
      }))
    ]
  }];
}

function normalizeAiDecision(item) {
  const action = cleanText(item?.action, 40);
  if (![
    "confirm_algorithm",
    "mark_insufficient",
    "mark_uncertain",
    "cap_score",
    "request_rerun"
  ].includes(action)) {
    return null;
  }
  const doNotScoreAbove = action !== "confirm_algorithm" && Number.isFinite(Number(item?.doNotScoreAbove))
    ? Math.max(0, Math.min(100, Number(item.doNotScoreAbove)))
    : null;
  return {
    action,
    reasonCode: cleanText(item?.reasonCode, 80),
    confidence: Number.isFinite(Number(item?.confidence))
      ? Math.max(0, Math.min(1, Number(item.confidence)))
      : null,
    doNotScoreAbove,
    requireUserReview: Boolean(item?.requireUserReview),
    note: cleanText(item?.note, 500)
  };
}

function normalizeAiArbitration(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const rawDecisions = Array.isArray(source) ? source : source.decisions;
  const decisions = Array.isArray(rawDecisions)
    ? rawDecisions.map(normalizeAiDecision).filter(Boolean).slice(0, 8)
    : [];
  return { decisions };
}

function markInsufficient(analysis, applied, reason = "证据不足，不能形成有效动作分析。") {
  analysis.captureQuality = "insufficient";
  analysis.repCount = 0;
  analysis.overallScore = 0;
  analysis.safetyLevel = "not_applicable";
  analysis.strengths = [];
  analysis.keyframes = [];
  analysis.contactSheet = null;
  ensureIssue(analysis, {
    code: "INSUFFICIENT_EVIDENCE",
    severity: "red",
    title: "画面证据不足",
    observation: reason,
    correction: "请重新拍摄，确保全身、器械和负重路径清晰入镜。"
  });
  applied.push("mark_insufficient");
}

function markActionClassUncertain(analysis, applied) {
  const diagnostics = analysis.diagnostics || {};
  const selectedFamily = diagnostics.selectedFamily || analysis.family || "selected";
  const detectedFamily = diagnostics.detectedFamily || "unknown";
  ensureIssue(analysis, {
    code: "ACTION_CLASS_UNCERTAIN",
    severity: "yellow",
    title: "动作类别证据不稳定",
    observation: `用户选择的动作族为 ${selectedFamily}，但当前视频低置信信号更接近 ${detectedFamily}，不适合给出高确定性技术结论。`,
    correction: "请确认动作类型是否选择正确；必要时重新拍摄完整工作组，或使用 ROI 锁定主训练者。",
    confidence: number(diagnostics.movementSignature?.confidence, 0)
  });
  analysis.overallScore = clampScore(analysis.overallScore, UNCERTAIN_ACTION_SCORE_CAP);
  if (analysis.safetyLevel === "green") analysis.safetyLevel = "yellow";
  applied.push("mark_action_class_uncertain");
}

function markActiveWindowUncertain(analysis, applied) {
  const activeWindow = analysis.diagnostics?.activeTrainingWindow || {};
  ensureIssue(analysis, {
    code: "ACTIVE_WINDOW_UNCERTAIN",
    severity: "yellow",
    title: "有效训练片段不稳定",
    observation: "系统只能低置信截取部分训练片段，本次计数和阶段图需要复核。",
    correction: "建议从动作开始前 1-2 秒拍到动作结束后 1-2 秒，避免中途走近镜头或遮挡。",
    confidence: number(activeWindow.confidence, 0)
  });
  analysis.overallScore = clampScore(analysis.overallScore, LOW_WINDOW_SCORE_CAP);
  if (analysis.safetyLevel === "green") analysis.safetyLevel = "yellow";
  applied.push("mark_active_window_uncertain");
}

function markCountUnstable(analysis, applied) {
  ensureIssue(analysis, {
    code: "COUNT_UNSTABLE",
    severity: "yellow",
    title: "重复次数证据不稳定",
    observation: "系统检测到部分周期不完整或边界不清，当前次数只能作为估计值。",
    correction: "请用固定机位拍完整一组，并保持每次动作起止位置清晰。"
  });
  analysis.overallScore = clampScore(analysis.overallScore, COUNT_UNSTABLE_SCORE_CAP);
  if (analysis.safetyLevel === "green") analysis.safetyLevel = "yellow";
  applied.push("mark_count_unstable");
}

function applyAiDecision(analysis, decision, applied, reasonCodes) {
  if (decision.action === "mark_insufficient") {
    if (reasonCodes.has("INSUFFICIENT_EVIDENCE") || decision.reasonCode === "INSUFFICIENT_EVIDENCE") {
      markInsufficient(analysis, applied, decision.note || "AI 仲裁认为证据不足。");
    } else {
      analysis.overallScore = clampScore(analysis.overallScore, LOW_WINDOW_SCORE_CAP);
      if (analysis.safetyLevel === "green") analysis.safetyLevel = "yellow";
      applied.push("reject_ai_mark_insufficient");
    }
    return;
  }
  if (decision.action === "mark_uncertain" || decision.action === "request_rerun") {
    if (decision.reasonCode === "ACTION_CLASS_UNCERTAIN") {
      markActionClassUncertain(analysis, applied);
    } else if (decision.reasonCode === "ACTIVE_WINDOW_UNCERTAIN") {
      markActiveWindowUncertain(analysis, applied);
    }
  }
  if (decision.doNotScoreAbove !== null) {
    analysis.overallScore = clampScore(analysis.overallScore, decision.doNotScoreAbove);
    if (decision.doNotScoreAbove < 80 && analysis.safetyLevel === "green") {
      analysis.safetyLevel = "yellow";
    }
    applied.push(`cap_score_${decision.doNotScoreAbove}`);
  }
}

function applyArbitration(analysis, reasons, aiArbitration = null, attemptLogs = []) {
  const result = JSON.parse(JSON.stringify(analysis));
  const applied = [];
  const reasonCodes = new Set(reasons.map((item) => item.code));

  if (reasonCodes.has("INSUFFICIENT_EVIDENCE")) {
    markInsufficient(result, applied);
  }
  if (reasonCodes.has("ACTION_CLASS_UNCERTAIN")) {
    markActionClassUncertain(result, applied);
  }
  if (reasonCodes.has("ACTIVE_WINDOW_UNCERTAIN")) {
    markActiveWindowUncertain(result, applied);
  }
  if (reasonCodes.has("COUNT_UNSTABLE") && !issueByCode(result, "COUNT_UNSTABLE")) {
    markCountUnstable(result, applied);
  }

  for (const decision of aiArbitration?.decisions || []) {
    applyAiDecision(result, decision, applied, reasonCodes);
  }

  const aiLogs = Array.isArray(aiArbitration?.logs) ? aiArbitration.logs : attemptLogs;
  const requireUserReview = Boolean(
    reasons.some((item) => item.severity !== "info")
    || (aiArbitration?.decisions || []).some((item) => item.requireUserReview)
  );

  result.diagnostics = {
    ...(result.diagnostics || {}),
    aiArbitration: {
      needed: reasons.length > 0,
      source: aiArbitration?.source || (aiArbitration ? "ai" : "deterministic"),
      reasons,
      decisions: aiArbitration?.decisions || [],
      attempts: aiLogs.map((item) => item.details || {}),
      applied: [...new Set(applied)],
      requireUserReview
    }
  };
  result.calculationLogs = [
    ...(Array.isArray(result.calculationLogs) ? result.calculationLogs : []),
    ...aiLogs,
    calculationLog(
    "ai",
    "低置信度仲裁规则落地",
    reasons.length
      ? `触发 ${reasons.length} 个仲裁原因，应用 ${[...new Set(applied)].length} 个保护动作。`
      : "没有触发低置信度仲裁，算法结果直接进入报告阶段。",
    {
      source: result.diagnostics.aiArbitration.source,
      reasons,
      decisions: aiArbitration?.decisions || [],
      applied: [...new Set(applied)],
      requireUserReview,
      aiAttempts: aiLogs.map((item) => item.details || {})
    },
    requireUserReview ? "warning" : "done"
    )
  ].slice(-80);

  return result;
}

async function arbitrateAnalysis({ client, visionClient, job, video, analysis, artifactDirectory }) {
  const reasons = detectArbitrationReasonsV2(analysis);
  if (!reasons.length) {
    return {
      ...analysis,
      calculationLogs: appendCalculationLog(analysis, calculationLog(
        "ai",
        "AI 仲裁未触发",
        "算法置信度满足当前阈值，没有调用视觉或文本模型做低置信度复核。",
        {
          reasonCount: 0,
          source: "deterministic"
        }
      )),
      diagnostics: {
        ...(analysis.diagnostics || {}),
        aiArbitration: {
          needed: false,
          source: "deterministic",
          reasons: [],
          decisions: [],
          applied: [],
          requireUserReview: false
        }
      }
    };
  }

  let aiArbitration = null;
  const attemptLogs = [];
  const images = collectArbitrationImages(analysis, artifactDirectory);
  if (visionClient && images.length) {
    const imageDetails = summarizeArbitrationImages(images);
    try {
      const response = await visionClient.complete(buildVisionArbitrationPrompt({
        job,
        video,
        analysis,
        reasons,
        images
      }), {
        json: true,
        maxTokens: 1024,
        temperature: 0
      });
      aiArbitration = normalizeAiArbitration(JSON.parse(response.content));
      aiArbitration.source = "vision";
      attemptLogs.push(calculationLog(
        "ai",
        "GLM-4V 图像仲裁完成",
        `已调用 GLM-4V-Flash 查看 ${images.length} 张证据图，返回 ${aiArbitration.decisions.length} 条仲裁决策。`,
        {
          provider: "zhipu",
          mode: "vision",
          model: response.model || "glm-4v-flash",
          request: {
            reasonCodes: reasons.map((item) => item.code),
            imageCount: images.length,
            images: imageDetails,
            json: true,
            maxTokens: 1024,
            temperature: 0
          },
          response: {
            usage: response.usage || null,
            decisions: aiArbitration.decisions
          }
        }
      ));
      aiArbitration.logs = attemptLogs;
    } catch (error) {
      attemptLogs.push(calculationLog(
        "ai",
        "GLM-4V 图像仲裁失败",
        "视觉模型调用失败或返回内容无法解析，准备回退到文本仲裁。",
        {
          provider: "zhipu",
          mode: "vision",
          request: {
            reasonCodes: reasons.map((item) => item.code),
            imageCount: images.length,
            images: imageDetails,
            json: true,
            maxTokens: 1024,
            temperature: 0
          },
          error: {
            code: error.code || "VISION_ARBITRATION_FAILED",
            statusCode: error.statusCode || null,
            message: cleanText(error.message, 300)
          }
        },
        "warning"
      ));
      aiArbitration = null;
    }
  } else {
    attemptLogs.push(calculationLog(
      "ai",
      "GLM-4V 图像仲裁跳过",
      visionClient ? "没有可发送给视觉模型的关键帧或拼图。" : "服务端未配置可用的视觉模型客户端。",
      {
        provider: "zhipu",
        mode: "vision",
        imageCount: images.length,
        hasVisionClient: Boolean(visionClient)
      },
      "skipped"
    ));
  }

  if (client) {
    if (!aiArbitration) {
      try {
        const response = await client.complete(buildArbitrationPrompt({
          job,
          video,
          analysis,
          reasons
        }), {
          json: true,
          maxTokens: 1200,
          temperature: 0
        });
        aiArbitration = normalizeAiArbitration(JSON.parse(response.content));
        attemptLogs.push(calculationLog(
          "ai",
          "文本 AI 仲裁完成",
          `已使用文本模型复核结构化诊断，返回 ${aiArbitration.decisions.length} 条仲裁决策。`,
          {
            provider: "deepseek",
            mode: "text",
            model: response.model || null,
            request: {
              reasonCodes: reasons.map((item) => item.code),
              json: true,
              maxTokens: 1200,
              temperature: 0
            },
            response: {
              usage: response.usage || null,
              decisions: aiArbitration.decisions
            }
          }
        ));
        aiArbitration.logs = attemptLogs;
      } catch (error) {
        attemptLogs.push(calculationLog(
          "ai",
          "文本 AI 仲裁失败",
          "文本模型调用失败或返回内容无法解析，保留算法确定性保护规则。",
          {
            provider: "deepseek",
            mode: "text",
            request: {
              reasonCodes: reasons.map((item) => item.code),
              json: true,
              maxTokens: 1200,
              temperature: 0
            },
            error: {
              code: error.code || "TEXT_ARBITRATION_FAILED",
              statusCode: error.statusCode || null,
              message: cleanText(error.message, 300)
            }
          },
          "warning"
        ));
        aiArbitration = null;
      }
    }
  } else if (!aiArbitration) {
    attemptLogs.push(calculationLog(
      "ai",
      "文本 AI 仲裁跳过",
      "服务端未配置文本模型客户端，保留算法确定性保护规则。",
      {
        provider: "deepseek",
        mode: "text",
        hasTextClient: false
      },
      "skipped"
    ));
  }

  return applyArbitration(analysis, reasons, aiArbitration, attemptLogs);
}

async function arbitrateAnalyses({ client, visionClient, job, analyses, artifactDirectories = [] }) {
  const reviewed = [];
  for (let index = 0; index < analyses.length; index += 1) {
    reviewed.push(await arbitrateAnalysis({
      client,
      visionClient,
      job,
      video: (job.videos || [])[index] || {},
      analysis: analyses[index],
      artifactDirectory: artifactDirectories[index]
    }));
  }
  return reviewed;
}

module.exports = {
  applyArbitration,
  arbitrateAnalyses,
  arbitrateAnalysis,
  buildArbitrationPrompt,
  buildVisionArbitrationPrompt,
  collectArbitrationImages,
  detectArbitrationReasons: detectArbitrationReasonsV2,
  normalizeAiArbitration
};
