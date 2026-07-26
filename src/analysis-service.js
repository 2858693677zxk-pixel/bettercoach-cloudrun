const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const { collectArbitrationImages } = require("./ai-arbitration-service");
const { getActionPromptProfile } = require("./action-prompts");
const { actionRubric } = require("./action-rubrics");
const { getExercise } = require("./exercise-catalog");
const { generateReports, summarizeVolume } = require("./report-service");
const { repairMojibake } = require("./text-encoding");

function calculationLog(stage, title, summary, details = {}, status = "done") {
  return {
    stage,
    status,
    title,
    summary,
    details,
    at: new Date().toISOString()
  };
}

function appendJobLog(storage, userKey, jobId, entry) {
  const current = storage.getJobByKey(userKey, jobId);
  return storage.updateJobByKey(userKey, jobId, {
    calculationLogs: [
      ...(Array.isArray(current.calculationLogs) ? current.calculationLogs : []),
      entry
    ].slice(-120)
  });
}

function updateJobWithLog(storage, userKey, jobId, patch, entry = null) {
  if (!entry) return storage.updateJobByKey(userKey, jobId, patch);
  const current = storage.getJobByKey(userKey, jobId);
  return storage.updateJobByKey(userKey, jobId, {
    ...patch,
    calculationLogs: [
      ...(Array.isArray(current.calculationLogs) ? current.calculationLogs : []),
      entry
    ].slice(-120)
  });
}

function runProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env || {}) },
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Video analysis timed out. Please shorten the video and retry."));
    }, options.timeoutMs || 20 * 60 * 1000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(stderr.trim() || stdout.trim() || `瑙嗛鍒嗘瀽杩涚▼閫€鍑虹爜 ${code}`));
    });
  });
}

function publicAnalysisSummary(analyses, videos) {
  const sourceVideos = Array.isArray(videos) && videos.length
    ? videos
    : analyses.map((item) => ({ poseEngineMode: item?._engine || "rtmo" }));
  const scores = analyses
    .map((item) => Number(item.overallScore))
    .filter((item) => Number.isFinite(item) && item > 0);
  const applicable = analyses.filter((item) =>
    item && item.captureQuality !== "insufficient" && item.safetyLevel !== "not_applicable"
  );
  return {
    averageScore: scores.length
      ? Math.round(scores.reduce((sum, item) => sum + item, 0) / scores.length)
      : 0,
    safetyLevel: !applicable.length
      ? "not_applicable"
      : analyses.some((item) => item.safetyLevel === "red")
      ? "red"
      : analyses.some((item) => item.safetyLevel === "yellow")
        ? "yellow"
        : "green",
    volume: summarizeVolume(videos),
    videos: sourceVideos.map((video, videoIndex) => {
      const engineAnalyses = analyses.filter((item, analysisIndex) =>
        Number(item?._sourceVideoIndex ?? analysisIndex) === videoIndex
      );
      const primary = engineAnalyses[0] || {};
      const toPublicAnalysis = (analysis) => {
      const publicAnalysis = { ...(analysis || {}) };
      if (publicAnalysis.actionType) {
        publicAnalysis.actionName = getExercise(publicAnalysis.actionType).name;
      }
      delete publicAnalysis._sourceVideoIndex;
      delete publicAnalysis._engine;
      delete publicAnalysis.cameraAdvice;
      publicAnalysis.issues = (publicAnalysis.issues || []).map((issue) => {
        const correction = sanitizeUserFeedbackText(issue?.correction || "", 220);
        return {
          ...issue,
          correction: correction || (issue?.code === "INSUFFICIENT_EVIDENCE"
            ? "当前证据不足，暂不判断动作细节。"
            : "")
        };
      });
      return publicAnalysis;
      };
      const publicPrimary = toPublicAnalysis(primary);
      return {
        ...publicPrimary,
        engineMode: video?.poseEngineMode || (engineAnalyses.length > 1 ? "both" : primary?._engine || "rtmo"),
        engineResults: engineAnalyses.map((item) => ({
          ...toPublicAnalysis(item),
          engine: item._engine || item.metadata?.requestedPoseEngine || item.metadata?.poseBackend || "unknown",
          evidenceScope: "engine_only"
        }))
      };
    })
  };
}

function poseEnginesForVideo(video) {
  if (video?.poseEngineMode === "both") return ["rtmo", "mediapipe"];
  if (video?.poseEngineMode === "mediapipe") return ["mediapipe"];
  return ["rtmo"];
}

function analyzerBackendForEngine(engine) {
  return engine === "mediapipe" ? "mediapipe" : "rtmlib";
}

function prefixArtifactPaths(analysis, engine) {
  const prefixed = { ...(analysis || {}) };
  if (prefixed.contactSheet) prefixed.contactSheet = `${engine}/${prefixed.contactSheet}`;
  prefixed.keyframes = (prefixed.keyframes || []).map((frame) => ({
    ...frame,
    image: frame.image ? `${engine}/${frame.image}` : frame.image
  }));
  prefixed.annotatedVideos = Object.fromEntries(
    Object.entries(prefixed.annotatedVideos || {}).map(([key, value]) => {
      const info = typeof value === "string" ? { filename: value } : { ...value };
      return [key, {
        ...info,
        filename: info.filename ? `${engine}/${info.filename}` : info.filename
      }];
    })
  );
  return prefixed;
}

function buildStandardImpact(analysis, rubric) {
  const issues = Array.isArray(analysis?.issues) ? analysis.issues : [];
  return {
    actionType: rubric.actionType,
    actionName: rubric.name,
    phaseGuide: rubric.phaseGuide,
    standardRole: rubric.standardRole,
    flow: [
      "骨骼模型先找出身体关键点和每次动作的起止时间。",
      "Python 可计算规则根据角度、路径、支撑和持续时间触发问题。",
      "动作专项标准补充动作目的和教练语言，AI 只负责把已有结果讲清楚。"
    ],
    checkedAreas: (rubric.mainMetrics || []).map((label) => ({
      label,
      role: "专项解释依据"
    })),
    triggeredRules: issues.slice(0, 10).map((issue) => ({
      code: issue.code || "",
      title: issue.title || issue.code || "动作问题",
      observation: issue.observation || "",
      correction: issue.correction || "",
      repIndexes: issue.repIndexes || [],
      stage: issue.stage || "",
      timeRangesMs: issue.timeRangesMs || []
    })),
    needsCalibration: true,
    calibrationNote: "需要继续把专项文字标准和 Python 阈值合并成同一份机器可读配置，并用教练标注视频校准每个动作。"
  };
}

function buildFrameFeedback(analysis, rubric) {
  const judgments = Array.isArray(analysis?.frameJudgments) ? analysis.frameJudgments : [];
  const phaseJudgments = Array.isArray(analysis?.phaseJudgments) ? analysis.phaseJudgments : [];
  const reps = Array.isArray(analysis?.repEvents) ? analysis.repEvents : [];
  const issues = Array.isArray(analysis?.issues) ? analysis.issues : [];
  const guide = rubric.phaseGuide || {};
  const timeline = phaseJudgments.length ? phaseJudgments : judgments;
  const nearestStability = (timeMs) => judgments.reduce((nearest, item) => {
    if (!nearest) return item;
    return Math.abs(Number(item.timeMs) - timeMs) < Math.abs(Number(nearest.timeMs) - timeMs)
      ? item
      : nearest;
  }, null);
  return timeline.map((phaseItem) => {
    const timeMs = Number(phaseItem.timeMs) || 0;
    const judgment = nearestStability(timeMs) || {};
    const rep = reps.find((item) => Number(item.startTimeMs) <= timeMs && timeMs <= Number(item.endTimeMs));
    const keyTime = Number(rep?.keyTimeMs || rep?.startTimeMs || 0);
    const phaseKey = phaseItem.phase || (!rep ? "between_reps" : Math.abs(timeMs - keyTime) <= 220
      ? "key"
      : timeMs < keyTime ? "to_key" : "return");
    const issue = issues.find((item) => {
      const inTime = (item.timeRangesMs || []).some(([start, end]) => Number(start) <= timeMs && timeMs <= Number(end));
      const inRep = rep && (item.repIndexes || []).map(Number).includes(Number(rep.repIndex));
      return inTime || (inRep && !(item.timeRangesMs || []).length);
    });
    const phaseLabel = {
      between_reps: "等待下一次动作",
      to_key: guide.toKey || "动作前半程",
      key: guide.key || "动作转折位置",
      return: guide.return || "动作回程"
    }[phaseKey];
    const defaultCue = {
      between_reps: "固定好支撑，再开始下一次",
      to_key: guide.toKey || "平稳完成动作前半程",
      key: "在这里停稳一下，不要用惯性带过去",
      return: guide.return || "控制动作回到起始位"
    }[phaseKey];
    return {
      frameIndex: phaseItem.frameIndex ?? judgment.frameIndex ?? null,
      timeMs,
      repIndex: phaseItem.repIndex ?? rep?.repIndex ?? judgment.repIndex ?? null,
      phase: phaseKey,
      phaseLabel,
      rangeStatus: phaseItem.rangeStatus || rep?.rangeStatus || null,
      phaseBasis: phaseItem.phaseBasis || rep?.phaseRule || "rep_turning_point",
      cue: sanitizeUserFeedbackText(issue?.correction || issue?.observation || defaultCue, 220),
      issueCode: issue?.code || null,
      issueTitle: issue?.title || null,
      stabilityState: judgment.state || "unknown",
      stabilityText: {
        stable: "躯干稳定",
        watch: "躯干开始偏移",
        unstable: "躯干正在明显晃动",
        recovering: "躯干正在回到稳定位置",
        unknown: "当前看不清躯干状态",
        not_evaluated: "这个动作不评价躯干稳定性"
      }[judgment.state] || "当前躯干状态未知"
    };
  });
}

function cleanText(value, maxLength = 300) {
  return repairMojibake(value)
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

function cleanList(value, maxItems = 5, maxLength = 220) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => cleanText(item, maxLength))
    .filter(Boolean)
    .slice(0, maxItems);
}

function finiteNumber(value, fallback = null) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function videoDataUrl(filePath, maxBytes = 24 * 1024 * 1024) {
  if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
  const bytes = fs.readFileSync(filePath);
  if (!bytes.length || bytes.length > maxBytes) return null;
  return `data:video/mp4;base64,${bytes.toString("base64")}`;
}

function compactAlgorithmResult(analysis) {
  const diagnostics = analysis?.diagnostics || {};
  return {
    actionType: analysis?.actionType || "",
    actionName: analysis?.actionName || "",
    family: analysis?.family || "",
    captureQuality: analysis?.captureQuality || "",
    confidence: analysis?.confidence ?? null,
    overallScore: analysis?.overallScore ?? null,
    safetyLevel: analysis?.safetyLevel || "",
    poseCoverage: analysis?.metadata?.poseCoverage ?? null,
    poseBackend: analysis?.metadata?.poseBackend || "",
    poseFusion: diagnostics.poseFusion ? {
      selectedEngine: diagnostics.poseFusion.selectedEngine,
      confidence: diagnostics.poseFusion.confidence,
      reasons: (diagnostics.poseFusion.reasons || []).slice(0, 6)
    } : null,
    measurements: {
      angles: analysis?.measurements?.angles || null,
      ranges: analysis?.measurements?.ranges || null,
      asymmetry: analysis?.measurements?.asymmetry || null,
      repSegmentation: diagnostics.repSegmentation || null
    },
    strengths: cleanList(analysis?.strengths, 6, 180),
    issues: (analysis?.issues || []).slice(0, 10).map((issue) => ({
      code: issue.code || "",
      title: issue.title || "",
      severity: issue.severity || "",
      observation: issue.observation || "",
      correction: issue.correction || "",
      confidence: issue.confidence ?? null,
      repIndexes: issue.repIndexes || []
    })),
    repEvents: (analysis?.repEvents || []).slice(0, 10).map((rep) => ({
      repIndex: rep.repIndex,
      startTimeMs: rep.startTimeMs,
      keyTimeMs: rep.keyTimeMs,
      endTimeMs: rep.endTimeMs,
      quality: rep.quality,
      signalAmplitude: rep.signalAmplitude,
      durationSeconds: rep.durationSeconds,
      counterRule: rep.counterRule,
      sourceEngine: rep.sourceEngine
    }))
  };
}

function issueAdjustment(issue) {
  const title = cleanText(issue?.title || issue?.code || "动作问题", 80);
  const correction = cleanText(issue?.correction || issue?.observation || "", 220);
  if (title && correction) return `${title}：${correction}`;
  return title || correction;
}

const FILMING_GUIDANCE_PATTERN = /拍摄|重拍|补拍|机位|镜头|入镜|取景|相机|摄像|画面角度/i;
const USER_METRIC_PATTERN = /覆盖率|置信度|可见度|识别率/i;

function sanitizeUserFeedbackText(value, maximum = 220) {
  const text = cleanText(value, maximum * 2);
  const sentences = text.match(/[^。！？!?；;]+[。！？!?；;]?/g) || [];
  return sentences
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item) => !FILMING_GUIDANCE_PATTERN.test(item) && !USER_METRIC_PATTERN.test(item))
    .join("")
    .slice(0, maximum);
}

function sanitizeUserFeedbackList(value, maximumItems, maximumLength) {
  const seen = new Set();
  return cleanList(value, maximumItems * 2, maximumLength * 2)
    .map((item) => sanitizeUserFeedbackText(item, maximumLength))
    .filter(Boolean)
    .filter((item) => {
      const key = item.replace(/[\s，。！？!?；;：:、]/g, "").toLowerCase();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, maximumItems);
}

const REVIEW_SECTION_TITLES = Object.freeze({
  movement_task: "动作任务与主要轨迹",
  range_endpoints: "活动范围与关键位置",
  stability_compensation: "支撑、稳定与代偿",
  tempo_consistency: "节奏、控制与重复一致性",
  load_safety: "负荷适配与安全边界"
});

const REVIEW_STATUSES = new Set([
  "meets_standard",
  "mostly_meets",
  "needs_attention",
  "limited_evidence",
  "not_assessed"
]);

function reviewIssueCategory(issue) {
  const value = `${issue?.code || ""} ${issue?.title || ""}`.toUpperCase();
  if (/RANGE|DEPTH|TOP|BOTTOM|ENDPOINT|EXTENSION|FLEXION|ANGLE/.test(value)) {
    return "range_endpoints";
  }
  if (/STABILITY|SWAY|TRUNK|TORSO|SUPPORT|FOOT|HIP_LIFT|MOMENTUM|LEAN/.test(value)) {
    return "stability_compensation";
  }
  if (/TEMPO|CONTROL|RETURN|ECCENTRIC|PAUSE|CYCLE|REP_|TIMING/.test(value)) {
    return "tempo_consistency";
  }
  if (/SAFETY|PAIN|DANGEROUS|INSUFFICIENT_EVIDENCE|ACTION_MISMATCH/.test(value)) {
    return "load_safety";
  }
  return "movement_task";
}

function reviewIssueContext(issue) {
  const reps = Array.isArray(issue?.repIndexes) && issue.repIndexes.length
    ? `第 ${issue.repIndexes.join("、")} 次`
    : "";
  const stageLabels = {
    startPosition: "起始位",
    keyPosition: "关键位",
    endPosition: "回位",
    to_key: "进入关键位阶段",
    return: "回放阶段",
    between_reps: "重复之间"
  };
  const rawStage = cleanText(issue?.stage || "", 50);
  const stage = stageLabels[rawStage] || rawStage;
  return [reps, stage].filter(Boolean).join("，");
}

function fallbackReviewSection(key, issues, strengths, rubric, captureQuality) {
  const relevant = issues.filter((issue) => reviewIssueCategory(issue) === key);
  const limited = captureQuality === "insufficient";
  const metricText = cleanList(rubric?.mainMetrics, 5, 80).join("、");
  const title = REVIEW_SECTION_TITLES[key];
  let assessment = "当前可见证据没有触发这一维度的明确问题，但这不等同于临床评估或绝对无误。";
  if (limited) assessment = "当前证据不足，暂不对这一维度作确定判断。";
  else if (relevant.length) {
    assessment = relevant.slice(0, 3).map((issue) => {
      const context = reviewIssueContext(issue);
      return `${context ? `${context}：` : ""}${issue.observation || issue.title || issue.code}`;
    }).join("；");
  } else if (key === "movement_task" && strengths.length) {
    assessment = strengths.slice(0, 3).join("；");
  } else if (key === "range_endpoints") {
    assessment = "当前规则没有发现明确的起始位、关键位或回位缺失。";
  } else if (key === "tempo_consistency") {
    assessment = "当前规则没有发现明显的失控回放、节奏中断或重复周期不完整。";
  }

  return {
    key,
    title,
    status: limited ? "limited_evidence" : relevant.length ? "needs_attention" : "mostly_meets",
    assessment: sanitizeUserFeedbackText(assessment, 420),
    evidence: metricText
      ? `本维度主要依据：${metricText}。`
      : "本维度仅依据当前视频中可见的结构化动作证据。",
    interpretation: relevant.length
      ? "这说明该技术环节尚未在当前重复中稳定达到动作规则要求。"
      : "未触发问题表示当前证据支持继续保持，不代表能够排除不可见平面的偏差。"
  };
}

function fallbackReviewRecommendation(issue, index) {
  const title = cleanText(issue?.title || issue?.code || `技术问题 ${index + 1}`, 100);
  const context = reviewIssueContext(issue);
  return {
    priority: issue?.severity === "red" ? "high" : index === 0 ? "primary" : "secondary",
    category: reviewIssueCategory(issue),
    title,
    rationale: sanitizeUserFeedbackText(
      `${context ? `${context}出现该问题。` : ""}${issue?.observation || "该环节尚未稳定达到动作标准。"}`,
      300
    ),
    execution: sanitizeUserFeedbackText(issue?.correction || "降低动作速度，在完整可控的范围内重复。", 260),
    successCriteria: `连续重复中不再出现“${title}”，并能完成清晰的起始位、关键位和回位。`
  };
}

function actionReviewFallback(analysis, reason = null) {
  const actionName = analysis?.actionType
    ? getExercise(analysis.actionType).name
    : cleanText(analysis?.actionName || "本次动作", 80);
  const score = finiteNumber(analysis?.overallScore, null);
  const issues = Array.isArray(analysis?.issues) ? analysis.issues : [];
  const rubric = actionRubric(analysis?.actionType || "other", actionName);
  const adjustments = sanitizeUserFeedbackList(issues.map(issueAdjustment), 4, 260);
  const positives = cleanList(analysis?.strengths, 4, 220);
  const firstCorrection = sanitizeUserFeedbackText(issues[0]?.correction || "", 180);
  const captureQuality = cleanText(analysis?.captureQuality || "", 40);
  const evidenceLimits = [];

  if (captureQuality === "insufficient") {
    evidenceLimits.push("当前证据不足，暂不判断不可见的动作细节。");
  }
  if (reason) {
    evidenceLimits.push("AI 点评暂不可用，本次反馈由动作规则生成。");
  }

  const firstIssue = issues[0] || null;
  const movementAnalysis = firstIssue
    ? `${reviewIssueContext(firstIssue) ? `${reviewIssueContext(firstIssue)}，` : ""}${firstIssue.observation || firstIssue.title || "动作出现了需要调整的地方"}。${firstIssue.correction ? `可以先这样改：${firstIssue.correction}` : ""}`
    : `${positives.slice(0, 2).join("；") || "本组能够形成可识别的完整动作过程"}。继续保持拉伸位、收缩位和回程节奏一致。`;

  return {
    source: "rule_fallback",
    model: null,
    summary: score === null
      ? `${actionName}已经完成分析。下面只说明视频里能确认的动作表现和最值得先改的一点。`
      : `${actionName}这组动作已经完成分析。分数只作参考，更重要的是具体动作过程和下一组怎样调整。`,
    headline: firstIssue ? `先解决：${cleanText(firstIssue.title || firstIssue.code, 80)}` : "整体动作可以继续保持",
    overview: score === null
      ? `${actionName}已经完成分析。下面只说明视频里能确认的动作表现和最值得先改的一点。`
      : `这组${actionName}能够被稳定识别。${firstIssue ? "目前最值得先处理的是一个明确的动作细节，而不是同时改很多地方。" : "当前没有触发明确问题，重点是把同样的幅度和控制保持到整组结束。"}`,
    movementAnalysis: sanitizeUserFeedbackText(movementAnalysis, 760),
    mainAdjustment: firstIssue
      ? sanitizeUserFeedbackText(`${firstIssue.title || "主要问题"}：${firstIssue.observation || "该动作细节没有稳定出现"}。${firstIssue.correction || "降低重量或速度后重新完成动作。"}`, 620)
      : "本次没有触发明确的主要问题。不要为了追求更大幅度或更快速度主动改变当前动作。",
    nextSetPlan: firstIssue
      ? sanitizeUserFeedbackText(`下一组先保持当前或更轻的重量，只盯住这一点：${firstIssue.correction || firstIssue.title}。连续几次都能做稳，再考虑加重。`, 420)
      : "下一组保持当前重量和节奏，确认每次都能从拉伸位进入收缩位，再控制回到拉伸位。",
    simpleTerms: ["拉伸位", "收缩位", "向心阶段", "离心阶段"],
    sections: ["movement_task", "range_endpoints", "stability_compensation", "tempo_consistency"]
      .map((key) => fallbackReviewSection(key, issues, positives, rubric, captureQuality)),
    recommendations: captureQuality === "insufficient"
      ? []
      : issues.slice(0, 4).map(fallbackReviewRecommendation).filter((item) => item.execution),
    coachingCues: [firstCorrection || "保持支撑稳定，让主要关节沿既定路径完成起始位、关键位和回位。"],
    progressionCriteria: captureQuality === "insufficient"
      ? "证据不足时不据此调整负荷。"
      : issues.length
        ? `先在当前或更低负荷下，使“${cleanText(issues[0].title || issues[0].code, 100)}”不再连续出现，再考虑增加负荷。`
        : "当前动作路径、活动范围和稳定性能够在整组内重复保持时，再考虑增加最小可用负荷。",
    positives: positives.length ? positives : ["动作主路径可以被稳定识别。"],
    adjustments: captureQuality === "insufficient"
      ? ["当前证据不足，暂不判断动作细节。"]
      : adjustments.length ? adjustments : ["本次没有触发明确动作问题，继续保持当前路径和节奏。"],
    cues: [firstCorrection || "下一组先把动作做慢，确认起始位充分拉伸、发力位稳定，再追求负重或训练量。"],
    nextSetFocus: issues[0]
      ? `下一组优先修正：${cleanText(issues[0].title || issues[0].code, 80)}。`
      : "下一组保持当前动作路径，重点看疲劳阶段是否还能维持同样幅度。",
    evidenceLimits
  };
}

function actionReviewEvidence(job, video, analysis) {
  const comparison = analysis?.diagnostics?.poseEngineComparison || null;
  const rubric = actionRubric(
    analysis?.actionType || video?.actionType || "other",
    analysis?.actionName || video?.actionName || ""
  );
  return {
    jobTitle: job?.title || "",
    actionType: analysis?.actionType || video?.actionType || "",
    actionName: getExercise(analysis?.actionType || video?.actionType || "other").name,
    cameraAngle: analysis?.cameraAngle || video?.cameraAngle || "",
    planned: {
      sets: video?.sets || 0,
      reps: video?.reps || 0,
      loadKg: video?.loadKg || 0,
      rpe: video?.rpe || 0
    },
    result: {
      overallScore: analysis?.overallScore ?? null,
      safetyLevel: analysis?.safetyLevel || "",
      captureQuality: analysis?.captureQuality || "",
      poseCoverage: analysis?.metadata?.poseCoverage ?? null,
      averagePoseQuality: analysis?.metadata?.averagePoseQuality ?? null
    },
    movementStandard: rubric,
    standardImpact: buildStandardImpact(analysis, rubric),
    measurements: {
      angles: analysis?.measurements?.angles || null,
      ranges: analysis?.measurements?.ranges || null,
      asymmetry: analysis?.measurements?.asymmetry || null,
      stability: analysis?.measurements?.stability || null
    },
    repetitionSummary: (analysis?.repEvents || []).slice(0, 12).map((rep) => ({
      repIndex: rep.repIndex,
      quality: rep.quality,
      durationSeconds: rep.durationSeconds,
      signalAmplitude: rep.signalAmplitude
    })),
    strengths: cleanList(analysis?.strengths, 6, 220),
    issues: (analysis?.issues || []).slice(0, 8).map((issue) => ({
      code: issue.code || "",
      title: issue.title || "",
      severity: issue.severity || "",
      stage: issue.stage || "",
      observation: issue.observation || "",
      correction: issue.correction || "",
      confidence: issue.confidence ?? null,
      repIndexes: issue.repIndexes || []
    })),
    autoActionDetection: analysis?.diagnostics?.autoActionDetection || null,
    poseFusion: analysis?.diagnostics?.poseFusion ? {
      selectedEngine: analysis.diagnostics.poseFusion.selectedEngine,
      confidence: analysis.diagnostics.poseFusion.confidence,
      reasons: (analysis.diagnostics.poseFusion.reasons || []).slice(0, 6)
    } : null,
    glmVideoPrior: analysis?.diagnostics?.glmVideoPrior || null,
    glmVisualReview: analysis?.diagnostics?.glmVisualReview || null,
    poseEngineComparison: comparison ? {
      enabled: comparison.enabled,
      primaryBackend: comparison.primaryBackend,
      secondaryBackend: comparison.secondaryBackend,
      primary: comparison.primary,
      secondary: comparison.secondary,
      topDivergentJoints: (comparison.topDivergentJoints || []).slice(0, 5),
      recommendation: comparison.recommendation,
      error: comparison.error || null
    } : null
  };
}

function summarizeVisualImages(images) {
  return images.map((item) => ({
    label: item.label,
    filename: item.filename,
    timeMs: item.timeMs,
    approxBytes: item.dataUrl
      ? Math.round(Math.max(0, item.dataUrl.length - item.dataUrl.indexOf(",") - 1) * 0.75)
      : 0
  }));
}

function buildGlmVisualReviewMessages({ job, video, analysis, images }) {
  const text = [
    "你是健身动作视频的视觉观察员。",
    "请先只根据图片证据判断动作，不要参考算法次数、算法评分或算法问题列表。",
    "图片可能包含一张四宫格关键帧和若干阶段帧；它们来自用户上传视频。",
    "任务：估计动作类型、有效次数、肉眼能看到的动作优点和主要技术问题。",
    "如果只能看到关键帧、无法可靠估计完整次数，visualRepEstimate 必须返回 null，不要返回 0。",
    "不要输出医疗诊断、康复处方或长期训练计划。",
    "如果图片不足以判断，请在 evidenceLimits 中说明，不要强行下结论。",
    "必须只返回 JSON 对象，不要 Markdown。",
    "JSON 字段：summary 字符串；visualAction 字符串；visualRepEstimate 数字或 null；positives 字符串数组；visualIssues 字符串数组；adjustments 字符串数组；confidence 0-1；evidenceLimits 字符串数组。",
    "所有内容使用简体中文。",
    "",
    JSON.stringify({
      job: {
        mode: job?.mode || "",
        bodyPart: job?.bodyPart || ""
      },
      video: {
        id: video?.id || "",
        actionType: video?.actionType || analysis?.actionType || "",
        actionName: video?.actionName || analysis?.actionName || "",
        cameraAngle: video?.cameraAngle || analysis?.cameraAngle || "",
        originalName: video?.originalName || ""
      },
      imageLabels: images.map((item) => ({
        label: item.label,
        filename: item.filename,
        timeMs: item.timeMs
      }))
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

function normalizeGlmVisualReview(raw, fallback = {}) {
  const review = raw && typeof raw === "object" ? raw : {};
  const confidence = finiteNumber(review.confidence, null);
  const visualRepEstimate = finiteNumber(review.visualRepEstimate, null);
  return {
    summary: cleanText(review.summary || fallback.summary || "", 320),
    visualAction: cleanText(review.visualAction || fallback.visualAction || "", 80),
    visualRepEstimate: visualRepEstimate === null || visualRepEstimate <= 0
      ? null
      : Math.max(0, Math.round(visualRepEstimate)),
    positives: cleanList(review.positives, 4, 180),
    visualIssues: cleanList(review.visualIssues, 5, 220),
    adjustments: cleanList(review.adjustments, 5, 220),
    confidence: confidence === null ? null : Math.max(0, Math.min(1, confidence)),
    evidenceLimits: cleanList(review.evidenceLimits, 4, 220)
  };
}

async function generateGlmVisualReview({ visionClient, job, video, analysis, artifactDirectory }) {
  const images = collectArbitrationImages(analysis, artifactDirectory);
  const fallback = {
    summary: "GLM 视觉先验未生成。",
    visualAction: cleanText(analysis?.actionName || analysis?.actionType || "", 80)
  };

  if (!visionClient) {
    const review = {
      source: "skipped",
      provider: "zhipu",
      model: null,
      imageCount: images.length,
      ...normalizeGlmVisualReview(fallback, fallback),
      evidenceLimits: ["服务端未配置 GLM 视觉模型，已仅使用骨骼算法和文本点评。"]
    };
    return {
      review,
      log: calculationLog(
        "ai",
        "GLM 视觉先验跳过",
        "服务端未配置可用的 GLM 视觉模型客户端。",
        { provider: "zhipu", imageCount: images.length },
        "warning"
      )
    };
  }

  if (!images.length) {
    const review = {
      source: "skipped",
      provider: "zhipu",
      model: null,
      imageCount: 0,
      ...normalizeGlmVisualReview(fallback, fallback),
      evidenceLimits: ["没有可发送给 GLM 的关键帧或拼图，已跳过视觉先验。"]
    };
    return {
      review,
      log: calculationLog(
        "ai",
        "GLM 视觉先验跳过",
        "没有可发送给 GLM 的关键帧或拼图。",
        { provider: "zhipu", imageCount: 0 },
        "warning"
      )
    };
  }

  try {
    const response = await visionClient.complete(buildGlmVisualReviewMessages({
      job,
      video,
      analysis,
      images
    }), {
      json: true,
      temperature: 0,
      maxTokens: 1000
    });
    const parsed = parseJsonObject(response.content);
    const normalized = normalizeGlmVisualReview(parsed, fallback);
    const review = {
      source: "glm_vision",
      provider: "zhipu",
      model: response.model || null,
      usage: response.usage || null,
      imageCount: images.length,
      images: summarizeVisualImages(images),
      ...normalized
    };
    return {
      review,
      log: calculationLog(
        "ai",
        "GLM 视觉先验完成",
        `GLM 已先查看 ${images.length} 张关键帧证据，并返回视觉观察。`,
        {
          provider: "zhipu",
          model: response.model || null,
          usage: response.usage || null,
          imageCount: images.length,
          visualRepEstimate: review.visualRepEstimate,
          confidence: review.confidence,
          images: summarizeVisualImages(images)
        }
      )
    };
  } catch (error) {
    const review = {
      source: "failed",
      provider: "zhipu",
      model: null,
      imageCount: images.length,
      ...normalizeGlmVisualReview(fallback, fallback),
      error: {
        code: error.code || "GLM_VISUAL_REVIEW_FAILED",
        message: cleanText(error.message || "GLM visual review failed", 300)
      },
      evidenceLimits: [`GLM 视觉先验失败，已仅使用骨骼算法和文本点评：${cleanText(error.code || error.message || "unknown error", 120)}`]
    };
    return {
      review,
      log: calculationLog(
        "ai",
        "GLM 视觉先验失败",
        "GLM 视觉模型调用失败，主分析不会因此失败。",
        {
          provider: "zhipu",
          imageCount: images.length,
          errorCode: review.error.code,
          error: review.error.message
        },
        "warning"
      )
    };
  }
}

function buildGlmVideoPriorMessages({ job, video, rubric, proxy, dataUrl }) {
  const text = [
    "你是健身动作视频的视觉观察员。请完整观看用户上传视频的低码率代理，不要只根据单帧下结论。",
    "用户已经选择了动作和拍摄方向；你需要按这个动作的关键点先给出自己的视觉判断。",
    "第一轮只基于完整视频视觉证据，不要参考后续算法结果。",
    "计数必须严格：只统计完整周期。完整周期必须同时包含起始拉伸/顶部、目标收缩/底部、再回到起始/顶部；半程、停顿、预备动作、镜头开始或结束处残缺动作都不要计入。",
    "高位下拉只在肘角从 >135° 的拉伸位进入 <90° 的收缩位、并可控回到拉伸位时计 1 次；硬拉/RDL 只在 top-bottom-top 完整完成时计 1 次。无法确认时返回较低估计或 null，并在 evidenceLimits 说明。",
    "如果视频代理分辨率较低、遮挡明显或无法可靠计数，请明确写入 evidenceLimits。",
    "不要输出医疗诊断、康复处方或长期训练计划。",
    "必须只返回 JSON 对象，不要 Markdown。",
    "JSON 字段：summary 字符串；visualAction 字符串；actionMatch confirm/uncertain/mismatch；visualRepEstimate 数字或 null；positives 字符串数组；visualIssues 字符串数组；adjustments 字符串数组；confidence 0-1；evidenceLimits 字符串数组。",
    "所有内容使用简体中文。",
    "",
    JSON.stringify({
      job: {
        mode: job?.mode || "",
        bodyPart: job?.bodyPart || ""
      },
      video: {
        id: video?.id || "",
        selectedActionType: video?.actionType || "",
        selectedActionName: video?.actionName || "",
        selectedCameraAngle: video?.cameraAngle || "",
        originalName: video?.originalName || ""
      },
      actionRubric: rubric,
      proxy: {
        filename: proxy?.filename || "",
        durationSeconds: proxy?.durationSeconds ?? null,
        fps: proxy?.fps ?? null,
        width: proxy?.width ?? null,
        height: proxy?.height ?? null
      }
    })
  ].join("\n");

  return [{
    role: "user",
    content: [
      { type: "text", text },
      { type: "video_url", video_url: { url: dataUrl } }
    ]
  }];
}

function normalizeGlmVideoPrior(raw, fallback = {}) {
  const review = raw && typeof raw === "object" ? raw : {};
  const confidence = finiteNumber(review.confidence, null);
  const repEstimate = finiteNumber(review.visualRepEstimate, null);
  const actionMatch = cleanText(review.actionMatch, 20);
  return {
    summary: cleanText(review.summary || fallback.summary || "", 360),
    visualAction: cleanText(review.visualAction || fallback.visualAction || "", 100),
    actionMatch: ["confirm", "uncertain", "mismatch"].includes(actionMatch) ? actionMatch : "uncertain",
    visualRepEstimate: repEstimate === null || repEstimate <= 0 ? null : Math.max(0, Math.round(repEstimate)),
    positives: cleanList(review.positives, 5, 200),
    visualIssues: cleanList(review.visualIssues, 6, 240),
    adjustments: cleanList(review.adjustments, 6, 240),
    confidence: confidence === null ? null : Math.max(0, Math.min(1, confidence)),
    evidenceLimits: cleanList(review.evidenceLimits, 5, 240)
  };
}

async function generateGlmVideoPrior({ visionClient, job, video, rubric, proxy, artifactDirectory }) {
  const fallback = {
    summary: "GLM 完整视频先验未生成。",
    visualAction: cleanText(video?.actionName || video?.actionType || "", 80)
  };
  if (!visionClient) {
    return {
      review: {
        source: "skipped",
        provider: "zhipu",
        model: null,
        proxy: proxy || null,
        ...normalizeGlmVideoPrior(fallback, fallback),
        evidenceLimits: ["服务端未配置 GLM 视频模型，已仅使用算法结果。"]
      },
      log: calculationLog(
        "ai",
        "GLM 完整视频先验跳过",
        "服务端未配置可用的 GLM 视频模型客户端。",
        { provider: "zhipu" },
        "warning"
      )
    };
  }
  const dataUrl = videoDataUrl(path.resolve(artifactDirectory, proxy?.filename || ""));
  if (!dataUrl) {
    return {
      review: {
        source: "skipped",
        provider: "zhipu",
        model: null,
        proxy: proxy || null,
        ...normalizeGlmVideoPrior(fallback, fallback),
        evidenceLimits: ["完整视频代理不存在或过大，未发送给 GLM。"]
      },
      log: calculationLog(
        "ai",
        "GLM 完整视频先验跳过",
        "完整视频代理不存在或超过发送大小限制。",
        { provider: "zhipu", proxy },
        "warning"
      )
    };
  }

  try {
    const response = await visionClient.complete(buildGlmVideoPriorMessages({
      job,
      video,
      rubric,
      proxy,
      dataUrl
    }), {
      json: true,
      temperature: 0,
      maxTokens: 1000
    });
    const normalized = normalizeGlmVideoPrior(parseJsonObject(response.content), fallback);
    return {
      review: {
        source: "glm_video",
        provider: "zhipu",
        model: response.model || null,
        usage: response.usage || null,
        proxy,
        actionRubric: rubric,
        ...normalized
      },
      log: calculationLog(
        "ai",
        "GLM 完整视频先验完成",
        "GLM 已查看完整视频代理并返回第一轮视觉判断。",
        {
          provider: "zhipu",
          model: response.model || null,
          usage: response.usage || null,
          proxy,
          visualRepEstimate: normalized.visualRepEstimate,
          confidence: normalized.confidence
        }
      )
    };
  } catch (error) {
    return {
      review: {
        source: "failed",
        provider: "zhipu",
        model: null,
        proxy,
        ...normalizeGlmVideoPrior(fallback, fallback),
        error: {
          code: error.code || "GLM_VIDEO_PRIOR_FAILED",
          message: cleanText(error.message || "GLM video prior failed", 300)
        },
        evidenceLimits: [`GLM 完整视频先验失败，已继续使用算法结果：${cleanText(error.code || error.message || "unknown error", 120)}`]
      },
      log: calculationLog(
        "ai",
        "GLM 完整视频先验失败",
        "GLM 完整视频调用失败，主分析继续执行。",
        {
          provider: "zhipu",
          proxy,
          errorCode: error.code || "GLM_VIDEO_PRIOR_FAILED",
          error: cleanText(error.message, 300)
        },
        "warning"
      )
    };
  }
}

function buildGlmAlgorithmJudgmentMessages({ job, video, analysis, rubric }) {
  const text = [
    "你是健身动作分析的二次仲裁器。",
    "你会收到：用户选择的动作/方向、该动作关键点、GLM 第一轮完整视频视觉判断、算法结构化结果。",
    "任务：比较第一轮视觉判断和算法结果，给出最终技术判断。不要写给用户的长篇点评，后续会交给 DeepSeek 改写。",
    "如果视觉判断和算法冲突，说明冲突点和你采信哪一边；不要凭空改算法角度数据。",
    "如果算法次数来自 motion-tracker，且骨骼覆盖率较高、没有明显遮挡或动作类型冲突，除非完整视频证据非常清楚，不要用 GLM 粗略估计覆盖 motion-tracker 次数。",
    "finalRepCount 必须代表完整重复次数，不得把半程、过渡、预备位或结束残段计入。",
    "不要输出医疗诊断、康复处方或长期训练计划。",
    "必须只返回 JSON 对象，不要 Markdown。",
    "JSON 字段：summary 字符串；finalActionMatch confirm/uncertain/mismatch；finalRepCount 数字或 null；repCountReason 字符串；positives 字符串数组；issues 字符串数组；adjustments 字符串数组；confidence 0-1；evidenceLimits 字符串数组。",
    "所有内容使用简体中文。",
    "",
    JSON.stringify({
      job: {
        mode: job?.mode || "",
        bodyPart: job?.bodyPart || ""
      },
      video: {
        id: video?.id || "",
        selectedActionType: video?.actionType || "",
        selectedActionName: video?.actionName || "",
        selectedCameraAngle: video?.cameraAngle || "",
        originalName: video?.originalName || ""
      },
      actionRubric: rubric,
      glmVideoPrior: analysis?.diagnostics?.glmVideoPrior || null,
      algorithm: compactAlgorithmResult(analysis)
    })
  ].join("\n");

  return [{ role: "user", content: text }];
}

function normalizeGlmAlgorithmJudgment(raw, fallback = {}) {
  const review = raw && typeof raw === "object" ? raw : {};
  const confidence = finiteNumber(review.confidence, null);
  const repCount = finiteNumber(review.finalRepCount, null);
  const finalActionMatch = cleanText(review.finalActionMatch, 20);
  return {
    summary: cleanText(review.summary || fallback.summary || "", 420),
    finalActionMatch: ["confirm", "uncertain", "mismatch"].includes(finalActionMatch) ? finalActionMatch : "uncertain",
    finalRepCount: repCount === null || repCount < 0 ? null : Math.round(repCount),
    repCountReason: cleanText(review.repCountReason || "", 240),
    positives: cleanList(review.positives, 5, 200),
    issues: cleanList(review.issues, 6, 240),
    adjustments: cleanList(review.adjustments, 6, 240),
    confidence: confidence === null ? null : Math.max(0, Math.min(1, confidence)),
    evidenceLimits: cleanList(review.evidenceLimits, 5, 240)
  };
}

async function generateGlmAlgorithmJudgment({ visionClient, job, video, analysis, rubric }) {
  const fallback = {
    summary: "GLM 二次仲裁未生成。"
  };
  if (!visionClient) {
    return {
      review: {
        source: "skipped",
        provider: "zhipu",
        model: null,
        ...normalizeGlmAlgorithmJudgment(fallback, fallback),
        evidenceLimits: ["服务端未配置 GLM 视频模型，已直接使用算法结果生成用户点评。"]
      },
      log: calculationLog(
        "ai",
        "GLM 二次仲裁跳过",
        "服务端未配置可用的 GLM 视频模型客户端。",
        { provider: "zhipu" },
        "warning"
      )
    };
  }
  try {
    const response = await visionClient.complete(buildGlmAlgorithmJudgmentMessages({
      job,
      video,
      analysis,
      rubric
    }), {
      json: true,
      temperature: 0,
      maxTokens: 1000
    });
    const normalized = normalizeGlmAlgorithmJudgment(parseJsonObject(response.content), fallback);
    return {
      review: {
        source: "glm_algorithm_judgment",
        provider: "zhipu",
        model: response.model || null,
        usage: response.usage || null,
        actionRubric: rubric,
        ...normalized
      },
      log: calculationLog(
        "ai",
        "GLM 二次仲裁完成",
        "GLM 已基于完整视频先验和算法结果返回最终技术判断。",
        {
          provider: "zhipu",
          model: response.model || null,
          usage: response.usage || null,
          finalRepCount: normalized.finalRepCount,
          finalActionMatch: normalized.finalActionMatch,
          confidence: normalized.confidence
        }
      )
    };
  } catch (error) {
    return {
      review: {
        source: "failed",
        provider: "zhipu",
        model: null,
        ...normalizeGlmAlgorithmJudgment(fallback, fallback),
        error: {
          code: error.code || "GLM_ALGORITHM_JUDGMENT_FAILED",
          message: cleanText(error.message || "GLM algorithm judgment failed", 300)
        },
        evidenceLimits: [`GLM 二次仲裁失败，已直接使用算法结果：${cleanText(error.code || error.message || "unknown error", 120)}`]
      },
      log: calculationLog(
        "ai",
        "GLM 二次仲裁失败",
        "GLM 二次仲裁调用失败，DeepSeek 将直接使用算法结果生成点评。",
        {
          provider: "zhipu",
          errorCode: error.code || "GLM_ALGORITHM_JUDGMENT_FAILED",
          error: cleanText(error.message, 300)
        },
        "warning"
      )
    };
  }
}

function buildActionReviewMessages({ job, video, analysis }) {
  const evidence = actionReviewEvidence(job, video, analysis);
  const actionPrompt = getActionPromptProfile(evidence.actionType);
  const actionInstructions = Array.isArray(actionPrompt.instructions)
    ? actionPrompt.instructions.filter(Boolean)
    : [];
  return [
    {
      role: "system",
      content: [
        "你是一名会把复杂动作讲明白的健身教练，读者是第一次系统学习动作的健身新手。",
        "用第二人称“你”自然表达，像教练看完一组后进行复盘，不要像检测报告、论文或开发日志。",
        "只使用最基础的训练词：起始位、拉伸位、收缩位、向心阶段（发力阶段）、离心阶段（控制重量回去的阶段）、躯干、骨盆、肩、肘、腕、髋、膝、脚。除非非用不可，不要使用运动链、承重线、终末位、关节力矩、矢状面、代偿模式等词。",
        "你只能依据用户上传视频的结构化分析结果（骨骼算法结果）进行判断；如果 evidence 中带有可用的 GLM 复核结论，可以把它作为辅助证据，但不要依赖它。",
        "当前主链路以 motion-tracker 骨骼周期定位和本地规则评分为准，DeepSeek 的任务是把结构化结论改写成用户能执行的反馈。",
        "不要输出、估算或强调本组做了多少次；如需讨论训练量，只能引用用户自己填写的计划组次。",
        "不要写“我认为”“系统判断”“算法判断”“没有进入目标范围”“未达到标准区间”“该维度”“证据支持”等检测报告式句子。直接说清楚哪一处身体、在动作哪个阶段、出现了什么。",
        "如果辅助视觉复核和算法结果冲突，必须以稳定的骨骼周期和标注视频为主，并说明证据限制。",
        "不要输出医疗诊断、康复处方、饮食建议或长期训练计划。",
        "如果证据不足，只说明哪些动作细节暂时不能判断；不要向用户输出覆盖率、置信度百分比或拍摄方法。",
        "输出应是一段完整教练复盘：先概括这组动作，再按实际动作顺序说明拉伸位、发力过程、收缩位和回程；只挑最重要的一个问题重点解释，其他问题只有明显且不重复时才补充。",
        "不要强制按“做得对、需要改、下一组怎么做”分类，也不要机械地给每个维度凑一句话。",
        "每个问题必须具体到身体部位和阶段。例如不要说“没有进入目标范围”，要说“在收缩位，肘还停在肩线附近，没有继续向身体两侧下拉”；不要说“稳定性不足”，要说“第 2 次发力后半段，躯干向后摆动帮助把重量拉下”。",
        "不要罗列原始角度或阈值。角度只用于你理解证据，给新手的正文应描述动作现象。",
        "coachingCues 只输出 1-2 条能在训练时马上执行的短口令，不要写抽象口号。",
        "只有结构化证据明确支持时，才讨论向心阶段、离心阶段、等长停顿、活动范围、运动链、支撑面或代偿。二维视频不能用于诊断损伤，也不能断言具体肌肉激活程度。",
        "nextSetPlan 只说明下一组最值得尝试的一件事，包括怎么做和怎样知道自己做到了，不开具长期计划。",
        "不同字段不得重复同一内容，也不要输出机位、镜头、重拍、补拍或其他拍摄建议。",
        ...(actionInstructions.length
          ? [`当前动作专属规则（${actionPrompt.version}）：`, ...actionInstructions]
          : []),
        "必须只返回 JSON 对象，不要 Markdown，不要代码块。",
        "JSON 字段：headline 字符串；overview 字符串；movementAnalysis 字符串；mainAdjustment 字符串；nextSetPlan 字符串；coachingCues 字符串数组；evidenceLimits 字符串数组。",
        "所有内容使用简体中文。movementAnalysis 是核心正文，应为 120-360 字，连贯说明动作过程，不能只是几个短句拼接。"
      ].join("\n")
    },
    {
      role: "user",
      content: JSON.stringify(evidence)
    }
  ];
}

function parseJsonObject(text) {
  const raw = String(text || "").trim();
  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start >= 0 && end > start) {
      return JSON.parse(raw.slice(start, end + 1));
    }
    throw new Error("DeepSeek did not return a JSON object");
  }
}

function normalizeActionReview(raw, fallback) {
  const review = raw && typeof raw === "object" ? raw : {};
  const fallbackSummary = sanitizeUserFeedbackText(fallback.summary, 520);
  const sections = (Array.isArray(review.sections) ? review.sections : [])
    .map((item) => {
      const key = String(item?.key || "").trim();
      const title = sanitizeUserFeedbackText(item?.title || REVIEW_SECTION_TITLES[key] || "技术分析", 80);
      const assessment = sanitizeUserFeedbackText(item?.assessment, 520);
      if (!assessment) return null;
      return {
        key: REVIEW_SECTION_TITLES[key] ? key : "movement_task",
        title,
        status: REVIEW_STATUSES.has(item?.status) ? item.status : "not_assessed",
        assessment,
        evidence: sanitizeUserFeedbackText(item?.evidence, 360),
        interpretation: sanitizeUserFeedbackText(item?.interpretation, 360)
      };
    })
    .filter(Boolean)
    .slice(0, 5);
  const recommendations = (Array.isArray(review.recommendations) ? review.recommendations : [])
    .map((item, index) => {
      const title = sanitizeUserFeedbackText(item?.title, 100);
      const execution = sanitizeUserFeedbackText(item?.execution, 300);
      if (!title || !execution) return null;
      return {
        priority: ["high", "primary", "secondary"].includes(item?.priority)
          ? item.priority
          : index === 0 ? "primary" : "secondary",
        category: REVIEW_SECTION_TITLES[item?.category] ? item.category : "movement_task",
        title,
        rationale: sanitizeUserFeedbackText(item?.rationale, 360),
        execution,
        successCriteria: sanitizeUserFeedbackText(item?.successCriteria, 300)
      };
    })
    .filter(Boolean)
    .slice(0, 4);
  const normalized = {
    headline: sanitizeUserFeedbackText(review.headline, 100) || fallback.headline,
    overview: sanitizeUserFeedbackText(review.overview || review.summary, 560) || fallback.overview || fallbackSummary,
    movementAnalysis: sanitizeUserFeedbackText(review.movementAnalysis, 1000) || fallback.movementAnalysis,
    mainAdjustment: sanitizeUserFeedbackText(review.mainAdjustment, 720) || fallback.mainAdjustment,
    nextSetPlan: sanitizeUserFeedbackText(review.nextSetPlan || review.progressionCriteria || review.nextSetFocus, 520) || fallback.nextSetPlan,
    simpleTerms: sanitizeUserFeedbackList(review.simpleTerms, 6, 40).length
      ? sanitizeUserFeedbackList(review.simpleTerms, 6, 40)
      : fallback.simpleTerms,
    summary: sanitizeUserFeedbackText(review.overview || review.summary, 560) || fallbackSummary,
    sections: sections.length ? sections : fallback.sections,
    recommendations: recommendations.length ? recommendations : fallback.recommendations,
    coachingCues: sanitizeUserFeedbackList(review.coachingCues || review.cues, 3, 140),
    progressionCriteria: sanitizeUserFeedbackText(review.progressionCriteria || review.nextSetFocus, 300)
      || sanitizeUserFeedbackText(fallback.progressionCriteria, 300),
    evidenceLimits: sanitizeUserFeedbackList(review.evidenceLimits, 4, 240),
    // Legacy fields remain available for older report consumers.
    positives: sanitizeUserFeedbackList(review.positives, 4, 220),
    adjustments: sanitizeUserFeedbackList(review.adjustments, 4, 260),
    cues: sanitizeUserFeedbackList(review.coachingCues || review.cues, 3, 140),
    nextSetFocus: sanitizeUserFeedbackText(review.progressionCriteria || review.nextSetFocus, 300)
      || sanitizeUserFeedbackText(fallback.nextSetFocus, 300)
  };

  if (!normalized.coachingCues.length) normalized.coachingCues = fallback.coachingCues;
  if (!normalized.positives.length) normalized.positives = sanitizeUserFeedbackList(fallback.positives, 4, 220);
  if (!normalized.adjustments.length) {
    normalized.adjustments = normalized.recommendations.map((item) => item.execution);
  }
  if (!normalized.cues.length) normalized.cues = normalized.coachingCues;
  if (!normalized.evidenceLimits.length) {
    normalized.evidenceLimits = sanitizeUserFeedbackList(fallback.evidenceLimits, 2, 180);
  }
  return normalized;
}

async function generateActionReview({ client, job, video, analysis }) {
  const fallback = actionReviewFallback(analysis);
  if (!client) return fallback;

  try {
    const response = await client.complete(buildActionReviewMessages({ job, video, analysis }), {
      json: true,
      temperature: 0.2,
      maxTokens: 2600
    });
    const parsed = parseJsonObject(response.content);
    return {
      source: "deepseek",
      model: response.model || null,
      usage: response.usage || null,
      ...normalizeActionReview(parsed, fallback)
    };
  } catch (error) {
    return {
      ...actionReviewFallback(analysis, error.code || error.message || "unknown error"),
      error: {
        code: error.code || "ACTION_REVIEW_FAILED",
        message: cleanText(error.message || "DeepSeek action review failed", 300)
      }
    };
  }
}

async function prepareGlmVideoProxy({
  config,
  storage,
  userKey,
  job,
  video,
  scriptPath = path.join(config.rootDirectory, "python", "prepare_glm_video.py")
}) {
  const artifactDirectory = storage.resolveJobPathByKey(userKey, job.id, "artifacts", video.id);
  fs.mkdirSync(artifactDirectory, { recursive: true });
  const inputPath = path.join(artifactDirectory, "glm-video-proxy.input.json");
  const outputPath = path.join(artifactDirectory, "glm-video-proxy.output.json");
  fs.writeFileSync(inputPath, JSON.stringify({
    videoPath: storage.resolveJobPathByKey(userKey, job.id, "uploads", video.storedFilename),
    outputDir: artifactDirectory,
    maxBytes: 14 * 1024 * 1024
  }), "utf8");
  try {
    await runProcess(config.analyzerPython, [scriptPath, inputPath, outputPath], {
      cwd: config.rootDirectory,
      timeoutMs: 10 * 60 * 1000
    });
    const payload = JSON.parse(fs.readFileSync(outputPath, "utf8"));
    if (!payload.ok) {
      throw new Error(payload.error?.message || "GLM video proxy failed");
    }
    return payload.result;
  } finally {
    fs.rmSync(inputPath, { force: true });
    fs.rmSync(outputPath, { force: true });
  }
}

function createAnalysisService(options) {
  const {
    storage,
    config,
    client,
    visionClient,
    analyzerScript = path.join(config.rootDirectory, "python", "analyze_video.py"),
    validatorScript = path.join(
      config.rootDirectory,
      "vendor",
      "xiaoyu-coach-skill",
      "xiaoyu-coach",
      "scripts",
      "validate_report.py"
    )
  } = options;
  const queue = [];
  const queuedKeys = new Set();
  const activeItems = new Map();
  const concurrency = Math.max(1, Number(config.analysisConcurrency) || 1);
  const maxQueuedJobs = Math.max(1, Number(config.analysisMaxQueuedJobs) || 32);
  const maxPendingPerUser = Math.max(1, Number(config.analysisMaxPendingPerUser) || 8);
  const maxConcurrentPerUser = Math.max(1, Number(config.analysisMaxConcurrentPerUser) || 1);

  function queueKey(userKey, jobId) {
    return `${userKey}:${jobId}`;
  }

  async function analyzeOne(userKey, job, video, index, engine = "rtmo") {
    const jobDirectory = storage.jobDirectoryByKey(userKey, job.id);
    const artifactDirectory = storage.resolveJobPathByKey(
      userKey,
      job.id,
      "artifacts",
      video.id,
      engine
    );
    fs.mkdirSync(artifactDirectory, { recursive: true });
    const inputPath = path.join(jobDirectory, `analyzer-${video.id}-${engine}.input.json`);
    const outputPath = path.join(jobDirectory, `analyzer-${video.id}-${engine}.output.json`);
    fs.writeFileSync(inputPath, JSON.stringify({
      videoPath: storage.resolveJobPathByKey(
        userKey,
        job.id,
        "uploads",
        video.storedFilename
      ),
      outputDir: artifactDirectory,
      actionType: video.actionType,
      cameraAngle: video.cameraAngle,
      targetRoi: video.targetRoi || null,
      poseBackend: analyzerBackendForEngine(engine),
      strictPoseBackend: true,
      poseEngineCompare: false,
      annotatedVideoMode: video.annotatedVideoMode || "selected"
    }), "utf8");

    updateJobWithLog(storage, userKey, job.id, {
      status: "processing",
      progress: Math.round(8 + (index / Math.max(1, job.videos.length)) * 66),
      phase: `Analyzing video ${index + 1}/${job.videos.length}`
    }, calculationLog(
      "queue",
      `开始分析第 ${index + 1} 个视频`,
      "已写入 Python 分析输入文件，并准备启动本地姿态识别进程。",
      {
        videoId: video.id,
        originalName: video.originalName,
        actionType: video.actionType,
        actionName: video.actionName,
        cameraAngle: video.cameraAngle,
        analyzerScript,
        poseBackend: analyzerBackendForEngine(engine),
        poseEngine: engine,
        poseEngineCompare: false,
        analyzerSampleFps: config.analyzerSampleFps,
        annotatedVideoMode: video.annotatedVideoMode || config.annotatedVideoMode,
        poseRecheckLowConfidence: config.poseRecheckLowConfidence,
        poseRecheckMaxWindows: config.poseRecheckMaxWindows,
        rtmlibPoseModelConfigured: Boolean(config.rtmlibPoseModel),
        rtmlibOneStage: Boolean(config.rtmlibOneStage)
      }
    ));

    const progressBase = Math.round(8 + (index / Math.max(1, job.videos.length)) * 66);
    const progressCeiling = Math.round(8 + ((index + 0.85) / Math.max(1, job.videos.length)) * 66);
    const startedAt = Date.now();
    const heartbeat = setInterval(() => {
      const elapsedSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
      const expectedSeconds = Math.max(45, Math.min(120, Math.round((Number(video.size) || 0) / (1024 * 1024) * 12 + 45)));
      const progress = Math.min(
        progressCeiling,
        progressBase + Math.round((progressCeiling - progressBase) * Math.min(0.92, elapsedSeconds / expectedSeconds))
      );
      try {
        storage.updateJobByKey(userKey, job.id, {
          status: "processing",
          progress,
          phase: `骨骼分析中，已运行 ${elapsedSeconds} 秒；本地模型通常需要 40-90 秒。`
        });
      } catch {
        // The job may be deleted while the analyzer is still running.
      }
    }, 5000);

    try {
      const processResult = await runProcess(config.analyzerPython, [analyzerScript, inputPath, outputPath], {
        cwd: config.rootDirectory,
        env: {
          POSE_BACKEND: analyzerBackendForEngine(engine),
          POSE_ENGINE_COMPARE: "false",
          ANALYSIS_SAMPLE_FPS: String(config.analyzerSampleFps),
          ANNOTATED_VIDEO_MODE: video.annotatedVideoMode || config.annotatedVideoMode,
          POSE_RECHECK_LOW_CONFIDENCE: config.poseRecheckLowConfidence ? "true" : "false",
          POSE_RECHECK_MAX_WINDOWS: String(config.poseRecheckMaxWindows),
          MMPOSE_CONFIG: config.mmposeConfig,
          MMPOSE_CHECKPOINT: config.mmposeCheckpoint,
          RTMLIB_DET_MODEL: config.rtmlibDetModel,
          RTMLIB_POSE_MODEL: config.rtmlibPoseModel,
          RTMLIB_MODE: config.rtmlibMode,
          RTMLIB_BACKEND: config.rtmlibBackend,
          RTMLIB_DEVICE: config.rtmlibDevice,
          RTMLIB_ONE_STAGE: config.rtmlibOneStage ? "true" : "false",
          RTMLIB_POSE_INPUT_SIZE: config.rtmlibPoseInputSize,
          RTMLIB_DET_INPUT_SIZE: config.rtmlibDetInputSize
        }
      });
      const payload = JSON.parse(fs.readFileSync(outputPath, "utf8"));
      if (!payload.ok) {
        throw new Error(payload.error?.message || "Video analysis failed");
      }
      appendJobLog(storage, userKey, job.id, calculationLog(
        "algorithm",
        `第 ${index + 1} 个视频算法完成`,
        `Python 输出 ${payload.result?.calculationLogs?.length || 0} 条算法步骤日志，动作分析完成。`,
        {
          videoId: video.id,
          actionType: video.actionType,
          overallScore: payload.result?.overallScore ?? null,
          safetyLevel: payload.result?.safetyLevel || null,
          captureQuality: payload.result?.captureQuality || null,
          stderr: processResult.stderr.trim().slice(0, 1000),
          stdout: processResult.stdout.trim().slice(0, 1000)
        },
        payload.result?.captureQuality === "good" ? "done" : "warning"
      ));
      return prefixArtifactPaths(payload.result, engine);
    } finally {
      clearInterval(heartbeat);
      fs.rmSync(inputPath, { force: true });
      fs.rmSync(outputPath, { force: true });
    }
  }

  async function processJob(item) {
    const { userKey, jobId } = item;
    const job = storage.getJobByKey(userKey, jobId);
    if (!Array.isArray(job.videos) || !job.videos.length) {
      throw new Error("No videos available for analysis in this job");
    }

    const comparisonMode = job.videos.some((video) => video.poseEngineMode === "both");
    const runGlmVideoReview = Boolean(config.glmVideoReviewEnabled && visionClient && !comparisonMode);
    const glmVideoPriors = [];
    const rubrics = [];
    job.videos.forEach((video) => {
      rubrics.push(actionRubric(video.actionType, video.actionName));
    });

    if (runGlmVideoReview) {
      updateJobWithLog(storage, userKey, jobId, {
        status: "reviewing",
        progress: 6,
        phase: "Running optional GLM full-video review"
      }, calculationLog(
        "ai",
        "开始可选 GLM 完整视频复核",
        "已显式开启 GLM_VIDEO_REVIEW_ENABLED，将生成完整视频低码率代理并调用 GLM。",
        {
          videoCount: job.videos.length,
          visionClientConfigured: Boolean(visionClient)
        }
      ));

      for (let index = 0; index < job.videos.length; index += 1) {
        const video = job.videos[index];
        const rubric = rubrics[index];
        let proxy = null;
        try {
          proxy = await prepareGlmVideoProxy({ config, storage, userKey, job, video });
        } catch (error) {
          proxy = {
            error: {
              code: "GLM_PROXY_FAILED",
              message: cleanText(error.message || "GLM video proxy failed", 300)
            }
          };
        }
        const prior = await generateGlmVideoPrior({
          visionClient,
          job,
          video,
          rubric,
          proxy,
          artifactDirectory: storage.resolveJobPathByKey(userKey, job.id, "artifacts", video.id)
        });
        glmVideoPriors.push(prior.review);
        appendJobLog(storage, userKey, jobId, prior.log);
      }

      appendJobLog(storage, userKey, jobId, calculationLog(
        "ai",
        "可选 GLM 完整视频复核完成",
        "每个视频已完成或记录 GLM 完整视频复核结果，随后进入骨骼算法计算。",
        {
          videos: glmVideoPriors.map((prior, index) => ({
            videoId: job.videos[index]?.id,
            source: prior.source,
            model: prior.model || null,
            actionMatch: prior.actionMatch,
            visualRepEstimate: prior.visualRepEstimate ?? null,
            confidence: prior.confidence ?? null,
            proxy: prior.proxy ? {
              filename: prior.proxy.filename,
              durationSeconds: prior.proxy.durationSeconds,
              bytes: prior.proxy.bytes,
              error: prior.proxy.error?.code || null
            } : null,
            error: prior.error?.code || null
          }))
        },
        glmVideoPriors.some((prior) => ["failed", "skipped"].includes(prior.source)) ? "warning" : "done"
      ));
    } else {
      updateJobWithLog(storage, userKey, jobId, {
        status: "processing",
        progress: 6,
        phase: "Running local pose analysis"
      }, calculationLog(
        "algorithm",
        "跳过 GLM 视频复核",
        "GLM 免费模型存在限流风险，主链路改为 motion-tracker 骨骼周期 + 本地规则 + DeepSeek 文本点评。",
        {
          glmVideoReviewEnabled: Boolean(config.glmVideoReviewEnabled),
          visionClientConfigured: Boolean(visionClient),
          videoCount: job.videos.length
        }
      ));
    }

    const analyses = [];
    for (let index = 0; index < job.videos.length; index += 1) {
      const engines = poseEnginesForVideo(job.videos[index]);
      for (const engine of engines) {
      const analysis = await analyzeOne(userKey, job, job.videos[index], index, engine);
      const actualBackend = analysis?.metadata?.poseBackend || analysis?.diagnostics?.poseBackend || "none";
      const expectedBackend = analyzerBackendForEngine(engine);
      if (actualBackend !== expectedBackend) {
        throw new Error(`Pose engine mismatch: requested ${expectedBackend}, received ${actualBackend}`);
      }
      analyses.push({
        ...analysis,
        _sourceVideoIndex: index,
        _engine: engine,
        metadata: {
          ...(analysis.metadata || {}),
          requestedPoseEngine: engine,
          actualPoseBackend: actualBackend,
          evidenceScope: "engine_only"
        },
        standardImpact: buildStandardImpact(analysis, rubrics[index]),
        frameFeedback: buildFrameFeedback(analysis, rubrics[index]),
        diagnostics: {
          ...(analysis.diagnostics || {}),
          ...(glmVideoPriors[index] ? { glmVideoPrior: glmVideoPriors[index] } : {})
        },
        calculationLogs: [
          ...(Array.isArray(analysis.calculationLogs) ? analysis.calculationLogs : []),
          ...(glmVideoPriors[index] ? [calculationLog(
            "ai",
            "GLM 完整视频先验并入算法结果",
            "算法结果已附加 GLM 第一轮完整视频判断，供二次仲裁使用。",
            {
              source: glmVideoPriors[index].source,
              actionMatch: glmVideoPriors[index].actionMatch,
              visualRepEstimate: glmVideoPriors[index].visualRepEstimate ?? null
            },
            ["failed", "skipped"].includes(glmVideoPriors[index].source) ? "warning" : "done"
          )] : [])
        ].slice(-120)
      });
      }
    }

    let judgedAnalyses = analyses;
    if (runGlmVideoReview) {
      updateJobWithLog(storage, userKey, jobId, {
        status: "reviewing",
        progress: 78,
        phase: "Running optional GLM algorithm judgment"
      }, calculationLog(
        "ai",
        "开始可选 GLM 二次仲裁",
        "将 GLM 完整视频复核和骨骼算法结构化结果交给 GLM，生成辅助技术判断。",
        {
          videoCount: analyses.length,
          visionClientConfigured: Boolean(visionClient)
        }
      ));

      judgedAnalyses = [];
      for (let index = 0; index < analyses.length; index += 1) {
        const judgment = await generateGlmAlgorithmJudgment({
          visionClient,
          job,
          video: job.videos[index],
          analysis: analyses[index],
          rubric: rubrics[index]
        });
        judgedAnalyses.push({
          ...analyses[index],
          diagnostics: {
            ...(analyses[index].diagnostics || {}),
            glmAlgorithmJudgment: judgment.review
          },
          calculationLogs: [
            ...(Array.isArray(analyses[index].calculationLogs) ? analyses[index].calculationLogs : []),
            judgment.log
          ].slice(-120)
        });
      }

      appendJobLog(storage, userKey, jobId, calculationLog(
        "ai",
        "可选 GLM 二次仲裁完成",
        "每个视频已附加 GLM 辅助技术判断；若 GLM 不可用，则保留算法结果。",
        {
          videos: judgedAnalyses.map((analysis, index) => ({
            videoId: job.videos[index]?.id,
            source: analysis.diagnostics?.glmAlgorithmJudgment?.source || null,
            model: analysis.diagnostics?.glmAlgorithmJudgment?.model || null,
            finalRepCount: analysis.diagnostics?.glmAlgorithmJudgment?.finalRepCount ?? null,
            finalActionMatch: analysis.diagnostics?.glmAlgorithmJudgment?.finalActionMatch || null,
            confidence: analysis.diagnostics?.glmAlgorithmJudgment?.confidence ?? null,
            error: analysis.diagnostics?.glmAlgorithmJudgment?.error?.code || null
          }))
        },
        judgedAnalyses.some((analysis) =>
          ["failed", "skipped"].includes(analysis.diagnostics?.glmAlgorithmJudgment?.source)
        ) ? "warning" : "done"
      ));
    }

    updateJobWithLog(storage, userKey, jobId, {
      status: "reviewing",
      progress: 80,
      phase: "Generating user-facing action review"
    }, calculationLog(
      "ai",
      "开始生成动作点评",
      "将骨骼算法结构化结果交给 DeepSeek，生成面向用户的动作点评和下一组调整建议。",
      {
        videoCount: judgedAnalyses.length,
        textClientConfigured: Boolean(client)
      }
    ));
    const jobBeforeActionReview = storage.getJobByKey(userKey, jobId);
    const actionReviewedAnalyses = [];
    for (let index = 0; index < judgedAnalyses.length; index += 1) {
      const sourceVideo = job.videos[judgedAnalyses[index]._sourceVideoIndex ?? index];
      const aiInputDigest = crypto
        .createHash("sha256")
        .update(JSON.stringify(actionReviewEvidence(jobBeforeActionReview, sourceVideo, judgedAnalyses[index])))
        .digest("hex");
      const analysisForReview = {
        ...judgedAnalyses[index],
        metadata: {
          ...(judgedAnalyses[index].metadata || {}),
          aiInputDigest
        }
      };
      const review = await generateActionReview({
        client,
        job: jobBeforeActionReview,
        video: sourceVideo,
        analysis: analysisForReview
      });
      actionReviewedAnalyses.push({
        ...analysisForReview,
        actionReview: review
      });
    }

    appendJobLog(storage, userKey, jobId, calculationLog(
      "ai",
      "动作点评生成完成",
      "每个视频已附加用户视角点评；如果 DeepSeek 不可用，则使用规则兜底建议。",
      {
        videos: actionReviewedAnalyses.map((analysis, index) => ({
          videoId: job.videos[analysis._sourceVideoIndex ?? index]?.id,
          engine: analysis._engine || null,
          source: analysis.actionReview?.source || null,
          model: analysis.actionReview?.model || null,
          fallback: analysis.actionReview?.source === "rule_fallback",
          error: analysis.actionReview?.error?.code || null
        }))
      },
      actionReviewedAnalyses.some((analysis) => analysis.actionReview?.source === "rule_fallback")
        ? "warning"
        : "done"
    ));

    updateJobWithLog(storage, userKey, jobId, {
      status: "reporting",
      progress: 82,
      phase: "Generating movement review report"
    }, calculationLog(
      "report",
      "开始生成报告",
      "将最终算法结果交给报告模块生成 HTML 和 Markdown 复盘。",
      {
        videoCount: actionReviewedAnalyses.length,
        reportUsesTextModel: Boolean(client)
      }
    ));
    const latestJob = storage.getJobByKey(userKey, jobId);
    const history = storage.listJobsByKey(userKey).filter((item) =>
      item.id !== jobId && item.status === "completed"
    );
    const reportDirectory = storage.resolveJobPathByKey(userKey, jobId, "reports");
    const reportJob = {
      ...latestJob,
      videos: actionReviewedAnalyses.map((item, index) => ({
        ...(latestJob.videos[item._sourceVideoIndex ?? index] || {}),
        actionName: `${item.actionName || "action"} · ${item._engine === "mediapipe" ? "MediaPipe" : "RTMO"}`
      }))
    };
    const report = await generateReports({
      client,
      job: reportJob,
      analyses: actionReviewedAnalyses,
      history,
      reportDirectory,
      assetUrl: (videoId, filename) => `../artifacts/${videoId}/${filename}`,
      validatorPython: config.analyzerPython,
      validatorScript
    });
    const analysis = publicAnalysisSummary(actionReviewedAnalyses, latestJob.videos);
    storage.deleteUploadsByKey(userKey, jobId);

    updateJobWithLog(storage, userKey, jobId, {
      status: "completed",
      progress: 100,
      phase: "Analysis completed",
      analysis,
      report: {
        source: report.source,
        model: report.model || null,
        narrative: report.narrative,
        files: report.files,
        validation: report.validation
      },
      error: null,
      uploadsRetained: false,
      completedAt: new Date().toISOString()
    }, calculationLog(
      "done",
      "分析任务完成",
      "上传、骨骼算法计算、动作点评、报告生成全部完成。",
      {
        averageScore: analysis.averageScore,
        safetyLevel: analysis.safetyLevel,
        reportSource: report.source,
        reportModel: report.model || null,
        reportFiles: report.files
      }
    ));
  }

  function takeNextFairItem() {
    if (!queue.length) return null;
    const activeByUser = [...activeItems.values()].reduce((counts, item) => {
      counts.set(item.userKey, (counts.get(item.userKey) || 0) + 1);
      return counts;
    }, new Map());
    const fairIndex = queue.findIndex((item) =>
      (activeByUser.get(item.userKey) || 0) < maxConcurrentPerUser
    );
    return fairIndex >= 0 ? queue.splice(fairIndex, 1)[0] : null;
  }

  function markFailed(item, error) {
    try {
      updateJobWithLog(storage, item.userKey, item.jobId, {
        status: "failed",
        phase: "Analysis failed",
        error: {
          code: "ANALYSIS_FAILED",
          message: String(error.message || "Analysis failed").slice(0, 1000)
        }
      }, calculationLog(
        "error",
        "分析任务失败",
        "本次任务在算法、AI 复核或报告生成过程中失败。",
        {
          message: String(error.message || "Analysis failed").slice(0, 1000)
        },
        "error"
      ));
    } catch {
      // The user may delete a running job. No record remains to update.
    }
  }

  function drain() {
    while (activeItems.size < concurrency && queue.length) {
      const item = takeNextFairItem();
      if (!item) break;
      const key = queueKey(item.userKey, item.jobId);
      activeItems.set(key, item);
      Promise.resolve()
        .then(() => options.processJobOverride ? options.processJobOverride(item) : processJob(item))
        .catch((error) => markFailed(item, error))
        .finally(() => {
          activeItems.delete(key);
          queuedKeys.delete(key);
          setImmediate(drain);
        });
    }
  }

  function pendingForUser(userKey) {
    return queue.filter((item) => item.userKey === userKey).length
      + [...activeItems.values()].filter((item) => item.userKey === userKey).length;
  }

  function capacityForUser(userKey) {
    if (pendingForUser(userKey) >= maxPendingPerUser) {
      return { accepted: false, code: "USER_ANALYSIS_LIMIT" };
    }
    if (queue.length >= maxQueuedJobs) {
      return { accepted: false, code: "ANALYSIS_QUEUE_FULL" };
    }
    return { accepted: true, code: null };
  }

  function enqueueByKey(userKey, jobId, enqueueOptions = {}) {
    const key = queueKey(userKey, jobId);
    if (queuedKeys.has(key)) return false;
    if (!enqueueOptions.recovery && !capacityForUser(userKey).accepted) return false;
    queuedKeys.add(key);
    queue.push({ userKey, jobId, enqueuedAt: Date.now() });
    drain();
    return true;
  }

  function enqueue(userId, jobId) {
    return enqueueByKey(storage.userKey(userId), jobId);
  }

  function recover() {
    const jobs = storage.recoverInterruptedJobs();
    jobs.forEach((item) => enqueueByKey(item.userKey, item.jobId, { recovery: true }));
    return jobs.length;
  }

  return {
    enqueue,
    enqueueByKey,
    async runNow(item) {
      try {
        await processJob(item);
      } catch (error) {
        markFailed(item, error);
        throw error;
      }
    },
    canAccept(userId) {
      return capacityForUser(storage.userKey(userId));
    },
    recover,
    status() {
      return {
        running: activeItems.size > 0,
        active: activeItems.size,
        concurrency,
        maxConcurrentPerUser,
        queued: queue.length,
        maxQueued: maxQueuedJobs,
        availableQueueSlots: Math.max(0, maxQueuedJobs - queue.length),
        overloaded: queue.length >= maxQueuedJobs,
        queueMode: "memory"
      };
    }
  };
}

module.exports = {
  createAnalysisService,
  actionReviewFallback,
  buildStandardImpact,
  buildFrameFeedback,
  buildActionReviewMessages,
  buildGlmAlgorithmJudgmentMessages,
  buildGlmVisualReviewMessages,
  buildGlmVideoPriorMessages,
  generateActionReview,
  generateGlmAlgorithmJudgment,
  generateGlmVisualReview,
  generateGlmVideoPrior,
  normalizeActionReview,
  normalizeGlmAlgorithmJudgment,
  normalizeGlmVisualReview,
  normalizeGlmVideoPrior,
  prepareGlmVideoProxy,
  publicAnalysisSummary,
  runProcess
};


