const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { repairMojibake } = require("./text-encoding");

const FORBIDDEN_TERMS = [
  "MediaPipe",
  "Mediapipe",
  "OpenPose",
  "OpenCV",
  "tracking pollution",
  "invalid frame",
  "model recognition error",
  "pose engine",
  "tracking failure",
  "骨架追踪",
  "骨骼追踪污染",
  "污染帧",
  "数据作废",
  "筛选帧",
  "自动判断"
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

function cleanText(value, maximum = 4000) {
  let text = repairMojibake(value).trim();
  for (const term of FORBIDDEN_TERMS) {
    text = text.replaceAll(term, "动作分析");
  }
  return text.slice(0, maximum);
}

function cleanList(value, maximumItems = 12, maximumLength = 1000) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => cleanText(item, maximumLength))
    .filter(Boolean)
    .slice(0, maximumItems);
}

function number(value, fallback = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function calculateVideoVolume(video) {
  const sets = Math.max(0, number(video.sets));
  const reps = Math.max(0, number(video.reps));
  const loadKg = Math.max(0, number(video.loadKg));
  return {
    sets,
    reps,
    loadKg,
    repetitions: sets * reps,
    tonnageKg: sets * reps * loadKg
  };
}

function summarizeVolume(videos) {
  return videos.reduce((total, video) => {
    const item = calculateVideoVolume(video);
    total.sets += item.sets;
    total.repetitions += item.repetitions;
    total.tonnageKg += item.tonnageKg;
    return total;
  }, { sets: 0, repetitions: 0, tonnageKg: 0 });
}

function severityLabel(severity) {
  if (severity === "red") return "先停止";
  if (severity === "yellow") return "需优化";
  return "可继续";
}

function issueContext(issue) {
  const reps = Array.isArray(issue.repIndexes) && issue.repIndexes.length
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
  const stage = issue.stage ? ` ${stageLabels[issue.stage] || issue.stage}` : "";
  return reps || stage ? `${reps}${stage}：` : "";
}

function fallbackNarrative(job, analyses, history = []) {
  const videos = job.videos || [];
  const scoreValues = analyses.map((item) => number(item.overallScore)).filter((item) => item > 0);
  const averageScore = scoreValues.length
    ? Math.round(scoreValues.reduce((sum, item) => sum + item, 0) / scoreValues.length)
    : 0;
  const allIssues = analyses.flatMap((item, index) =>
    (item.issues || []).map((issue) => ({
      ...issue,
      actionName: item.actionName,
      videoId: videos[index] && videos[index].id
    }))
  );
  const red = allIssues.filter((item) => item.severity === "red");
  const yellow = allIssues.filter((item) => item.severity === "yellow");
  const insufficient = analyses.filter((item) =>
    item.captureQuality === "insufficient"
    || (item.issues || []).some((issue) => issue.code === "INSUFFICIENT_EVIDENCE")
  );
  const volume = summarizeVolume(videos);
  const previous = history.find((item) =>
    item.id !== job.id
    && item.status === "completed"
    && (item.bodyPart === job.bodyPart || item.videos?.some((video) =>
      videos.some((current) => current.actionType === video.actionType)
    ))
  );
  const previousScore = number(previous?.analysis?.averageScore);
  const scoreChange = previousScore && averageScore ? averageScore - previousScore : null;

  return {
    title: cleanText(job.title || `${job.bodyPart || "训练"}动作复盘`, 120),
    overall: insufficient.length === analyses.length
      ? "当前证据不足，不能形成有效动作评分或进阶建议。"
      : insufficient.length
        ? `本次有 ${insufficient.length} 个视频画面证据不足，其余动作仅按可见结构化证据给出保守反馈。`
        : red.length
      ? `本次训练有 ${red.length} 项需要立即停止并确认的问题。先处理动作安全与控制，再讨论加重。`
      : yellow.length
        ? `本次动作整体可以继续训练，但有 ${yellow.length} 项质量问题需要优先修正。平均动作评分 ${averageScore || "暂无"}。`
        : `本次动作整体稳定，当前视频中没有明显需要立即停止的问题。继续保持当前技术标准。`,
    safetyPriorities: allIssues.length
      ? allIssues.slice(0, 8).map((item) => ({
        level: item.severity,
        title: `${item.actionName}：${item.title}`,
        detail: `${issueContext(item)}${item.observation}`,
        next: item.code === "INSUFFICIENT_EVIDENCE"
          ? "当前证据不足，暂不判断动作细节。"
          : item.correction
      }))
      : [{
        level: "green",
        title: "当前动作整体稳定",
        detail: "现有视频中未见明显失控或需要立即停止的动作表现。",
        next: "保持当前负重和节奏标准，疲劳后动作变形时结束该组。"
      }],
    actionSummaries: analyses.map((analysis, index) => {
      const review = analysis.actionReview || {};
      return {
        videoId: videos[index]?.id || String(index),
        actionName: analysis.actionName,
        summary: review.summary || (analysis.captureQuality === "insufficient"
          ? "当前证据不足，暂不判断动作细节。"
          : `动作评分 ${analysis.overallScore || "暂无"}。`),
        headline: cleanText(review.headline, 120),
        overview: cleanText(review.overview || review.summary, 700),
        movementAnalysis: cleanText(review.movementAnalysis, 1200),
        mainAdjustment: cleanText(review.mainAdjustment, 800),
        nextSetPlan: cleanText(review.nextSetPlan, 600),
        positives: cleanList(review.positives?.length ? review.positives : analysis.strengths, 2, 220),
        improvements: cleanList(
          review.adjustments?.length
            ? review.adjustments
            : (analysis.issues || []).map((item) => `${issueContext(item)}${item.title}`),
          1,
          240
        ),
        cue: analysis.captureQuality === "insufficient"
          ? "当前证据不足，暂不判断动作细节。"
          : cleanList(review.cues, 1, 140)[0]
            || analysis.issues?.[0]?.correction
            || "保持当前动作路径和节奏。",
        sections: (Array.isArray(review.sections) ? review.sections : []).slice(0, 5).map((section) => ({
          title: cleanText(section.title || "技术分析", 100),
          status: cleanText(section.status || "not_assessed", 40),
          assessment: cleanText(section.assessment, 600),
          evidence: cleanText(section.evidence, 400),
          interpretation: cleanText(section.interpretation, 400)
        })),
        recommendations: (Array.isArray(review.recommendations) ? review.recommendations : []).slice(0, 4).map((item) => ({
          priority: cleanText(item.priority || "secondary", 30),
          title: cleanText(item.title || "技术修正", 100),
          rationale: cleanText(item.rationale, 400),
          execution: cleanText(item.execution, 350),
          successCriteria: cleanText(item.successCriteria, 350)
        })),
        coachingCues: cleanList(review.coachingCues?.length ? review.coachingCues : review.cues, 3, 160),
        progressionCriteria: cleanText(review.progressionCriteria || review.nextSetFocus, 350),
        evidenceLimits: cleanList(review.evidenceLimits, 4, 260)
      };
    }),
    volumeAnalysis: volume.sets
      ? `共记录 ${volume.sets} 组、${volume.repetitions} 次计划重复${volume.tonnageKg ? `，估算训练吨位 ${Math.round(volume.tonnageKg)}kg` : ""}。`
      : "本次未填写完整的组数、次数或重量，因此不计算训练吨位。",
    historyComparison: previous
      ? scoreChange === null
        ? `已找到同一用户的历史训练记录，可结合本次动作问题继续追踪。`
        : `与最近同类训练相比，平均动作评分${scoreChange >= 0 ? "提高" : "下降"} ${Math.abs(scoreChange)} 分。`
      : "这是当前用户的首份同类训练记录，后续报告会在相同动作或部位范围内比较。",
    // Per-action reviews are the single source of user-facing strengths and fixes.
    positives: [],
    improvements: [],
    redFlags: red.length
      ? red.map((item) => `${item.actionName}：${issueContext(item)}${item.observation}`)
      : ["本次没有明确需要立即停止的问题；出现急性疼痛、麻木、头晕、胸痛或动作失控时仍应立即停止。"],
    progressionConditions: analyses.map((analysis, index) => {
      const video = videos[index] || {};
      if (analysis.captureQuality === "insufficient") {
        return `${analysis.actionName}：当前证据不足，暂不评分，也不建议加重。`;
      }
      const nextLoad = video.loadKg ? Math.round((number(video.loadKg) + Math.max(1, number(video.loadKg) * 0.025)) * 2) / 2 : null;
      const volumeText = video.sets || video.reps
        ? `${video.sets || "-"} 组${video.reps ? ` × ${video.reps}` : ""}`
        : "原计划训练量";
      return `${analysis.actionName}：先用当前重量完成 ${volumeText}，关键问题不再出现且最后一组节奏稳定${nextLoad ? `，再尝试 ${nextLoad}kg 的第一组` : "，再增加最小可用负重"}。`;
    }),
    filmingAdvice: [],
    conversationPoints: [
      "哪一个动作在训练中最吃力，吃力发生在动作的哪个阶段？",
      "本次是否出现疼痛、麻木、头晕或动作失去控制？"
    ],
    coachClose: insufficient.length === analyses.length
        ? "当前证据不能作为训练质量判断依据，暂不提供进阶建议。"
      : red.length
        ? "先停止相关动作并完成线下评估。安全问题没有解决前，不建议增加负重。"
      : "下次训练只改最重要的一到两个问题。先把当前负重做稳，再用清晰标准决定是否进阶。"
  };
}

function buildAiPrompt(job, analyses, history) {
  const evidence = {
    job: {
      mode: job.mode,
      title: job.title,
      traineeName: job.traineeName,
      trainingDate: job.trainingDate,
      bodyPart: job.bodyPart,
      notes: job.notes
    },
    videos: (job.videos || []).map((video, index) => ({
      id: video.id,
      actionType: video.actionType,
      actionName: analyses[index]?.actionName,
      sets: video.sets,
      reps: video.reps,
      loadKg: video.loadKg,
      cameraAngle: video.cameraAngle,
      notes: video.notes,
      analysis: analyses[index]
    })),
    history: history.slice(0, 5).map((item) => ({
      id: item.id,
      date: item.trainingDate,
      bodyPart: item.bodyPart,
      averageScore: item.analysis?.averageScore,
      volume: item.analysis?.volume,
      actions: item.videos?.map((video) => video.actionType)
    }))
  };

  return [
    {
      role: "system",
      content: [
        "你是面向 0-3 年训练者的力量训练教练报告编辑。",
        "只允许依据提供的结构化动作证据写结论，不得声称看见证据中没有的细节。",
        "安全优先于表现；不做医疗诊断。高风险问题必须直接建议停止、减重或线下求助。",
        "输出严格 JSON 对象，不要 Markdown 代码块。",
        "JSON 字段必须包括：title, overall, safetyPriorities, actionSummaries, volumeAnalysis, historyComparison, positives, improvements, redFlags, progressionConditions, conversationPoints, coachClose。",
        "safetyPriorities 每项包含 level(red/yellow/green), title, detail, next。",
        "actionSummaries 每项包含 videoId, actionName, summary, positives[], improvements[], cue。",
        "如果 issues 包含 repIndexes、stage 或 timeRangesMs，优先写清第几次和哪个阶段；不得向用户展示置信度或覆盖率百分比。",
        "不要在给用户看的 title、overall、detail、next、summary、cue 等文本里使用交通灯分级说法；直接说明哪里没做好以及怎么优化。",
        "不要输出下一次训练计划、训练动作安排、器械清单、组数处方或饮食建议；只评价当前上传视频中的动作表现。",
        "禁止输出内部工具、模型、跟踪失败、无效帧等技术术语。",
        "进阶条件只允许作为当前动作是否可以继续加重的判定条件，不得变成训练计划。",
        "不得提供机位、镜头、拍摄方法、重拍或补拍建议。画面证据不足时只说明暂不判断对应动作细节。",
        "动作结论必须复用已有 actionReview，不要在不同字段中换一种说法重复相同内容。"
      ].join("\n")
    },
    {
      role: "user",
      content: JSON.stringify(evidence)
    }
  ];
}

function normalizeNarrative(raw, fallback, videos) {
  const source = raw && typeof raw === "object" ? raw : {};
  const summaryById = new Map(
    (Array.isArray(source.actionSummaries) ? source.actionSummaries : [])
      .map((item) => [String(item?.videoId || ""), item])
  );

  return {
    title: cleanText(source.title || fallback.title, 120),
    overall: fallback.overall,
    // Safety conclusions remain deterministic and tied to measured rule issues.
    safetyPriorities: fallback.safetyPriorities,
    actionSummaries: fallback.actionSummaries.map((item, index) => {
      const candidate = summaryById.get(String(item.videoId)) || source.actionSummaries?.[index] || {};
      return {
        ...item,
        actionName: cleanText(candidate.actionName || item.actionName, 120),
        videoId: videos[index]?.id || item.videoId
      };
    }),
    volumeAnalysis: fallback.volumeAnalysis,
    historyComparison: fallback.historyComparison,
    positives: fallback.positives,
    improvements: fallback.improvements,
    redFlags: fallback.redFlags,
    progressionConditions: fallback.progressionConditions,
    filmingAdvice: [],
    conversationPoints: fallback.conversationPoints,
    coachClose: fallback.coachClose
  };
}

async function generateNarrative(client, job, analyses, history) {
  const fallback = fallbackNarrative(job, analyses, history);
  return {
    narrative: fallback,
    source: client ? "action_review_reuse" : "rule_fallback"
  };
}

function listHtml(items, emptyText = "暂无") {
  const values = cleanList(items, 20, 1000);
  if (!values.length) return `<p class="muted">${escapeHtml(emptyText)}</p>`;
  return `<ul>${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function reportStyles() {
  return `
    :root{--ink:#172019;--muted:#647066;--line:#d9e1d6;--panel:#f4f7f1;--accent:#476c35;--acid:#d9ff43;--red:#9e382b;--yellow:#8b651f}
    *{box-sizing:border-box}body{margin:0;background:#edf1e9;color:var(--ink);font-family:"Microsoft YaHei UI",sans-serif;line-height:1.72}
    main{width:min(1040px,calc(100% - 28px));margin:20px auto 60px;background:white;padding:clamp(20px,4vw,54px);box-shadow:0 24px 80px rgba(23,32,25,.12)}
    header{border-bottom:3px solid var(--ink);padding-bottom:22px}h1{font-size:clamp(28px,6vw,48px);line-height:1.1;margin:0 0 16px}h2{font-size:24px;border-top:1px solid var(--line);padding-top:26px;margin-top:36px}
    h3{margin:0 0 10px}.meta{display:flex;flex-wrap:wrap;gap:8px 22px;color:var(--muted);font-size:14px}.score{display:inline-flex;background:var(--acid);padding:7px 12px;font-weight:800;margin:12px 0}
    table{width:100%;border-collapse:collapse;margin:12px 0}th,td{border:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}th{background:var(--panel)}
    .priority{border-left:5px solid var(--yellow);padding:13px 15px;background:#fff9e9;margin:10px 0}.priority.red{border-color:var(--red);background:#fff0ed}.priority.green{border-color:var(--accent);background:#eff8ea}
    .action-card{border:1px solid var(--line);margin:18px 0;overflow:hidden}.action-card>h3{padding:14px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
    .action-body{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:18px;padding:16px}.action-body img{width:100%;height:auto;border:1px solid var(--line)}
    .technical-section{padding:12px 0;border-top:1px solid var(--line)}.technical-section h4{margin:0 0 6px}.technical-section p{margin:6px 0}.recommendation{border-left:4px solid var(--yellow);background:#fff9e9;padding:12px 14px;margin:10px 0}.recommendation.high{border-color:var(--red);background:#fff0ed}.recommendation p{margin:6px 0}
    .muted{color:var(--muted)}.callout{background:#eff8ea;border-left:5px solid var(--accent);padding:14px 16px}.warn{background:#fff7e8;border-left:5px solid var(--yellow);padding:14px 16px}
    @media(max-width:700px){main{width:100%;margin:0;padding:20px 14px}.action-body{display:block;padding:12px}.action-body img{margin-bottom:12px}table{font-size:13px}th,td{padding:8px}}
  `;
}

function renderHtmlReport({ job, analyses, narrative, assetUrl }) {
  const scoreValues = analyses.map((item) => number(item.overallScore)).filter((item) => item > 0);
  const averageScore = scoreValues.length
    ? Math.round(scoreValues.reduce((sum, item) => sum + item, 0) / scoreValues.length)
    : 0;

  const safetyHtml = narrative.safetyPriorities.map((item) => `
    <article class="priority ${escapeHtml(item.level)}">
      <h3>${severityLabel(item.level)} ${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.detail)}</p>
      <strong>处理方式：</strong> ${escapeHtml(item.next)}
    </article>`).join("");

  const actionRows = analyses.map((analysis, index) => {
    const video = job.videos[index] || {};
    return `<tr><td>${index + 1}</td><td>${escapeHtml(analysis.actionName)}</td><td>${escapeHtml(video.loadKg || "-")}</td><td>${escapeHtml(video.sets || "-")} × ${escapeHtml(video.reps || "-")}</td><td>${escapeHtml(analysis.captureQuality)}</td></tr>`;
  }).join("");

  const actionCards = narrative.actionSummaries.map((summary, index) => {
    const analysis = analyses[index] || {};
    const image = analysis.contactSheet
      ? `<img src="${escapeHtml(assetUrl(job.videos[index]?.id, analysis.contactSheet))}" alt="${escapeHtml(summary.actionName)}四阶段动作复盘">`
      : "<p class=\"muted\">当前没有可用的四阶段画面。</p>";
    const coachNarrative = summary.movementAnalysis || summary.mainAdjustment || summary.nextSetPlan || summary.headline
      ? `<div class="coach-narrative">
          ${summary.headline ? `<h4>${escapeHtml(summary.headline)}</h4>` : ""}
          <p>${escapeHtml(summary.overview || summary.summary)}</p>
          ${summary.movementAnalysis ? `<p>${escapeHtml(summary.movementAnalysis)}</p>` : ""}
          ${summary.mainAdjustment ? `<p><strong>最值得先改的一点：</strong>${escapeHtml(summary.mainAdjustment)}</p>` : ""}
          ${summary.nextSetPlan ? `<p><strong>下一组可以这样试：</strong>${escapeHtml(summary.nextSetPlan)}</p>` : ""}
        </div>`
      : "";
    const technicalSections = coachNarrative || (summary.sections?.length
      ? `<div class="technical-sections">${summary.sections.map((section) => `
          <section class="technical-section">
            <h4>${escapeHtml(section.title)}</h4>
            <p>${escapeHtml(section.assessment)}</p>
            ${section.interpretation ? `<p class="muted"><strong>技术含义：</strong>${escapeHtml(section.interpretation)}</p>` : ""}
            ${section.evidence ? `<p class="muted"><strong>判断依据：</strong>${escapeHtml(section.evidence)}</p>` : ""}
          </section>`).join("")}</div>`
      : `<strong>已确认表现</strong>${listHtml(summary.positives)}<strong>主要修正</strong>${listHtml(summary.improvements)}`);
    const recommendations = summary.recommendations?.length
      ? `<h4>优先修正与验收标准</h4>${summary.recommendations.map((item, itemIndex) => `
          <div class="recommendation ${escapeHtml(item.priority)}">
            <strong>${itemIndex + 1}. ${escapeHtml(item.title)}</strong>
            ${item.rationale ? `<p><strong>为什么：</strong>${escapeHtml(item.rationale)}</p>` : ""}
            <p><strong>怎么做：</strong>${escapeHtml(item.execution)}</p>
            ${item.successCriteria ? `<p><strong>完成标准：</strong>${escapeHtml(item.successCriteria)}</p>` : ""}
          </div>`).join("")}`
      : "";
    return `<article class="action-card">
      <h3>${index + 1}. ${escapeHtml(summary.actionName)}</h3>
      <div class="action-body">
        <div>${image}</div>
        <div>
          ${coachNarrative ? "" : `<p>${escapeHtml(summary.summary)}</p>`}
          ${technicalSections}
          ${recommendations}
          ${summary.coachingCues?.length ? `<p class="callout"><strong>执行口令：</strong>${escapeHtml(summary.coachingCues.join("；"))}</p>` : ""}
          ${!summary.nextSetPlan && summary.progressionCriteria ? `<p class="warn"><strong>负荷与进阶判断：</strong>${escapeHtml(summary.progressionCriteria)}</p>` : ""}
          ${summary.evidenceLimits?.length ? `<p class="muted"><strong>证据边界：</strong>${escapeHtml(summary.evidenceLimits.join("；"))}</p>` : ""}
        </div>
      </div>
    </article>`;
  }).join("");

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escapeHtml(narrative.title)}</title><style>${reportStyles()}</style></head>
<body><main>
  <header><h1>${escapeHtml(narrative.title)}</h1><div class="meta"><span>训练者：${escapeHtml(job.traineeName || "本地用户")}</span><span>日期：${escapeHtml(job.trainingDate || job.createdAt?.slice(0, 10))}</span><span>模式：${job.mode === "single" ? "单动作评估" : "训练日复盘"}</span></div>${averageScore ? `<div class="score">动作平均评分 ${averageScore}</div>` : ""}</header>
  <section><h2>总评</h2><p>${escapeHtml(narrative.overall)}</p></section>
  <section><h2>安全优先级</h2>${safetyHtml}</section>
  <section><h2>本次动作内容</h2><table><thead><tr><th>#</th><th>动作</th><th>重量 kg</th><th>组 × 次</th><th>画面质量</th></tr></thead><tbody>${actionRows}</tbody></table></section>
  <section><h2>重点动作图文复盘</h2>${actionCards}</section>
  <section><h2>动作修正与进阶条件</h2>${listHtml(narrative.progressionConditions)}</section>
  <section><h2>教练收束</h2><div class="warn">${escapeHtml(narrative.coachClose)}</div></section>
  <section><h2>下次训练重点</h2>${listHtml(narrative.conversationPoints)}</section>
</main></body></html>`;
}

function renderMarkdownReport({ job, analyses, narrative }) {
  const movementNotes = narrative.actionSummaries.map((item, index) => {
    const lines = [
      `### ${index + 1}. ${item.actionName}`,
      "",
      ...(item.headline ? [`**${item.headline}**`, ""] : []),
      item.overview || item.summary,
      ""
    ];
    if (item.movementAnalysis) lines.push(item.movementAnalysis, "");
    if (item.mainAdjustment) lines.push(`**最值得先改的一点：** ${item.mainAdjustment}`, "");
    if (item.nextSetPlan) lines.push(`**下一组可以这样试：** ${item.nextSetPlan}`, "");
    if (!item.movementAnalysis && item.sections?.length) {
      lines.push("#### 分维度技术分析", "");
      item.sections.forEach((section) => {
        lines.push(`**${section.title}**`, "", section.assessment);
        if (section.interpretation) lines.push("", `- 技术含义：${section.interpretation}`);
        if (section.evidence) lines.push(`- 判断依据：${section.evidence}`);
        lines.push("");
      });
    } else if (!item.movementAnalysis) {
      lines.push("**已确认表现**", ...(item.positives.length ? item.positives.map((text) => `- ${text}`) : ["- 暂无可确认项"]), "");
      lines.push("**主要修正**", ...(item.improvements.length ? item.improvements.map((text) => `- ${text}`) : ["- 暂无明确修正项"]), "");
    }
    if (item.recommendations?.length) {
      lines.push("#### 优先修正与验收标准", "");
      item.recommendations.forEach((recommendation, recommendationIndex) => {
        lines.push(`${recommendationIndex + 1}. **${recommendation.title}**`);
        if (recommendation.rationale) lines.push(`   - 为什么：${recommendation.rationale}`);
        lines.push(`   - 怎么做：${recommendation.execution}`);
        if (recommendation.successCriteria) lines.push(`   - 完成标准：${recommendation.successCriteria}`);
      });
      lines.push("");
    }
    if (item.coachingCues?.length) lines.push(`**执行口令：** ${item.coachingCues.join("；")}`, "");
    if (!item.nextSetPlan && item.progressionCriteria) lines.push(`**负荷与进阶判断：** ${item.progressionCriteria}`, "");
    if (item.evidenceLimits?.length) lines.push(`**证据边界：** ${item.evidenceLimits.join("；")}`);
    return lines.join("\n");
  }).join("\n\n");

  return `# ${narrative.title}

- 训练者：${job.traineeName || "本地用户"}
- 日期：${job.trainingDate || job.createdAt?.slice(0, 10)}
- 模式：${job.mode === "single" ? "单动作评估" : "训练日复盘"}

## 总体复盘

${narrative.overall}

## 动作逐项复盘

${movementNotes}

## 需要优先处理的问题

${narrative.redFlags.map((item) => `- ${item}`).join("\n")}

## 阶段性建议与进阶条件

${narrative.progressionConditions.map((item) => `- ${item}`).join("\n")}

## 沟通要点

${narrative.conversationPoints.map((item) => `- ${item}`).join("\n")}

## 教练收束

${narrative.coachClose}
`;
}

function validateReport(validatorPython, validatorScript, reportPath) {
  if (!validatorPython || !validatorScript || !fs.existsSync(validatorScript)) {
    return { ok: true, output: "Validator unavailable" };
  }
  const result = spawnSync(validatorPython, [validatorScript, reportPath], {
    encoding: "utf8",
    timeout: 30000,
    windowsHide: true
  });
  return {
    ok: result.status === 0,
    output: `${result.stdout || ""}${result.stderr || ""}`.trim()
  };
}

async function generateReports(options) {
  const {
    client,
    job,
    analyses,
    history,
    reportDirectory,
    assetUrl,
    validatorPython,
    validatorScript
  } = options;
  const generated = await generateNarrative(client, job, analyses, history);
  fs.mkdirSync(reportDirectory, { recursive: true });
  const htmlPath = path.join(reportDirectory, "report.html");
  const markdownPath = path.join(reportDirectory, "report.md");
  fs.writeFileSync(htmlPath, renderHtmlReport({
    job,
    analyses,
    narrative: generated.narrative,
    assetUrl
  }), "utf8");
  fs.writeFileSync(markdownPath, renderMarkdownReport({
    job,
    analyses,
    narrative: generated.narrative
  }), "utf8");

  const htmlValidation = validateReport(validatorPython, validatorScript, htmlPath);
  const markdownValidation = validateReport(validatorPython, validatorScript, markdownPath);
  if (!htmlValidation.ok || !markdownValidation.ok) {
    throw new Error(`报告校验失败：${htmlValidation.output}\n${markdownValidation.output}`);
  }

  return {
    ...generated,
    narrative: generated.narrative,
    files: {
      html: "report.html",
      markdown: "report.md"
    },
    validation: {
      html: htmlValidation.output,
      markdown: markdownValidation.output
    }
  };
}

module.exports = {
  FORBIDDEN_TERMS,
  calculateVideoVolume,
  fallbackNarrative,
  generateReports,
  normalizeNarrative,
  renderHtmlReport,
  renderMarkdownReport,
  summarizeVolume
};
