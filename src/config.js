const fs = require("node:fs");
const path = require("node:path");

function unquote(value) {
  const trimmed = value.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === "\"" && last === "\"") || (first === "'" && last === "'")) {
      return trimmed.slice(1, -1);
    }
  }
  return trimmed;
}

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return {};

  return fs.readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .reduce((values, line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) return values;
      const separator = trimmed.indexOf("=");
      if (separator <= 0) return values;
      const key = trimmed.slice(0, separator).trim();
      const value = unquote(trimmed.slice(separator + 1));
      if (/^[A-Z][A-Z0-9_]*$/.test(key)) values[key] = value;
      return values;
    }, {});
}

function boundedInteger(value, minimum, maximum, fallback) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) {
    return fallback;
  }
  return number;
}

function boundedNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < minimum || number > maximum) {
    return fallback;
  }
  return number;
}

function truthy(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(String(value).trim().toLowerCase());
}

function option(value, allowed, fallback) {
  const normalized = String(value || "").trim().toLowerCase();
  return allowed.includes(normalized) ? normalized : fallback;
}

function resolveConfig(environment = {}, rootDirectory = path.resolve(__dirname, "..")) {
  const maxUploadMb = boundedInteger(environment.MAX_UPLOAD_MB, 1, 2048, 250);
  return {
    apiKey: String(environment.DEEPSEEK_API_KEY || "").trim(),
    baseUrl: String(environment.DEEPSEEK_BASE_URL || "https://api.deepseek.com")
      .trim()
      .replace(/\/+$/, ""),
    model: String(environment.DEEPSEEK_MODEL || "deepseek-v4-flash").trim(),
    timeoutMs: boundedInteger(environment.DEEPSEEK_TIMEOUT_MS, 1000, 120000, 30000),
    zhipuApiKey: String(environment.ZHIPU_API_KEY || "").trim(),
    zhipuBaseUrl: String(environment.ZHIPU_BASE_URL || "https://open.bigmodel.cn/api/paas/v4")
      .trim()
      .replace(/\/+$/, ""),
    zhipuVisionModel: String(environment.ZHIPU_VISION_MODEL || "glm-4.6v-flash").trim(),
    zhipuTimeoutMs: boundedInteger(environment.ZHIPU_TIMEOUT_MS, 1000, 120000, 30000),
    visionArbitrationEnabled: truthy(environment.VISION_ARBITRATION_ENABLED, false),
    glmVideoReviewEnabled: truthy(environment.GLM_VIDEO_REVIEW_ENABLED, false),
    port: boundedInteger(environment.PORT, 1, 65535, 8788),
    host: String(environment.HOST || "127.0.0.1").trim(),
    httpsCertFile: String(environment.HTTPS_CERT_FILE || "").trim(),
    httpsKeyFile: String(environment.HTTPS_KEY_FILE || "").trim(),
    signingSecret: String(environment.AI_COACH_SIGNING_SECRET || "").trim(),
    analyzerPython: String(environment.ANALYZER_PYTHON || "python").trim(),
    analyzerSampleFps: boundedNumber(environment.ANALYSIS_SAMPLE_FPS, 4, 15, 10),
    analysisConcurrency: boundedInteger(environment.ANALYSIS_CONCURRENCY, 1, 8, 2),
    analysisMaxQueuedJobs: boundedInteger(environment.ANALYSIS_MAX_QUEUED_JOBS, 1, 500, 32),
    analysisMaxPendingPerUser: boundedInteger(environment.ANALYSIS_MAX_PENDING_PER_USER, 1, 100, 8),
    analysisMaxConcurrentPerUser: boundedInteger(environment.ANALYSIS_MAX_CONCURRENT_PER_USER, 1, 8, 1),
    analysisQueueMode: option(environment.ANALYSIS_QUEUE_MODE, ["memory", "filesystem"], "memory"),
    annotatedVideoMode: option(environment.ANNOTATED_VIDEO_MODE, ["selected", "all", "none"], "selected"),
    realtimeAnalysisEnabled: truthy(environment.REALTIME_ANALYSIS_ENABLED, false),
    realtimePoseEngine: option(
      environment.REALTIME_POSE_ENGINE,
      ["mediapipe-web", "tfjs-movenet", "server"],
      "mediapipe-web"
    ),
    realtimeMaxFps: boundedInteger(environment.REALTIME_MAX_FPS, 1, 30, 12),
    realtimeFeedbackMinIntervalMs: boundedInteger(
      environment.REALTIME_FEEDBACK_MIN_INTERVAL_MS,
      500,
      10000,
      2500
    ),
    poseBackend: String(environment.POSE_BACKEND || "rtmlib").trim().toLowerCase(),
    poseEngineCompare: truthy(environment.POSE_ENGINE_COMPARE, false),
    poseRecheckLowConfidence: String(environment.POSE_RECHECK_LOW_CONFIDENCE || "true").trim() !== "false",
    poseRecheckMaxWindows: boundedInteger(environment.POSE_RECHECK_MAX_WINDOWS, 0, 100, 8),
    mmposeConfig: String(environment.MMPOSE_CONFIG || "").trim(),
    mmposeCheckpoint: String(environment.MMPOSE_CHECKPOINT || "").trim(),
    rtmlibDetModel: String(environment.RTMLIB_DET_MODEL || "").trim(),
    rtmlibPoseModel: String(environment.RTMLIB_POSE_MODEL || "").trim(),
    rtmlibMode: String(environment.RTMLIB_MODE || "balanced").trim(),
    rtmlibBackend: String(environment.RTMLIB_BACKEND || "onnxruntime").trim(),
    rtmlibDevice: String(environment.RTMLIB_DEVICE || "cuda").trim(),
    rtmlibOneStage: String(environment.RTMLIB_ONE_STAGE || "false").trim() === "true",
    rtmlibPoseInputSize: String(environment.RTMLIB_POSE_INPUT_SIZE || "").trim(),
    rtmlibDetInputSize: String(environment.RTMLIB_DET_INPUT_SIZE || "").trim(),
    maxUploadBytes: maxUploadMb * 1024 * 1024,
    retentionDays: boundedInteger(environment.DATA_RETENTION_DAYS, 1, 3650, 30),
    dataDirectory: path.resolve(
      String(environment.DATA_DIRECTORY || path.join(rootDirectory, "data"))
    ),
    rootDirectory: path.resolve(rootDirectory)
  };
}

function loadConfig(rootDirectory = path.resolve(__dirname, "..")) {
  const local = loadEnvFile(path.join(rootDirectory, ".env.local"));
  return resolveConfig({ ...local, ...process.env }, rootDirectory);
}

module.exports = {
  loadEnvFile,
  resolveConfig,
  loadConfig
};
