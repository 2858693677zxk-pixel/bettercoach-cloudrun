const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const { pipeline } = require("node:stream");
const Busboy = require("busboy");

const { createAnalysisService } = require("./src/analysis-service");
const { verifyAiCoachToken } = require("./src/auth-token");
const { buildDeepSeekMessages, normalizeChatRequest } = require("./src/chat-service");
const { loadConfig } = require("./src/config");
const { createDeepSeekClient } = require("./src/deepseek-client");
const { CAMERA_ANGLES, EXERCISES, getExercise } = require("./src/exercise-catalog");
const { createFileAnalysisQueue } = require("./src/file-analysis-queue");
const { assertUserId, createStorage } = require("./src/storage");
const { normalizeTextEncoding } = require("./src/text-encoding");
const { createZhipuVisionClient } = require("./src/zhipu-vision-client");

const BODY_LIMIT = 64 * 1024;
const MAX_VIDEOS = 10;
const MOTION_EXPERIMENT_CASES = [
  { id: "lat_pulldown", title: "高位下拉", primaryCase: "lat_pulldown" },
  { id: "row", title: "坐姿划船", primaryCase: "row" },
  { id: "romanian_deadlift", title: "硬拉 / RDL 代理样本", primaryCase: "romanian_deadlift" }
];
const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp"
};
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".m4v", ".webm", ".avi"]);

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

class HttpError extends Error {
  constructor(message, code, statusCode) {
    super(message);
    this.name = "HttpError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff"
  });
  response.end(body);
}

function isNonEmptyFile(filePath) {
  if (!filePath) return false;
  try {
    const stat = fs.statSync(filePath);
    return stat.isFile() && stat.size > 0;
  } catch {
    return false;
  }
}

function parseByteRange(rangeHeader, size) {
  const match = String(rangeHeader || "").match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return null;
  let start = match[1] === "" ? null : Number(match[1]);
  let end = match[2] === "" ? null : Number(match[2]);
  if (
    (start !== null && (!Number.isInteger(start) || start < 0))
    || (end !== null && (!Number.isInteger(end) || end < 0))
  ) {
    return null;
  }
  if (start === null && end === null) return null;
  if (start === null) {
    const suffixLength = Math.min(end, size);
    start = size - suffixLength;
    end = size - 1;
  } else if (end === null || end >= size) {
    end = size - 1;
  }
  if (start >= size || end < start) {
    return { unsatisfiable: true };
  }
  return { start, end };
}

function sendFile(request, response, filePath, options = {}) {
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    throw new HttpError("File not found", "NOT_FOUND", 404);
  }
  const stat = fs.statSync(filePath);
  const range = parseByteRange(request.headers.range, stat.size);
  const headers = {
    "content-type": options.contentType
      || MIME_TYPES[path.extname(filePath).toLowerCase()]
      || "application/octet-stream",
    "content-length": stat.size,
    "accept-ranges": "bytes",
    "cache-control": options.cacheControl || "private, max-age=60",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer"
  };
  if (options.downloadName) {
    headers["content-disposition"] = `attachment; filename="${options.downloadName}"`;
  }
  if (range?.unsatisfiable) {
    response.writeHead(416, {
      ...headers,
      "content-range": `bytes */${stat.size}`,
      "content-length": 0
    });
    response.end();
    return;
  }
  if (range) {
    const length = range.end - range.start + 1;
    response.writeHead(206, {
      ...headers,
      "content-length": length,
      "content-range": `bytes ${range.start}-${range.end}/${stat.size}`
    });
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    pipeline(fs.createReadStream(filePath, { start: range.start, end: range.end }), response, () => {});
    return;
  }
  response.writeHead(200, headers);
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  pipeline(fs.createReadStream(filePath), response, () => {});
}

async function readJson(request) {
  let size = 0;
  let tooLarge = false;
  const chunks = [];
  for await (const chunk of request) {
    size += chunk.length;
    if (size > BODY_LIMIT) tooLarge = true;
    else chunks.push(chunk);
  }
  if (tooLarge) {
    throw new HttpError("璇锋眰鍐呭杩囧ぇ", "REQUEST_TOO_LARGE", 413);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  } catch {
    throw new HttpError("璇锋眰鍐呭涓嶆槸鏈夋晥 JSON", "INVALID_JSON", 400);
  }
}

function requestAccessToken(request, url) {
  const authorization = String(request.headers.authorization || "");
  const bearer = authorization.match(/^Bearer\s+(.+)$/i);
  return bearer ? bearer[1] : String(url.searchParams.get("access_token") || "");
}

function requestUserId(request, url, config) {
  const userId = assertUserId(request.headers["x-xiaoyu-user"] || url.searchParams.get("user"));
  verifyAiCoachToken(requestAccessToken(request, url), config.signingSecret, userId);
  return userId;
}

function resolveStaticPath(publicDirectory, pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    throw new HttpError("Invalid request path", "INVALID_PATH", 400);
  }
  const segments = decoded.replace(/\\/g, "/").split("/");
  if (segments.includes("..")) {
    throw new HttpError("Forbidden path", "FORBIDDEN_PATH", 403);
  }
  const relative = decoded === "/" ? "index.html" : decoded.replace(/^[/\\]+/, "");
  const root = path.resolve(publicDirectory);
  const filePath = path.resolve(root, relative);
  if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) {
    throw new HttpError("Forbidden path", "FORBIDDEN_PATH", 403);
  }
  return filePath;
}

function serveStatic(request, response, publicDirectory, pathname) {
  sendFile(request, response, resolveStaticPath(publicDirectory, pathname), {
    cacheControl: "no-cache"
  });
}

function safeFileName(filename) {
  const extension = path.extname(String(filename || "")).toLowerCase();
  return `${crypto.randomUUID()}${VIDEO_EXTENSIONS.has(extension) ? extension : ".mp4"}`;
}

function parseMultipart(request, options) {
  const temporaryDirectory = path.join(
    options.dataDirectory,
    "incoming",
    crypto.randomUUID()
  );
  fs.mkdirSync(temporaryDirectory, { recursive: true });

  return new Promise((resolve, reject) => {
    let busboy;
    try {
      busboy = Busboy({
        headers: request.headers,
        limits: {
          files: MAX_VIDEOS,
          fileSize: options.maxUploadBytes,
          fields: 20,
          fieldSize: BODY_LIMIT
        }
      });
    } catch {
      fs.rmSync(temporaryDirectory, { recursive: true, force: true });
      reject(new HttpError("Use multipart/form-data to upload videos", "INVALID_UPLOAD", 400));
      return;
    }

    const fields = {};
    const files = [];
    const writes = [];
    let totalBytes = 0;
    let failed = false;

    function fail(error) {
      if (failed) return;
      failed = true;
      reject(error);
    }

    busboy.on("field", (name, value) => {
      fields[name] = value;
    });
    busboy.on("file", (fieldName, stream, info) => {
      const extension = path.extname(info.filename || "").toLowerCase();
      if (!VIDEO_EXTENSIONS.has(extension) && !String(info.mimeType || "").startsWith("video/")) {
        stream.resume();
        fail(new HttpError("Only common video files are supported", "UNSUPPORTED_VIDEO", 415));
        return;
      }
      const storedFilename = safeFileName(info.filename);
      const temporaryPath = path.join(temporaryDirectory, storedFilename);
      const output = fs.createWriteStream(temporaryPath, { flags: "wx" });
      const record = {
        fieldName,
        originalName: path.basename(info.filename || "video"),
        mimeType: String(info.mimeType || "application/octet-stream"),
        storedFilename,
        temporaryPath,
        size: 0,
        truncated: false
      };
      files.push(record);
      stream.on("data", (chunk) => {
        record.size += chunk.length;
        totalBytes += chunk.length;
        if (totalBytes > options.maxUploadBytes * MAX_VIDEOS) {
          fail(new HttpError("涓婁紶瑙嗛鎬婚噺杩囧ぇ", "UPLOAD_TOO_LARGE", 413));
        }
      });
      stream.on("limit", () => {
        record.truncated = true;
        fail(new HttpError("鍗曚釜瑙嗛瓒呰繃涓婁紶澶у皬闄愬埗", "VIDEO_TOO_LARGE", 413));
      });
      writes.push(new Promise((writeResolve, writeReject) => {
        output.on("finish", writeResolve);
        output.on("error", writeReject);
      }));
      stream.pipe(output);
    });
    busboy.on("filesLimit", () => {
      fail(new HttpError(`At most ${MAX_VIDEOS} videos per upload`, "TOO_MANY_VIDEOS", 400));
    });
    busboy.on("error", (error) => fail(error));
    busboy.on("close", async () => {
      try {
        await Promise.all(writes);
        if (failed) return;
        resolve({ fields, files, temporaryDirectory });
      } catch (error) {
        fail(error);
      }
    });
    request.pipe(busboy);
  }).catch((error) => {
    fs.rmSync(temporaryDirectory, { recursive: true, force: true });
    throw error;
  });
}

function normalizeCloudUploadFile(value, index) {
  const item = typeof value === "string" ? { fileID: value } : (value || {});
  const fileID = String(item.fileID || item.cloudID || item.cloudId || "").trim();
  if (!fileID || fileID.length > 512 || !/^cloud:\/\//i.test(fileID)) {
    throw new HttpError("Cloud storage file ID is invalid", "INVALID_CLOUD_FILE_ID", 400);
  }
  const sourceName = String(
    item.name || item.filename || item.originalName || fileID.split("/").pop() || ""
  ).trim();
  const originalName = path.basename(sourceName || `video_${index}.mp4`);
  const extension = path.extname(originalName).toLowerCase();
  let tempFileURL = "";
  try {
    const parsed = new URL(String(item.tempFileURL || item.downloadURL || "").trim());
    if (parsed.protocol === "https:") tempFileURL = parsed.toString();
  } catch {
    tempFileURL = "";
  }
  return {
    fileID,
    tempFileURL,
    originalName: VIDEO_EXTENSIONS.has(extension) ? originalName : `${originalName || `video_${index}`}.mp4`,
    mimeType: String(item.mimeType || item.type || "video/mp4").slice(0, 100)
  };
}

function downloadCloudTempFile(url, targetPath, maximumBytes, redirects = 0) {
  if (!url) {
    return Promise.reject(new HttpError(
      "Cloud storage tempFileURL is required",
      "CLOUD_TEMP_URL_REQUIRED",
      400
    ));
  }
  if (redirects > 3) {
    return Promise.reject(new HttpError(
      "Cloud storage video redirect limit exceeded",
      "CLOUD_FILE_DOWNLOAD_FAILED",
      502
    ));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      if (error) {
        fs.rmSync(targetPath, { force: true });
        reject(error);
        return;
      }
      resolve(result);
    };
    const cloudRequest = https.get(url, (incoming) => {
      const statusCode = Number(incoming.statusCode || 0);
      if ([301, 302, 303, 307, 308].includes(statusCode) && incoming.headers.location) {
        incoming.resume();
        const redirected = new URL(incoming.headers.location, url).toString();
        downloadCloudTempFile(redirected, targetPath, maximumBytes, redirects + 1)
          .then((result) => finish(null, result))
          .catch((error) => finish(error));
        return;
      }
      if (statusCode < 200 || statusCode >= 300) {
        incoming.resume();
        finish(new HttpError(
          `Cloud storage video download failed (${statusCode})`,
          "CLOUD_FILE_DOWNLOAD_FAILED",
          502
        ));
        return;
      }
      const output = fs.createWriteStream(targetPath);
      let bytes = 0;
      incoming.on("data", (chunk) => {
        bytes += chunk.length;
        if (bytes > maximumBytes) {
          cloudRequest.destroy();
          incoming.destroy();
          output.destroy();
          finish(new HttpError("Video exceeds upload size limit", "VIDEO_TOO_LARGE", 413));
        }
      });
      incoming.on("error", (error) => finish(error));
      output.on("error", (error) => finish(error));
      output.on("finish", () => finish(null, { bytes }));
      incoming.pipe(output);
    });
    cloudRequest.setTimeout(120000, () => {
      cloudRequest.destroy();
      finish(new HttpError(
        "Cloud storage video download timed out",
        "CLOUD_FILE_DOWNLOAD_FAILED",
        504
      ));
    });
    cloudRequest.on("error", (error) => finish(error));
  });
}

function writeDownloadedCloudFile(targetPath, downloaded) {
  const content = downloaded && Object.prototype.hasOwnProperty.call(downloaded, "fileContent")
    ? downloaded.fileContent
    : downloaded;
  if (Buffer.isBuffer(content)) {
    fs.writeFileSync(targetPath, content);
    return;
  }
  if (content instanceof ArrayBuffer) {
    fs.writeFileSync(targetPath, Buffer.from(content));
    return;
  }
  if (ArrayBuffer.isView(content)) {
    fs.writeFileSync(targetPath, Buffer.from(content.buffer, content.byteOffset, content.byteLength));
    return;
  }
  if (typeof content === "string") {
    fs.writeFileSync(targetPath, content);
    return;
  }
  if (!fs.existsSync(targetPath)) {
    throw new HttpError("Cloud storage video download failed", "CLOUD_FILE_DOWNLOAD_FAILED", 502);
  }
}

async function parseCloudFileUpload(body, options) {
  const metadata = parseUploadMetadata(JSON.stringify(body && body.metadata || {}));
  const rawFiles = Array.isArray(body && body.files)
    ? body.files
    : Array.isArray(body && body.fileIDs)
      ? body.fileIDs
      : [];
  if (!rawFiles.length || rawFiles.length > MAX_VIDEOS || rawFiles.length !== metadata.videos.length) {
    throw new HttpError("Cloud video count does not match task metadata", "INVALID_VIDEO_COUNT", 400);
  }

  const temporaryDirectory = path.join(options.dataDirectory, "incoming", crypto.randomUUID());
  fs.mkdirSync(temporaryDirectory, { recursive: true });
  try {
    const files = [];
    for (let index = 0; index < rawFiles.length; index += 1) {
      const source = normalizeCloudUploadFile(rawFiles[index], index);
      const storedFilename = safeFileName(source.originalName);
      const temporaryPath = path.join(temporaryDirectory, storedFilename);
      if (source.tempFileURL) {
        await downloadCloudTempFile(source.tempFileURL, temporaryPath, options.maxUploadBytes);
      } else if (typeof options.cloudFileDownloader === "function") {
        const downloaded = await options.cloudFileDownloader(source.fileID, source, temporaryPath);
        writeDownloadedCloudFile(temporaryPath, downloaded);
      } else {
        throw new HttpError(
          "Cloud storage tempFileURL is required",
          "CLOUD_TEMP_URL_REQUIRED",
          400
        );
      }
      const stat = fs.statSync(temporaryPath);
      if (!stat.size) throw new HttpError("Cloud storage video is empty", "EMPTY_CLOUD_FILE", 400);
      if (stat.size > options.maxUploadBytes) {
        throw new HttpError("Video exceeds upload size limit", "VIDEO_TOO_LARGE", 413);
      }
      files.push({
        fieldName: `video_${index}`,
        originalName: source.originalName,
        mimeType: source.mimeType,
        storedFilename,
        temporaryPath,
        size: stat.size,
        truncated: false
      });
    }
    return { metadata, fields: { metadata: JSON.stringify(metadata) }, files, temporaryDirectory };
  } catch (error) {
    fs.rmSync(temporaryDirectory, { recursive: true, force: true });
    throw error;
  }
}

function parseUploadMetadata(value) {
  let metadata;
  try {
    metadata = JSON.parse(String(value || ""));
  } catch {
    throw new HttpError("Missing valid task metadata", "INVALID_METADATA", 400);
  }
  if (!metadata || typeof metadata !== "object" || !Array.isArray(metadata.videos)) {
    throw new HttpError("浠诲姟淇℃伅缂哄皯瑙嗛鍒楄〃", "INVALID_METADATA", 400);
  }
  if (!metadata.videos.length || metadata.videos.length > MAX_VIDEOS) {
    throw new HttpError(`Choose 1-${MAX_VIDEOS} videos`, "INVALID_VIDEO_COUNT", 400);
  }
  metadata.availableEquipment = normalizeAvailableEquipment(metadata.availableEquipment);
  return metadata;
}

function normalizeAvailableEquipment(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  return value.slice(0, 60).map((item) => {
    const equipmentId = String(item && (item.equipmentId || item.id) || "").trim().slice(0, 100);
    const name = String(item && item.name || "").trim().slice(0, 80);
    if (!equipmentId || !name || seen.has(equipmentId)) return null;
    seen.add(equipmentId);
    return {
      equipmentId,
      name,
      bodyPartCategory: String(item.bodyPartCategory || "").trim().slice(0, 30),
      bodyParts: (Array.isArray(item.bodyParts) ? item.bodyParts : [])
        .map((part) => String(part || "").trim().slice(0, 30))
        .filter(Boolean)
        .slice(0, 5),
      targetMuscles: String(item.targetMuscles || "").trim().slice(0, 200),
      locationLabel: String(item.locationLabel || "").trim().slice(0, 50),
      defaultSets: Math.max(1, Math.min(20, Number(item.defaultSets) || 4)),
      defaultReps: String(item.defaultReps || "10 reps").trim().slice(0, 20),
      defaultRestSeconds: Math.max(0, Math.min(600, Number(item.defaultRestSeconds) || 75)),
      pinnedNote: String(item.pinnedNote || "").trim().slice(0, 300)
    };
  }).filter(Boolean);
}

function normalizeVideo(metadata, file, index) {
  const requestedActionType = String(metadata.actionType || "auto_detect").trim();
  const autoDetect = ["", "auto", "auto_detect", "detect", "unknown"].includes(requestedActionType.toLowerCase());
  const exercise = autoDetect
    ? { id: "auto_detect", name: "自动识别动作" }
    : getExercise(requestedActionType);
  const number = (value, maximum) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? Math.min(parsed, maximum) : 0;
  };
  const normalizeTargetRoi = (value) => {
    const source = Array.isArray(value)
      ? value
      : value && typeof value === "object"
        ? [value.x, value.y, value.width, value.height]
        : [];
    const parsed = source.slice(0, 4).map(Number);
    if (parsed.length !== 4 || parsed.some((item) => !Number.isFinite(item))) return null;
    const [x, y, width, height] = parsed;
    if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > 1 || y + height > 1) return null;
    return { x, y, width, height };
  };
  const requestedAnnotatedVideoMode = String(metadata.annotatedVideoMode || "").trim().toLowerCase();
  const annotatedVideoMode = metadata.returnAnnotatedVideo === false
    ? "none"
    : ["selected", "all", "none"].includes(requestedAnnotatedVideoMode)
      ? requestedAnnotatedVideoMode
      : "";
  const requestedPoseEngineMode = String(metadata.poseEngineMode || "").trim().toLowerCase();
  const poseEngineMode = ["both", "mediapipe"].includes(requestedPoseEngineMode)
    ? requestedPoseEngineMode
    : ["rtmo", "rtmlib"].includes(requestedPoseEngineMode)
      ? "rtmo"
      : metadata.poseEngineCompare !== false ? "both" : "rtmo";
  return {
    id: crypto.randomUUID(),
    index,
    originalName: file.originalName,
    storedFilename: file.storedFilename,
    mimeType: file.mimeType,
    size: file.size,
    actionType: exercise.id,
    actionName: exercise.name,
    poseEngineMode,
    poseEngineCompare: metadata.poseEngineCompare !== false,
    annotatedVideoMode,
    cameraAngle: CAMERA_ANGLES.some((item) => item.id === metadata.cameraAngle)
      ? metadata.cameraAngle
      : "unknown",
    loadKg: number(metadata.loadKg, 2000),
    sets: number(metadata.sets, 100),
    reps: number(metadata.reps, 1000),
    rpe: number(metadata.rpe, 10),
    notes: String(metadata.notes || "").trim().slice(0, 1000),
    targetRoi: normalizeTargetRoi(metadata.targetRoi)
  };
}

function moveUploadedFiles(storage, userId, job, metadata, upload) {
  const byField = new Map(upload.files.map((file) => [file.fieldName, file]));
  const videos = metadata.videos.map((item, index) => {
    const file = byField.get(`video_${index}`);
    if (!file || !file.size || file.truncated) {
      throw new HttpError(`绗?${index + 1} 涓棰戞湭瀹屾暣涓婁紶`, "MISSING_VIDEO", 400);
    }
    const video = normalizeVideo(item || {}, file, index);
    const destination = storage.resolveJobPath(
      userId,
      job.id,
      "uploads",
      video.storedFilename
    );
    fs.renameSync(file.temporaryPath, destination);
    return video;
  });
  return videos;
}

const COUNT_RESULT_FIELDS = new Set([
  "repCount",
  "repCountSource",
  "repEvents",
  "repIndexes",
  "repSegmentation",
  "visualRepEstimate",
  "finalRepCount",
  "repCountReason",
  "primaryRepCount",
  "secondaryRepCount",
  "fusedRepCount",
  "counterRule"
]);

function stripCountResultFields(value) {
  if (typeof value === "string") return normalizeTextEncoding(value);
  if (Array.isArray(value)) return value.map(stripCountResultFields);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !COUNT_RESULT_FIELDS.has(key))
      .map(([key, item]) => [key, stripCountResultFields(item)])
  );
}

function publicJob(job, userId, accessToken = "") {
  const encodedUser = encodeURIComponent(userId);
  const accessQuery = accessToken
    ? `&access_token=${encodeURIComponent(accessToken)}`
    : "";
  const videos = (job.videos || []).map(({ storedFilename, ...video }) => video);
  const analysis = job.analysis
    ? {
      ...stripCountResultFields(job.analysis),
      videos: (job.analysis.videos || []).map((item, index) => {
        const videoId = videos[index]?.id;
        const base = `/api/analyses/${job.id}/artifacts/${videoId}`;
        const artifactUrl = (filename) => `${base}/${String(filename).split("/").map(encodeURIComponent).join("/")}?user=${encodedUser}${accessQuery}`;
        const decorateArtifacts = (source) => {
          const visibleItem = stripCountResultFields(source);
          const annotatedVideos = Object.fromEntries(
          Object.entries(source.annotatedVideos || {}).map(([key, value]) => {
            const info = typeof value === "string" ? { filename: value } : { ...value };
            return [key, {
              ...info,
              url: info.filename
                ? artifactUrl(info.filename)
                : null
            }];
          })
        );
        return {
          ...visibleItem,
          annotatedVideos,
          contactSheetUrl: source.contactSheet
            ? artifactUrl(source.contactSheet)
            : null,
          keyframes: (source.keyframes || []).map((frame) => ({
            ...frame,
            imageUrl: artifactUrl(frame.image)
          }))
        };
        };
        return {
          ...decorateArtifacts(item),
          engineResults: (item.engineResults || []).map(decorateArtifacts)
        };
      })
    }
    : null;
  return {
    ...stripCountResultFields(job),
    userKey: undefined,
    videos,
    analysis,
    report: job.report
      ? {
        ...normalizeTextEncoding(job.report),
        urls: {
          html: `/api/analyses/${job.id}/reports/report.html?user=${encodedUser}${accessQuery}`,
          markdown: `/api/analyses/${job.id}/reports/report.md?user=${encodedUser}${accessQuery}`
        }
      }
      : null
  };
}

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function publicMotionFile(caseId, directory, filename) {
  if (!filename || path.basename(filename) !== filename) return null;
  const filePath = path.join(directory, filename);
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
  return {
    filename,
    url: `/api/motion-tracker-experiments/${encodeURIComponent(caseId)}/files/${encodeURIComponent(filename)}`
  };
}

function compactPrimaryResult(result) {
  const video = result?.result || result;
  if (!video || typeof video !== "object") return null;
  return {
    actionType: video.actionType || null,
    actionName: video.actionName || null,
    score: video.overallScore ?? video.score ?? null,
    safetyLevel: video.safetyLevel || null,
    captureQuality: video.captureQuality || null,
    poseCoverage: video.measurements?.poseCoverage ?? video.metadata?.poseCoverage ?? null,
    averageConfidence: video.measurements?.averagePoseConfidence ?? video.confidence ?? null,
    issues: Array.isArray(video.issues) ? video.issues : []
  };
}

function publicMotionExperimentCase(config, caseInfo) {
  const directory = path.join(config.rootDirectory, "output", "motion_tracker_glm", caseInfo.id);
  const summary = safeReadJson(path.join(directory, "summary.json"));
  const glmReview = safeReadJson(path.join(directory, "glm_review.json"));
  const primary = compactPrimaryResult(safeReadJson(
    path.join(config.rootDirectory, "output", "primary_compare", caseInfo.primaryCase, "output.json")
  ));
  const exists = Boolean(summary);
  const keyframes = (Array.isArray(summary?.keyframes) ? summary.keyframes : [])
    .map((frame) => {
      const file = publicMotionFile(caseInfo.id, directory, frame.image);
      return file ? { ...frame, ...file } : null;
    })
    .filter(Boolean);
  const annotated = summary?.annotatedVideo?.filename
    ? publicMotionFile(caseInfo.id, directory, summary.annotatedVideo.filename)
    : null;
  const contactSheet = summary?.contactSheet
    ? publicMotionFile(caseInfo.id, directory, summary.contactSheet)
    : null;

  return {
    id: caseInfo.id,
    title: caseInfo.title,
    available: exists,
    summaryUrl: exists
      ? `/api/motion-tracker-experiments/${encodeURIComponent(caseInfo.id)}/files/summary.json`
      : null,
    glmReviewUrl: glmReview
      ? `/api/motion-tracker-experiments/${encodeURIComponent(caseInfo.id)}/files/glm_review.json`
      : null,
    annotatedVideo: annotated
      ? {
        ...annotated,
        codec: summary.annotatedVideo?.codec || null,
        browserOptimized: Boolean(summary.annotatedVideo?.browserOptimized)
      }
      : null,
    contactSheet,
    keyframes,
    motionTracker: summary
      ? {
        engine: summary.engine,
        actionType: summary.actionType,
        inputVideoName: summary.inputVideo ? path.basename(summary.inputVideo) : null,
        durationSeconds: summary.durationSeconds ?? null,
        fps: summary.fps ?? null,
        sampleFps: summary.sampleFps ?? null,
        sampledFrames: summary.sampledFrames ?? null,
        poseFrames: summary.poseFrames ?? null,
        poseCoverage: summary.poseCoverage ?? null,
        averageConfidence: summary.averageConfidence ?? null,
        score: summary.score ?? null,
        safetyLevel: summary.safetyLevel || null,
        elapsedSeconds: summary.elapsedSeconds ?? null,
        issues: Array.isArray(summary.issues) ? summary.issues : [],
        strengths: Array.isArray(summary.strengths) ? summary.strengths : [],
        metricSummary: summary.metricSummary || {}
      }
      : null,
    glm: glmReview
      ? {
        model: glmReview.model || null,
        usage: glmReview.usage || null,
        result: glmReview.result || null
      }
      : null,
    primary,
    note: caseInfo.id === "romanian_deadlift"
      ? "桌面未找到硬拉视频，本组使用项目 vendor 里的 dumbbell RDL 侧面样本作为硬拉类代理测试。"
      : null
  };
}

function listPublicMotionExperiments(config) {
  const docPath = path.join(config.rootDirectory, "docs", "motion-tracker-glm-action-debug.md");
  return {
    generatedAt: new Date().toISOString(),
    docUrl: fs.existsSync(docPath) ? "/api/motion-tracker-experiments/doc" : null,
    cases: MOTION_EXPERIMENT_CASES.map((item) => publicMotionExperimentCase(config, item))
  };
}

function createApplicationRequestHandler(options = {}) {
  const loadedConfig = options.config || loadConfig(__dirname);
  const config = {
    analyzerPython: "python",
    dataDirectory: path.join(__dirname, "data"),
    host: "127.0.0.1",
    httpsCertFile: "",
    httpsKeyFile: "",
    maxUploadBytes: 250 * 1024 * 1024,
    poseBackend: "rtmlib",
    poseEngineCompare: false,
    poseRecheckLowConfidence: true,
    poseRecheckMaxWindows: 8,
    mmposeConfig: "",
    mmposeCheckpoint: "",
    rtmlibDetModel: "",
    rtmlibPoseModel: "",
    rtmlibMode: "balanced",
    rtmlibBackend: "onnxruntime",
    rtmlibDevice: "cuda",
    rtmlibOneStage: false,
    rtmlibPoseInputSize: "",
    rtmlibDetInputSize: "",
    retentionDays: 30,
    analyzerSampleFps: 10,
    analysisConcurrency: 2,
    analysisMaxQueuedJobs: 32,
    analysisMaxPendingPerUser: 8,
    analysisMaxConcurrentPerUser: 1,
    analysisQueueMode: "memory",
    annotatedVideoMode: "selected",
    realtimeAnalysisEnabled: false,
    realtimePoseEngine: "mediapipe-web",
    realtimeMaxFps: 12,
    realtimeFeedbackMinIntervalMs: 2500,
    zhipuApiKey: "",
    zhipuBaseUrl: "https://open.bigmodel.cn/api/paas/v4",
    zhipuVisionModel: "glm-4.6v-flash",
    zhipuTimeoutMs: 30000,
    visionArbitrationEnabled: false,
    glmVideoReviewEnabled: false,
    rootDirectory: __dirname,
    ...loadedConfig
  };
  const client = options.client || createDeepSeekClient(config);
  const visionClient = options.visionClient
    || ((config.visionArbitrationEnabled || config.glmVideoReviewEnabled) && config.zhipuApiKey
      ? createZhipuVisionClient(config)
      : null);
  const storage = options.storage || createStorage(config.dataDirectory);
  const cloudFileDownloader = options.cloudFileDownloader || null;
  const analysisService = options.analysisService || (
    config.analysisQueueMode === "filesystem"
      ? createFileAnalysisQueue({ storage, config })
      : createAnalysisService({
        storage,
        config,
        client,
        visionClient
      })
  );
  function assertAnalysisCapacity(userId) {
    if (!analysisService.canAccept) return;
    const capacity = analysisService.canAccept(userId);
    if (capacity?.accepted) return;
    const perUser = capacity?.code === "USER_ANALYSIS_LIMIT";
    throw new HttpError(
      perUser
        ? "该用户等待中的分析任务过多，请等待已有任务完成后再上传"
        : "当前分析队列已满，请稍后重试",
      capacity?.code || "ANALYSIS_QUEUE_FULL",
      429
    );
  }
  const publicDirectory = path.join(config.rootDirectory, "public");
  storage.cleanupExpired(config.retentionDays);
  if (options.autoRecover) analysisService.recover();

  return async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");

      if (request.method === "GET" && url.pathname === "/api/health") {
        sendJson(response, 200, {
          ok: true,
          data: {
            status: "ready",
            model: config.model,
            apiKeyConfigured: Boolean(config.apiKey),
            analyzerConfigured: Boolean(config.analyzerPython),
            visionArbitration: {
              enabled: Boolean(config.visionArbitrationEnabled),
              mainPipelineEnabled: Boolean(config.glmVideoReviewEnabled),
              model: config.zhipuVisionModel,
              apiKeyConfigured: Boolean(config.zhipuApiKey)
            },
            videoAnalysis: {
              supported: Boolean(config.analyzerPython),
              poseBackend: config.poseBackend,
              poseEngineCompare: config.poseEngineCompare,
              analyzerSampleFps: config.analyzerSampleFps,
              annotatedVideoMode: config.annotatedVideoMode,
              poseRecheckLowConfidence: config.poseRecheckLowConfidence,
              poseRecheckMaxWindows: config.poseRecheckMaxWindows,
              rtmlibConfigured: isNonEmptyFile(config.rtmlibPoseModel),
              mmposeConfigured: Boolean(config.mmposeConfig || config.mmposeCheckpoint)
            },
            realtimeAnalysis: {
              enabled: Boolean(config.realtimeAnalysisEnabled),
              supported: config.realtimePoseEngine === "mediapipe-web",
              engine: config.realtimePoseEngine,
              maxFps: config.realtimeMaxFps,
              feedbackMinIntervalMs: config.realtimeFeedbackMinIntervalMs,
              voiceEnabled: false
            },
            queue: analysisService.status ? analysisService.status() : null
          }
        });
        return;
      }

      if (request.method === "GET" && url.pathname === "/api/catalog") {
        sendJson(response, 200, {
          ok: true,
          data: {
            exercises: EXERCISES,
            cameraAngles: CAMERA_ANGLES,
            maxVideos: MAX_VIDEOS,
            maxUploadMb: Math.round(config.maxUploadBytes / 1024 / 1024)
          }
        });
        return;
      }

      if (request.method === "POST" && url.pathname === "/api/chat") {
        if (config.signingSecret) requestUserId(request, url, config);
        const body = await readJson(request);
        const normalized = normalizeChatRequest(body);
        const result = await client.complete(buildDeepSeekMessages(normalized));
        sendJson(response, 200, {
          ok: true,
          data: {
            reply: result.content,
            model: result.model,
            usage: result.usage
          }
        });
        return;
      }

      if (request.method === "POST" && url.pathname === "/api/analyses") {
        const userId = requestUserId(request, url, config);
        const accessToken = requestAccessToken(request, url);
        assertAnalysisCapacity(userId);
        const upload = await parseMultipart(request, config);
        let created;
        try {
          const metadata = parseUploadMetadata(upload.fields.metadata);
          created = storage.createJob(userId, metadata);
          const videos = moveUploadedFiles(storage, userId, created.job, metadata, upload);
          const job = storage.updateJob(userId, created.job.id, {
            mode: metadata.mode === "single" ? "single" : "training_day",
            videos,
            status: "queued",
            progress: 4,
            phase: "瑙嗛宸叉帴鏀讹紝绛夊緟鍒嗘瀽",
            error: null,
            calculationLogs: [
              calculationLog(
                "upload",
                "接收上传视频",
                `已接收 ${videos.length} 个视频，任务进入本地分析队列。`,
                {
                  mode: metadata.mode === "single" ? "single" : "training_day",
                  title: metadata.title || "",
                  videos: videos.map((video) => ({
                    videoId: video.id,
                    originalName: video.originalName,
                    size: video.size,
                    mimeType: video.mimeType,
                    actionType: video.actionType,
                    actionName: video.actionName,
                    cameraAngle: video.cameraAngle,
                    loadKg: video.loadKg,
                    sets: video.sets,
                    reps: video.reps,
                    targetRoi: video.targetRoi
                  }))
                }
              )
            ]
          });
          if (analysisService.enqueue(userId, job.id) === false) {
            throw new HttpError("当前分析队列已满，请稍后重试", "ANALYSIS_QUEUE_FULL", 429);
          }
          sendJson(response, 202, { ok: true, data: publicJob(job, userId, accessToken) });
        } catch (error) {
          if (created) {
            try {
              storage.deleteJob(userId, created.job.id);
            } catch {
              // Ignore cleanup errors and return the original request error.
            }
          }
          throw error;
        } finally {
          fs.rmSync(upload.temporaryDirectory, { recursive: true, force: true });
        }
        return;
      }

      if (request.method === "POST" && url.pathname === "/api/analyses/cloud-files") {
        const userId = requestUserId(request, url, config);
        const accessToken = requestAccessToken(request, url);
        assertAnalysisCapacity(userId);
        const body = await readJson(request);
        let upload;
        let created;
        try {
          upload = await parseCloudFileUpload(body, {
            dataDirectory: config.dataDirectory,
            maxUploadBytes: config.maxUploadBytes,
            cloudFileDownloader
          });
          const metadata = upload.metadata;
          created = storage.createJob(userId, metadata);
          const videos = moveUploadedFiles(storage, userId, created.job, metadata, upload);
          const job = storage.updateJob(userId, created.job.id, {
            mode: metadata.mode === "single" ? "single" : "training_day",
            videos,
            status: "queued",
            progress: 4,
            phase: "云存储视频已接收，等待分析",
            error: null,
            calculationLogs: [
              calculationLog(
                "upload",
                "接收云存储视频",
                `已接收 ${videos.length} 个云存储视频，任务进入分析队列。`,
                {
                  mode: metadata.mode === "single" ? "single" : "training_day",
                  title: metadata.title || "",
                  videos: videos.map((video) => ({
                    videoId: video.id,
                    originalName: video.originalName,
                    size: video.size,
                    mimeType: video.mimeType,
                    actionType: video.actionType,
                    actionName: video.actionName,
                    cameraAngle: video.cameraAngle,
                    loadKg: video.loadKg,
                    sets: video.sets,
                    reps: video.reps,
                    targetRoi: video.targetRoi
                  }))
                }
              )
            ]
          });
          if (analysisService.enqueue(userId, job.id) === false) {
            throw new HttpError("当前分析队列已满，请稍后重试", "ANALYSIS_QUEUE_FULL", 429);
          }
          sendJson(response, 202, { ok: true, data: publicJob(job, userId, accessToken) });
        } catch (error) {
          if (created) {
            try {
              storage.deleteJob(userId, created.job.id);
            } catch {
              // Preserve the original upload error.
            }
          }
          throw error;
        } finally {
          if (upload) fs.rmSync(upload.temporaryDirectory, { recursive: true, force: true });
        }
        return;
      }

      if (request.method === "GET" && url.pathname === "/api/analyses") {
        const userId = requestUserId(request, url, config);
        const accessToken = requestAccessToken(request, url);
        sendJson(response, 200, {
          ok: true,
          data: storage.listJobs(userId).map((job) => publicJob(job, userId, accessToken))
        });
        return;
      }

      const analysisMatch = url.pathname.match(/^\/api\/analyses\/([a-f0-9-]+)$/i);
      if (analysisMatch && request.method === "GET") {
        const userId = requestUserId(request, url, config);
        const accessToken = requestAccessToken(request, url);
        sendJson(response, 200, {
          ok: true,
          data: publicJob(storage.getJob(userId, analysisMatch[1]), userId, accessToken)
        });
        return;
      }
      if (analysisMatch && request.method === "DELETE") {
        const userId = requestUserId(request, url, config);
        storage.deleteJob(userId, analysisMatch[1]);
        sendJson(response, 200, { ok: true, data: { deleted: true } });
        return;
      }

      const retryMatch = url.pathname.match(/^\/api\/analyses\/([a-f0-9-]+)\/retry$/i);
      if (retryMatch && request.method === "POST") {
        const userId = requestUserId(request, url, config);
        const accessToken = requestAccessToken(request, url);
        assertAnalysisCapacity(userId);
        const current = storage.getJob(userId, retryMatch[1]);
        if (current.status !== "failed") {
          throw new HttpError("Current job does not need retry", "JOB_NOT_RETRYABLE", 409);
        }
        const job = storage.updateJob(userId, current.id, {
          status: "queued",
          progress: 4,
          phase: "绛夊緟閲嶆柊鍒嗘瀽",
          error: null,
          analysis: null,
          report: null,
          calculationLogs: [
            ...(Array.isArray(current.calculationLogs) ? current.calculationLogs : []),
            calculationLog(
              "queue",
              "重新加入分析队列",
              "用户触发重试，旧分析结果已清空，任务等待重新计算。",
              {
                previousStatus: current.status,
                videoCount: (current.videos || []).length
              }
            )
          ].slice(-120)
        });
        if (analysisService.enqueue(userId, job.id) === false) {
          throw new HttpError("当前分析队列已满，请稍后重试", "ANALYSIS_QUEUE_FULL", 429);
        }
        sendJson(response, 202, { ok: true, data: publicJob(job, userId, accessToken) });
        return;
      }

      const feedbackMatch = url.pathname.match(/^\/api\/analyses\/([a-f0-9-]+)\/feedback$/i);
      if (feedbackMatch && request.method === "POST") {
        const userId = requestUserId(request, url, config);
        const body = await readJson(request);
        const current = storage.getJob(userId, feedbackMatch[1]);
        const feedback = {
          id: crypto.randomUUID(),
          type: ["helpful", "missed", "incorrect"].includes(body.type)
            ? body.type
            : "helpful",
          videoId: String(body.videoId || "").slice(0, 60),
          note: String(body.note || "").trim().slice(0, 1000),
          createdAt: new Date().toISOString()
        };
        const job = storage.updateJob(userId, current.id, {
          feedback: [...(current.feedback || []), feedback].slice(-50)
        });
        sendJson(response, 201, {
          ok: true,
          data: { feedback, count: job.feedback.length }
        });
        return;
      }

      const artifactMatch = url.pathname.match(
        /^\/api\/analyses\/([a-f0-9-]+)\/artifacts\/([a-f0-9-]+)(?:\/(rtmo|mediapipe))?\/([^/]+)$/i
      );
      if (artifactMatch && ["GET", "HEAD"].includes(request.method)) {
        const userId = requestUserId(request, url, config);
        storage.getJob(userId, artifactMatch[1]);
        const engine = artifactMatch[3] || null;
        const filename = decodeURIComponent(artifactMatch[4]);
        if (path.basename(filename) !== filename || !/\.(?:jpe?g|png|webp|mp4)$/i.test(filename)) {
          throw new HttpError("璇佹嵁鍥剧墖璺緞鏃犳晥", "INVALID_ARTIFACT", 400);
        }
        sendFile(
          request,
          response,
          storage.resolveJobPath(
            userId,
            artifactMatch[1],
            "artifacts",
            artifactMatch[2],
            ...(engine ? [engine] : []),
            filename
          )
        );
        return;
      }

      const reportMatch = url.pathname.match(
        /^\/api\/analyses\/([a-f0-9-]+)\/reports\/(report\.(?:html|md))$/i
      );
      if (reportMatch && ["GET", "HEAD"].includes(request.method)) {
        const userId = requestUserId(request, url, config);
        const job = storage.getJob(userId, reportMatch[1]);
        if (job.status !== "completed") {
          throw new HttpError("鎶ュ憡灏氭湭鐢熸垚", "REPORT_NOT_READY", 409);
        }
        const filePath = storage.resolveJobPath(userId, job.id, "reports", reportMatch[2]);
        if (reportMatch[2].endsWith(".html") && request.method === "GET") {
          const encodedUser = encodeURIComponent(userId);
          const accessToken = requestAccessToken(request, url);
          const accessQuery = accessToken
            ? `&access_token=${encodeURIComponent(accessToken)}`
            : "";
          const html = fs.readFileSync(filePath, "utf8").replace(
            /src="\.\.\/artifacts\/([^"]+)"/g,
            `src="../artifacts/$1?user=${encodedUser}${accessQuery}"`
          );
          response.writeHead(200, {
            "content-type": "text/html; charset=utf-8",
            "content-length": Buffer.byteLength(html),
            "cache-control": "no-store",
            "x-content-type-options": "nosniff"
          });
          response.end(html);
          return;
        }
        sendFile(request, response, filePath, {
          cacheControl: "no-store",
          downloadName: reportMatch[2].endsWith(".md") ? `xiaoyu-coach-${job.id}.md` : null
        });
        return;
      }

      if (request.method === "GET" && url.pathname === "/api/motion-tracker-experiments") {
        sendJson(response, 200, {
          ok: true,
          data: listPublicMotionExperiments(config)
        });
        return;
      }

      if (["GET", "HEAD"].includes(request.method) && url.pathname === "/api/motion-tracker-experiments/doc") {
        sendFile(
          request,
          response,
          path.join(config.rootDirectory, "docs", "motion-tracker-glm-action-debug.md"),
          { cacheControl: "no-store" }
        );
        return;
      }

      const motionFileMatch = url.pathname.match(
        /^\/api\/motion-tracker-experiments\/([^/]+)\/files\/([^/]+)$/i
      );
      if (motionFileMatch && ["GET", "HEAD"].includes(request.method)) {
        const caseId = decodeURIComponent(motionFileMatch[1]);
        const filename = decodeURIComponent(motionFileMatch[2]);
        const caseInfo = MOTION_EXPERIMENT_CASES.find((item) => item.id === caseId);
        if (!caseInfo) {
          throw new HttpError("Motion-tracker experiment not found", "NOT_FOUND", 404);
        }
        if (path.basename(filename) !== filename || !/\.(?:jpe?g|png|webp|mp4|json)$/i.test(filename)) {
          throw new HttpError("Invalid motion-tracker experiment file", "INVALID_ARTIFACT", 400);
        }
        sendFile(
          request,
          response,
          path.join(config.rootDirectory, "output", "motion_tracker_glm", caseInfo.id, filename),
          { cacheControl: filename.endsWith(".json") ? "no-store" : "private, max-age=60" }
        );
        return;
      }

      if (url.pathname.startsWith("/api/")) {
        throw new HttpError("API endpoint not found", "NOT_FOUND", 404);
      }
      if (["GET", "HEAD"].includes(request.method)) {
        serveStatic(request, response, publicDirectory, url.pathname);
        return;
      }
      throw new HttpError("resource not found", "NOT_FOUND", 404);
    } catch (error) {
      const statusCode = Number(error.statusCode) || 500;
      const exposed = statusCode < 500 || error.code;
      sendJson(response, statusCode, {
        ok: false,
        error: {
          code: exposed ? (error.code || "REQUEST_FAILED") : "INTERNAL_ERROR",
          message: exposed ? error.message : "Service temporarily unavailable"
        }
      });
    }
  };
}

function createApplicationServer(options = {}) {
  return http.createServer(createApplicationRequestHandler(options));
}

function createHttpsApplicationServer(options = {}) {
  const tls = options.tls || loadTlsOptions(options.config || loadConfig(__dirname));
  return https.createServer(tls, createApplicationRequestHandler(options));
}

function loadTlsOptions(config) {
  if (!config.httpsCertFile || !config.httpsKeyFile) return null;
  return {
    cert: fs.readFileSync(path.resolve(config.httpsCertFile)),
    key: fs.readFileSync(path.resolve(config.httpsKeyFile))
  };
}

if (require.main === module) {
  const config = loadConfig(__dirname);
  const tls = loadTlsOptions(config);
  const server = tls
    ? createHttpsApplicationServer({ config, autoRecover: true, tls })
    : createApplicationServer({ config, autoRecover: true });
  const protocol = tls ? "https" : "http";
  server.listen(config.port, config.host, () => {
    console.log(`Xiaoyu Coach is running at ${protocol}://${config.host}:${config.port}`);
  });
}

module.exports = {
  BODY_LIMIT,
  MAX_VIDEOS,
  createApplicationRequestHandler,
  createApplicationServer,
  createHttpsApplicationServer,
  loadTlsOptions,
  normalizeAvailableEquipment,
  parseCloudFileUpload,
  parseUploadMetadata,
  publicJob,
  resolveStaticPath
};


