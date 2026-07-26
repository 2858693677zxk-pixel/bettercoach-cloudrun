const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

class StorageError extends Error {
  constructor(message, code = "STORAGE_ERROR", statusCode = 500) {
    super(message);
    this.name = "StorageError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

function assertUserId(userId) {
  const value = String(userId || "").trim();
  if (!/^[a-zA-Z0-9_-]{16,100}$/.test(value)) {
    throw new StorageError("缺少有效的本地用户标识", "INVALID_USER_ID", 401);
  }
  return value;
}

function assertJobId(jobId) {
  const value = String(jobId || "").trim();
  if (!/^[a-f0-9-]{20,50}$/i.test(value)) {
    throw new StorageError("任务标识无效", "INVALID_JOB_ID", 400);
  }
  return value;
}

function userKey(userId) {
  return crypto.createHash("sha256").update(assertUserId(userId)).digest("hex").slice(0, 32);
}

function atomicWriteJson(filePath, value) {
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2), "utf8");
  fs.renameSync(temporary, filePath);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function createStorage(rootDirectory) {
  const root = path.resolve(rootDirectory);
  fs.mkdirSync(root, { recursive: true });

  function assertUserKey(value) {
    const normalized = String(value || "").trim();
    if (!/^[a-f0-9]{32}$/.test(normalized)) {
      throw new StorageError("内部用户标识无效", "INVALID_USER_KEY", 500);
    }
    return normalized;
  }

  function userDirectoryByKey(key) {
    return path.join(root, "users", assertUserKey(key));
  }

  function userDirectory(userId) {
    return userDirectoryByKey(userKey(userId));
  }

  function jobDirectoryByKey(key, jobId) {
    return path.join(userDirectoryByKey(key), "jobs", assertJobId(jobId));
  }

  function jobDirectory(userId, jobId) {
    return jobDirectoryByKey(userKey(userId), jobId);
  }

  function jobFileByKey(key, jobId) {
    return path.join(jobDirectoryByKey(key, jobId), "job.json");
  }

  function jobFile(userId, jobId) {
    return jobFileByKey(userKey(userId), jobId);
  }

  function createJob(userId, metadata) {
    const normalizedUserId = assertUserId(userId);
    const id = crypto.randomUUID();
    const directory = jobDirectory(normalizedUserId, id);
    fs.mkdirSync(path.join(directory, "uploads"), { recursive: true });
    fs.mkdirSync(path.join(directory, "artifacts"), { recursive: true });
    fs.mkdirSync(path.join(directory, "reports"), { recursive: true });
    const now = new Date().toISOString();
    const job = {
      id,
      userKey: userKey(normalizedUserId),
      mode: metadata.mode === "single" ? "single" : "training_day",
      title: String(metadata.title || "").trim().slice(0, 120),
      traineeName: String(metadata.traineeName || "").trim().slice(0, 80),
      trainingDate: String(metadata.trainingDate || "").trim().slice(0, 20),
      bodyPart: String(metadata.bodyPart || "").trim().slice(0, 80),
      notes: String(metadata.notes || "").trim().slice(0, 3000),
      availableEquipment: Array.isArray(metadata.availableEquipment)
        ? metadata.availableEquipment.slice(0, 60)
        : [],
      status: "uploading",
      progress: 0,
      phase: "正在接收视频",
      videos: [],
      analysis: null,
      report: null,
      error: null,
      createdAt: now,
      updatedAt: now
    };
    atomicWriteJson(jobFile(normalizedUserId, id), job);
    return { job, directory };
  }

  function getJob(userId, jobId) {
    const file = jobFile(userId, jobId);
    if (!fs.existsSync(file)) {
      throw new StorageError("分析任务不存在", "JOB_NOT_FOUND", 404);
    }
    return readJson(file);
  }

  function getJobByKey(key, jobId) {
    const file = jobFileByKey(key, jobId);
    if (!fs.existsSync(file)) {
      throw new StorageError("分析任务不存在", "JOB_NOT_FOUND", 404);
    }
    return readJson(file);
  }

  function updateJob(userId, jobId, patch) {
    const current = getJob(userId, jobId);
    const next = {
      ...current,
      ...patch,
      id: current.id,
      userKey: current.userKey,
      updatedAt: new Date().toISOString()
    };
    atomicWriteJson(jobFile(userId, jobId), next);
    return next;
  }

  function updateJobByKey(key, jobId, patch) {
    const current = getJobByKey(key, jobId);
    const next = {
      ...current,
      ...patch,
      id: current.id,
      userKey: current.userKey,
      updatedAt: new Date().toISOString()
    };
    atomicWriteJson(jobFileByKey(key, jobId), next);
    return next;
  }

  function listJobsByKey(key) {
    const jobsRoot = path.join(userDirectoryByKey(key), "jobs");
    if (!fs.existsSync(jobsRoot)) return [];
    return fs.readdirSync(jobsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => {
        const file = path.join(jobsRoot, entry.name, "job.json");
        if (!fs.existsSync(file)) return null;
        try {
          return readJson(file);
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)));
  }

  function listJobs(userId) {
    return listJobsByKey(userKey(userId));
  }

  function deleteJob(userId, jobId) {
    const directory = jobDirectory(userId, jobId);
    if (!fs.existsSync(directory)) {
      throw new StorageError("分析任务不存在", "JOB_NOT_FOUND", 404);
    }
    const expectedParent = path.join(userDirectory(userId), "jobs");
    const resolved = path.resolve(directory);
    if (!resolved.startsWith(`${path.resolve(expectedParent)}${path.sep}`)) {
      throw new StorageError("拒绝删除目标目录", "UNSAFE_DELETE", 500);
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  }

  function deleteUploadsByKey(key, jobId) {
    const directory = path.resolve(jobDirectoryByKey(key, jobId), "uploads");
    const expectedJobDirectory = path.resolve(jobDirectoryByKey(key, jobId));
    if (!directory.startsWith(`${expectedJobDirectory}${path.sep}`)) {
      throw new StorageError("拒绝删除任务目录之外的上传文件", "UNSAFE_DELETE", 500);
    }
    fs.rmSync(directory, { recursive: true, force: true });
    fs.mkdirSync(directory, { recursive: true });
  }

  function resolveJobPath(userId, jobId, ...segments) {
    const directory = path.resolve(jobDirectory(userId, jobId));
    const target = path.resolve(directory, ...segments);
    if (target !== directory && !target.startsWith(`${directory}${path.sep}`)) {
      throw new StorageError("拒绝访问任务目录之外的文件", "FORBIDDEN_PATH", 403);
    }
    return target;
  }

  function resolveJobPathByKey(key, jobId, ...segments) {
    const directory = path.resolve(jobDirectoryByKey(key, jobId));
    const target = path.resolve(directory, ...segments);
    if (target !== directory && !target.startsWith(`${directory}${path.sep}`)) {
      throw new StorageError("拒绝访问任务目录之外的文件", "FORBIDDEN_PATH", 403);
    }
    return target;
  }

  function recoverInterruptedJobs() {
    const usersRoot = path.join(root, "users");
    if (!fs.existsSync(usersRoot)) return [];
    const recovered = [];
    for (const userEntry of fs.readdirSync(usersRoot, { withFileTypes: true })) {
      if (!userEntry.isDirectory()) continue;
      const jobsRoot = path.join(usersRoot, userEntry.name, "jobs");
      if (!fs.existsSync(jobsRoot)) continue;
      for (const jobEntry of fs.readdirSync(jobsRoot, { withFileTypes: true })) {
        if (!jobEntry.isDirectory()) continue;
        const file = path.join(jobsRoot, jobEntry.name, "job.json");
        if (!fs.existsSync(file)) continue;
        try {
          const job = readJson(file);
          if (["queued", "processing", "reporting"].includes(job.status)) {
            const next = {
              ...job,
              status: "queued",
              phase: "服务重启后等待恢复",
              updatedAt: new Date().toISOString()
            };
            atomicWriteJson(file, next);
            recovered.push({ userKey: userEntry.name, jobId: job.id });
          }
        } catch {
          // A damaged job remains isolated and is not automatically retried.
        }
      }
    }
    return recovered;
  }

  function cleanupExpired(retentionDays) {
    if (!Number.isFinite(retentionDays) || retentionDays <= 0) return 0;
    const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000;
    let removed = 0;
    const usersRoot = path.join(root, "users");
    if (!fs.existsSync(usersRoot)) return removed;
    for (const userEntry of fs.readdirSync(usersRoot, { withFileTypes: true })) {
      if (!userEntry.isDirectory()) continue;
      const jobsRoot = path.join(usersRoot, userEntry.name, "jobs");
      if (!fs.existsSync(jobsRoot)) continue;
      for (const jobEntry of fs.readdirSync(jobsRoot, { withFileTypes: true })) {
        const directory = path.join(jobsRoot, jobEntry.name);
        const file = path.join(directory, "job.json");
        if (!jobEntry.isDirectory() || !fs.existsSync(file)) continue;
        try {
          const job = readJson(file);
          if (new Date(job.updatedAt || job.createdAt).getTime() < cutoff) {
            fs.rmSync(directory, { recursive: true, force: true });
            removed += 1;
          }
        } catch {
          // Do not delete unreadable records automatically.
        }
      }
    }
    return removed;
  }

  return {
    root,
    cleanupExpired,
    createJob,
    deleteJob,
    deleteUploadsByKey,
    getJob,
    getJobByKey,
    jobDirectory,
    jobDirectoryByKey,
    listJobs,
    listJobsByKey,
    recoverInterruptedJobs,
    resolveJobPath,
    resolveJobPathByKey,
    updateJob,
    updateJobByKey,
    userDirectory,
    userDirectoryByKey,
    userKey
  };
}

module.exports = {
  StorageError,
  assertJobId,
  assertUserId,
  createStorage,
  userKey
};
