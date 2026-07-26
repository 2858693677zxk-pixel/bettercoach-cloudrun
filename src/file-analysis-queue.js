"use strict";

const fs = require("node:fs");
const path = require("node:path");

function ensureDirectories(root) {
  fs.mkdirSync(path.join(root, "pending"), { recursive: true });
  fs.mkdirSync(path.join(root, "active"), { recursive: true });
}

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function atomicWriteJson(filePath, value) {
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2), "utf8");
  fs.renameSync(temporary, filePath);
}

function queueKey(userKey, jobId) {
  return `${userKey}:${jobId}`;
}

function itemFileName(item) {
  const enqueuedAt = Number(item.enqueuedAt) || Date.now();
  return `${String(enqueuedAt).padStart(13, "0")}-${item.userKey}-${item.jobId}.json`;
}

function listJsonFiles(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => path.join(directory, entry.name));
}

function createFileAnalysisQueue({ storage, config, workerId = `worker-${process.pid}` }) {
  const root = path.join(config.dataDirectory, "analysis-queue");
  const pendingDirectory = path.join(root, "pending");
  const activeDirectory = path.join(root, "active");
  const maxQueuedJobs = Math.max(1, Number(config.analysisMaxQueuedJobs) || 32);
  const maxPendingPerUser = Math.max(1, Number(config.analysisMaxPendingPerUser) || 8);
  const maxConcurrentPerUser = Math.max(1, Number(config.analysisMaxConcurrentPerUser) || 1);
  ensureDirectories(root);

  function readItems(directory) {
    return listJsonFiles(directory)
      .map((filePath) => {
        const item = safeReadJson(filePath);
        if (!item || !item.userKey || !item.jobId) return null;
        return { ...item, filePath };
      })
      .filter(Boolean)
      .sort((left, right) => (Number(left.enqueuedAt) || 0) - (Number(right.enqueuedAt) || 0));
  }

  function pendingItems() {
    return readItems(pendingDirectory);
  }

  function activeItems() {
    return readItems(activeDirectory);
  }

  function allItems() {
    return [...pendingItems(), ...activeItems()];
  }

  function pendingForUser(userKey) {
    return allItems().filter((item) => item.userKey === userKey).length;
  }

  function hasItem(userKey, jobId) {
    const key = queueKey(userKey, jobId);
    return allItems().some((item) => queueKey(item.userKey, item.jobId) === key);
  }

  function capacityForUser(userKey) {
    if (pendingForUser(userKey) >= maxPendingPerUser) {
      return { accepted: false, code: "USER_ANALYSIS_LIMIT" };
    }
    if (pendingItems().length >= maxQueuedJobs) {
      return { accepted: false, code: "ANALYSIS_QUEUE_FULL" };
    }
    return { accepted: true, code: null };
  }

  function enqueueByKey(userKey, jobId, enqueueOptions = {}) {
    if (hasItem(userKey, jobId)) return false;
    if (!enqueueOptions.recovery && !capacityForUser(userKey).accepted) return false;
    const item = {
      userKey,
      jobId,
      enqueuedAt: Date.now(),
      attempts: Number(enqueueOptions.attempts) || 0
    };
    atomicWriteJson(path.join(pendingDirectory, itemFileName(item)), item);
    return true;
  }

  function enqueue(userId, jobId) {
    return enqueueByKey(storage.userKey(userId), jobId);
  }

  function activeByUser() {
    return activeItems().reduce((counts, item) => {
      counts.set(item.userKey, (counts.get(item.userKey) || 0) + 1);
      return counts;
    }, new Map());
  }

  function claimNext() {
    const counts = activeByUser();
    const candidate = pendingItems().find((item) =>
      (counts.get(item.userKey) || 0) < maxConcurrentPerUser
    );
    if (!candidate) return null;
    const claimed = {
      userKey: candidate.userKey,
      jobId: candidate.jobId,
      enqueuedAt: candidate.enqueuedAt,
      attempts: Number(candidate.attempts) || 0,
      workerId,
      claimedAt: Date.now()
    };
    const activePath = path.join(activeDirectory, `${workerId}-${path.basename(candidate.filePath)}`);
    try {
      fs.renameSync(candidate.filePath, activePath);
      atomicWriteJson(activePath, claimed);
      return { ...claimed, filePath: activePath };
    } catch {
      return null;
    }
  }

  function complete(item) {
    if (item?.filePath) fs.rmSync(item.filePath, { force: true });
  }

  function fail(item) {
    complete(item);
  }

  function recover() {
    const jobs = storage.recoverInterruptedJobs();
    jobs.forEach((item) => enqueueByKey(item.userKey, item.jobId, { recovery: true }));
    return jobs.length;
  }

  function status() {
    const pending = pendingItems();
    const active = activeItems();
    return {
      running: active.length > 0,
      active: active.length,
      concurrency: Number(config.analysisConcurrency) || 1,
      maxConcurrentPerUser,
      queued: pending.length,
      maxQueued: maxQueuedJobs,
      availableQueueSlots: Math.max(0, maxQueuedJobs - pending.length),
      overloaded: pending.length >= maxQueuedJobs,
      queueMode: "filesystem"
    };
  }

  return {
    enqueue,
    enqueueByKey,
    canAccept(userId) {
      return capacityForUser(storage.userKey(userId));
    },
    claimNext,
    complete,
    fail,
    recover,
    status
  };
}

module.exports = {
  createFileAnalysisQueue
};
