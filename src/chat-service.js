const { buildCoachPrompt, buildDeveloperReviewPrompt } = require("./coach-prompt");

const MAX_MESSAGE_LENGTH = 8000;
const MAX_HISTORY_MESSAGES = 12;
const CHAT_MODES = new Set(["coach", "developer_review"]);

class ChatRequestError extends Error {
  constructor(message, code = "INVALID_REQUEST") {
    super(message);
    this.name = "ChatRequestError";
    this.code = code;
    this.statusCode = 400;
  }
}

function boundedText(value, maximum, label, allowEmpty = true) {
  const text = String(value || "").trim();
  if (!allowEmpty && !text) {
    throw new ChatRequestError(`请输入${label}`, "MESSAGE_REQUIRED");
  }
  if (text.length > maximum) {
    throw new ChatRequestError(`${label}不能超过 ${maximum} 个字符`, "TEXT_TOO_LONG");
  }
  return text;
}

function normalizeProfile(profile) {
  const source = profile && typeof profile === "object" && !Array.isArray(profile)
    ? profile
    : {};
  return {
    name: boundedText(source.name, 40, "称呼"),
    experience: boundedText(source.experience, 120, "训练经验"),
    goal: boundedText(source.goal, 300, "训练目标"),
    equipment: boundedText(source.equipment, 300, "可用器械"),
    constraints: boundedText(source.constraints, 300, "限制与不适")
  };
}

function normalizeHistory(history) {
  if (history === undefined) return [];
  if (!Array.isArray(history)) {
    throw new ChatRequestError("历史消息格式不正确", "INVALID_HISTORY");
  }

  const normalized = history.map((item) => {
    if (!item || !["user", "assistant"].includes(item.role)) {
      throw new ChatRequestError("历史消息角色只允许 user 或 assistant", "INVALID_ROLE");
    }
    return {
      role: item.role,
      content: boundedText(item.content, MAX_MESSAGE_LENGTH, "历史消息", false)
    };
  });

  return normalized.slice(-MAX_HISTORY_MESSAGES);
}

function normalizeMode(mode) {
  const value = String(mode || "coach").trim();
  return CHAT_MODES.has(value) ? value : "coach";
}

function normalizeChatRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new ChatRequestError("请求内容必须是 JSON 对象");
  }
  return {
    mode: normalizeMode(body.mode),
    message: boundedText(body.message, MAX_MESSAGE_LENGTH, "问题", false),
    profile: normalizeProfile(body.profile),
    history: normalizeHistory(body.history)
  };
}

function buildDeepSeekMessages(request) {
  const systemPrompt = request.mode === "developer_review"
    ? buildDeveloperReviewPrompt(request.profile)
    : buildCoachPrompt(request.profile);
  return [
    { role: "system", content: systemPrompt },
    ...request.history,
    { role: "user", content: request.message }
  ];
}

module.exports = {
  ChatRequestError,
  MAX_MESSAGE_LENGTH,
  MAX_HISTORY_MESSAGES,
  buildDeepSeekMessages,
  normalizeChatRequest
};
