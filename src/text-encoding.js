const iconv = require("iconv-lite");

const MOJIBAKE_MARKERS = [
  /[\uE000-\uF8FF]/g,
  /銆\?/g,
  /锛\?/g,
  /鈥\?/g,
  /€\?/g,
  /(?:鐨|鍔ㄤ|璁|瑙|缁|闂|绋|鎷|杩|寤鸿|璇|浣|鏁)/g
];

const DAMAGED_PUNCTUATION = new Map([
  ["銆?", "。"],
  ["锛?", "，"],
  ["鈥?", "“"],
  ["€?", "。"]
]);

const EXERCISE_TERM_TRANSLATIONS = [
  [/\blat pulldown\b/gi, "高位下拉"],
  [/\bbench press\b/gi, "卧推"],
  [/\bromanian deadlift\b/gi, "罗马尼亚硬拉"],
  [/\bhack squat\b/gi, "哈克深蹲"],
  [/\bhip thrust\b/gi, "臀桥"],
  [/\blateral raise\b/gi, "侧平举"],
  [/\bface pull\b/gi, "面拉"],
  [/\bpull[ -]?up\b/gi, "引体向上"],
  [/\bchest press\b/gi, "推胸"],
  [/\bleg extension\b/gi, "腿屈伸"],
  [/\bleg curl\b/gi, "腿弯举"],
  [/\bbiceps curl\b/gi, "肱二头肌弯举"],
  [/\btriceps extension\b/gi, "肱三头肌拉伸"],
  [/\bgoblet squat\b/gi, "高脚杯深蹲"],
  [/\bbarbell squat\b/gi, "杠铃深蹲"],
  [/\bdumbbell press\b/gi, "哑铃卧推"],
  [/\bshoulder press\b/gi, "肩上推举"]
];

function translateExerciseTerms(value) {
  return EXERCISE_TERM_TRANSLATIONS.reduce(
    (text, [pattern, replacement]) => text.replace(pattern, replacement),
    String(value ?? "")
  );
}

function mojibakeScore(value) {
  const text = String(value ?? "");
  return MOJIBAKE_MARKERS.reduce((score, pattern) => {
    const matches = text.match(pattern);
    return score + (matches ? matches.length : 0);
  }, 0);
}

function protectDamagedPunctuation(value) {
  let text = value;
  const protectedValues = [];
  for (const [broken, repaired] of DAMAGED_PUNCTUATION) {
    while (text.includes(broken)) {
      const token = `__BETTERCOACH_PUNCT_${protectedValues.length}__`;
      protectedValues.push([token, repaired]);
      text = text.replace(broken, token);
    }
  }
  return { text, protectedValues };
}

function repairMojibake(value) {
  const original = String(value ?? "");
  const originalScore = mojibakeScore(original);
  if (originalScore < 2 && !/[\uE000-\uF8FF]|銆\?|锛\?|鈥\?|€\?/.test(original)) {
    return translateExerciseTerms(original);
  }

  const { text, protectedValues } = protectDamagedPunctuation(original);
  let candidate;
  try {
    candidate = iconv.decode(iconv.encode(text, "gb18030"), "utf8");
  } catch {
    return "";
  }
  for (const [token, repaired] of protectedValues) {
    candidate = candidate.replaceAll(token, repaired);
  }

  if (candidate.includes("\uFFFD")) return "";
  return mojibakeScore(candidate) < originalScore
    ? translateExerciseTerms(candidate)
    : "";
}

function normalizeTextEncoding(value) {
  if (typeof value === "string") return repairMojibake(value);
  if (Array.isArray(value)) return value.map(normalizeTextEncoding);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, normalizeTextEncoding(item)])
  );
}

module.exports = {
  mojibakeScore,
  normalizeTextEncoding,
  repairMojibake,
  translateExerciseTerms
};
