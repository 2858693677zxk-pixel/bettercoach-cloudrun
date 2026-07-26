const crypto = require("node:crypto");

class AuthError extends Error {
  constructor(message, code = "UNAUTHORIZED", statusCode = 401) {
    super(message);
    this.name = "AuthError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

function timingSafeEqual(left, right) {
  const leftBuffer = Buffer.from(String(left || ""));
  const rightBuffer = Buffer.from(String(right || ""));
  return leftBuffer.length === rightBuffer.length
    && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

function verifyAiCoachToken(token, secret, expectedUserId, now = Date.now()) {
  const normalizedSecret = String(secret || "");
  if (!normalizedSecret) return null;
  const [payload, receivedSignature, extra] = String(token || "").split(".");
  if (!payload || !receivedSignature || extra) {
    throw new AuthError("AI 教练访问令牌无效");
  }
  const expectedSignature = crypto
    .createHmac("sha256", normalizedSecret)
    .update(payload)
    .digest("base64url");
  if (!timingSafeEqual(receivedSignature, expectedSignature)) {
    throw new AuthError("AI 教练访问令牌签名无效");
  }
  let claims;
  try {
    claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    throw new AuthError("AI 教练访问令牌格式无效");
  }
  if (String(claims.sub || "") !== String(expectedUserId || "")) {
    throw new AuthError("AI 教练访问令牌与当前会员不匹配", "FORBIDDEN", 403);
  }
  if (!Number(claims.exp) || Number(claims.exp) <= Math.floor(now / 1000)) {
    throw new AuthError("AI 教练访问令牌已过期", "TOKEN_EXPIRED", 401);
  }
  return claims;
}

module.exports = {
  AuthError,
  verifyAiCoachToken
};
