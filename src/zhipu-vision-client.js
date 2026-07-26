class ZhipuVisionError extends Error {
  constructor(message, code, statusCode) {
    super(message);
    this.name = "ZhipuVisionError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

function upstreamError(status) {
  if (status === 401 || status === 403) {
    return new ZhipuVisionError(
      "智谱视觉模型鉴权失败，请检查服务端 API Key。",
      "ZHIPU_AUTH_FAILED",
      502
    );
  }
  if (status === 429) {
    return new ZhipuVisionError(
      "智谱视觉模型请求过于频繁或额度不足，请稍后重试。",
      "ZHIPU_RATE_LIMITED",
      503
    );
  }
  return new ZhipuVisionError(
    "智谱视觉模型暂时不可用，请稍后重试。",
    "ZHIPU_UPSTREAM_ERROR",
    502
  );
}

function createZhipuVisionClient(config, dependencies = {}) {
  const fetchImpl = dependencies.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("This Node.js runtime does not provide fetch");
  }

  const baseUrl = String(config.zhipuBaseUrl || "").replace(/\/+$/, "");
  const model = String(config.zhipuVisionModel || "glm-4.6v-flash").trim();
  const timeoutMs = Number.isFinite(config.zhipuTimeoutMs)
    ? config.zhipuTimeoutMs
    : config.timeoutMs;

  return {
    async complete(messages, options = {}) {
      if (!config.zhipuApiKey) {
        throw new ZhipuVisionError(
          "服务端尚未配置智谱 API Key。",
          "MISSING_ZHIPU_API_KEY",
          503
        );
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs || 30000);

      try {
        const requestedMaxTokens = Number.isFinite(options.maxTokens) ? options.maxTokens : 1024;
        const requestBody = {
          model,
          messages,
          temperature: Number.isFinite(options.temperature) ? options.temperature : 0,
          max_tokens: Math.max(1, Math.min(1024, Math.floor(requestedMaxTokens))),
          stream: false
        };
        if (options.json) {
          requestBody.response_format = { type: "json_object" };
        }

        const response = await fetchImpl(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${config.zhipuApiKey}`
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal
        });

        if (!response.ok) throw upstreamError(response.status);

        let payload;
        try {
          payload = await response.json();
        } catch {
          throw new ZhipuVisionError(
            "智谱视觉模型返回了无法解析的内容。",
            "ZHIPU_INVALID_RESPONSE",
            502
          );
        }

        const content = payload
          && payload.choices
          && payload.choices[0]
          && payload.choices[0].message
          && payload.choices[0].message.content;
        if (typeof content !== "string" || !content.trim()) {
          throw new ZhipuVisionError(
            "智谱视觉模型没有返回有效回复。",
            "ZHIPU_INVALID_RESPONSE",
            502
          );
        }

        return {
          content: content.trim(),
          model: payload.model || model,
          usage: payload.usage || null
        };
      } catch (error) {
        if (error instanceof ZhipuVisionError) throw error;
        if (error.name === "AbortError" || controller.signal.aborted) {
          throw new ZhipuVisionError(
            "智谱视觉模型请求超时，请稍后重试。",
            "ZHIPU_TIMEOUT",
            504
          );
        }
        throw new ZhipuVisionError(
          "无法连接智谱视觉模型，请检查网络。",
          "ZHIPU_UNAVAILABLE",
          502
        );
      } finally {
        clearTimeout(timer);
      }
    }
  };
}

module.exports = {
  ZhipuVisionError,
  createZhipuVisionClient
};
