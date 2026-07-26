class DeepSeekError extends Error {
  constructor(message, code, statusCode) {
    super(message);
    this.name = "DeepSeekError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

function upstreamError(status) {
  if (status === 401 || status === 403) {
    return new DeepSeekError(
      "DeepSeek API 鉴权失败，请检查本地 API Key。",
      "DEEPSEEK_AUTH_FAILED",
      502
    );
  }
  if (status === 429) {
    return new DeepSeekError(
      "DeepSeek 请求过于频繁或额度不足，请稍后重试。",
      "DEEPSEEK_RATE_LIMITED",
      503
    );
  }
  return new DeepSeekError(
    "DeepSeek 服务暂时不可用，请稍后重试。",
    "DEEPSEEK_UPSTREAM_ERROR",
    502
  );
}

function createDeepSeekClient(config, dependencies = {}) {
  const fetchImpl = dependencies.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("This Node.js runtime does not provide fetch");
  }

  const baseUrl = String(config.baseUrl || "").replace(/\/+$/, "");

  return {
    async complete(messages, options = {}) {
      if (!config.apiKey) {
        throw new DeepSeekError(
          "服务端尚未配置 DeepSeek API Key。",
          "MISSING_API_KEY",
          503
        );
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), config.timeoutMs);

      try {
        const requestBody = {
          model: config.model,
          messages,
          temperature: Number.isFinite(options.temperature) ? options.temperature : 0.35,
          max_tokens: Number.isFinite(options.maxTokens) ? options.maxTokens : 1600,
          stream: false
        };
        if (options.json) {
          requestBody.response_format = { type: "json_object" };
        }

        const response = await fetchImpl(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${config.apiKey}`
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal
        });

        if (!response.ok) throw upstreamError(response.status);

        let payload;
        try {
          payload = await response.json();
        } catch {
          throw new DeepSeekError(
            "DeepSeek 返回了无法解析的内容。",
            "DEEPSEEK_INVALID_RESPONSE",
            502
          );
        }

        const content = payload
          && payload.choices
          && payload.choices[0]
          && payload.choices[0].message
          && payload.choices[0].message.content;
        if (typeof content !== "string" || !content.trim()) {
          throw new DeepSeekError(
            "DeepSeek 没有返回有效回复。",
            "DEEPSEEK_INVALID_RESPONSE",
            502
          );
        }

        return {
          content: content.trim(),
          model: payload.model || config.model,
          usage: payload.usage || null
        };
      } catch (error) {
        if (error instanceof DeepSeekError) throw error;
        if (error.name === "AbortError" || controller.signal.aborted) {
          throw new DeepSeekError(
            "DeepSeek 请求超时，请稍后重试。",
            "DEEPSEEK_TIMEOUT",
            504
          );
        }
        throw new DeepSeekError(
          "无法连接 DeepSeek 服务，请检查网络。",
          "DEEPSEEK_UNAVAILABLE",
          502
        );
      } finally {
        clearTimeout(timer);
      }
    }
  };
}

module.exports = {
  DeepSeekError,
  createDeepSeekClient
};
