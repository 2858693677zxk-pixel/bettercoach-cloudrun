ARG NODE_IMAGE=node:20-bookworm-slim
FROM ${NODE_IMAGE}

ARG NPM_REGISTRY=https://registry.npmmirror.com
ARG PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
ARG PIP_MEDIAPIPE_INDEX_URL=https://pypi.org/simple

RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/xiaoyu-venv \
    && /opt/xiaoyu-venv/bin/pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" --upgrade pip \
    && /opt/xiaoyu-venv/bin/pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
      "numpy<2" \
      "onnxruntime==1.23.2" \
      "opencv-python-headless>=4.10,<5" \
      "rtmlib==0.0.15" \
    && /opt/xiaoyu-venv/bin/pip install --no-cache-dir --index-url "${PIP_MEDIAPIPE_INDEX_URL}" \
      "mediapipe==0.10.21"

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev --registry="${NPM_REGISTRY}"

COPY python ./python
COPY src ./src
COPY server.js ./
COPY models/rtmo-m.onnx /opt/xiaoyu-models/rtmo-m.onnx
COPY cloudrun/start-container.sh /usr/local/bin/start-xiaoyu-coach

RUN chmod +x /usr/local/bin/start-xiaoyu-coach \
    && test -s /opt/xiaoyu-models/rtmo-m.onnx \
    && mkdir -p /data

ENV HOST=0.0.0.0 \
    PORT=80 \
    ANALYZER_PYTHON=/opt/xiaoyu-venv/bin/python \
    POSE_BACKEND=rtmlib \
    POSE_ENGINE_COMPARE=false \
    POSE_RECHECK_LOW_CONFIDENCE=true \
    POSE_RECHECK_MAX_WINDOWS=8 \
    RTMLIB_BACKEND=onnxruntime \
    RTMLIB_DEVICE=cpu \
    RTMLIB_ONE_STAGE=true \
    RTMLIB_POSE_INPUT_SIZE=640x640 \
    RTMLIB_POSE_MODEL=/opt/xiaoyu-models/rtmo-m.onnx \
    DATA_DIRECTORY=/data \
    MAX_UPLOAD_MB=90

EXPOSE 80

CMD ["/usr/local/bin/start-xiaoyu-coach"]
