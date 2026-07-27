#!/bin/sh
set -eu

export DATA_DIRECTORY="${DATA_DIRECTORY:-/data}"
export POSE_BACKEND="rtmlib"
export POSE_BACKEND_STRICT="true"
export POSE_ENGINE_COMPARE="false"
export RTMLIB_BACKEND="onnxruntime"
export RTMLIB_DEVICE="cpu"
export RTMLIB_ONE_STAGE="true"
export RTMLIB_POSE_INPUT_SIZE="640x640"
export RTMLIB_POSE_MODEL="/opt/xiaoyu-models/rtmo-m.onnx"

mkdir -p "${DATA_DIRECTORY}"

exec node server.js
