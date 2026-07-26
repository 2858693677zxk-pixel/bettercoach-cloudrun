#!/usr/bin/env python3
"""Create a compact full-duration video proxy for GLM video review."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2


PROFILES = [
    {"width": 480, "fps": 6},
    {"width": 384, "fps": 5},
    {"width": 320, "fps": 4},
    {"width": 288, "fps": 3},
]


def even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def render_proxy(
    video_path: Path,
    destination: Path,
    *,
    target_width: int,
    target_fps: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("video cannot be opened for GLM proxy")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("invalid video metadata for GLM proxy")

    output_width = min(target_width, width)
    output_height = even(round(height * output_width / max(1, width)))
    step = max(1, int(round(source_fps / max(1.0, target_fps))))
    effective_fps = source_fps / step

    if destination.exists():
        destination.unlink()
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, effective_fps),
        (even(output_width), output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot open GLM proxy writer")

    frame_index = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % step == 0:
                frame = cv2.resize(frame, (even(output_width), output_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if written == 0:
        raise RuntimeError("GLM proxy contains no frames")

    return {
        "filename": destination.name,
        "frames": written,
        "fps": round(effective_fps, 2),
        "width": even(output_width),
        "height": output_height,
        "bytes": destination.stat().st_size,
        "durationSeconds": round(frame_count / source_fps, 2),
        "sourceFps": round(source_fps, 2),
        "sourceFrames": frame_count,
    }


def prepare(payload: dict[str, Any]) -> dict[str, Any]:
    video_path = Path(payload["videoPath"]).resolve()
    output_dir = Path(payload["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(payload.get("maxBytes") or 14 * 1024 * 1024)

    last_result: dict[str, Any] | None = None
    for profile in PROFILES:
        destination = output_dir / "glm_full_video_proxy.mp4"
        result = render_proxy(
            video_path,
            destination,
            target_width=int(profile["width"]),
            target_fps=float(profile["fps"]),
        )
        result["profile"] = profile
        last_result = result
        if int(result["bytes"]) <= max_bytes:
            return result

    assert last_result is not None
    last_result["warning"] = f"proxy exceeds target size {max_bytes}"
    return last_result


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: prepare_glm_video.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        result = prepare(json.loads(input_path.read_text(encoding="utf-8-sig")))
        output_path.write_text(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as error:
        output_path.write_text(
            json.dumps({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
