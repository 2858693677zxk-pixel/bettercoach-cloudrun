"""Optional secondary pose-engine comparison for uploaded videos.

The primary analyzer remains authoritative. This module runs a lightweight
MediaPipe 33-landmark pass and compares pose coverage, confidence, and major
joint angles against the primary pose frames.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np


SECONDARY_BACKEND = "motion_tracker_mediapipe"
MIN_ANGLE_CONFIDENCE = 0.20

LANDMARK = mp.solutions.pose.PoseLandmark
JOINT_CHAINS = {
    "left_knee": (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
    "right_knee": (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
    "left_hip": (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
    "right_hip": (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
    "left_elbow": (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
    "right_elbow": (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
    "left_shoulder": (LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP),
    "right_shoulder": (LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP),
}


@dataclass
class SecondaryPoseFrame:
    frame_index: int
    time_ms: int
    landmarks: list[list[float]]
    quality: float


def _finite_numbers(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _percentile(values: Iterable[float], percentile: float, fallback: float = 0.0) -> float:
    finite = _finite_numbers(values)
    if not finite:
        return fallback
    return float(np.percentile(np.asarray(finite, dtype=float), percentile))


def _point(landmarks: list[list[float]], index: int) -> np.ndarray:
    return np.asarray(landmarks[int(index)][:2], dtype=float)


def _visibility(landmarks: list[list[float]], index: int) -> float:
    if int(index) >= len(landmarks) or len(landmarks[int(index)]) < 4:
        return 0.0
    return float(landmarks[int(index)][3])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def joint_angle(landmarks: list[list[float]], joint: str) -> float | None:
    chain = JOINT_CHAINS[joint]
    confidence = min(_visibility(landmarks, int(index)) for index in chain)
    if confidence < MIN_ANGLE_CONFIDENCE:
        return None
    value = _angle(
        _point(landmarks, int(chain[0])),
        _point(landmarks, int(chain[1])),
        _point(landmarks, int(chain[2])),
    )
    return value if math.isfinite(value) else None


def angle_series(frames: Iterable[Any]) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {joint: {} for joint in JOINT_CHAINS}
    for frame in frames:
        landmarks = getattr(frame, "landmarks", None)
        frame_index = getattr(frame, "frame_index", None)
        if landmarks is None or frame_index is None:
            continue
        for joint in JOINT_CHAINS:
            value = joint_angle(landmarks, joint)
            if value is not None:
                series[joint][int(frame_index)] = float(value)
    return series


def _landmarks_from_result(result: Any) -> list[list[float]]:
    landmarks = []
    for landmark in result.pose_landmarks.landmark:
        landmarks.append([
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 1.0)),
        ])
    return landmarks


def extract_secondary_pose_frames(video_path: Path, fps: float, step: int) -> list[SecondaryPoseFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("secondary pose compare could not open video")

    frames: list[SecondaryPoseFrame] = []
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                landmarks = _landmarks_from_result(result)
                chain_indices = [
                    int(LANDMARK.LEFT_SHOULDER),
                    int(LANDMARK.RIGHT_SHOULDER),
                    int(LANDMARK.LEFT_HIP),
                    int(LANDMARK.RIGHT_HIP),
                    int(LANDMARK.LEFT_KNEE),
                    int(LANDMARK.RIGHT_KNEE),
                    int(LANDMARK.LEFT_ANKLE),
                    int(LANDMARK.RIGHT_ANKLE),
                    int(LANDMARK.LEFT_ELBOW),
                    int(LANDMARK.RIGHT_ELBOW),
                    int(LANDMARK.LEFT_WRIST),
                    int(LANDMARK.RIGHT_WRIST),
                ]
                quality = float(np.mean([landmarks[index][3] for index in chain_indices]))
                frames.append(SecondaryPoseFrame(
                    frame_index=frame_index,
                    time_ms=int(round(frame_index * 1000.0 / fps)) if fps > 0 else 0,
                    landmarks=landmarks,
                    quality=quality,
                ))
            frame_index += 1
    finally:
        pose.close()
        capture.release()
    return frames


def _joint_delta_summary(
    primary_frames: list[Any],
    secondary_frames: list[SecondaryPoseFrame],
) -> dict[str, dict[str, float | int]]:
    primary_series = angle_series(primary_frames)
    secondary_series = angle_series(secondary_frames)
    summary: dict[str, dict[str, float | int]] = {}
    for joint in JOINT_CHAINS:
        shared = sorted(set(primary_series[joint]) & set(secondary_series[joint]))
        deltas = [
            abs(primary_series[joint][frame_index] - secondary_series[joint][frame_index])
            for frame_index in shared
        ]
        if not deltas:
            summary[joint] = {"count": 0}
            continue
        summary[joint] = {
            "count": len(deltas),
            "meanAbsDelta": round(float(np.mean(deltas)), 2),
            "medianAbsDelta": round(float(np.median(deltas)), 2),
            "p95AbsDelta": round(_percentile(deltas, 95), 2),
        }
    return summary


def _top_divergent_joints(summary: dict[str, dict[str, float | int]]) -> list[dict[str, Any]]:
    candidates = [
        {
            "joint": joint,
            "count": int(values.get("count", 0)),
            "meanAbsDelta": float(values.get("meanAbsDelta", 0.0)),
            "medianAbsDelta": float(values.get("medianAbsDelta", 0.0)),
            "p95AbsDelta": float(values.get("p95AbsDelta", 0.0)),
        }
        for joint, values in summary.items()
        if int(values.get("count", 0)) > 0
    ]
    return sorted(
        candidates,
        key=lambda item: (
            item["medianAbsDelta"],
            item["meanAbsDelta"],
            item["p95AbsDelta"],
        ),
        reverse=True,
    )[:5]


def _recommendation(
    primary_pose_coverage: float,
    secondary_pose_coverage: float,
    primary_average_confidence: float,
    secondary_average_confidence: float,
    top_divergent_joints: list[dict[str, Any]],
) -> str:
    worst_delta = max((item["medianAbsDelta"] for item in top_divergent_joints), default=0.0)
    if worst_delta >= 30.0:
        return "needs_manual_review"
    if (
        secondary_pose_coverage >= primary_pose_coverage + 0.15
        or secondary_average_confidence >= primary_average_confidence + 0.12
    ):
        return "import_secondary_metrics"
    return "keep_primary"


def build_pose_engine_comparison(
    *,
    video_path: Path,
    primary_frames: list[Any],
    primary_backend: str | None,
    primary_pose_coverage: float,
    primary_average_confidence: float,
    fps: float,
    frame_count: int,
    step: int,
) -> dict[str, Any]:
    started = time.time()
    expected_samples = max(1, int(math.ceil(frame_count / max(1, step))))
    secondary_frames = extract_secondary_pose_frames(video_path, fps, max(1, step))
    secondary_pose_coverage = min(1.0, len(secondary_frames) / expected_samples)
    secondary_average_confidence = float(np.mean([frame.quality for frame in secondary_frames])) if secondary_frames else 0.0
    delta_summary = _joint_delta_summary(primary_frames, secondary_frames)
    top_divergent = _top_divergent_joints(delta_summary)
    recommendation = _recommendation(
        primary_pose_coverage,
        secondary_pose_coverage,
        primary_average_confidence,
        secondary_average_confidence,
        top_divergent,
    )
    return {
        "enabled": True,
        "primaryBackend": primary_backend or "unknown",
        "secondaryBackend": SECONDARY_BACKEND,
        "primary": {
            "poseCoverage": round(float(primary_pose_coverage), 3),
            "averageConfidence": round(float(primary_average_confidence), 3),
            "poseFrames": len(primary_frames),
        },
        "secondary": {
            "poseCoverage": round(float(secondary_pose_coverage), 3),
            "averageConfidence": round(float(secondary_average_confidence), 3),
            "poseFrames": len(secondary_frames),
            "expectedSamples": expected_samples,
        },
        "jointAngleDeltaSummary": delta_summary,
        "topDivergentJoints": top_divergent,
        "runtimeMs": int(round((time.time() - started) * 1000)),
        "recommendation": recommendation,
    }


def failed_pose_engine_comparison(
    *,
    primary_backend: str | None,
    primary_pose_coverage: float,
    primary_average_confidence: float,
    primary_frame_count: int,
    runtime_ms: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "primaryBackend": primary_backend or "unknown",
        "secondaryBackend": SECONDARY_BACKEND,
        "primary": {
            "poseCoverage": round(float(primary_pose_coverage), 3),
            "averageConfidence": round(float(primary_average_confidence), 3),
            "poseFrames": int(primary_frame_count),
        },
        "secondary": {
            "poseCoverage": 0,
            "averageConfidence": 0,
            "poseFrames": 0,
            "expectedSamples": 0,
        },
        "jointAngleDeltaSummary": {},
        "topDivergentJoints": [],
        "runtimeMs": int(runtime_ms),
        "recommendation": "keep_primary",
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:500],
        },
    }
