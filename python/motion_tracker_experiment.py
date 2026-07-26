"""Motion-tracker based exercise experiment runner.

This script is intentionally separate from the production xiaoyu-coach
analyzer. It uses the deployed E:\\motion-tracker MediaPipe 33-point backend as
an alternate pose engine, then applies lightweight exercise rules for debugging
model selection and future action onboarding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_MOTION_TRACKER_ROOT = Path(r"E:\motion-tracker")


@dataclass
class MotionSample:
    frame_index: int
    time_ms: int
    confidence: float
    pose: Any
    angles: dict[str, float | None]
    posture: dict[str, float | None]
    metrics: dict[str, float | None]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def clean_number(value: Any, digits: int = 3) -> float | None:
    if not finite(value):
        return None
    return round(float(value), digits)


def numeric_stats(values: Iterable[Any]) -> dict[str, float | int] | None:
    numbers = [float(item) for item in values if finite(item)]
    if not numbers:
        return None
    arr = np.asarray(numbers, dtype=float)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p05": round(float(np.percentile(arr, 5)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "range": round(float(np.percentile(arr, 95) - np.percentile(arr, 5)), 2),
    }


def avg(values: Iterable[Any], fallback: float | None = None) -> float | None:
    numbers = [float(item) for item in values if finite(item)]
    if not numbers:
        return fallback
    return float(np.mean(numbers))


def keypoint(pose: Any, name: str) -> Any | None:
    if pose is None:
        return None
    return pose.get_keypoint(name)


def visible_xy(pose: Any, name: str, min_visibility: float = 0.45) -> tuple[float, float] | None:
    kp = keypoint(pose, name)
    if kp is None or float(getattr(kp, "visibility", 0.0) or 0.0) < min_visibility:
        return None
    return float(kp.x), float(kp.y)


def midpoint(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return float(np.mean([item[0] for item in points])), float(np.mean([item[1] for item in points]))


def torso_lean_2d(pose: Any) -> float | None:
    shoulders = [visible_xy(pose, "left_shoulder"), visible_xy(pose, "right_shoulder")]
    hips = [visible_xy(pose, "left_hip"), visible_xy(pose, "right_hip")]
    shoulder_mid = midpoint([item for item in shoulders if item is not None])
    hip_mid = midpoint([item for item in hips if item is not None])
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if abs(dy) < 1e-6:
        return None
    return abs(math.degrees(math.atan2(dx, dy)))


def shoulder_hip_angle_2d(pose: Any) -> float | None:
    shoulders = [visible_xy(pose, "left_shoulder"), visible_xy(pose, "right_shoulder")]
    hips = [visible_xy(pose, "left_hip"), visible_xy(pose, "right_hip")]
    shoulder_mid = midpoint([item for item in shoulders if item is not None])
    hip_mid = midpoint([item for item in hips if item is not None])
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return abs(math.degrees(math.atan2(dy, dx)))


def elbow_below_shoulder(pose: Any) -> float | None:
    values: list[float] = []
    for side in ("left", "right"):
        shoulder = visible_xy(pose, f"{side}_shoulder")
        elbow = visible_xy(pose, f"{side}_elbow")
        if shoulder is not None and elbow is not None:
            values.append(elbow[1] - shoulder[1])
    return avg(values)


def forearm_tilt_ratio(pose: Any) -> float | None:
    values: list[float] = []
    for side in ("left", "right"):
        elbow = visible_xy(pose, f"{side}_elbow")
        wrist = visible_xy(pose, f"{side}_wrist")
        if elbow is None or wrist is None:
            continue
        dx = abs(wrist[0] - elbow[0])
        dy = abs(wrist[1] - elbow[1])
        values.append(dx / max(0.015, dy))
    return avg(values)


def bilateral_angle(angles: dict[str, float | None], joint: str) -> float | None:
    return avg([angles.get(f"left_{joint}"), angles.get(f"right_{joint}")])


def open_video_writer(destination: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "mp4v"):
        if destination.exists():
            destination.unlink()
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*codec),
            max(1.0, float(fps)),
            size,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("Could not open a browser-compatible video writer")


def resize_frame(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        output = frame
    else:
        scale = max_width / max(1, width)
        output = cv2.resize(frame, (max_width, int(round(height * scale))))
    out_h, out_w = output.shape[:2]
    if out_w % 2 or out_h % 2:
        output = output[: out_h - (out_h % 2), : out_w - (out_w % 2)]
    return output


def series(samples: list[MotionSample], name: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = None
        if name in sample.angles:
            value = sample.angles.get(name)
        elif name in sample.posture:
            value = sample.posture.get(name)
        elif name in sample.metrics:
            value = sample.metrics.get(name)
        if finite(value):
            values.append(float(value))
    return values


def smooth(values: list[float], window: int = 3) -> list[float]:
    if len(values) < 3:
        return values[:]
    radius = max(1, window // 2)
    result = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result.append(float(np.mean(values[start:end])))
    return result


def segment_elbow_cycles(
    samples: list[MotionSample],
    *,
    extended_threshold: float,
    flexed_threshold: float,
    min_amplitude: float,
) -> list[dict[str, Any]]:
    values = [sample.metrics.get("elbow_angle") for sample in samples]
    usable = [float(item) if finite(item) else float("nan") for item in values]
    smoothed = smooth([item if math.isfinite(item) else 180.0 for item in usable], 5)
    events: list[dict[str, Any]] = []
    index = 0
    while index < len(smoothed):
        while index < len(smoothed) and smoothed[index] < extended_threshold:
            index += 1
        if index >= len(smoothed):
            break
        start = index
        min_index = start
        crossed_flexed = False
        while index < len(smoothed):
            if smoothed[index] < smoothed[min_index]:
                min_index = index
            if index > start and smoothed[index] <= flexed_threshold:
                crossed_flexed = True
            if (
                crossed_flexed
                and index > min_index + 2
                and smoothed[index] >= smoothed[min_index] + 5
            ):
                break
            if index - start > 80:
                break
            index += 1
        if index >= len(smoothed) or not crossed_flexed or smoothed[min_index] > flexed_threshold:
            break
        end = index
        while end + 1 < len(smoothed) and smoothed[end] < extended_threshold - 15:
            end += 1
            if end - min_index > 80:
                break
        amplitude = smoothed[start] - smoothed[min_index]
        if amplitude >= min_amplitude:
            events.append({
                "repIndex": len(events) + 1,
                "startSample": start,
                "keySample": min_index,
                "endSample": end,
                "startTimeMs": samples[start].time_ms,
                "keyTimeMs": samples[min_index].time_ms,
                "endTimeMs": samples[end].time_ms,
                "startElbowAngle": round(smoothed[start], 2),
                "keyElbowAngle": round(smoothed[min_index], 2),
                "endElbowAngle": round(smoothed[end], 2),
                "amplitude": round(amplitude, 2),
            })
        index = max(end + 1, index + 1)
    return events


def segment_hinge_cycles(samples: list[MotionSample]) -> list[dict[str, Any]]:
    values = [sample.metrics.get("torso_lean_2d") for sample in samples]
    usable = smooth([float(item) if finite(item) else 0.0 for item in values], 5)
    if not usable:
        return []
    top_threshold = max(10.0, float(np.percentile(usable, 25)) + 5)
    bottom_threshold = max(top_threshold + 25, float(np.percentile(usable, 80)))
    events: list[dict[str, Any]] = []
    index = 0
    while index < len(usable):
        while index < len(usable) and usable[index] > top_threshold:
            index += 1
        if index >= len(usable):
            break
        start = index
        max_index = start
        while index < len(usable):
            if usable[index] > usable[max_index]:
                max_index = index
            if index > start and usable[index] >= bottom_threshold:
                break
            if index - start > 100:
                break
            index += 1
        if index >= len(usable) or usable[max_index] < bottom_threshold:
            break
        end = index
        while end + 1 < len(usable) and usable[end] > top_threshold + 5:
            end += 1
            if end - max_index > 100:
                break
        if usable[end] > top_threshold + 12:
            index = max(end + 1, index + 1)
            continue
        amplitude = usable[max_index] - usable[start]
        if amplitude >= 25:
            events.append({
                "repIndex": len(events) + 1,
                "startSample": start,
                "keySample": max_index,
                "endSample": end,
                "startTimeMs": samples[start].time_ms,
                "keyTimeMs": samples[max_index].time_ms,
                "endTimeMs": samples[end].time_ms,
                "startTorsoLean": round(usable[start], 2),
                "bottomTorsoLean": round(usable[max_index], 2),
                "endTorsoLean": round(usable[end], 2),
                "amplitude": round(amplitude, 2),
            })
        index = max(end + 1, index + 1)
    return events


def issue(code: str, severity: str, title: str, observation: str, correction: str, confidence: float = 0.75) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "observation": observation,
        "correction": correction,
        "confidence": round(float(confidence), 3),
    }


def evaluate_action(action_type: str, samples: list[MotionSample], pose_coverage: float, average_confidence: float) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    strengths: list[str] = []
    if pose_coverage < 0.72:
        issues.append(issue(
            "MT_LOW_POSE_COVERAGE",
            "yellow",
            "Motion-tracker pose coverage is low",
            f"Only {pose_coverage:.0%} of sampled frames produced a usable 33-point skeleton.",
            "Reshoot with the full body and implement visible, or use the primary RTMLib track for this sample.",
            0.9,
        ))
    if average_confidence < 0.55:
        issues.append(issue(
            "MT_LOW_CONFIDENCE",
            "yellow",
            "Motion-tracker confidence is low",
            f"Average skeleton confidence is {average_confidence:.2f}.",
            "Use the annotated video to confirm whether MediaPipe is following the target person.",
            0.85,
        ))

    metrics = {
        "elbowAngle": numeric_stats(series(samples, "elbow_angle")),
        "hipAngle": numeric_stats(series(samples, "hip_angle")),
        "kneeAngle": numeric_stats(series(samples, "knee_angle")),
        "torsoLean2d": numeric_stats(series(samples, "torso_lean_2d")),
        "bodyLean": numeric_stats(series(samples, "body_lean")),
        "spineCurve": numeric_stats(series(samples, "spine_curve")),
        "elbowBelowShoulder": numeric_stats(series(samples, "elbow_below_shoulder")),
        "forearmTiltRatio": numeric_stats(series(samples, "forearm_tilt_ratio")),
    }
    rep_events: list[dict[str, Any]]
    if action_type == "lat_pulldown":
        rep_events = segment_elbow_cycles(
            samples,
            extended_threshold=135,
            flexed_threshold=90,
            min_amplitude=35,
        )
        min_elbow = metrics["elbowAngle"]["p05"] if metrics["elbowAngle"] else None
        torso_range = metrics["torsoLean2d"]["range"] if metrics["torsoLean2d"] else 0
        lowest_elbow = max(
            [sample.metrics.get("elbow_below_shoulder") for sample in samples if finite(sample.metrics.get("elbow_below_shoulder"))],
            default=None,
        )
        if not rep_events:
            issues.append(issue(
                "MT_LAT_NO_FULL_REP",
                "yellow",
                "No complete lat pulldown rep detected",
                "Motion-tracker did not see an elbow angle transition from >135 degrees to <90 degrees.",
                "Check whether the camera sees shoulder-elbow-wrist clearly and whether the top position is fully extended.",
            ))
        if min_elbow is not None and min_elbow > 90:
            issues.append(issue(
                "MT_LAT_RANGE_INCOMPLETE",
                "yellow",
                "Bottom contraction is not deep enough",
                f"The smallest bilateral elbow angle was about {min_elbow:.1f} degrees.",
                "Pull the elbows down until the forearm-upper-arm angle is below 90 degrees before returning.",
            ))
        else:
            strengths.append("Elbow flexion reached the configured lat-pulldown bottom threshold.")
        if lowest_elbow is not None and lowest_elbow < 0.02:
            issues.append(issue(
                "MT_LAT_ELBOW_PATH_LIMITED",
                "yellow",
                "Elbows did not clearly pass shoulder line",
                f"The largest elbow-below-shoulder offset was {lowest_elbow:.3f} in normalized image height.",
                "Drive elbows down and slightly back instead of only bending the wrists.",
            ))
        if torso_range and torso_range > 18:
            issues.append(issue(
                "MT_LAT_TORSO_SWING",
                "yellow",
                "Torso angle changes during the pull",
                f"2D torso lean range was about {torso_range:.1f} degrees.",
                "Keep the rib cage tall and avoid turning the pulldown into a backward lean.",
            ))
    elif action_type in {"row", "open_elbow_row"}:
        rep_events = segment_elbow_cycles(
            samples,
            extended_threshold=124,
            flexed_threshold=118,
            min_amplitude=10,
        )
        min_elbow = metrics["elbowAngle"]["p05"] if metrics["elbowAngle"] else None
        torso_range = metrics["torsoLean2d"]["range"] if metrics["torsoLean2d"] else 0
        if not rep_events:
            issues.append(issue(
                "MT_ROW_NO_FULL_REP",
                "yellow",
                "No complete row rep detected",
                "Motion-tracker did not see repeated elbow extension-to-flexion cycles.",
                "Confirm the side camera shows both elbows and wrists throughout the row.",
            ))
        if min_elbow is not None and min_elbow > 105:
            issues.append(issue(
                "MT_ROW_PULL_SHORT",
                "yellow",
                "Handle pull appears short",
                f"The smallest bilateral elbow angle was about {min_elbow:.1f} degrees.",
                "Finish the pull by bringing elbows behind the torso while keeping shoulders down.",
            ))
        else:
            strengths.append("Elbow flexion indicates the handle was pulled into a clear contracted position.")
        if torso_range and torso_range > 18:
            issues.append(issue(
                "MT_ROW_TORSO_SWING",
                "yellow",
                "Torso swing is large",
                f"2D torso lean range was about {torso_range:.1f} degrees.",
                "Allow a controlled reach at the front, but do not use backward body swing to finish the pull.",
            ))
    elif action_type in {"deadlift", "romanian_deadlift"}:
        rep_events = segment_hinge_cycles(samples)
        hip_range = metrics["hipAngle"]["range"] if metrics["hipAngle"] else 0
        torso_range = metrics["torsoLean2d"]["range"] if metrics["torsoLean2d"] else 0
        knee_range = metrics["kneeAngle"]["range"] if metrics["kneeAngle"] else 0
        spine_p95 = metrics["spineCurve"]["p95"] if metrics["spineCurve"] else None
        if not rep_events:
            issues.append(issue(
                "MT_HINGE_NO_FULL_REP",
                "yellow",
                "No complete hinge rep detected",
                "Motion-tracker did not see a clear top-to-bottom-to-top torso hinge cycle.",
                "Use a side 30-45 degree camera and include the full standing top and bottom position.",
            ))
        if max(float(hip_range or 0), float(torso_range or 0)) < 30:
            issues.append(issue(
                "MT_HINGE_RANGE_SMALL",
                "yellow",
                "Hinge range appears small",
                f"Hip angle range was {hip_range or 0:.1f} degrees and torso lean range was {torso_range or 0:.1f} degrees.",
                "Push hips back farther while keeping the spine braced, or confirm the full range is visible.",
            ))
        else:
            strengths.append("Motion-tracker sees a clear hip-hinge range.")
        if action_type == "romanian_deadlift" and knee_range and knee_range > 45:
            issues.append(issue(
                "MT_RDL_TOO_KNEE_DOMINANT",
                "yellow",
                "Knee bend is high for an RDL",
                f"Knee angle range was about {knee_range:.1f} degrees.",
                "Keep knees softly bent and make hip travel the main movement instead of squatting down.",
            ))
        if spine_p95 is not None and spine_p95 > 28:
            issues.append(issue(
                "MT_HINGE_BACK_LINE_REVIEW",
                "yellow",
                "Back line needs visual review",
                f"Motion-tracker spine-curve p95 was about {spine_p95:.1f} degrees.",
                "Use GLM/contact-sheet review to confirm whether this is real rounding or a side-view landmark artifact.",
            ))
    else:
        rep_events = []
        issues.append(issue(
            "MT_ACTION_UNSUPPORTED",
            "yellow",
            "Action is not configured in the motion-tracker experiment",
            f"No experiment rules are defined for {action_type}.",
            "Add action-specific signal, rep segmentation, and rule thresholds before trusting the output.",
        ))

    if not issues:
        strengths.append("No configured motion-tracker experiment rule was violated.")
    score = max(0, 100 - sum(18 if item["severity"] == "yellow" else 35 for item in issues))
    return {
        "repEvents": rep_events,
        "repCount": len(rep_events),
        "metrics": metrics,
        "issues": issues,
        "strengths": strengths,
        "score": score,
        "safetyLevel": "yellow" if issues else "green",
    }


def choose_key_samples(action_type: str, samples: list[MotionSample], rep_events: list[dict[str, Any]]) -> list[int]:
    if not samples:
        return []
    if rep_events:
        event = rep_events[len(rep_events) // 2]
        return sorted(set([
            int(event.get("startSample", 0)),
            int((int(event.get("startSample", 0)) + int(event.get("keySample", 0))) / 2),
            int(event.get("keySample", 0)),
            int(event.get("endSample", len(samples) - 1)),
        ]))
    if action_type in {"deadlift", "romanian_deadlift"}:
        values = [sample.metrics.get("torso_lean_2d") for sample in samples]
        valid = [(index, float(value)) for index, value in enumerate(values) if finite(value)]
        if valid:
            bottom = max(valid, key=lambda item: item[1])[0]
            return sorted(set([0, max(0, bottom // 2), bottom, len(samples) - 1]))
    return sorted(set([0, len(samples) // 3, int(len(samples) * 2 / 3), len(samples) - 1]))


def make_contact_sheet(image_paths: list[Path], destination: Path) -> None:
    if not image_paths:
        return
    cells = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = resize_frame(image, 520)
        h, w = image.shape[:2]
        canvas = np.zeros((300, 520, 3), dtype=np.uint8)
        scale = min(520 / max(1, w), 300 / max(1, h))
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        rh, rw = resized.shape[:2]
        y = (300 - rh) // 2
        x = (520 - rw) // 2
        canvas[y:y + rh, x:x + rw] = resized
        cells.append(canvas)
    if not cells:
        return
    while len(cells) < 4:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:4])])
    cv2.imwrite(str(destination), sheet)


def analyze_video(args: argparse.Namespace) -> dict[str, Any]:
    motion_root = Path(args.motion_tracker_root)
    if str(motion_root) not in sys.path:
        sys.path.insert(0, str(motion_root))
    from src.backends.mediapipe_backend import MediaPipeBackend
    from src.core.angle_calculator import AngleCalculator
    from src.visualization.skeleton_renderer import SkeletonRenderer

    video_path = Path(args.video).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    logs: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(source_fps / max(1.0, float(args.sample_fps)))))
    logs.append({
        "stage": "video",
        "summary": "Opened input video",
        "details": {
            "video": str(video_path),
            "fps": round(source_fps, 3),
            "frameCount": frame_count,
            "width": width,
            "height": height,
            "sampleStep": step,
        },
    })

    estimator = MediaPipeBackend(
        model_complexity=max(0, min(2, int(args.model_complexity))),
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
        static_image_mode=True,
    )
    if not estimator.initialize():
        capture.release()
        raise RuntimeError("motion-tracker MediaPipe backend failed to initialize")
    calculator = AngleCalculator(use_3d=True)
    renderer = SkeletonRenderer(show_keypoints=True, show_connections=True, show_labels=False)

    samples: list[MotionSample] = []
    keyframe_source: dict[int, np.ndarray] = {}
    writer: cv2.VideoWriter | None = None
    writer_codec = None
    annotated_video = output_dir / "motion_tracker_annotated.mp4"
    sampled = 0
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            sampled += 1
            time_ms = int(round(frame_index * 1000.0 / max(1e-6, source_fps)))
            pose = estimator.process_frame(frame)
            if pose is not None and pose.is_valid(0.35):
                angles = calculator.calculate_all_angles(pose)
                posture = calculator.calculate_posture_metrics(pose)
                metrics = {
                    "elbow_angle": bilateral_angle(angles, "elbow"),
                    "hip_angle": bilateral_angle(angles, "hip"),
                    "knee_angle": bilateral_angle(angles, "knee"),
                    "torso_lean_2d": torso_lean_2d(pose),
                    "shoulder_hip_angle_2d": shoulder_hip_angle_2d(pose),
                    "elbow_below_shoulder": elbow_below_shoulder(pose),
                    "forearm_tilt_ratio": forearm_tilt_ratio(pose),
                }
                samples.append(MotionSample(
                    frame_index=frame_index,
                    time_ms=time_ms,
                    confidence=float(pose.confidence),
                    pose=pose,
                    angles=angles,
                    posture=posture,
                    metrics=metrics,
                ))
                annotated = renderer.render(frame, pose, angles)
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (annotated.shape[1], max(52, int(annotated.shape[0] * 0.08))), (8, 12, 10), -1)
                annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
                cv2.putText(
                    annotated,
                    f"motion-tracker | {args.action} | t={time_ms}ms | conf={pose.confidence:.3f}",
                    (14, max(32, int(annotated.shape[0] * 0.045))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.55, annotated.shape[1] / 1350),
                    (220, 255, 100),
                    2,
                    cv2.LINE_AA,
                )
                annotated = resize_frame(annotated, int(args.max_video_width))
                if writer is None:
                    out_h, out_w = annotated.shape[:2]
                    writer, writer_codec = open_video_writer(annotated_video, float(args.sample_fps), (out_w, out_h))
                writer.write(annotated)
                keyframe_source[len(samples) - 1] = annotated
            frame_index += 1
    finally:
        capture.release()
        estimator.release()
        if writer is not None:
            writer.release()

    pose_coverage = len(samples) / max(1, sampled)
    average_confidence = avg([sample.confidence for sample in samples], 0.0) or 0.0
    logs.append({
        "stage": "pose",
        "summary": "Extracted motion-tracker pose samples",
        "details": {
            "sampledFrames": sampled,
            "poseFrames": len(samples),
            "poseCoverage": round(pose_coverage, 3),
            "averageConfidence": round(average_confidence, 3),
            "annotatedVideo": annotated_video.name if annotated_video.exists() else None,
            "annotatedVideoCodec": writer_codec,
        },
    })

    evaluation = evaluate_action(args.action, samples, pose_coverage, average_confidence)
    logs.append({
        "stage": "rules",
        "summary": "Applied motion-tracker experiment rules",
        "details": {
            "action": args.action,
            "repCount": evaluation["repCount"],
            "issueCodes": [item["code"] for item in evaluation["issues"]],
            "score": evaluation["score"],
        },
    })

    key_indices = choose_key_samples(args.action, samples, evaluation["repEvents"])
    keyframes = []
    keyframe_paths = []
    for index, sample_index in enumerate(key_indices[:4], start=1):
        sample_index = max(0, min(len(samples) - 1, sample_index))
        image = keyframe_source.get(sample_index)
        if image is None:
            continue
        image_name = f"keyframe_{index}.jpg"
        image_path = output_dir / image_name
        cv2.imwrite(str(image_path), image)
        keyframe_paths.append(image_path)
        keyframes.append({
            "label": f"keyframe_{index}",
            "sampleIndex": sample_index,
            "frameIndex": samples[sample_index].frame_index,
            "timeMs": samples[sample_index].time_ms,
            "image": image_name,
            "confidence": round(samples[sample_index].confidence, 3),
        })
    contact_sheet = output_dir / "contact_sheet.jpg"
    make_contact_sheet(keyframe_paths, contact_sheet)

    result = {
        "engine": "motion-tracker",
        "engineRoot": str(motion_root),
        "actionType": args.action,
        "inputVideo": str(video_path),
        "durationSeconds": round(frame_count / max(1e-6, source_fps), 3),
        "fps": round(source_fps, 3),
        "frameCount": frame_count,
        "width": width,
        "height": height,
        "sampleFps": float(args.sample_fps),
        "sampledFrames": sampled,
        "poseFrames": len(samples),
        "poseCoverage": round(pose_coverage, 3),
        "averageConfidence": round(average_confidence, 3),
        "repCount": evaluation["repCount"],
        "repEvents": evaluation["repEvents"],
        "score": evaluation["score"],
        "safetyLevel": evaluation["safetyLevel"],
        "issues": evaluation["issues"],
        "strengths": evaluation["strengths"],
        "metricSummary": evaluation["metrics"],
        "keyframes": keyframes,
        "contactSheet": contact_sheet.name if contact_sheet.exists() else None,
        "annotatedVideo": {
            "filename": annotated_video.name if annotated_video.exists() else None,
            "codec": writer_codec,
            "browserOptimized": writer_codec == "avc1",
        },
        "logs": logs,
        "elapsedSeconds": round(time.time() - started, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "samples.json").write_text(json.dumps([
        {
            "sampleIndex": index,
            "frameIndex": sample.frame_index,
            "timeMs": sample.time_ms,
            "confidence": round(sample.confidence, 3),
            "angles": {key: clean_number(value, 2) for key, value in sample.angles.items()},
            "posture": {key: clean_number(value, 2) for key, value in sample.posture.items()},
            "metrics": {key: clean_number(value, 3) for key, value in sample.metrics.items()},
        }
        for index, sample in enumerate(samples)
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a motion-tracker based action experiment")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--action", required=True, help="Action type: row, lat_pulldown, deadlift, romanian_deadlift")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--motion-tracker-root", default=str(DEFAULT_MOTION_TRACKER_ROOT))
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-video-width", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_video(args)
    print(json.dumps({
        "summary": str(Path(args.output).resolve() / "summary.json"),
        "actionType": result["actionType"],
        "repCount": result["repCount"],
        "score": result["score"],
        "issueCodes": [item["code"] for item in result["issues"]],
        "poseCoverage": result["poseCoverage"],
        "averageConfidence": result["averageConfidence"],
        "annotatedVideo": result["annotatedVideo"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
