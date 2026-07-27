#!/usr/bin/env python3
"""Local exercise-video analysis for the Xiaoyu Coach web application.

The worker extracts a compact pose time-series, selects movement-phase
keyframes, calculates explainable measurements, and writes user-facing
evidence images. It never produces medical diagnoses.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from stability_profiles import stability_profile


class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


LANDMARK = PoseLandmark
POSE_CONNECTIONS = {
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
}

DEFAULT_POSE_BACKEND = "rtmlib"
POSE_CONFIDENCE_FLOOR = 0.55
FIXED_FOOT_ACTIONS = {"hack_squat", "hip_thrust"}
POSE_COMPARE_SECONDARY_BACKEND = "motion_tracker_mediapipe"
MOTION_TRACKER_COUNT_FAMILIES = {
    "pull",
    "hinge",
    "isolation_elbow",
    "isolation_knee",
    "isolation_shoulder",
}
AUTO_ACTION_TYPES = {"", "auto", "auto_detect", "detect", "unknown"}
AUTO_ACTION_BY_FAMILY = {
    "squat": "barbell_squat",
    "hinge": "deadlift",
    "press": "bench_press",
    "pull": "row",
    "isolation_shoulder": "lateral_raise",
    "isolation_elbow": "biceps_curl",
    "isolation_knee": "leg_extension",
}


def truthy_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def bounded_float_config(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def annotated_video_mode() -> str:
    mode = str(os.environ.get("ANNOTATED_VIDEO_MODE") or "selected").strip().lower()
    return mode if mode in {"selected", "all", "none"} else "selected"


def pose_engine_compare_enabled(payload: dict[str, Any]) -> bool:
    if "poseEngineCompare" in payload:
        return truthy_config(payload.get("poseEngineCompare"))
    return truthy_config(os.environ.get("POSE_ENGINE_COMPARE"))


def failed_pose_engine_comparison_payload(
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
        "secondaryBackend": POSE_COMPARE_SECONDARY_BACKEND,
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


def build_pose_fusion(
    *,
    comparison: dict[str, Any] | None,
    backend_diagnostics: dict[str, Any],
    rep_diagnostics: dict[str, Any] | None,
    primary_rep_count: int = 0,
    secondary_rule_summary: dict[str, Any] | None = None,
    capture_quality: str,
    action_type: str,
    family: str,
) -> dict[str, Any]:
    primary = comparison.get("primary", {}) if comparison else {}
    secondary = comparison.get("secondary", {}) if comparison else {}
    primary_quality = (
        float(primary.get("poseCoverage") or 0.0) * 0.55
        + float(primary.get("averageConfidence") or 0.0) * 0.45
    )
    secondary_quality = (
        float(secondary.get("poseCoverage") or 0.0) * 0.55
        + float(secondary.get("averageConfidence") or 0.0) * 0.45
    )
    top_deltas = comparison.get("topDivergentJoints", []) if comparison else []
    worst_delta = max((float(item.get("medianAbsDelta") or 0.0) for item in top_deltas), default=0.0)
    target_switches = int(backend_diagnostics.get("targetSwitchCount") or 0)
    multi_person_frames = int(backend_diagnostics.get("multiPersonFrames") or 0)
    target_lock_confidence = float(backend_diagnostics.get("targetLockConfidence") or 1.0)
    target_tracking_risky = target_switches > 0 or target_lock_confidence < 0.55
    count_unstable = bool((rep_diagnostics or {}).get("countUnstable"))
    secondary_rep_count = int((secondary_rule_summary or {}).get("repCount") or 0)
    secondary_count_unstable = bool((secondary_rule_summary or {}).get("countUnstable"))
    secondary_count_usable = (
        bool(comparison)
        and not comparison.get("error")
        and secondary_rep_count > 0
        and not secondary_count_unstable
        and capture_quality != "insufficient"
        and not target_tracking_risky
        and family in MOTION_TRACKER_COUNT_FAMILIES
    )

    selected = "xiaoyuCoach"
    rep_count_source = "motionTracker" if secondary_count_usable else "xiaoyuCoach"
    fused_rep_count = secondary_rep_count if secondary_count_usable else primary_rep_count
    confidence = 0.62
    reasons: list[str] = []
    tradeoffs: list[str] = [
        "xiaoyu-coach 鏇撮€傚悎澶氫汉銆侀伄鎸°€佺洰鏍囬攣瀹氬拰鏈€缁堝姩浣滆鍒欒瘎鍒嗐€?",
        "motion-tracker 鏇撮€傚悎鍗曚汉娓呮櫚鐢婚潰涓嬬殑 MediaPipe 33 鐐瑰彲瑙嗗寲銆佽搴﹀鏍稿拰浜哄伐閫夋ā鍨嬨€?",
    ]

    if not comparison or comparison.get("error"):
        reasons.append("绗簩楠ㄩ寮曟搸涓嶅彲鐢紝鏈淇濈暀 xiaoyu-coach 涓荤粨鏋溿€?")
        confidence = 0.52
        rep_count_source = "xiaoyuCoach"
        fused_rep_count = primary_rep_count
    elif capture_quality == "insufficient":
        selected = "needsReview"
        reasons.append("涓昏棰戣瘉鎹笉瓒筹紝涓ゅ楠ㄩ閮戒笉閫傚悎鐩存帴缁欓珮缃俊鍒ゆ柇銆?")
        confidence = 0.35
        rep_count_source = "xiaoyuCoach"
        fused_rep_count = primary_rep_count
    elif target_tracking_risky:
        reasons.append("鐢婚潰瀛樺湪鐩爣鍒囨崲鎴栫洰鏍囬攣瀹氱疆淇″害涓嶈冻锛屼紭鍏堜繚鐣?xiaoyu-coach 鐨勭洰鏍囬攣瀹氳兘鍔涖€?")
        confidence = 0.72
        rep_count_source = "xiaoyuCoach"
        fused_rep_count = primary_rep_count
    elif worst_delta >= 30.0:
        selected = "hybridReview"
        reasons.append("涓ゅ寮曟搸鍦ㄥ叧閿叧鑺傝搴︿笂鍒嗘杈冨ぇ锛岄渶瑕佺粨鍚堟爣娉ㄨ棰戜汉宸ョ‘璁ゃ€?")
        confidence = 0.46
    elif (
        secondary_rep_count > primary_rep_count
        and not secondary_count_unstable
        and family in {"pull", "hinge", "isolation_elbow"}
    ):
        selected = "motionTracker"
        reasons.append("motion-tracker 鐨勫悓鍔ㄤ綔娆℃暟鍒囧垎鏇村畬鏁达紝涓旀病鏈夎Е鍙戞鏁颁笉绋冲畾銆?")
        confidence = 0.74
    elif secondary_quality >= primary_quality + 0.10 and family in {"pull", "hinge", "isolation_elbow"}:
        selected = "motionTracker"
        reasons.append("motion-tracker 瑕嗙洊鐜囨垨骞冲潎缃俊搴︽槑鏄炬洿楂橈紝涓旇鍔ㄤ綔涓昏渚濊禆涓婅偄/楂嬮摪閾捐搴︺€?")
        confidence = 0.72
    elif count_unstable and secondary_quality >= primary_quality - 0.05:
        selected = "hybridReview"
        reasons.append("涓绘ā鍨嬫鏁板垏鍒嗕笉绋冲畾锛屽缓璁敤 motion-tracker 鏍囨敞瑙嗛澶嶆牳鍏抽敭甯у拰娆℃暟銆?")
        confidence = 0.58
    else:
        reasons.append("涓绘ā鍨嬭鐩栫巼銆佺疆淇″害鎴栫洰鏍囬攣瀹氭洿绋筹紝鏈€缁堣瘎鍒嗕粛浠?xiaoyu-coach 涓哄噯銆?")
        confidence = 0.68

    if comparison:
        reasons.append(
            f"Quality: xiaoyu-coach {primary_quality:.2f} / motion-tracker {secondary_quality:.2f}; "
            f"max median joint delta {worst_delta:.1f} deg."
        )
    if multi_person_frames > 0 and not target_tracking_risky:
        reasons.append("鐢婚潰瀛樺湪鑳屾櫙浜虹墿锛屼絾涓荤洰鏍囬攣瀹氱ǔ瀹氾紱鏈涓嶅啀鍥犳闃绘 motion-tracker 娆℃暟鍒囧垎銆?")
    if secondary_rule_summary:
        reasons.append(f"娆℃暟鍒囧垎锛歺iaoyu-coach {primary_rep_count} 娆?/ motion-tracker {secondary_rep_count} 娆°€?")
    if rep_count_source == "motionTracker":
        reasons.append("鏈瀵圭敤鎴峰睍绀虹殑璇嗗埆娆℃暟閲囩敤 motion-tracker锛屽洜涓哄畠鐨勫懆鏈熷垏鍒嗙ǔ瀹氫笖鏇撮€傚悎璇ュ姩浣滄棌銆?")
    if action_type == "lat_pulldown":
        tradeoffs.append("楂樹綅涓嬫媺浼樺厛鐪嬭倶瑙掍粠 >135掳 鍒?<90掳 鐨勫畬鏁翠笅鎷夊懆鏈燂紝motion-tracker 鏍囨敞瑙嗛瀵逛汉宸ョ‘璁ゆ洿鐩磋銆?")
    elif action_type == "y_raise":
        tradeoffs.append("Y瀛椾晶骞充妇浼樺厛鐪嬭偐閮ㄤ粠浣庝綅鎶埌 110掳 浠ヤ笂鐨?Y 瀛楅《閮紝鍚屾椂澶嶆牳鑲橀儴鏄惁鍩烘湰浼哥洿鍜岃函骞叉槸鍚﹀€熷姏鎽嗗姩銆?")
    elif action_type in {"row", "open_elbow_row"}:
        tradeoffs.append("鍒掕埞鏍锋湰閲?xiaoyu-coach 瀹规槗浣庝及娆℃暟锛宮otion-tracker 瀵硅倶瑙掑懆鏈熸洿鏁忔劅锛屼絾韬共鎽嗗姩浠嶈浜ょ粰瑙勫垯澶嶆牳銆?")
    elif family == "hinge":
        tradeoffs.append("纭媺/RDL 瑕佸厛纭 top-bottom-top 瀹屾暣鍛ㄦ湡锛宮otion-tracker 鍙緟鍔╃湅楂嬮摪閾撅紝涓绘ā鍨嬩繚鐣欏畨鍏ㄨ鍒欍€?")

    return {
        "enabled": bool(comparison),
        "selectedEngine": selected,
        "confidence": round(confidence, 3),
        "primaryQuality": round(primary_quality, 3),
        "secondaryQuality": round(secondary_quality, 3),
        "primaryRepCount": int(primary_rep_count),
        "secondaryRepCount": int(secondary_rep_count),
        "repCountSource": rep_count_source,
        "fusedRepCount": int(fused_rep_count),
        "maxMedianJointDelta": round(worst_delta, 2),
        "scoreSource": "xiaoyuCoach_rules",
        "metricAssist": "motionTracker" if selected in {"motionTracker", "hybridReview"} else "xiaoyuCoach",
        "reasons": reasons,
        "tradeoffs": tradeoffs,
    }


def public_rep_events(rep_events: Iterable[dict[str, Any]], source_engine: str) -> list[dict[str, Any]]:
    public_events: list[dict[str, Any]] = []
    for event in rep_events:
        public_event = {
            key: value
            for key, value in event.items()
            if not str(key).startswith("pose")
        }
        public_event["sourceEngine"] = source_engine
        public_events.append(public_event)
    return public_events


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def calculation_log(
    stage: str,
    title: str,
    summary: str,
    details: dict[str, Any] | None = None,
    status: str = "done",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "title": title,
        "summary": summary,
        "details": json_safe(details or {}),
    }


def numeric_summary(values: Iterable[float], digits: int = 3) -> dict[str, Any]:
    valid = np.array([float(item) for item in values if np.isfinite(item)], dtype=float)
    if valid.size == 0:
        return {"count": 0}
    return {
        "count": int(valid.size),
        "min": round(float(np.min(valid)), digits),
        "p05": round(float(np.percentile(valid, 5)), digits),
        "median": round(float(np.percentile(valid, 50)), digits),
        "p95": round(float(np.percentile(valid, 95)), digits),
        "max": round(float(np.max(valid)), digits),
    }


def find_signal_peaks(
    signal: np.ndarray,
    prominence: float,
    distance: int,
    prominence_window: int | None = None,
) -> np.ndarray:
    """Return prominent local maxima without requiring SciPy."""
    values = np.asarray(signal, dtype=float)
    if values.size < 3:
        return np.asarray([], dtype=int)
    candidates: list[int] = []
    distance_window = max(2, int(distance))
    window = max(distance_window, int(prominence_window or distance_window))
    for index in range(1, values.size - 1):
        if not (values[index] > values[index - 1] and values[index] >= values[index + 1]):
            continue
        left = values[max(0, index - window):index]
        right = values[index + 1:min(values.size, index + window + 1)]
        if not left.size or not right.size:
            continue
        local_prominence = values[index] - max(float(np.min(left)), float(np.min(right)))
        if local_prominence >= prominence:
            candidates.append(index)

    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: values[item], reverse=True):
        if all(abs(index - kept) >= distance_window for kept in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=int)


ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "machine_chest_press": {"name": "machine chest press", "family": "press", "bodyPart": "chest", "stages": ["extended", "lower", "bottom", "press"]},
    "machine_crunch": {"name": "machine abdominal crunch", "family": "core_flexion", "bodyPart": "core", "stages": ["upright", "crunch", "contracted", "return"]},
    "standing_hip_abduction": {"name": "standing hip abduction", "family": "isolation_hip", "bodyPart": "glutes", "stages": ["neutral", "abduct", "top", "return"]},
    "seated_hip_abduction": {"name": "seated hip abduction", "family": "isolation_hip", "bodyPart": "glutes", "stages": ["closed", "open", "wide", "return"]},
    "chest_supported_row": {"name": "chest-supported machine row", "family": "pull", "bodyPart": "back", "stages": ["stretch", "pull", "peak", "return"]},
    "t_bar_row": {"name": "T-bar row", "family": "pull", "bodyPart": "back", "stages": ["stretch", "pull", "peak", "return"]},
    "plate_loaded_pulldown": {"name": "plate-loaded pronated pulldown", "family": "pull", "bodyPart": "back", "stages": ["top stretch", "pull", "bottom", "return"]},
    "plate_loaded_romanian_deadlift": {"name": "plate-loaded Romanian deadlift", "family": "hinge", "bodyPart": "hamstrings / glutes", "stages": ["top", "hinge", "bottom", "stand"]},
    "single_arm_pulldown": {"name": "single-arm pulldown", "family": "pull", "bodyPart": "back", "stages": ["top stretch", "pull", "bottom", "return"]},
    "hack_squat": {"name": "hack squat", "family": "squat", "bodyPart": "legs / glutes", "stages": ["top", "descent", "depth", "ascent"]},
    "hip_thrust": {"name": "hip thrust", "family": "hinge", "bodyPart": "glutes", "stages": ["bottom", "drive", "lockout", "return"]},
    "back_extension": {"name": "back extension", "family": "hinge", "bodyPart": "posterior chain", "stages": ["top", "hinge", "bottom", "return"]},
    "plate_loaded_rear_leg_raise": {"name": "plate-loaded rear leg raise", "family": "hinge", "bodyPart": "glutes / hamstrings", "stages": ["bottom", "extend", "top", "return"]},
    "preacher_curl": {"name": "preacher curl", "family": "isolation_elbow", "bodyPart": "arms", "stages": ["extended", "curl", "top", "lower"]},
    "single_arm_hammer_row": {"name": "single-arm hammer row", "family": "pull", "bodyPart": "back", "stages": ["stretch", "pull", "peak", "return"]},
    "barbell_squat": {"name": "barbell squat", "family": "squat", "bodyPart": "legs", "stages": ["start", "descent", "bottom", "ascent"]},
    "goblet_squat": {"name": "goblet squat", "family": "squat", "bodyPart": "legs", "stages": ["start", "descent", "bottom", "ascent"]},
    "deadlift": {"name": "deadlift", "family": "hinge", "bodyPart": "posterior chain", "stages": ["start", "hinge", "bottom", "stand"]},
    "romanian_deadlift": {"name": "romanian deadlift", "family": "hinge", "bodyPart": "hamstrings / glutes", "stages": ["start", "hinge", "bottom", "stand"]},
    "bench_press": {"name": "bench press", "family": "press", "bodyPart": "chest", "stages": ["lockout", "lower", "bottom", "press"]},
    "dumbbell_press": {"name": "dumbbell press", "family": "press", "bodyPart": "chest", "stages": ["start", "lower", "bottom", "press"]},
    "shoulder_press": {"name": "shoulder press", "family": "press", "bodyPart": "shoulders", "stages": ["start", "press", "top", "return"]},
    "push_up": {"name": "push-up", "family": "press", "bodyPart": "chest", "stages": ["start", "lower", "bottom", "press"]},
    "dip": {"name": "dip", "family": "press", "bodyPart": "chest / triceps", "stages": ["start", "lower", "bottom", "press"]},
    "row": {"name": "row", "family": "pull", "bodyPart": "back", "stages": ["stretch", "pull", "peak", "return"]},
    "open_elbow_row": {"name": "open-elbow row", "family": "pull", "bodyPart": "upper back", "stages": ["stretch", "pull", "peak", "return"]},
    "lat_pulldown": {"name": "lat pulldown", "family": "pull", "bodyPart": "back", "stages": ["top", "pull", "bottom", "return"]},
    "pull_up": {"name": "pull-up", "family": "pull", "bodyPart": "back", "stages": ["bottom", "pull", "top", "lower"]},
    "face_pull": {"name": "face pull", "family": "pull", "bodyPart": "shoulders / upper back", "stages": ["start", "pull", "peak", "return"]},
    "lateral_raise": {"name": "lateral raise", "family": "isolation_shoulder", "bodyPart": "shoulders", "stages": ["start", "raise", "top", "return"]},
    "y_raise": {"name": "Y raise", "family": "isolation_shoulder", "bodyPart": "shoulders", "stages": ["low start", "raise", "Y top", "lower"]},
    "fly": {"name": "fly", "family": "isolation_shoulder", "bodyPart": "chest / shoulders", "stages": ["stretch", "close", "peak", "return"]},
    "biceps_curl": {"name": "biceps curl", "family": "isolation_elbow", "bodyPart": "arms", "stages": ["start", "curl", "peak", "lower"]},
    "triceps_extension": {"name": "triceps extension", "family": "isolation_elbow", "bodyPart": "arms", "stages": ["start", "extend", "peak", "return"]},
    "leg_extension": {"name": "leg extension", "family": "isolation_knee", "bodyPart": "legs", "stages": ["start", "extend", "peak", "lower"]},
    "leg_curl": {"name": "leg curl", "family": "isolation_knee", "bodyPart": "legs", "stages": ["start", "curl", "peak", "lower"]},
    "other": {"name": "custom movement", "family": "general", "bodyPart": "full body", "stages": ["start", "middle", "key", "return"]},
}
LEGACY_STABILITY_ISSUE_CODES = {
    "TORSO_SWAY",
    "HACK_SQUAT_SUPPORT_SHIFT",
    "HIP_THRUST_TORSO_SHIFT",
    "HINGE_TRUNK_CONTROL",
    "MACHINE_CHEST_PRESS_SUPPORT_SHIFT",
    "HAMMER_ROW_TORSO_COMPENSATION",
    "CHEST_SUPPORTED_ROW_SUPPORT_SHIFT",
    "PULL_TORSO_COMPENSATION",
    "PREACHER_CURL_BODY_SWING",
    "ISOLATION_ELBOW_BODY_SWING",
    "ISOLATION_BODY_SWING",
    "HIP_ABDUCTION_TORSO_SWAY",
}


CAMERA_GUIDANCE = {
    "squat": "Use a side or side-rear angle with the full lower body and machine path visible.",
    "hinge": "Use a side angle that includes feet, hips, spine, and load path.",
    "press": "Use a side-front angle that shows contact point, wrists, elbows, shoulders, and safety setup.",
    "pull": "Use a front-oblique angle that shows trunk, elbows, and handle path.",
    "isolation_shoulder": "Use a front or front-oblique angle that shows both shoulders, elbows, and top position.",
    "isolation_elbow": "Use a side-front angle with full upper-arm and elbow visibility.",
    "isolation_knee": "Use a side-front angle that shows knee travel and pad contact.",
    "isolation_hip": "Use a front-oblique angle that shows the pelvis, both knees, and the working leg or machine pads.",
    "core_flexion": "Use a side angle that shows the shoulders, rib cage, pelvis, and machine support.",
    "general": "Use a stable camera angle that keeps the full working set visible.",
}


CHAIN_BY_FAMILY = {
    "squat": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.RIGHT_KNEE,
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_ANKLE,
    ],
    "hinge": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.RIGHT_KNEE,
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_ANKLE,
    ],
    "press": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_ELBOW,
        LANDMARK.LEFT_WRIST,
        LANDMARK.RIGHT_WRIST,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ],
    "pull": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_ELBOW,
        LANDMARK.LEFT_WRIST,
        LANDMARK.RIGHT_WRIST,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ],
    "isolation_shoulder": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_ELBOW,
        LANDMARK.LEFT_WRIST,
        LANDMARK.RIGHT_WRIST,
    ],
    "isolation_elbow": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_ELBOW,
        LANDMARK.LEFT_WRIST,
        LANDMARK.RIGHT_WRIST,
    ],
    "isolation_knee": [
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.RIGHT_KNEE,
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_ANKLE,
    ],
    "isolation_hip": [
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.RIGHT_KNEE,
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_ANKLE,
    ],
    "core_flexion": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ],
    "general": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ],
}

LOWER_BODY_FAMILIES = {"squat", "hinge", "isolation_knee", "isolation_hip"}
UPPER_BODY_FAMILIES = {"press", "pull", "isolation_shoulder", "isolation_elbow"}
MATCH_CHECK_FAMILIES = sorted(LOWER_BODY_FAMILIES | UPPER_BODY_FAMILIES)

LANDMARK_NAME_TO_INDEX = {
    item.name.lower(): int(item)
    for item in LANDMARK
}

SUPPORT_ANKLE_LANDMARKS = {
    "LEFT": int(LANDMARK.LEFT_ANKLE),
    "RIGHT": int(LANDMARK.RIGHT_ANKLE),
}
SUPPORT_FOOT_DETAIL_LANDMARKS = {
    "LEFT": [int(LANDMARK.LEFT_HEEL), int(LANDMARK.LEFT_FOOT_INDEX)],
    "RIGHT": [int(LANDMARK.RIGHT_HEEL), int(LANDMARK.RIGHT_FOOT_INDEX)],
}
DISTAL_HAND_LANDMARKS = [
    int(LANDMARK.LEFT_WRIST),
    int(LANDMARK.RIGHT_WRIST),
    int(LANDMARK.LEFT_PINKY),
    int(LANDMARK.RIGHT_PINKY),
    int(LANDMARK.LEFT_INDEX),
    int(LANDMARK.RIGHT_INDEX),
    int(LANDMARK.LEFT_THUMB),
    int(LANDMARK.RIGHT_THUMB),
]
FIXED_FOOT_DISPLAY_SUPPRESSED_LANDMARKS = {
    "hack_squat": DISTAL_HAND_LANDMARKS,
    "hip_thrust": [
        int(LANDMARK.LEFT_KNEE),
        int(LANDMARK.RIGHT_KNEE),
        *DISTAL_HAND_LANDMARKS,
    ],
}
HINGE_DISPLAY_SUPPRESSED_LANDMARKS = [
    int(LANDMARK.LEFT_HEEL),
    int(LANDMARK.RIGHT_HEEL),
    int(LANDMARK.LEFT_FOOT_INDEX),
    int(LANDMARK.RIGHT_FOOT_INDEX),
]
SEGMENTATION_FILTER_LANDMARKS = {
    int(LANDMARK.LEFT_KNEE),
    int(LANDMARK.RIGHT_KNEE),
    int(LANDMARK.LEFT_ANKLE),
    int(LANDMARK.RIGHT_ANKLE),
    int(LANDMARK.LEFT_HEEL),
    int(LANDMARK.RIGHT_HEEL),
    int(LANDMARK.LEFT_FOOT_INDEX),
    int(LANDMARK.RIGHT_FOOT_INDEX),
}
YOLO_PERSON_SEGMENTATION_LANDMARKS = set(SEGMENTATION_FILTER_LANDMARKS)


@dataclass
class PoseFrame:
    frame_index: int
    time_ms: int
    landmarks: list[list[float]]
    signal: float
    quality: float
    target_id: int = 0
    candidate_count: int = 1
    target_lock_confidence: float = 1.0
    person_bbox: list[float] | None = None
    landmark_mask_scores: list[float] | None = None


@dataclass
class PoseBackendResult:
    frames: list[PoseFrame]
    diagnostics: dict[str, Any]


@dataclass
class TargetTracker:
    center: tuple[float, float] | None = None
    bbox: list[float] | None = None
    selected_index: int | None = None
    frame_count: int = 0
    multi_person_frames: int = 0
    target_switch_count: int = 0
    target_lost_count: int = 0
    rejected_distractor_count: int = 0
    lock_confidences: list[float] = field(default_factory=list)


def point(landmarks: list[list[float]], index: int) -> np.ndarray:
    return np.array(landmarks[int(index)][:3], dtype=float)


def midpoint(landmarks: list[list[float]], left: int, right: int) -> np.ndarray:
    return (point(landmarks, left) + point(landmarks, right)) / 2.0


def visibility(landmarks: list[list[float]], index: int) -> float:
    return float(landmarks[int(index)][3])


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_valid(values: Iterable[float], fallback: float = 0.0) -> float:
    valid = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else fallback


def bilateral_angle(
    landmarks: list[list[float]],
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> tuple[float, float, float]:
    left_value = angle(point(landmarks, left[0]), point(landmarks, left[1]), point(landmarks, left[2]))
    right_value = angle(point(landmarks, right[0]), point(landmarks, right[1]), point(landmarks, right[2]))
    return left_value, right_value, average_valid([left_value, right_value], 180.0)


def movement_signal(landmarks: list[list[float]], family: str) -> float:
    if family in {"squat", "isolation_knee"}:
        _, _, knee = bilateral_angle(
            landmarks,
            (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
            (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
        )
        return 180.0 - knee
    if family == "hinge":
        _, _, hip = bilateral_angle(
            landmarks,
            (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
            (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
        )
        return 180.0 - hip
    if family in {"press", "pull", "isolation_elbow"}:
        _, _, elbow = bilateral_angle(
            landmarks,
            (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
            (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
        )
        return 180.0 - elbow
    if family == "isolation_shoulder":
        _, _, shoulder = bilateral_angle(
            landmarks,
            (LANDMARK.LEFT_HIP, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW),
            (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW),
        )
        return shoulder
    if family == "isolation_hip":
        shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        torso_length = max(0.12, float(np.linalg.norm(shoulders[:2] - hips[:2])))
        knee_distance = abs(
            float(point(landmarks, LANDMARK.LEFT_KNEE)[0])
            - float(point(landmarks, LANDMARK.RIGHT_KNEE)[0])
        )
        return knee_distance / torso_length * 100.0
    if family == "core_flexion":
        shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        trunk = shoulders[:2] - hips[:2]
        return math.degrees(math.atan2(abs(float(trunk[0])), max(1e-6, abs(float(trunk[1])))))

    shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
    wrists = midpoint(landmarks, LANDMARK.LEFT_WRIST, LANDMARK.RIGHT_WRIST)
    hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
    return float((np.linalg.norm(wrists[:2] - shoulders[:2]) + np.linalg.norm(shoulders[:2] - hips[:2])) * 100)


def side_landmark_indices(side: str, names: Iterable[str]) -> list[int]:
    prefix = "LEFT" if str(side).lower() == "left" else "RIGHT"
    return [int(getattr(LANDMARK, f"{prefix}_{name}")) for name in names]


def chain_visibility(landmarks: list[list[float]], indices: Iterable[int]) -> float:
    values = [visibility(landmarks, int(index)) for index in indices]
    return float(np.mean(values)) if values else 0.0


def best_side_visibility(landmarks: list[list[float]], names: Iterable[str]) -> float:
    return max(
        chain_visibility(landmarks, side_landmark_indices("left", names)),
        chain_visibility(landmarks, side_landmark_indices("right", names)),
    )


def frame_quality(landmarks: list[list[float]], family: str) -> float:
    trunk = chain_visibility(landmarks, [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ])

    if family in {"squat", "isolation_knee", "isolation_hip"}:
        visible_leg = best_side_visibility(landmarks, ["HIP", "KNEE", "ANKLE"])
        return float(0.28 * trunk + 0.72 * visible_leg)

    if family == "hinge":
        visible_hinge = max(
            chain_visibility(landmarks, side_landmark_indices("left", ["SHOULDER", "HIP", "KNEE"])),
            chain_visibility(landmarks, side_landmark_indices("right", ["SHOULDER", "HIP", "KNEE"])),
        )
        return float(0.38 * trunk + 0.62 * visible_hinge)

    if family in {"press", "pull", "isolation_elbow"}:
        visible_arm = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        return float(0.24 * trunk + 0.76 * visible_arm)

    if family == "isolation_shoulder":
        visible_shoulder_chain = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        return float(visible_shoulder_chain)

    if family == "core_flexion":
        return float(trunk)

    chain = CHAIN_BY_FAMILY.get(family, CHAIN_BY_FAMILY["general"])
    return chain_visibility(landmarks, chain)


def action_frame_quality(
    landmarks: list[list[float]],
    family: str,
    action_type: str = "other",
) -> float:
    if action_type in {"machine_chest_press", "chest_supported_row", "t_bar_row", "plate_loaded_pulldown"}:
        trunk = chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])
        visible_arm = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        return float(0.20 * trunk + 0.80 * visible_arm)

    if action_type in {"standing_hip_abduction", "seated_hip_abduction"}:
        pelvis = chain_visibility(landmarks, [LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP])
        visible_leg = best_side_visibility(landmarks, ["HIP", "KNEE", "ANKLE"])
        return float(0.28 * pelvis + 0.72 * visible_leg)

    if action_type == "machine_crunch":
        return chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])

    if action_type == "single_arm_pulldown":
        trunk = chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])
        working_arm = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        return float(0.18 * trunk + 0.82 * working_arm)

    if action_type == "single_arm_hammer_row":
        trunk = chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])
        working_arm = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        return float(0.24 * trunk + 0.76 * working_arm)

    if action_type == "preacher_curl":
        upper_arm = best_side_visibility(landmarks, ["SHOULDER", "ELBOW", "WRIST"])
        elbow_support_context = best_side_visibility(landmarks, ["SHOULDER", "ELBOW"])
        return float(0.18 * elbow_support_context + 0.82 * upper_arm)

    if action_type == "plate_loaded_rear_leg_raise":
        trunk = chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])
        working_leg = best_side_visibility(landmarks, ["SHOULDER", "HIP", "KNEE"])
        return float(0.25 * trunk + 0.75 * working_leg)

    if action_type == "hack_squat":
        trunk = chain_visibility(landmarks, [
            LANDMARK.LEFT_SHOULDER,
            LANDMARK.RIGHT_SHOULDER,
            LANDMARK.LEFT_HIP,
            LANDMARK.RIGHT_HIP,
        ])
        visible_leg = best_side_visibility(landmarks, ["HIP", "KNEE"])
        return float(0.22 * trunk + 0.78 * visible_leg)

    return frame_quality(landmarks, family)


METRIC_LANDMARKS = {
    "kneeAngle": [
        LANDMARK.LEFT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_HIP,
        LANDMARK.RIGHT_KNEE,
        LANDMARK.RIGHT_ANKLE,
    ],
    "hipAngle": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.LEFT_KNEE,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.RIGHT_HIP,
        LANDMARK.RIGHT_KNEE,
    ],
    "trunkLean": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.LEFT_HIP,
        LANDMARK.RIGHT_HIP,
    ],
    "wristStack": [
        LANDMARK.LEFT_WRIST,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_WRIST,
        LANDMARK.RIGHT_ELBOW,
    ],
    "elbowAngle": [
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.LEFT_WRIST,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.RIGHT_ELBOW,
        LANDMARK.RIGHT_WRIST,
    ],
    "shoulderAngle": [
        LANDMARK.LEFT_HIP,
        LANDMARK.LEFT_SHOULDER,
        LANDMARK.LEFT_ELBOW,
        LANDMARK.RIGHT_HIP,
        LANDMARK.RIGHT_SHOULDER,
        LANDMARK.RIGHT_ELBOW,
    ],
    "ankleSupport": [
        LANDMARK.LEFT_ANKLE,
        LANDMARK.RIGHT_ANKLE,
    ],
}

BILATERAL_METRIC_CHAINS = {
    "kneeAngle": [
        [LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE],
        [LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE],
    ],
    "hipAngle": [
        [LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE],
        [LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE],
    ],
    "elbowAngle": [
        [LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST],
        [LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST],
    ],
    "shoulderAngle": [
        [LANDMARK.LEFT_HIP, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW],
        [LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW],
    ],
    "wristStack": [
        [LANDMARK.LEFT_WRIST, LANDMARK.LEFT_ELBOW],
        [LANDMARK.RIGHT_WRIST, LANDMARK.RIGHT_ELBOW],
    ],
}


def landmarks_confidence(landmarks: list[list[float]], indices: Iterable[int]) -> float:
    values = [visibility(landmarks, int(index)) for index in indices]
    return float(min(values)) if values else 0.0


def metric_confidence(landmarks: list[list[float]], metric: str) -> float:
    if metric in BILATERAL_METRIC_CHAINS:
        return min(
            landmarks_confidence(landmarks, chain)
            for chain in BILATERAL_METRIC_CHAINS[metric]
        )
    return landmarks_confidence(landmarks, METRIC_LANDMARKS.get(metric, []))


def metric_side_confidences(landmarks: list[list[float]], metric: str) -> list[float]:
    chains = BILATERAL_METRIC_CHAINS.get(metric)
    if not chains:
        return [metric_confidence(landmarks, metric)]
    return [landmarks_confidence(landmarks, chain) for chain in chains]


def visible_landmark_points(
    landmarks: list[list[float]],
    indices: Iterable[int] | None = None,
    threshold: float = 0.05,
) -> list[tuple[float, float]]:
    selected = list(indices) if indices is not None else range(len(landmarks))
    points: list[tuple[float, float]] = []
    for index in selected:
        item = landmarks[int(index)]
        if len(item) >= 4 and float(item[3]) >= threshold:
            points.append((float(item[0]), float(item[1])))
    return points


def landmark_bbox(
    landmarks: list[list[float]],
    indices: Iterable[int] | None = None,
) -> list[float]:
    points = visible_landmark_points(landmarks, indices)
    if not points:
        return [0.0, 0.0, 1.0, 1.0]
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return [
        clamp(min(xs)),
        clamp(min(ys)),
        clamp(max(xs)),
        clamp(max(ys)),
    ]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 1e-8 else 0.0


def normalize_target_roi(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = [value.get("x"), value.get("y"), value.get("width"), value.get("height")]
    elif isinstance(value, (list, tuple)):
        raw = list(value[:4])
    else:
        return None
    try:
        x, y, width, height = [float(item) for item in raw]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if max(abs(x), abs(y), abs(width), abs(height)) > 1.5:
        # Pixel-style ROI is intentionally not accepted here because the analyzer
        # does not know whether the client measured against a transformed preview.
        return None
    x1 = clamp(x)
    y1 = clamp(y)
    x2 = clamp(x + width)
    y2 = clamp(y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def frame_metric_confidences(frame: PoseFrame) -> dict[str, float]:
    return {
        metric: round(metric_confidence(frame.landmarks, metric), 3)
        for metric in METRIC_LANDMARKS
    }


def summarized_metric_confidences(frames: list[PoseFrame]) -> dict[str, float]:
    if not frames:
        return {metric: 0.0 for metric in METRIC_LANDMARKS}
    return {
        metric: round(percentile(
            [metric_confidence(frame.landmarks, metric) for frame in frames],
            25,
            0.0,
        ), 3)
        for metric in METRIC_LANDMARKS
    }


def low_confidence_windows(
    frames: list[PoseFrame],
    threshold: float = POSE_CONFIDENCE_FLOOR,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for index, frame in enumerate(frames):
        confidence = frame.quality
        if confidence < threshold and active is None:
            active = {
                "startIndex": index,
                "startTimeMs": frame.time_ms,
                "minConfidence": confidence,
            }
        elif confidence < threshold and active is not None:
            active["minConfidence"] = min(active["minConfidence"], confidence)
        elif confidence >= threshold and active is not None:
            previous = frames[index - 1]
            active.update({
                "endIndex": index - 1,
                "endTimeMs": previous.time_ms,
                "minConfidence": round(active["minConfidence"], 3),
            })
            windows.append(active)
            active = None
    if active is not None:
        last = frames[-1]
        active.update({
            "endIndex": len(frames) - 1,
            "endTimeMs": last.time_ms,
            "minConfidence": round(active["minConfidence"], 3),
        })
        windows.append(active)
    return windows


def landmark_index_from_name(name: Any) -> int | None:
    raw = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    aliases = {
        "l_shoulder": "left_shoulder",
        "r_shoulder": "right_shoulder",
        "l_elbow": "left_elbow",
        "r_elbow": "right_elbow",
        "l_wrist": "left_wrist",
        "r_wrist": "right_wrist",
        "l_hip": "left_hip",
        "r_hip": "right_hip",
        "l_knee": "left_knee",
        "r_knee": "right_knee",
        "l_ankle": "left_ankle",
        "r_ankle": "right_ankle",
    }
    return LANDMARK_NAME_TO_INDEX.get(aliases.get(raw, raw))


def normalize_pose_landmark_priors(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    raw_items = value if isinstance(value, list) else [value]
    priors: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        frame_index = item.get("frameIndex", item.get("frame_index"))
        time_ms = item.get("timeMs", item.get("time_ms"))
        landmarks = item.get("landmarks") or item.get("keypoints") or {}
        if not isinstance(landmarks, dict):
            continue
        parsed_landmarks: dict[int, list[float]] = {}
        for name, point_value in landmarks.items():
            index = landmark_index_from_name(name)
            if index is None:
                continue
            if isinstance(point_value, dict):
                raw_point = [point_value.get("x"), point_value.get("y"), point_value.get("z", 0.0), point_value.get("confidence", 1.0)]
            elif isinstance(point_value, (list, tuple)):
                raw_point = list(point_value)
            else:
                continue
            if len(raw_point) < 2:
                continue
            try:
                x = clamp(float(raw_point[0]))
                y = clamp(float(raw_point[1]))
                z = float(raw_point[2]) if len(raw_point) > 2 and raw_point[2] is not None else 0.0
                confidence = clamp(float(raw_point[3])) if len(raw_point) > 3 and raw_point[3] is not None else 1.0
            except (TypeError, ValueError):
                continue
            parsed_landmarks[int(index)] = [x, y, z, max(0.75, confidence)]
        if not parsed_landmarks:
            continue
        try:
            parsed_frame_index = int(frame_index) if frame_index is not None else None
        except (TypeError, ValueError):
            parsed_frame_index = None
        try:
            parsed_time_ms = int(time_ms) if time_ms is not None else None
        except (TypeError, ValueError):
            parsed_time_ms = None
        priors.append({
            "frameIndex": parsed_frame_index,
            "timeMs": parsed_time_ms,
            "landmarks": parsed_landmarks,
            "source": str(item.get("source") or "vision_prior"),
        })
    return priors


def apply_pose_landmark_priors(
    frames: list[PoseFrame],
    priors: list[dict[str, Any]],
    *,
    max_frame_distance: int = 2,
    max_time_distance_ms: int = 180,
) -> int:
    if not frames or not priors:
        return 0

    applied = 0
    for prior in priors:
        target_frame: PoseFrame | None = None
        frame_index = prior.get("frameIndex")
        if frame_index is not None:
            candidates = [
                frame for frame in frames
                if abs(int(frame.frame_index) - int(frame_index)) <= max_frame_distance
            ]
            if candidates:
                target_frame = min(candidates, key=lambda frame: abs(int(frame.frame_index) - int(frame_index)))
        elif prior.get("timeMs") is not None:
            time_ms = int(prior["timeMs"])
            candidates = [
                frame for frame in frames
                if abs(int(frame.time_ms) - time_ms) <= max_time_distance_ms
            ]
            if candidates:
                target_frame = min(candidates, key=lambda frame: abs(int(frame.time_ms) - time_ms))
        if target_frame is None:
            continue
        for index, prior_point in (prior.get("landmarks") or {}).items():
            landmark_index = int(index)
            if landmark_index < 0 or landmark_index >= len(target_frame.landmarks):
                continue
            current = target_frame.landmarks[landmark_index]
            if current[3] < 0.82:
                target_frame.landmarks[landmark_index] = list(prior_point)
            else:
                current[0] = float(0.35 * current[0] + 0.65 * prior_point[0])
                current[1] = float(0.35 * current[1] + 0.65 * prior_point[1])
                current[2] = float(0.35 * current[2] + 0.65 * prior_point[2])
                current[3] = max(float(current[3]), float(prior_point[3]))
            applied += 1
    return applied


def limb_length_outlier(
    landmarks: list[list[float]],
    proximal: int,
    joint: int,
    distal: int,
    *,
    min_ratio: float = 0.25,
    max_ratio: float = 2.45,
) -> bool:
    if landmarks_confidence(landmarks, [proximal, joint, distal]) < 0.35:
        return False
    proximal_joint = float(np.linalg.norm(point(landmarks, proximal)[:2] - point(landmarks, joint)[:2]))
    joint_distal = float(np.linalg.norm(point(landmarks, joint)[:2] - point(landmarks, distal)[:2]))
    if proximal_joint < 0.015 or joint_distal < 0.015:
        return True
    ratio = joint_distal / max(1e-6, proximal_joint)
    return bool(ratio < min_ratio or ratio > max_ratio)


def downweight_anatomical_outliers(frames: list[PoseFrame], family: str, action_type: str) -> int:
    if not frames:
        return 0
    changed = 0
    check_legs = family in {"squat", "hinge", "isolation_knee"}
    check_arms = family in {"press", "pull", "isolation_shoulder", "isolation_elbow"} or action_type in {"single_arm_pulldown"}
    for frame in frames:
        marks = frame.landmarks
        for side in ("LEFT", "RIGHT"):
            if check_legs:
                hip = getattr(LANDMARK, f"{side}_HIP")
                knee = getattr(LANDMARK, f"{side}_KNEE")
                ankle = getattr(LANDMARK, f"{side}_ANKLE")
                if limb_length_outlier(marks, int(hip), int(knee), int(ankle), min_ratio=0.22, max_ratio=2.65):
                    marks[int(ankle)][3] = min(float(marks[int(ankle)][3]), 0.18)
                    changed += 1
            if check_arms:
                shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
                elbow = getattr(LANDMARK, f"{side}_ELBOW")
                wrist = getattr(LANDMARK, f"{side}_WRIST")
                if limb_length_outlier(marks, int(shoulder), int(elbow), int(wrist), min_ratio=0.20, max_ratio=2.85):
                    marks[int(wrist)][3] = min(float(marks[int(wrist)][3]), 0.18)
                    changed += 1
    return changed


def smooth_low_confidence_landmarks(
    frames: list[PoseFrame],
    threshold: float = POSE_CONFIDENCE_FLOOR,
    family: str = "general",
    action_type: str = "other",
    pose_landmark_priors: list[dict[str, Any]] | None = None,
) -> list[PoseFrame]:
    """Interpolate coordinates only; low confidence stays low for rule weighting."""
    if len(frames) < 3:
        return frames
    result = [
        PoseFrame(
            frame_index=frame.frame_index,
            time_ms=frame.time_ms,
            landmarks=[list(item) for item in frame.landmarks],
            signal=frame.signal,
            quality=frame.quality,
            target_id=frame.target_id,
            candidate_count=frame.candidate_count,
            target_lock_confidence=frame.target_lock_confidence,
            person_bbox=list(frame.person_bbox) if frame.person_bbox else None,
            landmark_mask_scores=list(frame.landmark_mask_scores) if frame.landmark_mask_scores else None,
        )
        for frame in frames
    ]
    downweight_anatomical_outliers(result, family, action_type)
    apply_pose_landmark_priors(result, pose_landmark_priors or [])
    landmark_count = len(result[0].landmarks)
    for landmark_index in range(landmark_count):
        for index in range(1, len(result) - 1):
            if result[index].landmarks[landmark_index][3] >= threshold:
                continue
            before = next(
                (
                    left for left in range(index - 1, -1, -1)
                    if result[left].landmarks[landmark_index][3] >= threshold
                ),
                None,
            )
            after = next(
                (
                    right for right in range(index + 1, len(result))
                    if result[right].landmarks[landmark_index][3] >= threshold
                ),
                None,
            )
            if before is None or after is None:
                continue
            ratio = (index - before) / max(1, after - before)
            for axis in range(3):
                start = result[before].landmarks[landmark_index][axis]
                end = result[after].landmarks[landmark_index][axis]
                result[index].landmarks[landmark_index][axis] = float(start + (end - start) * ratio)
    for frame in result:
        frame.quality = action_frame_quality(frame.landmarks, family, action_type)
    return result


def clone_pose_frame(frame: PoseFrame, landmarks: list[list[float]] | None = None) -> PoseFrame:
    return PoseFrame(
        frame_index=frame.frame_index,
        time_ms=frame.time_ms,
        landmarks=[list(item) for item in (landmarks if landmarks is not None else frame.landmarks)],
        signal=frame.signal,
        quality=frame.quality,
        target_id=frame.target_id,
        candidate_count=frame.candidate_count,
        target_lock_confidence=frame.target_lock_confidence,
        person_bbox=list(frame.person_bbox) if frame.person_bbox else None,
        landmark_mask_scores=list(frame.landmark_mask_scores) if frame.landmark_mask_scores else None,
    )


def recover_short_display_landmark_gaps(
    frames: list[PoseFrame],
    threshold: float = POSE_CONFIDENCE_FLOOR,
    max_gap_frames: int = 3,
) -> list[PoseFrame]:
    """Recover short visual gaps without changing the analysis evidence."""
    result = [clone_pose_frame(frame) for frame in frames]
    if len(result) < 3:
        return result
    landmark_count = len(result[0].landmarks)
    for landmark_index in range(landmark_count):
        cursor = 1
        while cursor < len(result) - 1:
            if result[cursor].landmarks[landmark_index][3] >= threshold:
                cursor += 1
                continue
            gap_start = cursor
            while cursor < len(result) - 1 and result[cursor].landmarks[landmark_index][3] < threshold:
                cursor += 1
            gap_end = cursor - 1
            gap_length = gap_end - gap_start + 1
            before = gap_start - 1
            after = cursor
            if (
                gap_length > max_gap_frames
                or after >= len(result)
                or result[before].landmarks[landmark_index][3] < threshold
                or result[after].landmarks[landmark_index][3] < threshold
            ):
                continue
            for gap_index in range(gap_start, gap_end + 1):
                ratio = (gap_index - before) / (after - before)
                for axis in range(3):
                    start = result[before].landmarks[landmark_index][axis]
                    end = result[after].landmarks[landmark_index][axis]
                    result[gap_index].landmarks[landmark_index][axis] = float(start + (end - start) * ratio)
                # Rendering visibility only. The analysis frames retain their
                # original low confidence and remain evidence-gated.
                result[gap_index].landmarks[landmark_index][3] = min(
                    0.58,
                    float(result[before].landmarks[landmark_index][3]),
                    float(result[after].landmarks[landmark_index][3]),
                )
    return result


def fixed_foot_action(action_type: str) -> bool:
    return str(action_type or "").strip() in FIXED_FOOT_ACTIONS


def segmentation_filter_enabled(action_type: str, family: str) -> bool:
    if str(os.environ.get("MEDIAPIPE_SEGMENTATION_FILTER") or "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return fixed_foot_action(action_type) or family in {"squat", "hinge", "isolation_knee"}


def yolo_person_segmentation_filter_enabled(action_type: str, family: str) -> bool:
    if not truthy_config(os.environ.get("YOLO_PERSON_SEGMENTATION_FILTER")):
        return False
    if fixed_foot_action(action_type) and not truthy_config(os.environ.get("YOLO_PERSON_SEGMENTATION_FIXED_FOOT")):
        return False
    return fixed_foot_action(action_type) or family in {"squat", "hinge", "isolation_knee"}


def sample_segmentation_mask(mask: np.ndarray, x: float, y: float, radius: int = 3) -> float:
    if mask is None or mask.size == 0:
        return 1.0
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return 0.0
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return 0.0
    height, width = mask.shape[:2]
    px = int(round(float(x) * max(1, width - 1)))
    py = int(round(float(y) * max(1, height - 1)))
    x1 = max(0, px - radius)
    x2 = min(width, px + radius + 1)
    y1 = max(0, py - radius)
    y2 = min(height, py + radius + 1)
    patch = np.asarray(mask[y1:y2, x1:x2], dtype=float)
    return float(np.max(patch)) if patch.size else 0.0


def apply_segmentation_mask_filter(
    landmarks: list[list[float]],
    segmentation_mask: np.ndarray | None,
    *,
    action_type: str,
    family: str,
    threshold: float = 0.35,
) -> tuple[list[float] | None, dict[str, Any]]:
    if segmentation_mask is None or not segmentation_filter_enabled(action_type, family):
        return None, {"enabled": False, "filtered": 0}

    mask_scores = [
        sample_segmentation_mask(segmentation_mask, float(item[0]), float(item[1]))
        for item in landmarks
    ]
    filtered: list[dict[str, Any]] = []
    for landmark_index in SEGMENTATION_FILTER_LANDMARKS:
        if landmark_index >= len(landmarks):
            continue
        score = float(mask_scores[landmark_index])
        if score >= threshold:
            continue
        current_confidence = float(landmarks[landmark_index][3])
        landmarks[landmark_index][3] = min(current_confidence, max(0.02, score * 0.25))
        filtered.append({
            "index": landmark_index,
            "maskScore": round(score, 3),
            "previousConfidence": round(current_confidence, 3),
            "nextConfidence": round(float(landmarks[landmark_index][3]), 3),
        })

    return mask_scores, {
        "enabled": True,
        "threshold": round(float(threshold), 3),
        "filtered": len(filtered),
        "filteredLandmarks": filtered[:24],
    }


def landmark_bbox_pixels(
    landmarks: list[list[float]],
    width: int,
    height: int,
    min_confidence: float = 0.18,
) -> list[float] | None:
    points = [
        (clamp(float(item[0])) * width, clamp(float(item[1])) * height)
        for item in landmarks
        if len(item) >= 4 and float(item[3]) >= min_confidence
    ]
    if len(points) < 4:
        return None
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def load_yolo_person_segmentation_model() -> tuple[Any | None, dict[str, Any]]:
    model_path = str(os.environ.get("YOLO_PERSON_SEGMENTATION_MODEL") or "yolo11n-seg.pt").strip()
    try:
        from ultralytics import YOLO

        return YOLO(model_path), {"loaded": True, "model": model_path}
    except Exception as error:
        return None, {
            "loaded": False,
            "model": model_path,
            "error": f"{type(error).__name__}: {str(error)[:300]}",
        }


def yolo_person_mask_for_frame(
    model: Any,
    frame: np.ndarray,
    landmarks: list[list[float]],
    *,
    confidence_threshold: float,
    image_size: int,
    mask_threshold: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = frame.shape[:2]
    try:
        results = model.predict(
            frame,
            classes=[0],
            imgsz=image_size,
            conf=confidence_threshold,
            verbose=False,
        )
    except Exception as error:
        return None, {
            "detected": False,
            "error": f"{type(error).__name__}: {str(error)[:300]}",
        }
    if not results:
        return None, {"detected": False, "instanceCount": 0}
    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, {"detected": False, "instanceCount": 0}

    boxes = np.asarray(result.boxes.xyxy.detach().cpu().numpy(), dtype=float)
    confidences = np.asarray(result.boxes.conf.detach().cpu().numpy(), dtype=float)
    masks = result.masks.data.detach().cpu().numpy()
    pose_box = landmark_bbox_pixels(landmarks, width, height)
    best_index = 0
    best_score = -1.0
    for index, box in enumerate(boxes):
        box_list = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        overlap = bbox_iou(pose_box, box_list) if pose_box else 0.0
        score = overlap * 2.0 + float(confidences[index]) + 0.08 * (
            max(0.0, box_list[2] - box_list[0]) * max(0.0, box_list[3] - box_list[1]) / max(1.0, width * height)
        )
        if score > best_score:
            best_score = score
            best_index = index

    mask = np.asarray(masks[best_index], dtype=np.float32)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (mask >= mask_threshold).astype(np.float32)
    return mask, {
        "detected": True,
        "instanceCount": int(len(boxes)),
        "selectedIndex": int(best_index),
        "selectedConfidence": round(float(confidences[best_index]), 3),
        "selectedIou": round(float(bbox_iou(pose_box, boxes[best_index].tolist())) if pose_box else 0.0, 3),
    }


def apply_yolo_person_mask_filter(
    landmarks: list[list[float]],
    person_mask: np.ndarray | None,
    *,
    action_type: str,
    family: str,
    threshold: float = 0.55,
) -> tuple[list[float] | None, dict[str, Any]]:
    if person_mask is None or not yolo_person_segmentation_filter_enabled(action_type, family):
        return None, {"enabled": False, "filtered": 0}

    mask_scores = [
        sample_segmentation_mask(person_mask, float(item[0]), float(item[1]), radius=5)
        for item in landmarks
    ]
    filtered: list[dict[str, Any]] = []
    for landmark_index in YOLO_PERSON_SEGMENTATION_LANDMARKS:
        if landmark_index >= len(landmarks):
            continue
        score = float(mask_scores[landmark_index])
        if score >= threshold:
            continue
        current_confidence = float(landmarks[landmark_index][3])
        landmarks[landmark_index][3] = min(current_confidence, max(0.01, score * 0.18))
        filtered.append({
            "index": landmark_index,
            "maskScore": round(score, 3),
            "previousConfidence": round(current_confidence, 3),
            "nextConfidence": round(float(landmarks[landmark_index][3]), 3),
        })

    return mask_scores, {
        "enabled": True,
        "threshold": round(float(threshold), 3),
        "filtered": len(filtered),
        "filteredLandmarks": filtered[:24],
    }


def combine_landmark_mask_scores(
    first: list[float] | None,
    second: list[float] | None,
) -> list[float] | None:
    if first is None:
        return list(second) if second is not None else None
    if second is None:
        return list(first)
    limit = min(len(first), len(second))
    combined = [min(float(first[index]), float(second[index])) for index in range(limit)]
    if len(first) > limit:
        combined.extend(float(item) for item in first[limit:])
    elif len(second) > limit:
        combined.extend(float(item) for item in second[limit:])
    return combined


def body_scale_from_frames(frames: list[PoseFrame]) -> float:
    widths: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        if landmarks_confidence(marks, [LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER]) < 0.35:
            continue
        widths.append(float(np.linalg.norm(
            point(marks, LANDMARK.LEFT_SHOULDER)[:2]
            - point(marks, LANDMARK.RIGHT_SHOULDER)[:2]
        )))
    return max(0.06, percentile(widths, 50, 0.18))


def support_ankle_candidate(frame: PoseFrame, side: str, action_type: str) -> tuple[float, float, float] | None:
    prefix = "LEFT" if side == "LEFT" else "RIGHT"
    hip = getattr(LANDMARK, f"{prefix}_HIP")
    knee = getattr(LANDMARK, f"{prefix}_KNEE")
    ankle = getattr(LANDMARK, f"{prefix}_ANKLE")
    marks = frame.landmarks
    ankle_confidence = visibility(marks, ankle)
    knee_confidence = visibility(marks, knee)
    hip_confidence = visibility(marks, hip)
    min_ankle_confidence = 0.48 if action_type == "hip_thrust" else 0.32
    if ankle_confidence < min_ankle_confidence or knee_confidence < 0.35:
        return None
    ankle_point = point(marks, ankle)
    knee_point = point(marks, knee)
    if float(ankle_point[1]) + 0.04 < float(knee_point[1]):
        return None
    if hip_confidence >= 0.35:
        hip_point = point(marks, hip)
        hip_knee = float(np.linalg.norm(hip_point[:2] - knee_point[:2]))
        knee_ankle = float(np.linalg.norm(knee_point[:2] - ankle_point[:2]))
        if hip_knee >= 0.035 and (knee_ankle < hip_knee * 0.28 or knee_ankle > hip_knee * 2.25):
            return None
    return float(ankle_point[0]), float(ankle_point[1]), float(ankle_confidence)


def build_fixed_foot_support(frames: list[PoseFrame], action_type: str) -> dict[str, Any]:
    if not fixed_foot_action(action_type):
        return {"enabled": False, "ignoredForScoring": False}
    if not frames:
        return {
            "enabled": True,
            "actionType": action_type,
            "mode": "hide_unreliable_foot",
            "ignoredForScoring": True,
            "anchors": {},
            "anchorCount": 0,
            "source": "active_window_pose",
        }

    body_scale = body_scale_from_frames(frames)
    anchors: dict[str, list[float]] = {}
    diagnostics: dict[str, Any] = {}
    min_samples = max(3, min(8, len(frames) // 4))
    for side in ("LEFT", "RIGHT"):
        candidates = [
            candidate
            for frame in frames
            if (candidate := support_ankle_candidate(frame, side, action_type)) is not None
        ]
        side_key = side.lower()
        if len(candidates) < min_samples:
            diagnostics[side_key] = {"candidateCount": len(candidates), "accepted": False}
            continue
        points = np.asarray([[item[0], item[1]] for item in candidates], dtype=float)
        center = np.median(points, axis=0)
        spread = max(
            float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10)),
            float(np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10)),
        )
        normalized_spread = spread / body_scale
        accepted = normalized_spread <= (0.68 if action_type == "hack_squat" else 0.42)
        diagnostics[side_key] = {
            "candidateCount": len(candidates),
            "accepted": accepted,
            "spreadRatio": round(normalized_spread, 3),
        }
        if not accepted:
            continue
        anchor_confidence = max(0.68, min(0.94, percentile([item[2] for item in candidates], 50, 0.68)))
        anchors[side_key] = [float(center[0]), float(center[1]), 0.0, float(anchor_confidence)]

    return {
        "enabled": True,
        "actionType": action_type,
        "mode": "lock_ankle_hide_foot_detail" if anchors else "hide_unreliable_foot",
        "ignoredForScoring": True,
        "anchors": anchors,
        "anchorCount": len(anchors),
        "source": "active_window_pose_median",
        "bodyScale": round(body_scale, 4),
        "diagnostics": diagnostics,
    }


def apply_fixed_foot_display_lock(
    frames: Iterable[PoseFrame],
    action_type: str,
    support: dict[str, Any] | None,
) -> list[PoseFrame]:
    if not fixed_foot_action(action_type):
        return [clone_pose_frame(frame) for frame in frames]

    support = support or {}
    anchors = support.get("anchors") or {}
    result: list[PoseFrame] = []
    for frame in frames:
        marks = [list(item) for item in frame.landmarks]
        for landmark_index in FIXED_FOOT_DISPLAY_SUPPRESSED_LANDMARKS.get(action_type, []):
            if landmark_index < len(marks):
                marks[landmark_index][3] = min(float(marks[landmark_index][3]), 0.05)
        for side in ("LEFT", "RIGHT"):
            side_key = side.lower()
            ankle_index = SUPPORT_ANKLE_LANDMARKS[side]
            anchor = anchors.get(side_key)
            if anchor:
                marks[ankle_index] = [
                    clamp(float(anchor[0])),
                    clamp(float(anchor[1])),
                    float(anchor[2]) if len(anchor) > 2 else 0.0,
                    max(0.68, min(0.94, float(anchor[3]) if len(anchor) > 3 else 0.74)),
                ]
            else:
                marks[ankle_index][3] = min(float(marks[ankle_index][3]), 0.05)
            for detail_index in SUPPORT_FOOT_DETAIL_LANDMARKS[side]:
                marks[detail_index][0] = marks[ankle_index][0]
                marks[detail_index][1] = marks[ankle_index][1]
                marks[detail_index][2] = marks[ankle_index][2]
                marks[detail_index][3] = 0.0
        result.append(clone_pose_frame(frame, marks))
    return result


def display_suppressed_landmarks(action_type: str, family: str) -> list[int]:
    # Face keypoints are not needed for movement coaching. Keep them available to
    # the pose backend internally, but never render nose/eye/ear landmarks in
    # annotated videos or evidence frames.
    suppressed: list[int] = list(range(0, 11))
    if family == "hinge":
        suppressed.extend(HINGE_DISPLAY_SUPPRESSED_LANDMARKS)
    if fixed_foot_action(action_type):
        suppressed.extend(FIXED_FOOT_DISPLAY_SUPPRESSED_LANDMARKS.get(action_type, []))
    return sorted(set(suppressed))


def apply_display_landmark_suppression(
    frames: Iterable[PoseFrame],
    action_type: str,
    family: str,
) -> list[PoseFrame]:
    suppressed = display_suppressed_landmarks(action_type, family)
    if not suppressed:
        return [clone_pose_frame(frame) for frame in frames]
    result: list[PoseFrame] = []
    for frame in frames:
        marks = [list(item) for item in frame.landmarks]
        for landmark_index in suppressed:
            if landmark_index < len(marks):
                marks[landmark_index][3] = min(float(marks[landmark_index][3]), 0.0)
        result.append(clone_pose_frame(frame, marks))
    return result


def smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values
    window = max(3, min(window, len(values) if len(values) % 2 else len(values) - 1))
    if window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def detect_repetitions(
    signal: np.ndarray,
    sample_fps: float,
    prominence_ratio: float = 0.18,
    min_prominence: float = 3.0,
    distance_seconds: float = 1.0,
    prominence_window_seconds: float | None = None,
) -> tuple[list[int], float]:
    if len(signal) < max(6, int(sample_fps * 2)):
        return [], 0.0
    signal_range = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if signal_range < 4.0:
        return [], signal_range
    prominence = max(float(min_prominence), signal_range * float(prominence_ratio))
    distance = max(3, int(sample_fps * float(distance_seconds)))
    prominence_window = (
        max(distance, int(sample_fps * float(prominence_window_seconds)))
        if prominence_window_seconds
        else distance
    )
    peaks = find_signal_peaks(signal, prominence=prominence, distance=distance, prominence_window=prominence_window)
    return [int(value) for value in peaks], signal_range


def upper_side_chain_confidence(landmarks: list[list[float]], side: str) -> float:
    prefix = "LEFT" if side == "left" else "RIGHT"
    return landmarks_confidence(landmarks, [
        getattr(LANDMARK, f"{prefix}_SHOULDER"),
        getattr(LANDMARK, f"{prefix}_ELBOW"),
        getattr(LANDMARK, f"{prefix}_WRIST"),
    ])


def visible_press_side(frames: list[PoseFrame]) -> str:
    left_values = [upper_side_chain_confidence(frame.landmarks, "left") for frame in frames]
    right_values = [upper_side_chain_confidence(frame.landmarks, "right") for frame in frames]
    left_score = percentile(left_values, 25, 0.0)
    right_score = percentile(right_values, 25, 0.0)
    return "left" if left_score >= right_score else "right"


def bench_press_motion_signal(frames: list[PoseFrame], camera_angle: str) -> np.ndarray:
    if camera_angle not in {"side", "side_front", "side_rear"}:
        return np.array([movement_signal(frame.landmarks, "press") for frame in frames], dtype=float)
    side = visible_press_side(frames)
    prefix = "LEFT" if side == "left" else "RIGHT"
    wrist_index = getattr(LANDMARK, f"{prefix}_WRIST")
    elbow_index = getattr(LANDMARK, f"{prefix}_ELBOW")
    fallback_prefix = "RIGHT" if side == "left" else "LEFT"
    fallback_wrist = getattr(LANDMARK, f"{fallback_prefix}_WRIST")
    fallback_elbow = getattr(LANDMARK, f"{fallback_prefix}_ELBOW")
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        use_wrist = wrist_index
        use_elbow = elbow_index
        confidence = upper_side_chain_confidence(marks, side)
        if confidence < 0.35:
            fallback_side = "right" if side == "left" else "left"
            if upper_side_chain_confidence(marks, fallback_side) > confidence:
                use_wrist, use_elbow = fallback_wrist, fallback_elbow
        wrist_y = float(marks[int(use_wrist)][1])
        elbow_y = float(marks[int(use_elbow)][1])
        values.append((0.68 * wrist_y + 0.32 * elbow_y) * 100.0)
    return np.array(values, dtype=float)


def lat_pulldown_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        side_values: list[float] = []
        weights: list[float] = []
        for side in ("LEFT", "RIGHT"):
            shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
            elbow = getattr(LANDMARK, f"{side}_ELBOW")
            wrist = getattr(LANDMARK, f"{side}_WRIST")
            confidence = landmarks_confidence(marks, [shoulder, elbow, wrist])
            if confidence < 0.12:
                continue
            elbow_angle = angle(
                point(marks, shoulder),
                point(marks, elbow),
                point(marks, wrist),
            )
            if not np.isfinite(elbow_angle):
                continue
            side_values.append(180.0 - float(elbow_angle))
            weights.append(max(0.05, confidence))
        if side_values:
            values.append(float(np.average(side_values, weights=weights)))
        else:
            values.append(movement_signal(marks, "pull"))
    return np.array(values, dtype=float)


def upper_limb_side_prefix(side: str) -> str:
    return "LEFT" if str(side).lower() == "left" else "RIGHT"


def upper_limb_side_confidence(landmarks: list[list[float]], side: str) -> float:
    prefix = upper_limb_side_prefix(side)
    return landmarks_confidence(landmarks, [
        getattr(LANDMARK, f"{prefix}_SHOULDER"),
        getattr(LANDMARK, f"{prefix}_ELBOW"),
        getattr(LANDMARK, f"{prefix}_WRIST"),
    ])


def single_arm_pulldown_side_elbow_angle(landmarks: list[list[float]], side: str) -> float:
    prefix = upper_limb_side_prefix(side)
    return angle(
        point(landmarks, getattr(LANDMARK, f"{prefix}_SHOULDER")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_ELBOW")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_WRIST")),
    )


def single_arm_pulldown_side_summary(frames: list[PoseFrame], side: str) -> dict[str, Any]:
    prefix = upper_limb_side_prefix(side)
    elbow_index = getattr(LANDMARK, f"{prefix}_ELBOW")
    flexions: list[float] = []
    elbow_y_values: list[float] = []
    confidences: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = upper_limb_side_confidence(marks, side)
        if confidence < 0.12:
            continue
        elbow_angle = single_arm_pulldown_side_elbow_angle(marks, side)
        if not np.isfinite(elbow_angle):
            continue
        flexions.append(180.0 - float(elbow_angle))
        elbow_y_values.append(float(marks[int(elbow_index)][1]))
        confidences.append(float(confidence))
    flexion_range = percentile(flexions, 95, 0.0) - percentile(flexions, 5, 0.0)
    elbow_descent = percentile(elbow_y_values, 95, 0.0) - percentile(elbow_y_values, 5, 0.0)
    confidence = percentile(confidences, 25, 0.0)
    return {
        "side": str(side).lower(),
        "confidence": round(float(confidence), 3),
        "flexionRange": round(float(flexion_range), 3),
        "peakFlexion": round(percentile(flexions, 95, 0.0), 3),
        "elbowDescentRange": round(float(elbow_descent), 3),
        "score": round(float(confidence * (flexion_range + elbow_descent * 120.0)), 3),
    }


def single_arm_pulldown_working_side(frames: list[PoseFrame]) -> dict[str, Any]:
    left = single_arm_pulldown_side_summary(frames, "left")
    right = single_arm_pulldown_side_summary(frames, "right")
    selected = right if float(right["score"]) > float(left["score"]) else left
    return {
        "side": selected["side"],
        "confidence": selected["confidence"],
        "left": left,
        "right": right,
    }


def single_arm_pulldown_motion_signal_for_side(frames: list[PoseFrame], side: str) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = upper_limb_side_confidence(marks, side)
        elbow_angle = single_arm_pulldown_side_elbow_angle(marks, side)
        if confidence < 0.12 or not np.isfinite(elbow_angle):
            values.append(movement_signal(marks, "pull"))
            continue
        values.append(180.0 - float(elbow_angle))
    return np.array(values, dtype=float)


def single_arm_pulldown_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    working = single_arm_pulldown_working_side(frames)
    return single_arm_pulldown_motion_signal_for_side(frames, str(working["side"]))


def single_arm_hammer_row_side_summary(frames: list[PoseFrame], side: str) -> dict[str, Any]:
    prefix = upper_limb_side_prefix(side)
    wrist_index = getattr(LANDMARK, f"{prefix}_WRIST")
    flexions: list[float] = []
    wrist_x_values: list[float] = []
    confidences: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = upper_limb_side_confidence(marks, side)
        if confidence < 0.12:
            continue
        elbow_angle = single_arm_pulldown_side_elbow_angle(marks, side)
        if not np.isfinite(elbow_angle):
            continue
        flexions.append(180.0 - float(elbow_angle))
        wrist_x_values.append(float(marks[int(wrist_index)][0]))
        confidences.append(float(confidence))
    flexion_range = percentile(flexions, 95, 0.0) - percentile(flexions, 5, 0.0)
    wrist_x_range = percentile(wrist_x_values, 95, 0.0) - percentile(wrist_x_values, 5, 0.0)
    confidence = percentile(confidences, 25, 0.0)
    return {
        "side": str(side).lower(),
        "confidence": round(float(confidence), 3),
        "flexionRange": round(float(flexion_range), 3),
        "peakFlexion": round(percentile(flexions, 95, 0.0), 3),
        "wristXRange": round(float(wrist_x_range), 3),
        "score": round(float(confidence * (flexion_range + wrist_x_range * 160.0)), 3),
    }


def single_arm_hammer_row_working_side(frames: list[PoseFrame]) -> dict[str, Any]:
    left = single_arm_hammer_row_side_summary(frames, "left")
    right = single_arm_hammer_row_side_summary(frames, "right")
    selected = right if float(right["score"]) > float(left["score"]) else left
    return {
        "side": selected["side"],
        "confidence": selected["confidence"],
        "left": left,
        "right": right,
    }


def single_arm_hammer_row_motion_signal_for_side(frames: list[PoseFrame], side: str) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = upper_limb_side_confidence(marks, side)
        elbow_angle = single_arm_pulldown_side_elbow_angle(marks, side)
        if confidence < 0.12 or not np.isfinite(elbow_angle):
            values.append(movement_signal(marks, "pull"))
            continue
        values.append(180.0 - float(elbow_angle))
    return np.array(values, dtype=float)


def single_arm_hammer_row_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    working = single_arm_hammer_row_working_side(frames)
    return single_arm_hammer_row_motion_signal_for_side(frames, str(working["side"]))


def lower_limb_side_confidence(landmarks: list[list[float]], side: str, *, include_ankle: bool = False) -> float:
    prefix = upper_limb_side_prefix(side)
    chain = [
        getattr(LANDMARK, f"{prefix}_SHOULDER"),
        getattr(LANDMARK, f"{prefix}_HIP"),
        getattr(LANDMARK, f"{prefix}_KNEE"),
    ]
    if include_ankle:
        chain.append(getattr(LANDMARK, f"{prefix}_ANKLE"))
    return landmarks_confidence(landmarks, chain)


def lower_limb_side_hip_angle(landmarks: list[list[float]], side: str) -> float:
    prefix = upper_limb_side_prefix(side)
    return angle(
        point(landmarks, getattr(LANDMARK, f"{prefix}_SHOULDER")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_HIP")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_KNEE")),
    )


def plate_loaded_rear_leg_raise_side_summary(frames: list[PoseFrame], side: str) -> dict[str, Any]:
    hip_angles: list[float] = []
    confidences: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = lower_limb_side_confidence(marks, side)
        if confidence < 0.12:
            continue
        hip_angle = lower_limb_side_hip_angle(marks, side)
        if not np.isfinite(hip_angle):
            continue
        hip_angles.append(float(hip_angle))
        confidences.append(float(confidence))
    bottom_angle = percentile(hip_angles, 5, 180.0)
    top_angle = percentile(hip_angles, 95, 0.0)
    angle_range = max(0.0, top_angle - bottom_angle)
    confidence = percentile(confidences, 25, 0.0)
    return {
        "side": str(side).lower(),
        "confidence": round(float(confidence), 3),
        "bottomHipAngle": round(float(bottom_angle), 3),
        "topHipAngle": round(float(top_angle), 3),
        "hipAngleRange": round(float(angle_range), 3),
        "score": round(float(confidence * (angle_range + max(0.0, top_angle - 118.0) * 0.25)), 3),
    }


def plate_loaded_rear_leg_raise_working_side(frames: list[PoseFrame]) -> dict[str, Any]:
    left = plate_loaded_rear_leg_raise_side_summary(frames, "left")
    right = plate_loaded_rear_leg_raise_side_summary(frames, "right")
    selected = right if float(right["score"]) > float(left["score"]) else left
    return {
        "side": selected["side"],
        "confidence": selected["confidence"],
        "left": left,
        "right": right,
    }


def plate_loaded_rear_leg_raise_motion_signal_for_side(frames: list[PoseFrame], side: str) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = lower_limb_side_confidence(marks, side)
        hip_angle = lower_limb_side_hip_angle(marks, side)
        if confidence < 0.12 or not np.isfinite(hip_angle):
            values.append(hip_extension_angle(marks))
            continue
        values.append(float(hip_angle))
    return np.array(values, dtype=float)


def plate_loaded_rear_leg_raise_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    working = plate_loaded_rear_leg_raise_working_side(frames)
    return plate_loaded_rear_leg_raise_motion_signal_for_side(frames, str(working["side"]))


def hip_extension_angle(landmarks: list[list[float]]) -> float:
    _, _, hip = bilateral_angle(
        landmarks,
        (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
        (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
    )
    return hip


def hip_thrust_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    return np.array([hip_extension_angle(frame.landmarks) for frame in frames], dtype=float)


def y_raise_side_prefix(side: str) -> str:
    return "LEFT" if str(side).lower() == "left" else "RIGHT"


def y_raise_side_landmarks(side: str, *, include_wrist: bool = False) -> list[int]:
    prefix = y_raise_side_prefix(side)
    indices = [
        getattr(LANDMARK, f"{prefix}_HIP"),
        getattr(LANDMARK, f"{prefix}_SHOULDER"),
        getattr(LANDMARK, f"{prefix}_ELBOW"),
    ]
    if include_wrist:
        indices.append(getattr(LANDMARK, f"{prefix}_WRIST"))
    return indices


def y_raise_side_confidence(
    landmarks: list[list[float]],
    side: str,
    *,
    include_wrist: bool = False,
) -> float:
    return landmarks_confidence(landmarks, y_raise_side_landmarks(side, include_wrist=include_wrist))


def y_raise_side_shoulder_angle(landmarks: list[list[float]], side: str) -> float:
    prefix = y_raise_side_prefix(side)
    return angle(
        point(landmarks, getattr(LANDMARK, f"{prefix}_HIP")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_SHOULDER")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_ELBOW")),
    )


def y_raise_side_elbow_angle(landmarks: list[list[float]], side: str) -> float:
    prefix = y_raise_side_prefix(side)
    return angle(
        point(landmarks, getattr(LANDMARK, f"{prefix}_SHOULDER")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_ELBOW")),
        point(landmarks, getattr(LANDMARK, f"{prefix}_WRIST")),
    )


def y_raise_side_samples(frames: list[PoseFrame], side: str) -> tuple[list[float], list[float]]:
    values: list[float] = []
    confidences: list[float] = []
    for frame in frames:
        confidence = y_raise_side_confidence(frame.landmarks, side)
        if confidence < 0.12:
            continue
        shoulder_angle = y_raise_side_shoulder_angle(frame.landmarks, side)
        if not np.isfinite(shoulder_angle):
            continue
        values.append(float(shoulder_angle))
        confidences.append(float(confidence))
    return values, confidences


def y_raise_side_summary(frames: list[PoseFrame], side: str) -> dict[str, Any]:
    values, confidences = y_raise_side_samples(frames, side)
    valid_ratio = len(values) / max(1, len(frames))
    angle_range = percentile(values, 95, 0.0) - percentile(values, 10, 0.0)
    top_angle = percentile(values, 95, 0.0)
    low_angle = percentile(values, 10, 0.0)
    confidence = percentile(confidences, 25, 0.0)
    wrist_above = y_raise_wrist_above_shoulder_score(frames, side)
    # Y raise is a one-arm cable movement in the reference video, so side choice
    # should be driven by the moving arm instead of averaging both arms.
    score = (
        max(0.0, angle_range) * clamp(valid_ratio / 0.55) * max(0.2, confidence)
        + max(0.0, top_angle - 80.0) * 0.15
        + max(0.0, wrist_above) * 20.0
    )
    return {
        "side": side,
        "score": round(float(score), 3),
        "angleRange": round(float(angle_range), 1),
        "topAngle": round(float(top_angle), 1),
        "lowAngle": round(float(low_angle), 1),
        "validRatio": round(float(valid_ratio), 3),
        "confidence": round(float(confidence), 3),
        "wristAboveShoulder": round(float(wrist_above), 3),
    }


def y_raise_working_side(frames: list[PoseFrame]) -> dict[str, Any]:
    left = y_raise_side_summary(frames, "left")
    right = y_raise_side_summary(frames, "right")
    selected = left if float(left["score"]) >= float(right["score"]) else right
    other = right if selected is left else left
    confidence = clamp(
        float(selected["confidence"]) * 0.45
        + float(selected["validRatio"]) * 0.35
        + max(0.0, float(selected["score"]) - float(other["score"])) / max(30.0, float(selected["score"])) * 0.20
    )
    return {
        "side": selected["side"],
        "confidence": round(float(confidence), 3),
        "left": left,
        "right": right,
    }


def y_raise_motion_signal_for_side(frames: list[PoseFrame], side: str) -> np.ndarray:
    raw: list[float] = []
    for frame in frames:
        confidence = y_raise_side_confidence(frame.landmarks, side)
        shoulder_angle = y_raise_side_shoulder_angle(frame.landmarks, side)
        raw.append(float(shoulder_angle) if confidence >= 0.12 and np.isfinite(shoulder_angle) else float("nan"))

    values = np.array(raw, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        indices = np.arange(len(values), dtype=float)
        return np.interp(indices, indices[valid], values[valid]).astype(float)

    return np.array([movement_signal(frame.landmarks, "isolation_shoulder") for frame in frames], dtype=float)


def y_raise_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    working = y_raise_working_side(frames)
    return y_raise_motion_signal_for_side(frames, str(working["side"]))


def hack_squat_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        confidences = [
            landmarks_confidence(frame.landmarks, side_landmark_indices("left", ["HIP", "KNEE"])),
            landmarks_confidence(frame.landmarks, side_landmark_indices("right", ["HIP", "KNEE"])),
        ]
        side_index = 0 if confidences[0] >= confidences[1] else 1
        if confidences[side_index] < 0.35:
            values.append(float("nan"))
            continue
        values.append(float(hack_squat_visible_depth(frame.landmarks, side_index) * 100.0))
    fallback = average_valid(values, 0.0)
    return np.asarray([fallback if not np.isfinite(value) else value for value in values], dtype=float)


def machine_chest_press_front_signal(frames: list[PoseFrame]) -> np.ndarray:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        shoulders = midpoint(marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        torso_length = max(0.12, float(np.linalg.norm(shoulders[:2] - hips[:2])))
        wrist_separation = abs(
            float(point(marks, LANDMARK.LEFT_WRIST)[0])
            - float(point(marks, LANDMARK.RIGHT_WRIST)[0])
        )
        values.append(wrist_separation / torso_length * 100.0)
    return np.asarray(values, dtype=float)


def standing_hip_abduction_motion_signal_for_side(
    frames: list[PoseFrame],
    side: str,
) -> tuple[np.ndarray, float]:
    prefix = upper_limb_side_prefix(side)
    hip_index = getattr(LANDMARK, f"{prefix}_HIP")
    ankle_index = getattr(LANDMARK, f"{prefix}_ANKLE")
    raw: list[float] = []
    confidences: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        confidence = landmarks_confidence(marks, [hip_index, ankle_index])
        shoulders = midpoint(marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        torso_length = max(0.12, float(np.linalg.norm(shoulders[:2] - hips[:2])))
        displacement = abs(
            float(point(marks, ankle_index)[0]) - float(point(marks, hip_index)[0])
        ) / torso_length * 100.0
        raw.append(displacement if confidence >= 0.12 else float("nan"))
        confidences.append(confidence)
    values = np.asarray(raw, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        indexes = np.arange(len(values), dtype=float)
        values = np.interp(indexes, indexes[valid], values[valid]).astype(float)
    else:
        values = np.asarray([movement_signal(frame.landmarks, "isolation_hip") for frame in frames], dtype=float)
    motion_range = percentile(values.tolist(), 95, 0.0) - percentile(values.tolist(), 5, 0.0)
    score = motion_range * percentile(confidences, 25, 0.0)
    return values, float(score)


def standing_hip_abduction_motion_signal(frames: list[PoseFrame]) -> np.ndarray:
    left, left_score = standing_hip_abduction_motion_signal_for_side(frames, "left")
    right, right_score = standing_hip_abduction_motion_signal_for_side(frames, "right")
    return right if right_score > left_score else left


def motion_signal_series(
    frames: list[PoseFrame],
    family: str,
    action_type: str,
    camera_angle: str,
) -> tuple[np.ndarray, str]:
    if action_type == "machine_chest_press":
        if camera_angle in {"front", "front_oblique"}:
            return machine_chest_press_front_signal(frames), "machine_chest_press_handle_separation"
        return bench_press_motion_signal(frames, camera_angle), "machine_chest_press_visible_arm_vertical_path"
    if action_type == "bench_press":
        return bench_press_motion_signal(frames, camera_angle), "bench_press_visible_arm_y"
    if action_type == "lat_pulldown":
        return lat_pulldown_motion_signal(frames), "lat_pulldown_elbow_flexion_angle"
    if action_type == "single_arm_pulldown":
        return single_arm_pulldown_motion_signal(frames), "single_arm_pulldown_working_side_elbow_flexion_angle"
    if action_type == "single_arm_hammer_row":
        return single_arm_hammer_row_motion_signal(frames), "single_arm_hammer_row_working_side_elbow_flexion_angle"
    if action_type in {"chest_supported_row", "plate_loaded_pulldown"}:
        return single_arm_pulldown_motion_signal(frames), f"{action_type}_best_visible_arm_elbow_flexion_angle"
    if action_type == "hack_squat":
        return hack_squat_motion_signal(frames), "hack_squat_hip_knee_visible_depth"
    if action_type == "hip_thrust":
        return hip_thrust_motion_signal(frames), "hip_thrust_hip_extension_angle"
    if action_type == "plate_loaded_rear_leg_raise":
        return plate_loaded_rear_leg_raise_motion_signal(frames), "plate_loaded_rear_leg_raise_working_hip_extension_angle"
    if action_type == "back_extension":
        return np.array([movement_signal(frame.landmarks, "hinge") for frame in frames], dtype=float), "back_extension_hip_flexion_angle"
    if action_type == "preacher_curl":
        return np.array([movement_signal(frame.landmarks, "isolation_elbow") for frame in frames], dtype=float), "preacher_curl_elbow_flexion_angle"
    if action_type == "plate_loaded_romanian_deadlift":
        return np.array([movement_signal(frame.landmarks, "core_flexion") for frame in frames], dtype=float), "plate_loaded_rdl_trunk_hinge_angle"
    if action_type == "machine_crunch":
        return np.array([movement_signal(frame.landmarks, "core_flexion") for frame in frames], dtype=float), "machine_crunch_trunk_flexion_angle"
    if action_type == "standing_hip_abduction":
        return standing_hip_abduction_motion_signal(frames), "standing_hip_abduction_working_leg_lateral_displacement"
    if action_type == "seated_hip_abduction":
        return np.array([movement_signal(frame.landmarks, "isolation_hip") for frame in frames], dtype=float), f"{action_type}_normalized_knee_separation"
    if action_type == "y_raise":
        return y_raise_motion_signal(frames), "y_raise_working_side_shoulder_angle"
    return np.array([movement_signal(frame.landmarks, family) for frame in frames], dtype=float), f"{family}_joint_angle"


def trunk_lean_degrees(landmarks: list[list[float]]) -> float:
    shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
    hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
    trunk_vector = shoulders[:2] - hips[:2]
    return math.degrees(math.atan2(abs(float(trunk_vector[0])), max(1e-6, abs(float(trunk_vector[1])))))


def shoulder_hip_line_angle_degrees(landmarks: list[list[float]]) -> float:
    shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
    hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
    trunk_vector = shoulders[:2] - hips[:2]
    return abs(math.degrees(math.atan2(float(trunk_vector[1]), float(trunk_vector[0]))))


def knee_flexion_score(landmarks: list[list[float]]) -> float:
    _, _, knee = bilateral_angle(
        landmarks,
        (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
        (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
    )
    if not np.isfinite(knee):
        return 0.0
    return clamp((180.0 - knee - 10.0) / 70.0)


def normalized_motion_energy(signal: np.ndarray, sample_fps: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    diff = np.abs(np.diff(values, prepend=values[0]))
    window = max(3, int(sample_fps * 0.35) | 1)
    smoothed = smooth_signal(diff, window)
    scale = float(np.percentile(smoothed, 90)) if smoothed.size else 0.0
    if scale < 1e-6:
        return np.zeros_like(values)
    return np.clip(smoothed / scale, 0.0, 1.0)


def posture_activity_scores(
    frames: list[PoseFrame],
    family: str,
    action_type: str,
) -> np.ndarray:
    scores: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        quality_scale = clamp((frame.quality - 0.18) / 0.52)
        if action_type == "bench_press":
            posture = torso_horizontal_score(marks)
        elif action_type in {"lat_pulldown", "single_arm_pulldown", "single_arm_hammer_row", "chest_supported_row", "plate_loaded_pulldown"}:
            posture = clamp((movement_signal(marks, "pull") - 8.0) / 75.0)
        elif action_type == "hip_thrust":
            posture = clamp((hip_extension_angle(marks) - 105.0) / 55.0)
        elif action_type == "plate_loaded_rear_leg_raise":
            hip_values = [
                value
                for value in (
                    lower_limb_side_hip_angle(marks, "left"),
                    lower_limb_side_hip_angle(marks, "right"),
                )
                if np.isfinite(value)
            ]
            posture = clamp(((max(hip_values) if hip_values else 105.0) - 105.0) / 55.0)
        elif action_type == "preacher_curl":
            posture = clamp((movement_signal(marks, "isolation_elbow") - 8.0) / 85.0)
        elif action_type == "y_raise":
            posture = clamp((movement_signal(marks, "isolation_shoulder") - 35.0) / 85.0)
        elif action_type == "machine_crunch":
            posture = clamp((movement_signal(marks, "core_flexion") - 8.0) / 55.0)
        elif action_type in {"standing_hip_abduction", "seated_hip_abduction"}:
            posture = clamp((movement_signal(marks, "isolation_hip") - 20.0) / 90.0)
        elif family in {"hinge", "pull"}:
            posture = clamp((trunk_lean_degrees(marks) - 18.0) / 42.0)
        elif family in {"squat", "isolation_knee", "isolation_hip"}:
            posture = knee_flexion_score(marks)
        elif family in {"press", "isolation_shoulder", "isolation_elbow"}:
            posture = clamp((movement_signal(marks, family) - 8.0) / 55.0)
        else:
            posture = frame.quality
        scores.append(float(posture) * quality_scale)
    return np.asarray(scores, dtype=float)


def target_bbox_for_frame(frame: PoseFrame) -> list[float]:
    if frame.person_bbox:
        return frame.person_bbox
    return landmark_bbox(frame.landmarks, CHAIN_BY_FAMILY.get("general", []))


def target_tracking_quality_scores(frames: list[PoseFrame]) -> np.ndarray:
    """Score target-lock/crop consistency, not biomechanical stability."""
    if not frames:
        return np.asarray([], dtype=float)
    bboxes = [target_bbox_for_frame(frame) for frame in frames]
    areas = np.asarray([bbox_area(bbox) for bbox in bboxes], dtype=float)
    box_widths = np.asarray([max(0.0, float(bbox[2]) - float(bbox[0])) for bbox in bboxes], dtype=float)
    box_heights = np.asarray([max(0.0, float(bbox[3]) - float(bbox[1])) for bbox in bboxes], dtype=float)
    centers = np.asarray([bbox_center(bbox) for bbox in bboxes], dtype=float)

    median_area = max(0.015, float(np.percentile(areas, 50)))
    median_width = max(0.03, float(np.percentile(box_widths, 50)))
    median_height = max(0.05, float(np.percentile(box_heights, 50)))
    median_center = np.median(centers, axis=0)
    scores: list[float] = []
    for area, width, height, center in zip(areas, box_widths, box_heights, centers):
        area_ratio = float(area / median_area)
        width_ratio = float(width / median_width)
        height_ratio = float(height / median_height)
        area_score = 1.0 if area_ratio <= 1.65 else clamp(1.0 - (area_ratio - 1.65) / 1.35)
        width_score = 1.0 if width_ratio <= 1.75 else clamp(1.0 - (width_ratio - 1.75) / 1.35)
        height_score = 1.0 if height_ratio <= 1.75 else clamp(1.0 - (height_ratio - 1.75) / 1.35)
        size_score = min(area_score, width_score, height_score)
        center_distance = float(np.linalg.norm(center - median_center))
        center_score = clamp(1.0 - center_distance / 0.42)
        scores.append(0.65 * area_score + 0.20 * size_score + 0.15 * center_score)
    return np.asarray(scores, dtype=float)


def reliable_target_window(
    frames: list[PoseFrame],
    tracking_quality: np.ndarray,
    sample_fps: float,
) -> tuple[int, int, dict[str, Any] | None]:
    if len(frames) < 8 or tracking_quality.size != len(frames):
        return 0, max(0, len(frames) - 1), None
    smoothed = smooth_signal(tracking_quality, max(3, int(sample_fps * 0.45) | 1))
    mask = smoothed >= 0.42
    segments = active_segments(mask, max(1, int(sample_fps * 1.0)))
    if not segments:
        return 0, len(frames) - 1, None
    start, end = max(segments, key=lambda item: item[1] - item[0])
    padding = max(1, int(sample_fps * 0.55))
    start = max(0, start - padding)
    end = min(len(frames) - 1, end + padding)
    trimmed_start = start
    trimmed_end = len(frames) - 1 - end
    min_meaningful_trim = max(2, int(sample_fps * 0.4))
    if trimmed_start < min_meaningful_trim and trimmed_end < min_meaningful_trim:
        return 0, len(frames) - 1, None
    return start, end, {
        "reliableTargetStartPoseIndex": start,
        "reliableTargetEndPoseIndex": end,
        "reliableTargetStartTimeMs": frames[start].time_ms,
        "reliableTargetEndTimeMs": frames[end].time_ms,
        "trackingQualityTrimmedStartFrames": trimmed_start,
        "trackingQualityTrimmedEndFrames": trimmed_end,
        "targetTrackingQualityMedian": round(float(np.median(smoothed)), 3),
        "targetTrackingQualityMin": round(float(np.min(smoothed)), 3),
    }


def merge_segments(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged: list[list[int]] = [[segments[0][0], segments[0][1]]]
    for start, end in segments[1:]:
        previous = merged[-1]
        if start - previous[1] - 1 <= max_gap:
            previous[1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def active_segments(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if bool(active) and start is None:
            start = index
        elif not bool(active) and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return merge_segments(segments, max_gap)


def select_active_training_window(
    frames: list[PoseFrame],
    raw_signal: np.ndarray,
    family: str,
    action_type: str,
    sample_fps: float,
) -> tuple[int, int, dict[str, Any]]:
    if len(frames) < max(8, int(sample_fps * 1.5)):
        end_index = max(0, len(frames) - 1)
        return 0, end_index, {
            "enabled": True,
            "reason": "too_few_pose_frames",
            "startPoseIndex": 0,
            "endPoseIndex": end_index,
            "trimmedStartFrames": 0,
            "trimmedEndFrames": 0,
            "confidence": 0.0,
        }

    posture = posture_activity_scores(frames, family, action_type)
    motion = normalized_motion_energy(raw_signal, sample_fps)
    tracking_quality = target_tracking_quality_scores(frames)
    tracked_motion = motion * tracking_quality
    tracked_posture = posture * tracking_quality

    # Pull/curl phase is driven by the limb signal, not by trunk visibility.
    # Keep the window around every clear elbow-flexion cycle so a temporary
    # landmark-quality dip cannot discard later repetitions.
    elbow_phase_actions = {
        "lat_pulldown",
        "single_arm_pulldown",
        "single_arm_hammer_row",
        "chest_supported_row",
        "plate_loaded_pulldown",
        "preacher_curl",
    }
    if action_type in elbow_phase_actions and len(raw_signal) >= 4:
        phase_signal = smooth_signal(raw_signal, max(3, int(sample_fps * 0.25) | 1))
        flexed_thresholds = {
            "lat_pulldown": 90.0,
            "single_arm_pulldown": 110.0,
            "single_arm_hammer_row": 110.0,
            "chest_supported_row": 112.0,
            "plate_loaded_pulldown": 110.0,
        }
        phase_events = segment_lat_pulldown_repetitions(
            frames,
            phase_signal,
            sample_fps,
            flexed_elbow_angle=flexed_thresholds.get(action_type, 105.0),
        )
        if phase_events:
            start = int(phase_events[0].get("poseStartIndex") or 0)
            end = int(phase_events[-1].get("poseEndIndex") or len(frames) - 1)
            padding = max(1, int(sample_fps * 0.45))
            start = max(0, start - padding)
            end = min(len(frames) - 1, end + padding)
            if end > start:
                return start, end, {
                    "enabled": True,
                    "reason": f"{action_type}_all_elbow_turning_points",
                    "startPoseIndex": start,
                    "endPoseIndex": end,
                    "startTimeMs": frames[start].time_ms,
                    "endTimeMs": frames[end].time_ms,
                    "trimmedStartFrames": start,
                    "trimmedEndFrames": len(frames) - 1 - end,
                    "confidence": round(float(np.median(tracking_quality[start : end + 1])), 3),
                    "turningPointCount": len(phase_events),
                    "fullSignalRange": round(float(np.percentile(phase_signal, 95) - np.percentile(phase_signal, 5)), 3),
                }
    if action_type == "bench_press":
        posture_gate = np.where(posture >= 0.16, 1.0, 0.35)
        scores = (0.82 * tracked_posture + 0.18 * tracked_motion) * posture_gate
        reason = "bench_horizontal_posture"
    elif action_type in {"lat_pulldown", "single_arm_pulldown", "plate_loaded_pulldown"}:
        scores = 0.44 * tracked_posture + 0.56 * tracked_motion
        reason = (
            "lat_elbow_flexion_motion"
            if action_type == "lat_pulldown"
            else "single_arm_pulldown_elbow_flexion_motion"
            if action_type == "single_arm_pulldown"
            else "plate_loaded_pulldown_elbow_flexion_motion"
        )
    elif action_type in {"single_arm_hammer_row", "chest_supported_row", "preacher_curl"}:
        scores = 0.34 * tracked_posture + 0.66 * tracked_motion
        reason = f"{action_type}_elbow_flexion_motion"
    elif action_type in {"machine_crunch", "standing_hip_abduction", "seated_hip_abduction", "plate_loaded_romanian_deadlift"}:
        scores = 0.30 * tracked_posture + 0.70 * tracked_motion
        reason = f"{action_type}_primary_motion"
    elif action_type == "hip_thrust":
        scores = 0.42 * tracked_posture + 0.58 * tracked_motion
        reason = "hip_thrust_hip_extension_motion"
    elif family in {"hinge", "pull"}:
        scores = 0.74 * tracked_posture + 0.26 * tracked_motion
        reason = "trunk_lean_posture"
    elif family in {"squat", "isolation_knee", "isolation_hip"}:
        scores = 0.68 * tracked_posture + 0.32 * tracked_motion
        reason = "knee_flexion_posture"
    else:
        scores = 0.58 * tracked_posture + 0.42 * tracked_motion
        reason = "joint_motion_activity"

    scores = smooth_signal(scores, max(3, int(sample_fps * 0.45) | 1))
    max_score = float(np.max(scores)) if scores.size else 0.0
    if max_score < 0.18:
        reliable_start, reliable_end, tracking_diagnostics = reliable_target_window(frames, tracking_quality, sample_fps)
        if tracking_diagnostics:
            return reliable_start, reliable_end, {
                "enabled": True,
                "reason": "target_tracking_quality_trim",
                "startPoseIndex": reliable_start,
                "endPoseIndex": reliable_end,
                "startTimeMs": frames[reliable_start].time_ms,
                "endTimeMs": frames[reliable_end].time_ms,
                "trimmedStartFrames": reliable_start,
                "trimmedEndFrames": len(frames) - 1 - reliable_end,
                "confidence": round(max_score, 3),
                **tracking_diagnostics,
            }
        end_index = len(frames) - 1
        return 0, end_index, {
            "enabled": True,
            "reason": "insufficient_activity_contrast",
            "startPoseIndex": 0,
            "endPoseIndex": end_index,
            "startTimeMs": frames[0].time_ms,
            "endTimeMs": frames[-1].time_ms,
            "trimmedStartFrames": 0,
            "trimmedEndFrames": 0,
            "confidence": round(max_score, 3),
        }

    threshold = max(0.20, min(max_score * 0.58, percentile(scores.tolist(), 60, 0.0) * 0.85))
    mask = scores >= threshold
    segments = active_segments(mask, max(1, int(sample_fps * 0.8)))
    min_samples = max(6, int(sample_fps * 1.2))
    full_signal_range = max(1e-6, float(np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5)))

    full_signal_threshold = {
        "single_arm_hammer_row": 35.0,
        "machine_chest_press": 20.0,
        "machine_crunch": 10.0,
        "standing_hip_abduction": 10.0,
        "seated_hip_abduction": 8.0,
        "chest_supported_row": 20.0,
        "plate_loaded_pulldown": 20.0,
        "plate_loaded_romanian_deadlift": 15.0,
    }.get(action_type)
    if (
        full_signal_threshold is not None
        and full_signal_range >= full_signal_threshold
        and float(np.median(tracking_quality)) >= 0.72
    ):
        end_index = len(frames) - 1
        return 0, end_index, {
            "enabled": True,
            "reason": f"{action_type}_full_primary_signal",
            "startPoseIndex": 0,
            "endPoseIndex": end_index,
            "startTimeMs": frames[0].time_ms,
            "endTimeMs": frames[-1].time_ms,
            "trimmedStartFrames": 0,
            "trimmedEndFrames": 0,
            "confidence": round(max_score, 3),
            "threshold": round(float(threshold), 3),
            "maxScore": round(max_score, 3),
            "targetTrackingQualityMedian": round(float(np.median(tracking_quality)), 3),
            "targetTrackingQualityMin": round(float(np.min(tracking_quality)), 3),
            "fullSignalRange": round(float(full_signal_range), 3),
        }

    candidates: list[tuple[float, int, int, float]] = []
    for start, end in segments:
        duration = end - start + 1
        if duration < min_samples:
            continue
        mean_score = float(np.mean(scores[start : end + 1]))
        segment_range = float(np.percentile(raw_signal[start : end + 1], 95) - np.percentile(raw_signal[start : end + 1], 5))
        range_score = clamp(segment_range / full_signal_range)
        candidates.append((duration * (0.72 * mean_score + 0.28 * range_score), start, end, mean_score))

    if not candidates:
        reliable_start, reliable_end, tracking_diagnostics = reliable_target_window(frames, tracking_quality, sample_fps)
        if tracking_diagnostics:
            return reliable_start, reliable_end, {
                "enabled": True,
                "reason": "target_tracking_quality_trim",
                "startPoseIndex": reliable_start,
                "endPoseIndex": reliable_end,
                "startTimeMs": frames[reliable_start].time_ms,
                "endTimeMs": frames[reliable_end].time_ms,
                "trimmedStartFrames": reliable_start,
                "trimmedEndFrames": len(frames) - 1 - reliable_end,
                "confidence": round(max_score, 3),
                **tracking_diagnostics,
            }
        end_index = len(frames) - 1
        return 0, end_index, {
            "enabled": True,
            "reason": "no_stable_active_segment",
            "startPoseIndex": 0,
            "endPoseIndex": end_index,
            "startTimeMs": frames[0].time_ms,
            "endTimeMs": frames[-1].time_ms,
            "trimmedStartFrames": 0,
            "trimmedEndFrames": 0,
            "confidence": round(max_score, 3),
        }

    _, start, end, mean_score = max(candidates, key=lambda item: item[0])
    padding = max(1, int(sample_fps * 0.75))
    start = max(0, start - padding)
    end = min(len(frames) - 1, end + padding)
    trimmed_start = start
    trimmed_end = len(frames) - 1 - end
    selected_duration = (frames[end].time_ms - frames[start].time_ms) / 1000.0
    full_duration = max(1e-6, (frames[-1].time_ms - frames[0].time_ms) / 1000.0)

    if selected_duration / full_duration > 0.92 or (trimmed_start < int(sample_fps) and trimmed_end < int(sample_fps)):
        start = 0
        end = len(frames) - 1
        trimmed_start = 0
        trimmed_end = 0
        reason = "full_video_active"

    return start, end, {
        "enabled": True,
        "reason": reason,
        "startPoseIndex": start,
        "endPoseIndex": end,
        "startTimeMs": frames[start].time_ms,
        "endTimeMs": frames[end].time_ms,
        "trimmedStartFrames": trimmed_start,
        "trimmedEndFrames": trimmed_end,
        "confidence": round(float(mean_score), 3),
        "threshold": round(float(threshold), 3),
        "maxScore": round(max_score, 3),
        "targetTrackingQualityMedian": round(float(np.median(tracking_quality)), 3),
        "targetTrackingQualityMin": round(float(np.min(tracking_quality)), 3),
    }


def family_group(family: str) -> str:
    if family in LOWER_BODY_FAMILIES:
        return "lower_body"
    if family in UPPER_BODY_FAMILIES:
        return "upper_body"
    return "general"


def movement_signature(
    frames: list[PoseFrame],
    sample_fps: float,
    preferred_family: str | None = None,
) -> dict[str, Any]:
    if len(frames) < 8:
        return {
            "detectedGroup": "general",
            "detectedFamily": "general",
            "confidence": 0,
            "familyRanges": {},
            "familyReps": {},
            "lowerBodyRange": 0,
            "upperBodyRange": 0,
        }

    family_ranges: dict[str, float] = {}
    family_reps: dict[str, int] = {}
    window = max(3, int(sample_fps * 0.25) | 1)
    for candidate in MATCH_CHECK_FAMILIES:
        signal = np.array([movement_signal(frame.landmarks, candidate) for frame in frames], dtype=float)
        smoothed = smooth_signal(signal, window)
        peaks, signal_range = detect_repetitions(smoothed, sample_fps)
        family_ranges[candidate] = round(signal_range, 1)
        family_reps[candidate] = len(peaks)

    detected_family = max(family_ranges, key=lambda item: family_ranges[item])
    if preferred_family in family_ranges:
        best_range = family_ranges[detected_family]
        preferred_range = family_ranges[str(preferred_family)]
        if abs(preferred_range - best_range) <= max(1.0, best_range * 0.03):
            detected_family = str(preferred_family)
    detected_group = family_group(detected_family)
    lower_peak = max(family_ranges.get(item, 0.0) for item in LOWER_BODY_FAMILIES)
    upper_peak = max(family_ranges.get(item, 0.0) for item in UPPER_BODY_FAMILIES)
    detected_peak = family_ranges.get(detected_family, 0.0)
    second_peak = max(
        [value for key, value in family_ranges.items() if key != detected_family] or [0.0]
    )
    confidence = min(0.98, max(0.0, (detected_peak - second_peak) / max(8.0, detected_peak)))

    return {
        "detectedGroup": detected_group,
        "detectedFamily": detected_family,
        "confidence": round(confidence, 3),
        "familyRanges": family_ranges,
        "familyReps": family_reps,
        "lowerBodyRange": round(lower_peak, 1),
        "upperBodyRange": round(upper_peak, 1),
    }


def vertical_motion_range(
    frames: list[PoseFrame],
    landmark: int,
    *,
    min_visibility: float = 0.45,
) -> float:
    values = [
        float(frame.landmarks[int(landmark)][1])
        for frame in frames
        if visibility(frame.landmarks, landmark) >= min_visibility
    ]
    if len(values) < 3:
        return 0.0
    return float(percentile(values, 95) - percentile(values, 5))


def horizontal_motion_range(
    frames: list[PoseFrame],
    landmark: int,
    *,
    min_visibility: float = 0.45,
) -> float:
    values = [
        float(frame.landmarks[int(landmark)][0])
        for frame in frames
        if visibility(frame.landmarks, landmark) >= min_visibility
    ]
    if len(values) < 3:
        return 0.0
    return float(percentile(values, 95) - percentile(values, 5))


def wrist_above_shoulder_score(frames: list[PoseFrame]) -> float:
    values: list[float] = []
    for frame in frames:
        marks = frame.landmarks
        for side in ("LEFT", "RIGHT"):
            wrist = getattr(LANDMARK, f"{side}_WRIST")
            shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
            if landmarks_confidence(marks, [wrist, shoulder]) < 0.45:
                continue
            values.append(float(marks[int(shoulder)][1]) - float(marks[int(wrist)][1]))
    return percentile(values, 80, 0.0) if values else 0.0


def infer_pull_action_type(frames: list[PoseFrame]) -> tuple[str, dict[str, Any]]:
    elbow_y_range = average_valid([
        vertical_motion_range(frames, LANDMARK.LEFT_ELBOW),
        vertical_motion_range(frames, LANDMARK.RIGHT_ELBOW),
    ])
    elbow_x_range = average_valid([
        horizontal_motion_range(frames, LANDMARK.LEFT_ELBOW),
        horizontal_motion_range(frames, LANDMARK.RIGHT_ELBOW),
    ])
    wrist_above = wrist_above_shoulder_score(frames)
    overhead_start = wrist_above >= 0.055
    vertical_dominant = (
        elbow_y_range >= 0.14
        and elbow_y_range >= elbow_x_range * 1.25
        and wrist_above > -0.08
    )
    if overhead_start or vertical_dominant:
        return "lat_pulldown", {
            "reason": "wrists_or_elbows_move_from_overhead_to_downward_pull",
            "wristAboveShoulderScore": round(float(wrist_above), 3),
            "elbowVerticalRange": round(float(elbow_y_range), 3),
            "elbowHorizontalRange": round(float(elbow_x_range), 3),
            "overheadStart": overhead_start,
            "verticalDominant": vertical_dominant,
        }
    return "row", {
        "reason": "pull_motion_without_clear_overhead_start",
        "wristAboveShoulderScore": round(float(wrist_above), 3),
        "elbowVerticalRange": round(float(elbow_y_range), 3),
        "elbowHorizontalRange": round(float(elbow_x_range), 3),
        "overheadStart": overhead_start,
        "verticalDominant": vertical_dominant,
    }


def infer_shoulder_action_type(frames: list[PoseFrame]) -> tuple[str, dict[str, Any]]:
    working = y_raise_working_side(frames)
    working_side = str(working["side"])
    signal = y_raise_motion_signal(frames)
    top_angle = percentile(signal.tolist(), 95, 0.0) if signal.size else 0.0
    low_angle = percentile(signal.tolist(), 10, 0.0) if signal.size else 0.0
    signal_range = top_angle - low_angle
    wrist_above = y_raise_wrist_above_shoulder_score(frames, working_side)
    working_prefix = y_raise_side_prefix(working_side)
    elbow_y_range = vertical_motion_range(frames, getattr(LANDMARK, f"{working_prefix}_ELBOW"))
    top_frame = frames[int(np.argmax(signal))] if signal.size else None
    top_outward_ratio = y_raise_wrist_outward_ratio(top_frame.landmarks, working_side) if top_frame else 0.0
    top_vertical_outward_ratio = y_raise_vertical_outward_ratio(top_frame.landmarks, working_side) if top_frame else 0.0
    y_shape = top_outward_ratio >= 0.35 and top_vertical_outward_ratio >= 1.0
    y_top = top_angle >= 108.0 and wrist_above >= 0.035 and y_shape
    y_range = signal_range >= 48.0 and wrist_above >= 0.015 and y_shape
    if y_top or y_range:
        return "y_raise", {
            "reason": "shoulder_isolation_finishes_in_high_y_position",
            "topShoulderAngle": round(float(top_angle), 1),
            "lowShoulderAngle": round(float(low_angle), 1),
            "shoulderSignalRange": round(float(signal_range), 1),
            "wristAboveShoulderScore": round(float(wrist_above), 3),
            "elbowVerticalRange": round(float(elbow_y_range), 3),
            "topWristOutwardRatio": round(float(top_outward_ratio), 3),
            "topVerticalOutwardRatio": round(float(top_vertical_outward_ratio), 3),
            "yShape": y_shape,
            "yTop": y_top,
            "yRange": y_range,
            "workingSide": working_side,
            "workingSideDiagnostics": working,
        }
    return "lateral_raise", {
        "reason": "shoulder_isolation_without_clear_high_y_finish",
        "topShoulderAngle": round(float(top_angle), 1),
        "lowShoulderAngle": round(float(low_angle), 1),
        "shoulderSignalRange": round(float(signal_range), 1),
        "wristAboveShoulderScore": round(float(wrist_above), 3),
        "elbowVerticalRange": round(float(elbow_y_range), 3),
        "topWristOutwardRatio": round(float(top_outward_ratio), 3),
        "topVerticalOutwardRatio": round(float(top_vertical_outward_ratio), 3),
        "yShape": y_shape,
        "yTop": y_top,
        "yRange": y_range,
        "workingSide": working_side,
        "workingSideDiagnostics": working,
    }


def infer_action_type_from_frames(
    frames: list[PoseFrame],
    sample_fps: float,
) -> dict[str, Any]:
    signature = movement_signature(frames, sample_fps, preferred_family=None)
    detected_family = str(signature.get("detectedFamily") or "general")
    confidence = float(signature.get("confidence") or 0.0)
    if detected_family == "isolation_shoulder":
        shoulder_action_type, shoulder_features = infer_shoulder_action_type(frames)
        pull_action_type, pull_features = infer_pull_action_type(frames)
        if shoulder_action_type == "y_raise":
            action_type, features = shoulder_action_type, shoulder_features
        elif pull_action_type == "lat_pulldown":
            action_type, features = pull_action_type, {
                "reason": "shoulder_signal_with_clear_overhead_downward_pull",
                "shoulderFeatures": shoulder_features,
                **pull_features,
            }
        else:
            action_type, features = shoulder_action_type, {
                "pullFeatures": pull_features,
                **shoulder_features,
            }
    elif detected_family in UPPER_BODY_FAMILIES:
        pull_action_type, pull_features = infer_pull_action_type(frames)
        if (
            pull_action_type == "lat_pulldown"
            or detected_family == "pull"
            or float(pull_features.get("elbowHorizontalRange") or 0.0) >= 0.035
        ):
            action_type, features = pull_action_type, pull_features
        else:
            action_type = AUTO_ACTION_BY_FAMILY.get(detected_family, "other")
            features = {"reason": f"dominant_{detected_family}_signal", **pull_features}
    else:
        action_type = AUTO_ACTION_BY_FAMILY.get(detected_family, "other")
        features = {"reason": f"dominant_{detected_family}_signal"}
    return {
        "enabled": True,
        "actionType": action_type,
        "actionName": ACTION_CATALOG.get(action_type, ACTION_CATALOG["other"])["name"],
        "family": ACTION_CATALOG.get(action_type, ACTION_CATALOG["other"])["family"],
        "confidence": round(confidence, 3),
        "signature": signature,
        "features": features,
    }


def movement_match_profile(
    frames: list[PoseFrame],
    expected_family: str,
    sample_fps: float,
) -> dict[str, Any]:
    expected_group = family_group(expected_family)
    if expected_group == "general" or len(frames) < 8:
        return {
            "expectedGroup": expected_group,
            "detectedGroup": expected_group,
            "mismatch": False,
            "confidence": 0,
            "familyRanges": {},
        }

    signature = movement_signature(frames, sample_fps, preferred_family=expected_family)
    family_ranges = signature["familyRanges"]
    family_reps = signature["familyReps"]
    lower_peak = signature["lowerBodyRange"]
    upper_peak = signature["upperBodyRange"]
    detected_group = signature["detectedGroup"]
    detected_family = signature["detectedFamily"]
    expected_peak = lower_peak if expected_group == "lower_body" else upper_peak
    opposite_peak = upper_peak if expected_group == "lower_body" else lower_peak
    dominance_ratio = opposite_peak / max(1.0, expected_peak)
    mismatch = (
        detected_group != expected_group
        and opposite_peak >= 14.0
        and (expected_peak < 8.0 or dominance_ratio >= 1.6)
    )
    if expected_family == "hinge" and lower_peak >= 20.0 and family_reps.get("hinge", 0) >= 1:
        mismatch = False
    confidence = min(0.98, max(0.0, (dominance_ratio - 1.0) / 1.8))

    return {
        "expectedGroup": expected_group,
        "detectedGroup": detected_group,
        "expectedFamily": expected_family,
        "detectedFamily": detected_family,
        "mismatch": mismatch,
        "confidence": round(confidence, 3) if mismatch else 0,
        "lowerBodyRange": round(lower_peak, 1),
        "upperBodyRange": round(upper_peak, 1),
        "dominanceRatio": round(dominance_ratio, 2),
        "familyRanges": family_ranges,
        "familyReps": family_reps,
    }


def local_minimum(signal: np.ndarray, start: int, end: int) -> int:
    start = max(0, start)
    end = min(len(signal) - 1, end)
    if end <= start:
        return start
    return start + int(np.argmin(signal[start : end + 1]))


def closest_value_index(signal: np.ndarray, start: int, end: int, target: float) -> int:
    start = max(0, start)
    end = min(len(signal) - 1, end)
    if end <= start:
        return start
    return start + int(np.argmin(np.abs(signal[start : end + 1] - target)))


def select_stages(signal: np.ndarray, peaks: list[int], sample_fps: float) -> list[int]:
    if len(signal) < 4:
        return [0] * 4

    if peaks:
        peak = peaks[len(peaks) // 2]
        period = int(np.median(np.diff(peaks))) if len(peaks) > 1 else int(sample_fps * 2.2)
        period = max(int(sample_fps), period)
        start = local_minimum(signal, peak - period, peak - 1)
        end = local_minimum(signal, peak + 1, peak + period)
        if end <= peak:
            end = min(len(signal) - 1, peak + max(2, period // 2))
        mid_down = closest_value_index(signal, start, peak, (signal[start] + signal[peak]) / 2)
        mid_up = closest_value_index(signal, peak, end, (signal[peak] + signal[end]) / 2)
        return [start, mid_down, peak, mid_up]

    low = int(np.argmin(signal))
    high = int(np.argmax(signal))
    start, end = sorted([low, high])
    if end - start < 3:
        return [0, len(signal) // 3, (2 * len(signal)) // 3, len(signal) - 1]
    mid = closest_value_index(signal, start, end, (signal[start] + signal[end]) / 2)
    return [start, mid, end, min(len(signal) - 1, end + max(1, (end - start) // 2))]


def segment_repetitions(
    frames: list[PoseFrame],
    signal: np.ndarray,
    peaks: list[int],
    sample_fps: float,
    primary_family: str,
) -> list[dict[str, Any]]:
    if not frames or len(signal) < 4:
        return []

    peak_indices = [int(item) for item in peaks if 0 <= int(item) < len(frames)]
    if not peak_indices:
        stage_indices = select_stages(signal, peaks, sample_fps)
        start, key, end = stage_indices[0], stage_indices[2], stage_indices[-1]
        if end <= start:
            return []
        peak_indices = [key]

    events: list[dict[str, Any]] = []
    for order, peak in enumerate(peak_indices, start=1):
        previous_peak = peak_indices[order - 2] if order > 1 else 0
        next_peak = peak_indices[order] if order < len(peak_indices) else len(frames) - 1
        left_boundary = int((previous_peak + peak) / 2) if order > 1 else 0
        right_boundary = int((peak + next_peak) / 2) if order < len(peak_indices) else len(frames) - 1
        start = local_minimum(signal, left_boundary, peak)
        end = local_minimum(signal, peak, right_boundary)
        if end <= start:
            start = max(0, left_boundary)
            end = min(len(frames) - 1, right_boundary)
        if end <= start:
            continue
        window = frames[start : end + 1]
        quality = average_valid([item.quality for item in window])
        start_signal = float(signal[start])
        key_signal = float(signal[peak])
        end_signal = float(signal[end])
        top_value = min(start_signal, end_signal)
        amplitude = key_signal - top_value
        duration_seconds = (end - start) / max(1e-6, sample_fps)
        descent_seconds = (peak - start) / max(1e-6, sample_fps)
        ascent_seconds = (end - peak) / max(1e-6, sample_fps)
        events.append({
            "repIndex": order,
            "startFrameIndex": frames[start].frame_index,
            "keyFrameIndex": frames[peak].frame_index,
            "endFrameIndex": frames[end].frame_index,
            "startTimeMs": frames[start].time_ms,
            "keyTimeMs": frames[peak].time_ms,
            "endTimeMs": frames[end].time_ms,
            "poseStartIndex": start,
            "poseKeyIndex": peak,
            "poseEndIndex": end,
            "quality": round(quality, 3),
            "primaryFamily": primary_family,
            "startSignal": round(start_signal, 3),
            "keySignal": round(key_signal, 3),
            "endSignal": round(end_signal, 3),
            "signalAmplitude": round(float(amplitude), 2),
            "durationSeconds": round(float(duration_seconds), 2),
            "descentSeconds": round(float(descent_seconds), 2),
            "ascentSeconds": round(float(ascent_seconds), 2),
        })
    return events


def segment_lat_pulldown_repetitions(
    frames: list[PoseFrame],
    signal: np.ndarray,
    sample_fps: float,
    *,
    extended_elbow_angle: float = 135.0,
    flexed_elbow_angle: float = 90.0,
    counter_rule: str = "elbow_angle_gt_135_to_lt_90",
) -> list[dict[str, Any]]:
    if not frames or len(signal) < 4:
        return []

    extended_flexion_threshold = 180.0 - float(extended_elbow_angle)
    flexed_flexion_threshold = 180.0 - float(flexed_elbow_angle)
    min_gap = max(1, int(sample_fps * 0.35))
    complete_events: list[dict[str, Any]] = []
    index = 0
    while index < len(signal):
        while index < len(signal) and float(signal[index]) > extended_flexion_threshold:
            index += 1
        if index >= len(signal):
            break
        start = index
        while index < len(signal) and float(signal[index]) < flexed_flexion_threshold:
            index += 1
        if index >= len(signal):
            break
        flex_cross = index
        start_search = max(start, flex_cross - max(2, int(sample_fps * 2.5)))
        pre_flexion = np.asarray(signal[start_search : flex_cross + 1], dtype=float)
        if pre_flexion.size:
            local_floor = float(np.min(pre_flexion))
            floor_margin = max(0.8, float(np.max(pre_flexion) - local_floor) * 0.02)
            floor_indexes = np.flatnonzero(pre_flexion <= local_floor + floor_margin)
            if floor_indexes.size:
                start = start_search + int(floor_indexes[-1])
        search_end = flex_cross
        while search_end + 1 < len(signal) and float(signal[search_end + 1]) > extended_flexion_threshold:
            search_end += 1
            if search_end - flex_cross > int(sample_fps * 4.0):
                break
        key = start + int(np.argmax(signal[start : search_end + 1]))
        return_index = next((
            candidate
            for candidate in range(key + 1, min(len(signal), key + int(sample_fps * 4.0) + 1))
            if float(signal[candidate]) <= extended_flexion_threshold
        ), None)
        end = return_index if return_index is not None else min(len(signal) - 1, key + max(1, int(sample_fps * 0.45)))
        if end <= start:
            end = min(len(signal) - 1, start + 1)
        start_signal = float(signal[start])
        key_signal = float(signal[key])
        end_signal = float(signal[end])
        amplitude = key_signal - start_signal
        if key - start >= min_gap and amplitude >= (flexed_flexion_threshold - extended_flexion_threshold):
            window = frames[start : end + 1]
            complete_events.append({
                "repIndex": len(complete_events) + 1,
                "startFrameIndex": frames[start].frame_index,
                "keyFrameIndex": frames[key].frame_index,
                "endFrameIndex": frames[end].frame_index,
                "startTimeMs": frames[start].time_ms,
                "keyTimeMs": frames[key].time_ms,
                "endTimeMs": frames[end].time_ms,
                "poseStartIndex": start,
                "poseKeyIndex": key,
                "poseEndIndex": end,
                "quality": round(average_valid([item.quality for item in window]), 3),
                "primaryFamily": "pull",
                "startSignal": round(start_signal, 3),
                "keySignal": round(key_signal, 3),
                "endSignal": round(end_signal, 3),
                "signalAmplitude": round(float(amplitude), 2),
                "durationSeconds": round(float((end - start) / max(1e-6, sample_fps)), 2),
                "descentSeconds": round(float((key - start) / max(1e-6, sample_fps)), 2),
                "ascentSeconds": round(float((end - key) / max(1e-6, sample_fps)), 2),
                "counterRule": counter_rule,
                "phaseRule": "elbow_angle_direction_reversal",
                "rangeStatus": "complete",
                "startElbowAngle": round(180.0 - start_signal, 1),
                "keyElbowAngle": round(180.0 - key_signal, 1),
                "endElbowAngle": round(180.0 - end_signal, 1),
                "startElbowAngleThreshold": int(round(float(extended_elbow_angle))),
                "keyElbowAngleThreshold": int(round(float(flexed_elbow_angle))),
            })
        index = max(end + 1, key + min_gap)

    # Full-range cycles remain the safest counter. If the clip contains no
    # complete cycle, keep a clear direction-reversal cycle as a partial rep so
    # its eccentric phase is still shown and range can be reported separately.
    if complete_events:
        return complete_events

    # The signal is elbow flexion (180 - elbow angle). A local maximum is
    # therefore the smallest upper-arm/forearm angle. Phase changes at that
    # turning point even when the athlete reverses before reaching full range.
    signal_range = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    prominence = max(8.0, signal_range * 0.08)
    distance = max(2, int(sample_fps * 0.65))
    peaks = [int(value) for value in find_signal_peaks(
        signal,
        prominence=prominence,
        distance=distance,
        prominence_window=max(distance, int(sample_fps * 1.4)),
    )]
    candidates = segment_repetitions(frames, signal, peaks, sample_fps, "pull")
    minimum_phase_samples = max(1, int(round(sample_fps * 0.12)))
    events: list[dict[str, Any]] = []
    for candidate in candidates:
        start = int(candidate.get("poseStartIndex") or 0)
        key = int(candidate.get("poseKeyIndex") or start)
        end = int(candidate.get("poseEndIndex") or key)
        amplitude = float(candidate.get("signalAmplitude") or 0.0)
        if (
            amplitude < 8.0
            or key - start < minimum_phase_samples
            or end - key < minimum_phase_samples
        ):
            continue
        start_elbow_angle = 180.0 - float(candidate.get("startSignal") or 0.0)
        key_elbow_angle = 180.0 - float(candidate.get("keySignal") or 0.0)
        end_elbow_angle = 180.0 - float(candidate.get("endSignal") or 0.0)
        range_complete = (
            start_elbow_angle >= float(extended_elbow_angle)
            and key_elbow_angle <= float(flexed_elbow_angle)
        )
        event = {
            **candidate,
            "repIndex": len(events) + 1,
            "counterRule": counter_rule,
            "phaseRule": "elbow_angle_direction_reversal",
            "rangeStatus": "complete" if range_complete else "insufficient",
            "startElbowAngle": round(start_elbow_angle, 1),
            "keyElbowAngle": round(key_elbow_angle, 1),
            "endElbowAngle": round(end_elbow_angle, 1),
            "startElbowAngleThreshold": int(round(float(extended_elbow_angle))),
            "keyElbowAngleThreshold": int(round(float(flexed_elbow_angle))),
        }
        events.append(event)
    return events


def segment_y_raise_repetitions(
    frames: list[PoseFrame],
    signal: np.ndarray,
    sample_fps: float,
) -> list[dict[str, Any]]:
    if not frames or len(signal) < 4:
        return []

    low_threshold = 60.0
    top_threshold = 110.0
    min_amplitude = 48.0
    min_gap = max(1, int(sample_fps * 0.25))
    max_cycle_samples = max(8, int(sample_fps * 6.0))
    events: list[dict[str, Any]] = []
    index = 0

    while index < len(signal):
        while index < len(signal) and float(signal[index]) > low_threshold:
            index += 1
        if index >= len(signal):
            break
        start = index

        while index < len(signal) and float(signal[index]) < top_threshold:
            index += 1
            if index - start > max_cycle_samples:
                break
        if index >= len(signal) or float(signal[index]) < top_threshold:
            break
        top_cross = index

        search_end = top_cross
        while search_end + 1 < len(signal) and float(signal[search_end + 1]) > low_threshold:
            search_end += 1
            if search_end - start > max_cycle_samples:
                break

        key = start + int(np.argmax(signal[start : search_end + 1]))
        return_index = None
        for candidate in range(key + 1, min(len(signal), key + max_cycle_samples + 1)):
            if float(signal[candidate]) <= low_threshold:
                return_index = candidate
                break
        end = return_index if return_index is not None else min(len(signal) - 1, search_end)
        if end <= start:
            end = min(len(signal) - 1, start + 1)

        window = frames[start : end + 1]
        quality = average_valid([item.quality for item in window])
        start_signal = float(signal[start])
        key_signal = float(signal[key])
        end_signal = float(signal[end])
        amplitude = key_signal - start_signal
        duration_seconds = (end - start) / max(1e-6, sample_fps)
        raise_seconds = (key - start) / max(1e-6, sample_fps)
        lower_seconds = max(0.0, (end - key) / max(1e-6, sample_fps))

        if key - start >= min_gap and amplitude >= min_amplitude:
            events.append({
                "repIndex": len(events) + 1,
                "startFrameIndex": frames[start].frame_index,
                "keyFrameIndex": frames[key].frame_index,
                "endFrameIndex": frames[end].frame_index,
                "startTimeMs": frames[start].time_ms,
                "keyTimeMs": frames[key].time_ms,
                "endTimeMs": frames[end].time_ms,
                "poseStartIndex": start,
                "poseKeyIndex": key,
                "poseEndIndex": end,
                "quality": round(quality, 3),
                "primaryFamily": "isolation_shoulder",
                "startSignal": round(start_signal, 3),
                "keySignal": round(key_signal, 3),
                "endSignal": round(end_signal, 3),
                "signalAmplitude": round(float(amplitude), 2),
                "durationSeconds": round(float(duration_seconds), 2),
                "descentSeconds": round(float(raise_seconds), 2),
                "ascentSeconds": round(float(lower_seconds), 2),
                "counterRule": "shoulder_angle_lt_60_to_gt_110_y_top",
                "startShoulderAngleThreshold": int(low_threshold),
                "keyShoulderAngleThreshold": int(top_threshold),
            })

        index = max(end + 1, key + min_gap)

    return events

def segment_hip_thrust_repetitions(
    frames: list[PoseFrame],
    signal: np.ndarray,
    sample_fps: float,
) -> list[dict[str, Any]]:
    if not frames or len(signal) < 4:
        return []

    values = np.asarray(signal, dtype=float)
    low_threshold = min(140.0, percentile(values.tolist(), 30, 120.0) + 5.0)
    top_threshold = max(low_threshold + 25.0, percentile(values.tolist(), 78, low_threshold + 25.0))
    return_threshold = low_threshold + 4.0
    min_amplitude = 25.0
    min_gap = max(1, int(sample_fps * 0.25))
    max_cycle_samples = max(8, int(sample_fps * 6.0))
    events: list[dict[str, Any]] = []
    index = 0

    while index < len(values):
        while index < len(values) and float(values[index]) > low_threshold:
            index += 1
        if index >= len(values):
            break
        start = index

        while index < len(values) and float(values[index]) < top_threshold:
            index += 1
            if index - start > max_cycle_samples:
                break
        if index >= len(values) or float(values[index]) < top_threshold:
            break
        top_cross = index

        search_end = top_cross
        while search_end + 1 < len(values) and float(values[search_end + 1]) > return_threshold:
            search_end += 1
            if search_end - start > max_cycle_samples:
                break

        key = start + int(np.argmax(values[start : search_end + 1]))
        return_index = None
        for candidate in range(key + 1, min(len(values), key + max_cycle_samples + 1)):
            if float(values[candidate]) <= return_threshold:
                return_index = candidate
                break
        end = return_index if return_index is not None else min(len(values) - 1, search_end)
        if end <= start:
            end = min(len(values) - 1, start + 1)

        window = frames[start : end + 1]
        quality = average_valid([item.quality for item in window])
        start_signal = float(values[start])
        key_signal = float(values[key])
        end_signal = float(values[end])
        amplitude = key_signal - min(start_signal, end_signal)
        duration_seconds = (end - start) / max(1e-6, sample_fps)
        raise_seconds = (key - start) / max(1e-6, sample_fps)
        lower_seconds = max(0.0, (end - key) / max(1e-6, sample_fps))

        if key - start >= min_gap and amplitude >= min_amplitude and duration_seconds >= 0.45:
            events.append({
                "repIndex": len(events) + 1,
                "startFrameIndex": frames[start].frame_index,
                "keyFrameIndex": frames[key].frame_index,
                "endFrameIndex": frames[end].frame_index,
                "startTimeMs": frames[start].time_ms,
                "keyTimeMs": frames[key].time_ms,
                "endTimeMs": frames[end].time_ms,
                "poseStartIndex": start,
                "poseKeyIndex": key,
                "poseEndIndex": end,
                "quality": round(quality, 3),
                "primaryFamily": "hinge",
                "startSignal": round(start_signal, 3),
                "keySignal": round(key_signal, 3),
                "endSignal": round(end_signal, 3),
                "signalAmplitude": round(float(amplitude), 2),
                "durationSeconds": round(float(duration_seconds), 2),
                "descentSeconds": round(float(raise_seconds), 2),
                "ascentSeconds": round(float(lower_seconds), 2),
                "counterRule": "hip_angle_bottom_top_bottom",
                "bottomHipAngleThreshold": round(float(low_threshold), 2),
                "topHipAngleThreshold": round(float(top_threshold), 2),
            })

        index = max(end + 1, key + min_gap)

    return events


def segment_hinge_repetitions(
    frames: list[PoseFrame],
    sample_fps: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not frames or len(frames) < 4:
        return [], np.asarray([], dtype=float)

    raw = np.asarray([shoulder_hip_line_angle_degrees(frame.landmarks) for frame in frames], dtype=float)
    signal = smooth_signal(raw, max(3, int(sample_fps * 0.35) | 1))
    top_threshold = max(10.0, percentile(signal.tolist(), 25, 0.0) + 5.0)
    bottom_threshold = max(top_threshold + 25.0, percentile(signal.tolist(), 80, top_threshold + 25.0))
    max_cycle_samples = max(8, int(sample_fps * 8.5))
    min_amplitude = 25.0
    events: list[dict[str, Any]] = []
    index = 0

    while index < len(signal):
        while index < len(signal) and float(signal[index]) > top_threshold:
            index += 1
        if index >= len(signal):
            break
        start = index
        max_index = start

        while index < len(signal):
            if float(signal[index]) > float(signal[max_index]):
                max_index = index
            if index > start and float(signal[index]) >= bottom_threshold:
                break
            if index - start > max_cycle_samples:
                break
            index += 1

        if index >= len(signal) or float(signal[max_index]) < bottom_threshold:
            break

        end = index
        while end + 1 < len(signal) and float(signal[end]) > top_threshold + 5.0:
            end += 1
            if end - max_index > max_cycle_samples:
                break
        if float(signal[end]) > top_threshold + 12.0:
            index = max(end + 1, index + 1)
            continue

        window = frames[start : end + 1]
        quality = average_valid([item.quality for item in window])
        start_signal = float(signal[start])
        key_signal = float(signal[max_index])
        end_signal = float(signal[end])
        amplitude = key_signal - start_signal
        duration_seconds = (end - start) / max(1e-6, sample_fps)
        descent_seconds = (max_index - start) / max(1e-6, sample_fps)
        ascent_seconds = (end - max_index) / max(1e-6, sample_fps)

        if amplitude >= min_amplitude and duration_seconds >= 0.55:
            events.append({
                "repIndex": len(events) + 1,
                "startFrameIndex": frames[start].frame_index,
                "keyFrameIndex": frames[max_index].frame_index,
                "endFrameIndex": frames[end].frame_index,
                "startTimeMs": frames[start].time_ms,
                "keyTimeMs": frames[max_index].time_ms,
                "endTimeMs": frames[end].time_ms,
                "poseStartIndex": start,
                "poseKeyIndex": max_index,
                "poseEndIndex": end,
                "quality": round(quality, 3),
                "primaryFamily": "hinge",
                "startSignal": round(start_signal, 3),
                "keySignal": round(key_signal, 3),
                "endSignal": round(end_signal, 3),
                "signalAmplitude": round(float(amplitude), 2),
                "durationSeconds": round(float(duration_seconds), 2),
                "descentSeconds": round(float(descent_seconds), 2),
                "ascentSeconds": round(float(ascent_seconds), 2),
                "counterRule": "shoulder_hip_line_top_bottom_top",
                "topTorsoLeanThreshold": round(float(top_threshold), 2),
                "bottomTorsoLeanThreshold": round(float(bottom_threshold), 2),
            })
        index = max(end + 1, index + 1)

    return events, signal


def validate_rep_events(
    events: list[dict[str, Any]],
    signal: np.ndarray,
    sample_fps: float,
    action_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if action_type != "bench_press":
        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for event in events:
            start = int(event.get("poseStartIndex", 0))
            key = int(event.get("poseKeyIndex", 0))
            end = int(event.get("poseEndIndex", 0))
            if not (0 <= start <= key < len(signal) and 0 <= end < len(signal)):
                rejected.append({"repIndex": event.get("repIndex"), "reason": "invalid_index"})
                continue
            duration = float(event.get("durationSeconds") or ((end - start) / max(1e-6, sample_fps)))
            amplitude = float(event.get("signalAmplitude") or 0.0)
            if duration < 0.35:
                rejected.append({"repIndex": event.get("repIndex"), "reason": "too_short", "duration": round(duration, 2)})
                continue
            max_duration = 8.0 if action_type in {
                "hip_thrust",
                "back_extension",
                "romanian_deadlift",
                "plate_loaded_romanian_deadlift",
                "plate_loaded_rear_leg_raise",
            } else 6.0
            if duration > max_duration:
                rejected.append({"repIndex": event.get("repIndex"), "reason": "too_long", "duration": round(duration, 2)})
                continue
            if amplitude < 3.0:
                rejected.append({
                    "repIndex": event.get("repIndex"),
                    "reason": "low_amplitude",
                    "amplitude": round(amplitude, 2),
                })
                continue
            valid.append(dict(event))

        for index, event in enumerate(valid, start=1):
            event["repIndex"] = index

        raw_count = len(events)
        valid_count = len(valid)
        count_unstable = bool(
            raw_count > 0
            and (
                valid_count == 0
                or abs(raw_count - valid_count) > max(1, int(round(raw_count * 0.45)))
            )
        )
        return valid, {
            "rawPeakCount": raw_count,
            "validRepCount": valid_count,
            "rejectedRepCount": len(rejected),
            "rejectedReps": rejected[:12],
            "countUnstable": count_unstable,
        }

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for event in events:
        start = int(event.get("poseStartIndex", 0))
        key = int(event.get("poseKeyIndex", 0))
        end = int(event.get("poseEndIndex", 0))
        if not (0 <= start <= key < len(signal) and 0 <= end < len(signal)):
            rejected.append({"repIndex": event.get("repIndex"), "reason": "invalid_index"})
            continue
        duration = (end - start) / max(1e-6, sample_fps)
        descent_duration = (key - start) / max(1e-6, sample_fps)
        ascent_duration = (end - key) / max(1e-6, sample_fps)
        top_value = min(float(signal[start]), float(signal[end]))
        amplitude = float(signal[key]) - top_value
        if duration < 0.45:
            rejected.append({"repIndex": event.get("repIndex"), "reason": "too_short", "duration": round(duration, 2)})
            continue
        if descent_duration < 0.25 or ascent_duration < 0.25:
            rejected.append({
                "repIndex": event.get("repIndex"),
                "reason": "incomplete_cycle",
                "descent": round(descent_duration, 2),
                "ascent": round(ascent_duration, 2),
            })
            continue
        if duration > 6.0:
            rejected.append({"repIndex": event.get("repIndex"), "reason": "too_long", "duration": round(duration, 2)})
            continue
        if amplitude < 4.5:
            rejected.append({"repIndex": event.get("repIndex"), "reason": "low_amplitude", "amplitude": round(amplitude, 2)})
            continue
        next_event = dict(event)
        next_event["signalAmplitude"] = round(amplitude, 2)
        next_event["durationSeconds"] = round(duration, 2)
        valid.append(next_event)

    for index, event in enumerate(valid, start=1):
        event["repIndex"] = index

    raw_count = len(events)
    valid_count = len(valid)
    count_unstable = bool(
        raw_count > 0
        and (
            valid_count == 0
            or abs(raw_count - valid_count) > max(1, int(round(valid_count * 0.35)))
        )
    )
    return valid, {
        "rawPeakCount": raw_count,
        "validRepCount": valid_count,
        "rejectedRepCount": len(rejected),
        "rejectedReps": rejected[:12],
        "countUnstable": count_unstable,
    }


def select_stages_from_rep_events(
    signal: np.ndarray,
    rep_events: list[dict[str, Any]],
    sample_fps: float,
) -> list[int]:
    if not rep_events:
        return select_stages(signal, [], sample_fps)
    candidates = sorted(
        rep_events,
        key=lambda item: (
            float(item.get("signalAmplitude", 0.0)),
            float(item.get("quality", 0.0)),
            float(item.get("durationSeconds", 0.0)),
        ),
        reverse=True,
    )
    selected = candidates[len(candidates) // 2] if len(candidates) > 2 else candidates[0]
    start = int(selected.get("poseStartIndex", 0))
    key = int(selected.get("poseKeyIndex", start))
    end = int(selected.get("poseEndIndex", key))
    mid_down = closest_value_index(signal, start, key, (float(signal[start]) + float(signal[key])) / 2.0)
    mid_up = closest_value_index(signal, key, end, (float(signal[key]) + float(signal[end])) / 2.0)
    return [start, mid_down, key, mid_up]


def secondary_action_rule_summary(
    frames: list[Any],
    *,
    family: str,
    action_type: str,
    camera_angle: str,
    sample_fps: float,
) -> dict[str, Any]:
    if not frames:
        return {
            "repCount": 0,
            "repEvents": [],
            "signalSource": None,
            "signalRange": 0,
            "countUnstable": True,
        }
    ordered = sorted(frames, key=lambda item: int(getattr(item, "frame_index", 0)))
    for frame in ordered:
        frame.signal = movement_signal(frame.landmarks, family)
        frame.quality = action_frame_quality(frame.landmarks, family, action_type)
    raw_signal_full, signal_source = motion_signal_series(ordered, family, action_type, camera_angle)
    active_frames = ordered
    raw_signal = raw_signal_full
    smoothed = smooth_signal(raw_signal, max(3, int(sample_fps * 0.25) | 1))
    raw_rep_events: list[dict[str, Any]]
    if action_type in {
        "lat_pulldown",
        "single_arm_pulldown",
        "single_arm_hammer_row",
        "chest_supported_row",
        "plate_loaded_pulldown",
    }:
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        flexed_thresholds = {
            "lat_pulldown": 90.0,
            "single_arm_pulldown": 110.0,
            "single_arm_hammer_row": 110.0,
            "chest_supported_row": 112.0,
            "plate_loaded_pulldown": 110.0,
        }
        counter_rules = {
            "lat_pulldown": "elbow_angle_gt_135_to_lt_90",
            "single_arm_pulldown": "single_arm_elbow_angle_gt_135_to_lt_110",
            "single_arm_hammer_row": "single_arm_row_elbow_angle_gt_135_to_lt_110",
            "chest_supported_row": "visible_arm_elbow_angle_gt_135_to_lt_112",
            "plate_loaded_pulldown": "visible_arm_elbow_angle_gt_135_to_lt_110",
        }
        raw_rep_events = segment_lat_pulldown_repetitions(
            active_frames,
            smoothed,
            sample_fps,
            flexed_elbow_angle=flexed_thresholds.get(action_type, 90.0),
            counter_rule=counter_rules.get(action_type, "elbow_angle_gt_135_to_lt_90"),
        )
    elif action_type == "preacher_curl":
        peaks, signal_range = detect_repetitions(
            smoothed,
            sample_fps,
            prominence_ratio=0.14,
            min_prominence=2.0,
            distance_seconds=0.65,
        )
        raw_rep_events = segment_repetitions(active_frames, smoothed, peaks, sample_fps, "isolation_elbow")
        for event in raw_rep_events:
            event["counterRule"] = "preacher_elbow_flexion_peak_cycle"
    elif action_type == "y_raise":
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        raw_rep_events = segment_y_raise_repetitions(active_frames, smoothed, sample_fps)
    elif action_type in {"hip_thrust", "plate_loaded_rear_leg_raise"}:
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        raw_rep_events = segment_hip_thrust_repetitions(active_frames, smoothed, sample_fps)
        if action_type == "plate_loaded_rear_leg_raise":
            for event in raw_rep_events:
                event["counterRule"] = "working_hip_extension_bottom_top_bottom"
    elif action_type == "back_extension":
        peaks, signal_range = detect_repetitions(
            smoothed,
            sample_fps,
            prominence_ratio=0.14,
            min_prominence=2.0,
            distance_seconds=0.65,
        )
        raw_rep_events = segment_repetitions(active_frames, smoothed, peaks, sample_fps, "hinge")
    elif family == "hinge":
        raw_rep_events, smoothed = segment_hinge_repetitions(active_frames, sample_fps)
        raw_signal = smoothed
        signal_source = "hinge_shoulder_hip_line_top_bottom_top"
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
    else:
        peaks, signal_range = detect_repetitions(smoothed, sample_fps)
        signature = movement_signature(active_frames, sample_fps, preferred_family=family)
        raw_rep_events = segment_repetitions(active_frames, smoothed, peaks, sample_fps, signature["detectedFamily"])
    rep_events, diagnostics = validate_rep_events(raw_rep_events, smoothed, sample_fps, action_type)
    return {
        "repCount": len(rep_events),
        "repEvents": [
            {
                key: value
                for key, value in event.items()
                if not key.startswith("pose")
            }
            for event in rep_events[:20]
        ],
        "signalSource": signal_source,
        "signalRange": round(float(signal_range), 3),
        "countUnstable": bool(diagnostics.get("countUnstable")),
        "diagnostics": diagnostics,
        "averageQuality": round(average_valid([getattr(item, "quality", 0.0) for item in active_frames]), 3),
    }


def percentile(values: list[float], value: float, fallback: float = 0.0) -> float:
    valid = np.array([item for item in values if np.isfinite(item)], dtype=float)
    if valid.size == 0:
        return fallback
    return float(np.percentile(valid, value))


def measurement_series(frames: list[PoseFrame]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {
        "leftKneeAngle": [],
        "rightKneeAngle": [],
        "leftElbowAngle": [],
        "rightElbowAngle": [],
        "leftShoulderAngle": [],
        "rightShoulderAngle": [],
        "leftHipAngle": [],
        "rightHipAngle": [],
        "trunkLean": [],
        "torsoX": [],
        "torsoSupportOffsetX": [],
        "leftAnkleX": [],
        "rightAnkleX": [],
        "leftWristStack": [],
        "rightWristStack": [],
        "leftElbowHeight": [],
        "rightElbowHeight": [],
        "leftWristAboveShoulder": [],
        "rightWristAboveShoulder": [],
        "hipDepth": [],
        "shoulderWidth": [],
    }

    for item in frames:
        marks = item.landmarks
        left_knee, right_knee, _ = bilateral_angle(
            marks,
            (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
            (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
        )
        left_elbow, right_elbow, _ = bilateral_angle(
            marks,
            (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
            (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
        )
        left_shoulder, right_shoulder, _ = bilateral_angle(
            marks,
            (LANDMARK.LEFT_HIP, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW),
            (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW),
        )
        left_hip, right_hip, _ = bilateral_angle(
            marks,
            (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
            (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
        )
        shoulders = midpoint(marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        knees = midpoint(marks, LANDMARK.LEFT_KNEE, LANDMARK.RIGHT_KNEE)
        shoulder_width = max(
            0.03,
            float(np.linalg.norm(
                point(marks, LANDMARK.LEFT_SHOULDER)[:2]
                - point(marks, LANDMARK.RIGHT_SHOULDER)[:2]
            )),
        )
        trunk_vector = shoulders[:2] - hips[:2]
        trunk_lean = math.degrees(math.atan2(abs(float(trunk_vector[0])), max(1e-6, abs(float(trunk_vector[1])))))

        series["leftKneeAngle"].append(left_knee)
        series["rightKneeAngle"].append(right_knee)
        series["leftElbowAngle"].append(left_elbow)
        series["rightElbowAngle"].append(right_elbow)
        series["leftShoulderAngle"].append(left_shoulder)
        series["rightShoulderAngle"].append(right_shoulder)
        series["leftHipAngle"].append(left_hip)
        series["rightHipAngle"].append(right_hip)
        series["trunkLean"].append(trunk_lean)
        torso_x = float((shoulders[0] + hips[0]) / 2)
        left_ankle_x = float(point(marks, LANDMARK.LEFT_ANKLE)[0])
        right_ankle_x = float(point(marks, LANDMARK.RIGHT_ANKLE)[0])
        ankle_visibility = min(
            visibility(marks, LANDMARK.LEFT_ANKLE),
            visibility(marks, LANDMARK.RIGHT_ANKLE),
        )
        series["torsoX"].append(torso_x)
        series["torsoSupportOffsetX"].append(
            torso_x - (left_ankle_x + right_ankle_x) / 2
            if ankle_visibility >= 0.55
            else float("nan")
        )
        series["leftAnkleX"].append(left_ankle_x)
        series["rightAnkleX"].append(right_ankle_x)
        series["leftWristStack"].append(
            abs(float(point(marks, LANDMARK.LEFT_WRIST)[0] - point(marks, LANDMARK.LEFT_ELBOW)[0]))
            / shoulder_width
        )
        series["rightWristStack"].append(
            abs(float(point(marks, LANDMARK.RIGHT_WRIST)[0] - point(marks, LANDMARK.RIGHT_ELBOW)[0]))
            / shoulder_width
        )
        series["leftElbowHeight"].append(
            float(point(marks, LANDMARK.LEFT_ELBOW)[1] - point(marks, LANDMARK.LEFT_SHOULDER)[1])
        )
        series["rightElbowHeight"].append(
            float(point(marks, LANDMARK.RIGHT_ELBOW)[1] - point(marks, LANDMARK.RIGHT_SHOULDER)[1])
        )
        series["leftWristAboveShoulder"].append(
            float(point(marks, LANDMARK.LEFT_SHOULDER)[1] - point(marks, LANDMARK.LEFT_WRIST)[1])
        )
        series["rightWristAboveShoulder"].append(
            float(point(marks, LANDMARK.RIGHT_SHOULDER)[1] - point(marks, LANDMARK.RIGHT_WRIST)[1])
        )
        series["hipDepth"].append(float(hips[1] - knees[1]))
        series["shoulderWidth"].append(shoulder_width)

    return series


def range_value(values: list[float]) -> float:
    return percentile(values, 95) - percentile(values, 5)


def signed_line_angle_degrees(start: np.ndarray, end: np.ndarray) -> float:
    """Return a signed angle from vertical in image coordinates."""

    vector = np.asarray(end[:2], dtype=float) - np.asarray(start[:2], dtype=float)
    return math.degrees(math.atan2(float(vector[0]), max(1e-6, -float(vector[1]))))


def wrapped_angle_delta_degrees(value: float, reference: float) -> float:
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def finite_interpolated(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros_like(array)
    indexes = np.arange(array.size)
    return np.interp(indexes, indexes[valid], array[valid])


def stable_baseline_window(
    angles: np.ndarray,
    confidences: np.ndarray,
    sample_fps: float,
) -> tuple[int, int]:
    if angles.size == 0:
        return 0, 0
    window = max(3, min(angles.size, int(round(max(0.55, min(0.9, angles.size / max(1.0, sample_fps))) * sample_fps))))
    search_end = max(window, min(angles.size, int(round(max(window, angles.size * 0.45)))))
    best = (float("inf"), 0, window - 1)
    for start in range(0, max(1, search_end - window + 1)):
        end = min(angles.size, start + window)
        local_angles = angles[start:end]
        local_confidence = confidences[start:end]
        if local_angles.size < 3 or float(np.mean(local_confidence >= POSE_CONFIDENCE_FLOOR)) < 0.7:
            continue
        spread = float(np.percentile(local_angles, 90) - np.percentile(local_angles, 10))
        confidence_penalty = max(0.0, POSE_CONFIDENCE_FLOOR - float(np.mean(local_confidence))) * 10.0
        score = spread + confidence_penalty
        if score < best[0]:
            best = (score, start, end - 1)
    return int(best[1]), int(best[2])


def frame_rep_context(
    time_ms: int,
    rep_events: Iterable[dict[str, Any]] | None,
) -> tuple[int | None, str]:
    for event in rep_events or []:
        start = int(event.get("startTimeMs") or 0)
        key = int(event.get("keyTimeMs") or start)
        end = int(event.get("endTimeMs") or key)
        if start <= time_ms <= end:
            return int(event.get("repIndex") or 0) or None, "to_key" if time_ms <= key else "return"
    return None, "between_reps"


def build_movement_phase_judgments(
    frames: Iterable[Any],
    rep_events: Iterable[dict[str, Any]] | None,
    sample_fps: float,
) -> list[dict[str, Any]]:
    """Build a limb-only phase timeline independent of trunk stability."""

    events = sorted(
        list(rep_events or []),
        key=lambda item: int(item.get("startTimeMs") or 0),
    )
    turn_window_ms = max(70, int(round(500.0 / max(1.0, sample_fps))))
    timeline: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda item: int(getattr(item, "frame_index", 0))):
        time_ms = int(getattr(frame, "time_ms", 0) or 0)
        active = next((
            event for event in events
            if int(event.get("startTimeMs") or 0) <= time_ms <= int(event.get("endTimeMs") or 0)
        ), None)
        if not active:
            phase = "between_reps"
            rep_index = None
            range_status = None
            phase_basis = "no_active_limb_cycle"
        else:
            key_time = int(active.get("keyTimeMs") or active.get("startTimeMs") or 0)
            if abs(time_ms - key_time) <= turn_window_ms:
                phase = "key"
            elif time_ms < key_time:
                phase = "to_key"
            else:
                phase = "return"
            rep_index = int(active.get("repIndex") or 0) or None
            range_status = active.get("rangeStatus") or "unknown"
            phase_basis = str(active.get("phaseRule") or "joint_angle_turning_point")
        timeline.append({
            "frameIndex": int(getattr(frame, "frame_index", 0)),
            "timeMs": time_ms,
            "repIndex": rep_index,
            "phase": phase,
            "rangeStatus": range_status,
            "phaseBasis": phase_basis,
        })
    return timeline


def expanded_person_mask(
    width: int,
    height: int,
    bboxes: Iterable[list[float] | None],
    margin_ratio: float = 0.16,
) -> np.ndarray:
    """Build a feature mask that excludes the athlete and nearby equipment."""

    mask = np.full((height, width), 255, dtype=np.uint8)
    for bbox in bboxes:
        if not bbox or len(bbox) < 4:
            continue
        left, top, right, bottom = [float(value) for value in bbox[:4]]
        box_width = max(0.02, right - left)
        box_height = max(0.02, bottom - top)
        x1 = max(0, int(round((left - box_width * margin_ratio) * width)))
        y1 = max(0, int(round((top - box_height * margin_ratio) * height)))
        x2 = min(width, int(round((right + box_width * margin_ratio) * width)))
        y2 = min(height, int(round((bottom + box_height * margin_ratio) * height)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
    return mask


def estimate_camera_motion(
    video_path: Path,
    frames: list[PoseFrame],
    max_dimension: int = 720,
) -> dict[str, Any]:
    """Estimate sampled-frame camera motion from background optical flow.

    The transform maps the first successfully tracked background view to each
    sampled frame. Athlete regions are excluded so exercise motion does not
    become camera motion.
    """

    ordered = sorted(frames, key=lambda item: item.frame_index)
    empty_summary = {
        "method": "background_optical_flow_ransac",
        "status": "unavailable",
        "availableFrames": 0,
        "frameCount": len(ordered),
        "coverage": 0.0,
        "reason": "no_pose_frames" if not ordered else "video_unavailable",
    }
    if not ordered:
        return {"frames": [], "summary": empty_summary}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"frames": [], "summary": empty_summary}

    target_frames = {int(item.frame_index): item for item in ordered}
    first_index = min(target_frames)
    last_index = max(target_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    current_index = first_index
    reference_gray: np.ndarray | None = None
    reference_bbox: list[float] | None = None
    reference_cumulative = np.eye(3, dtype=float)
    motions: list[dict[str, Any]] = []
    successful_pairs = 0
    inlier_ratios: list[float] = []

    try:
        while current_index <= last_index:
            ok, image = capture.read()
            if not ok:
                break
            pose_frame = target_frames.get(current_index)
            current_index += 1
            if pose_frame is None:
                continue

            height, width = image.shape[:2]
            resize_scale = min(1.0, float(max_dimension) / max(1.0, float(max(width, height))))
            if resize_scale < 1.0:
                image = cv2.resize(
                    image,
                    (max(1, int(round(width * resize_scale))), max(1, int(round(height * resize_scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canvas_height, canvas_width = gray.shape[:2]
            current_bbox = target_bbox_for_frame(pose_frame)

            if reference_gray is None:
                reference_gray = gray
                reference_bbox = current_bbox
                motions.append({
                    "frameIndex": int(pose_frame.frame_index),
                    "timeMs": int(pose_frame.time_ms),
                    "available": True,
                    "status": "anchor",
                    "confidence": 1.0,
                    "trackedPoints": 0,
                    "inlierRatio": 1.0,
                    "canvasWidth": canvas_width,
                    "canvasHeight": canvas_height,
                    "inverseAffine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "cumulativeRotationDeg": 0.0,
                    "cumulativeScale": 1.0,
                    "cumulativeTranslationX": 0.0,
                    "cumulativeTranslationY": 0.0,
                })
                continue

            mask = expanded_person_mask(
                canvas_width,
                canvas_height,
                (reference_bbox, current_bbox),
            )
            previous_points = cv2.goodFeaturesToTrack(
                reference_gray,
                maxCorners=320,
                qualityLevel=0.012,
                minDistance=8,
                mask=mask,
                blockSize=7,
            )
            available = False
            tracked_count = 0
            inlier_ratio = 0.0
            cumulative = reference_cumulative.copy()
            if previous_points is not None and len(previous_points) >= 12:
                current_points, status, errors = cv2.calcOpticalFlowPyrLK(
                    reference_gray,
                    gray,
                    previous_points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if current_points is not None and status is not None:
                    valid = status.reshape(-1).astype(bool)
                    if errors is not None:
                        valid &= errors.reshape(-1) < 24.0
                    source = previous_points.reshape(-1, 2)[valid]
                    destination = current_points.reshape(-1, 2)[valid]
                    tracked_count = int(len(source))
                    if tracked_count >= 12:
                        pair_affine, inliers = cv2.estimateAffinePartial2D(
                            source,
                            destination,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.5,
                            maxIters=2000,
                            confidence=0.99,
                            refineIters=10,
                        )
                        if pair_affine is not None and np.isfinite(pair_affine).all():
                            inlier_ratio = float(np.mean(inliers)) if inliers is not None else 0.0
                            scale = math.hypot(float(pair_affine[0, 0]), float(pair_affine[1, 0]))
                            if inlier_ratio >= 0.45 and 0.92 <= scale <= 1.08:
                                pair_matrix = np.vstack([pair_affine, [0.0, 0.0, 1.0]])
                                cumulative = pair_matrix @ reference_cumulative
                                available = True

            inverse = np.linalg.inv(cumulative)[:2]
            scale = math.hypot(float(cumulative[0, 0]), float(cumulative[1, 0]))
            rotation = math.degrees(math.atan2(float(cumulative[1, 0]), float(cumulative[0, 0])))
            confidence = min(1.0, tracked_count / 80.0) * inlier_ratio if available else 0.0
            motions.append({
                "frameIndex": int(pose_frame.frame_index),
                "timeMs": int(pose_frame.time_ms),
                "available": available,
                "status": "tracked" if available else "insufficient_background_features",
                "confidence": round(float(confidence), 3),
                "trackedPoints": tracked_count,
                "inlierRatio": round(float(inlier_ratio), 3),
                "canvasWidth": canvas_width,
                "canvasHeight": canvas_height,
                "inverseAffine": np.round(inverse, 8).tolist(),
                "cumulativeRotationDeg": round(float(rotation), 3),
                "cumulativeScale": round(float(scale), 5),
                "cumulativeTranslationX": round(float(cumulative[0, 2]) / max(1, canvas_width), 5),
                "cumulativeTranslationY": round(float(cumulative[1, 2]) / max(1, canvas_height), 5),
            })
            if available:
                successful_pairs += 1
                inlier_ratios.append(inlier_ratio)
                reference_gray = gray
                reference_bbox = current_bbox
                reference_cumulative = cumulative
    finally:
        capture.release()

    available_frames = sum(1 for item in motions if item.get("available"))
    coverage = available_frames / max(1, len(ordered))
    status = "ready" if len(motions) == len(ordered) and coverage >= 0.6 else "insufficient"
    reason = None if status == "ready" else "background_tracking_coverage_low"
    return {
        "frames": motions,
        "summary": {
            "method": "background_optical_flow_ransac",
            "status": status,
            "availableFrames": available_frames,
            "frameCount": len(ordered),
            "coverage": round(float(coverage), 3),
            "successfulPairs": successful_pairs,
            "medianInlierRatio": round(float(np.median(inlier_ratios)), 3) if inlier_ratios else 0.0,
            **({"reason": reason} if reason else {}),
        },
    }


def camera_compensated_landmarks(
    landmarks: list[list[float]],
    motion: dict[str, Any] | None,
) -> list[list[float]]:
    if not motion or not motion.get("available"):
        return landmarks
    matrix = np.asarray(motion.get("inverseAffine"), dtype=float)
    width = float(motion.get("canvasWidth") or 0.0)
    height = float(motion.get("canvasHeight") or 0.0)
    if matrix.shape != (2, 3) or width <= 0 or height <= 0 or not np.isfinite(matrix).all():
        return landmarks
    compensated: list[list[float]] = []
    for item in landmarks:
        values = list(item)
        pixel = np.asarray([float(values[0]) * width, float(values[1]) * height, 1.0], dtype=float)
        corrected = matrix @ pixel
        values[0] = float(corrected[0] / width)
        values[1] = float(corrected[1] / height)
        compensated.append(values)
    return compensated


def compress_frame_judgments(
    judgments: list[dict[str, Any]],
    sample_fps: float,
) -> list[dict[str, Any]]:
    if not judgments:
        return []
    step_ms = max(1, int(round(1000.0 / max(1.0, sample_fps))))
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        first = current[0]
        last = current[-1]
        angles = [abs(float(item.get("features", {}).get("trunkAngleDeltaSigned") or 0.0)) for item in current]
        scores = [float(item.get("score") or 0.0) for item in current]
        confidences = [float(item.get("confidence") or 0.0) for item in current]
        rep_indexes = sorted({int(item["repIndex"]) for item in current if item.get("repIndex") is not None})
        segments.append({
            "startTimeMs": int(first["timeMs"]),
            "endTimeMs": int(last["timeMs"]) + step_ms,
            "state": first["state"],
            "reasonCode": first["reasonCode"],
            "reason": first["reason"],
            "direction": first.get("direction") or "centered",
            "sampleCount": len(current),
            "durationMs": max(step_ms, int(last["timeMs"]) - int(first["timeMs"]) + step_ms),
            "maxAngleDeltaDeg": round(max(angles, default=0.0), 2),
            "maxScore": round(max(scores, default=0.0), 3),
            "confidence": round(average_valid(confidences), 3),
            **({"repIndexes": rep_indexes} if rep_indexes else {}),
        })
        current.clear()

    for item in judgments:
        grouping = (item.get("state"), item.get("reasonCode"), item.get("direction"))
        current_grouping = (
            current[0].get("state"),
            current[0].get("reasonCode"),
            current[0].get("direction"),
        ) if current else None
        if current and grouping != current_grouping:
            flush()
        current.append(item)
    flush()
    return segments


def build_stability_analysis(
    frames: list[PoseFrame],
    action_type: str,
    camera_angle: str,
    sample_fps: float,
    movement_values: Iterable[float] | None = None,
    rep_events: Iterable[dict[str, Any]] | None = None,
    camera_motion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = stability_profile(action_type)
    mode = str(profile.get("mode") or "disabled")
    supported_views = set(profile.get("supportedViews") or [])
    supported_view = camera_angle in supported_views
    base_payload: dict[str, Any] = {
        "profile": profile,
        "mode": mode,
        "view": camera_angle,
        "supportedView": supported_view,
        "frameJudgments": [],
        "judgmentSegments": [],
    }
    if not frames:
        return {
            **base_payload,
            "summary": {
                "mode": mode,
                "view": camera_angle,
                "supportedView": supported_view,
                "evaluated": False,
                "reason": "no_pose_frames",
            },
        }

    motion_by_frame = {
        int(item.get("frameIndex") or 0): item
        for item in (camera_motion or {}).get("frames") or []
    }
    camera_summary = dict((camera_motion or {}).get("summary") or {})
    camera_coverage = sum(
        1 for frame in frames if (motion_by_frame.get(int(frame.frame_index)) or {}).get("available")
    ) / max(1, len(frames))
    camera_compensation_applied = camera_coverage >= 0.6 and len(frames) >= 3
    camera_summary.update({
        "status": "applied" if camera_compensation_applied else camera_summary.get("status", "unavailable"),
        "applied": camera_compensation_applied,
        "activeFrameCoverage": round(float(camera_coverage), 3),
    })

    raw_features: list[dict[str, float]] = []
    confidences: list[float] = []
    for frame in frames:
        motion_item = motion_by_frame.get(int(frame.frame_index))
        marks = camera_compensated_landmarks(frame.landmarks, motion_item) if camera_compensation_applied else frame.landmarks
        raw_marks = frame.landmarks
        shoulders = midpoint(marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        hips = midpoint(marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        left_shoulder = point(marks, LANDMARK.LEFT_SHOULDER)
        right_shoulder = point(marks, LANDMARK.RIGHT_SHOULDER)
        left_hip = point(marks, LANDMARK.LEFT_HIP)
        right_hip = point(marks, LANDMARK.RIGHT_HIP)
        trunk_length = max(0.04, float(np.linalg.norm(shoulders[:2] - hips[:2])))
        shoulder_width = max(0.03, float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])))
        trunk_center = (shoulders[:2] + hips[:2]) / 2.0
        raw_shoulders = midpoint(raw_marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
        raw_hips = midpoint(raw_marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
        raw_features.append({
            "trunkAngle": signed_line_angle_degrees(hips, shoulders),
            "rawTrunkAngle": signed_line_angle_degrees(raw_hips, raw_shoulders),
            "shoulderTilt": math.degrees(math.atan2(
                float(right_shoulder[1] - left_shoulder[1]),
                max(1e-6, abs(float(right_shoulder[0] - left_shoulder[0]))),
            )),
            "pelvisTilt": math.degrees(math.atan2(
                float(right_hip[1] - left_hip[1]),
                max(1e-6, abs(float(right_hip[0] - left_hip[0]))),
            )),
            "trunkCenterX": float(trunk_center[0]),
            "trunkCenterY": float(trunk_center[1]),
            "shoulderCenterX": float(shoulders[0]),
            "shoulderCenterY": float(shoulders[1]),
            "pelvisCenterX": float(hips[0]),
            "pelvisCenterY": float(hips[1]),
            "shoulderWidth": shoulder_width,
            "trunkLength": trunk_length,
            "shoulderPelvisShear": float(shoulders[0] - hips[0]) / trunk_length,
            "cameraCompensated": bool(camera_compensation_applied and motion_item and motion_item.get("available")),
            "cameraRotationDeg": float((motion_item or {}).get("cumulativeRotationDeg") or 0.0),
            "cameraTranslationX": float((motion_item or {}).get("cumulativeTranslationX") or 0.0),
            "cameraTranslationY": float((motion_item or {}).get("cumulativeTranslationY") or 0.0),
            "cameraMotionConfidence": float((motion_item or {}).get("confidence") or 0.0),
        })
        confidences.append(metric_confidence(marks, "trunkLean"))

    angles = finite_interpolated([item["trunkAngle"] for item in raw_features])
    shoulder_tilts = finite_interpolated([item["shoulderTilt"] for item in raw_features])
    pelvis_tilts = finite_interpolated([item["pelvisTilt"] for item in raw_features])
    confidence_array = np.asarray(confidences, dtype=float)
    smooth_window = max(3, int(round(sample_fps * 0.25)) | 1)
    angles = smooth_signal(angles, smooth_window)
    shoulder_tilts = smooth_signal(shoulder_tilts, smooth_window)
    pelvis_tilts = smooth_signal(pelvis_tilts, smooth_window)
    baseline_start, baseline_end = stable_baseline_window(angles, confidence_array, sample_fps)
    baseline_slice = slice(baseline_start, baseline_end + 1)
    baseline_angle = float(np.median(angles[baseline_slice]))
    baseline_shoulder_tilt = float(np.median(shoulder_tilts[baseline_slice]))
    baseline_pelvis_tilt = float(np.median(pelvis_tilts[baseline_slice]))
    baseline_centers = np.asarray([
        [item["trunkCenterX"], item["trunkCenterY"]]
        for item in raw_features[baseline_start : baseline_end + 1]
    ], dtype=float)
    baseline_center = np.median(baseline_centers, axis=0)
    baseline_width = max(0.03, float(np.median([
        item["shoulderWidth"] for item in raw_features[baseline_start : baseline_end + 1]
    ])))

    angle_deltas = np.asarray([
        wrapped_angle_delta_degrees(value, baseline_angle) for value in angles
    ], dtype=float)
    shoulder_tilt_deltas = np.asarray([
        wrapped_angle_delta_degrees(value, baseline_shoulder_tilt) for value in shoulder_tilts
    ], dtype=float)
    pelvis_tilt_deltas = np.asarray([
        wrapped_angle_delta_degrees(value, baseline_pelvis_tilt) for value in pelvis_tilts
    ], dtype=float)

    primary_deltas = angle_deltas.copy()
    coupling_slope = 0.0
    max_coupled_angle = float(profile.get("maxCoupledAngleDeg") or 0.0)
    motion = np.asarray(list(movement_values) if movement_values is not None else [], dtype=float)
    if mode == "relative_coupled" and camera_angle == "side" and motion.size == angles.size:
        motion = finite_interpolated(motion.tolist())
        motion_span = float(np.percentile(motion, 95) - np.percentile(motion, 5))
        valid = confidence_array >= POSE_CONFIDENCE_FLOOR
        if motion_span > 1e-4 and int(np.count_nonzero(valid)) >= 5:
            normalized_motion = (motion - float(np.median(motion))) / motion_span
            coupling_slope, intercept = np.polyfit(normalized_motion[valid], angles[valid], 1)
            predicted = coupling_slope * normalized_motion + intercept
            predicted_center = float(np.median(predicted[valid]))
            allowed = max(0.0, max_coupled_angle)
            bounded_predicted = np.clip(predicted, predicted_center - allowed, predicted_center + allowed)
            primary_deltas = np.asarray([
                wrapped_angle_delta_degrees(value, expected)
                for value, expected in zip(angles, bounded_predicted)
            ], dtype=float)

    noise_angle = float(np.median(np.abs(np.diff(primary_deltas)))) / 0.6745 if primary_deltas.size > 1 else 0.0
    noise_tilt = max(
        float(np.median(np.abs(np.diff(shoulder_tilt_deltas)))) / 0.6745 if shoulder_tilt_deltas.size > 1 else 0.0,
        float(np.median(np.abs(np.diff(pelvis_tilt_deltas)))) / 0.6745 if pelvis_tilt_deltas.size > 1 else 0.0,
    )
    angle_threshold = max(float(profile.get("angleThresholdDeg") or 4.5), noise_angle * 3.5)
    tilt_threshold = max(float(profile.get("tiltThresholdDeg") or 4.0), noise_tilt * 3.5)
    minimum_frames = max(2, int(math.ceil(float(profile.get("minimumDurationMs") or 320) / 1000.0 * sample_fps)))
    recovery_frames = max(2, int(math.ceil(float(profile.get("recoveryDurationMs") or 240) / 1000.0 * sample_fps)))

    states = ["stable"] * len(frames)
    scores = np.zeros(len(frames), dtype=float)
    dominant_metrics = ["trunk_angle"] * len(frames)
    if mode in {"disabled", "primary_trunk_motion"} or not supported_view:
        state = "not_evaluated" if mode in {"disabled", "primary_trunk_motion"} else "unknown"
        reason_code = (
            "PRIMARY_TRUNK_MOTION" if mode == "primary_trunk_motion"
            else "STABILITY_DISABLED" if mode == "disabled"
            else "STABILITY_VIEW_UNSUPPORTED"
        )
        reason = (
            "躯干运动属于该动作的主要运动过程，不使用通用稳定性扣分。"
            if mode == "primary_trunk_motion"
            else "该动作尚未配置通用躯干稳定性判断。"
            if mode == "disabled"
            else "首期稳定性强判断只支持正侧面和正后方。"
        )
        judgments = []
        for frame, confidence, feature, angle_delta in zip(frames, confidences, raw_features, angle_deltas):
            rep_index, phase = frame_rep_context(int(frame.time_ms), rep_events)
            judgments.append({
                "frameIndex": int(frame.frame_index),
                "timeMs": int(frame.time_ms),
                "state": state,
                "mode": mode,
                "view": camera_angle,
                "reasonCode": reason_code,
                "reason": reason,
                "direction": "centered",
                "durationMs": 0,
                "confidence": round(float(confidence), 3),
                "repIndex": rep_index,
                "phase": phase,
                "score": None,
                "features": {
                    "trunkAngleSigned": round(float(feature["trunkAngle"]), 2),
                    "trunkAngleDeltaSigned": round(float(angle_delta), 2),
                },
                "thresholds": {
                    "angleDeg": round(float(angle_threshold), 2),
                    "tiltDeg": round(float(tilt_threshold), 2),
                },
                "baselineAngleDeg": round(float(baseline_angle), 2),
            })
        segments = compress_frame_judgments(judgments, sample_fps)
        return {
            **base_payload,
            "frameJudgments": judgments,
            "judgmentSegments": segments,
            "summary": {
                "mode": mode,
                "view": camera_angle,
                "supportedView": supported_view,
                "evaluated": False,
                "reason": reason_code,
                "baselineAngleDeg": round(float(baseline_angle), 2),
                "cameraCompensation": camera_summary,
            },
        }

    for index in range(len(frames)):
        if confidences[index] < POSE_CONFIDENCE_FLOOR:
            states[index] = "unknown"
            continue
        metric_scores = {
            "trunk_angle": abs(float(primary_deltas[index])) / max(1e-6, angle_threshold),
        }
        if mode == "relative_coupled" and camera_angle == "side" and max_coupled_angle > 0:
            coupled_excess = max(0.0, abs(float(angle_deltas[index])) - max_coupled_angle)
            metric_scores["coupled_angle_limit"] = coupled_excess / max(1e-6, angle_threshold)
        if camera_angle == "rear":
            metric_scores["shoulder_tilt"] = abs(float(shoulder_tilt_deltas[index])) / max(1e-6, tilt_threshold)
            metric_scores["pelvis_tilt"] = abs(float(pelvis_tilt_deltas[index])) / max(1e-6, tilt_threshold)
        dominant = max(metric_scores, key=metric_scores.get)
        score = float(metric_scores[dominant])
        scores[index] = score
        dominant_metrics[index] = dominant
        states[index] = "candidate" if score >= 1.0 else "watch" if score >= 0.65 else "stable"

    start = 0
    while start < len(states):
        if states[start] != "candidate":
            start += 1
            continue
        end = start
        while end + 1 < len(states) and states[end + 1] == "candidate":
            end += 1
        replacement = "unstable" if end - start + 1 >= minimum_frames else "watch"
        for index in range(start, end + 1):
            states[index] = replacement
        start = end + 1

    active_unstable = False
    recovering = 0
    for index, state in enumerate(states):
        if state == "unknown":
            active_unstable = False
            recovering = 0
            continue
        if state == "unstable":
            active_unstable = True
            recovering = 0
            continue
        if not active_unstable:
            continue
        if scores[index] <= 0.55:
            states[index] = "recovering"
            recovering += 1
            if recovering >= recovery_frames:
                active_unstable = False
                recovering = 0
        else:
            states[index] = "unstable"
            recovering = 0

    judgments: list[dict[str, Any]] = []
    previous_state = ""
    state_started_ms = 0
    reason_by_state = {
        "stable": ("TRUNK_STABLE", "躯干在动作基线和噪声死区内。"),
        "watch": ("TRUNK_WATCH", "躯干刚超过预警范围，持续时间尚不足以判定不稳定。"),
        "unstable": ("TRUNK_UNSTABLE", "躯干相对动作基线持续偏移，已达到不稳定判定条件。"),
        "recovering": ("TRUNK_RECOVERING", "躯干正在回到稳定范围，等待持续恢复。"),
        "unknown": ("TRUNK_EVIDENCE_LOW", "当前帧躯干关键点置信度不足，暂不判断。"),
    }
    for index, (frame, feature, confidence, state) in enumerate(zip(frames, raw_features, confidences, states)):
        time_ms = int(frame.time_ms)
        if state != previous_state:
            state_started_ms = time_ms
            previous_state = state
        reason_code, reason = reason_by_state[state]
        signed_delta = float(primary_deltas[index])
        direction = "screen_right" if signed_delta > 0.8 else "screen_left" if signed_delta < -0.8 else "centered"
        rep_index, phase = frame_rep_context(time_ms, rep_events)
        center_displacement = float(np.linalg.norm(
            np.asarray([feature["trunkCenterX"], feature["trunkCenterY"]]) - baseline_center
        )) / baseline_width
        judgments.append({
            "frameIndex": int(frame.frame_index),
            "timeMs": time_ms,
            "state": state,
            "mode": mode,
            "view": camera_angle,
            "reasonCode": reason_code,
            "reason": reason,
            "direction": direction,
            "durationMs": max(0, time_ms - state_started_ms),
            "confidence": round(float(confidence), 3),
            "repIndex": rep_index,
            "phase": phase,
            "score": round(float(scores[index]), 3),
            "dominantMetric": dominant_metrics[index],
            "features": {
                "trunkAngleSigned": round(float(feature["trunkAngle"]), 2),
                "rawTrunkAngleSigned": round(float(feature["rawTrunkAngle"]), 2),
                "trunkAngleDeltaSigned": round(float(angle_deltas[index]), 2),
                "relativeResidualDeg": round(float(primary_deltas[index]), 2),
                "trunkCenterDisplacement": round(center_displacement, 3),
                "shoulderTiltDelta": round(float(shoulder_tilt_deltas[index]), 2),
                "pelvisTiltDelta": round(float(pelvis_tilt_deltas[index]), 2),
                "cameraCompensated": bool(feature["cameraCompensated"]),
                "cameraRotationDeg": round(float(feature["cameraRotationDeg"]), 2),
                "cameraTranslationX": round(float(feature["cameraTranslationX"]), 4),
                "cameraTranslationY": round(float(feature["cameraTranslationY"]), 4),
                "cameraMotionConfidence": round(float(feature["cameraMotionConfidence"]), 3),
            },
            "thresholds": {
                "angleDeg": round(float(angle_threshold), 2),
                "tiltDeg": round(float(tilt_threshold), 2),
                "minimumDurationMs": int(profile.get("minimumDurationMs") or 320),
            },
            "baselineAngleDeg": round(float(baseline_angle), 2),
        })

    segments = compress_frame_judgments(judgments, sample_fps)
    unstable_segments = [item for item in segments if item.get("state") == "unstable"]
    unstable_duration_ms = sum(int(item.get("durationMs") or 0) for item in unstable_segments)
    return {
        **base_payload,
        "frameJudgments": judgments,
        "judgmentSegments": segments,
        "summary": {
            "mode": mode,
            "view": camera_angle,
            "supportedView": supported_view,
            "evaluated": True,
            "baselineAngleDeg": round(float(baseline_angle), 2),
            "baselineWindowMs": [
                int(frames[baseline_start].time_ms),
                int(frames[baseline_end].time_ms),
            ],
            "angleThresholdDeg": round(float(angle_threshold), 2),
            "tiltThresholdDeg": round(float(tilt_threshold), 2),
            "poseNoiseEstimateDeg": round(float(noise_angle), 3),
            "couplingSlopeDeg": round(float(coupling_slope), 3),
            "maxCoupledAngleDeg": round(float(max_coupled_angle), 2),
            "cameraCompensation": camera_summary,
            "frameCount": len(judgments),
            "unstableFrameCount": sum(1 for item in judgments if item.get("state") == "unstable"),
            "unstableDurationMs": unstable_duration_ms,
            "maxAngleDeltaDeg": round(max((abs(float(item["features"]["trunkAngleDeltaSigned"])) for item in judgments), default=0.0), 2),
            "maxStabilityScore": round(max((float(item.get("score") or 0.0) for item in judgments), default=0.0), 3),
        },
    }


def summarized_measurements(
    frames: list[PoseFrame],
    signal_range: float,
    peaks: list[int],
    action_type: str = "other",
) -> dict[str, Any]:
    series = measurement_series(frames)
    left_knee_min = percentile(series["leftKneeAngle"], 5, 180)
    right_knee_min = percentile(series["rightKneeAngle"], 5, 180)
    left_elbow_min = percentile(series["leftElbowAngle"], 5, 180)
    right_elbow_min = percentile(series["rightElbowAngle"], 5, 180)
    left_elbow_max = percentile(series["leftElbowAngle"], 95, 0)
    right_elbow_max = percentile(series["rightElbowAngle"], 95, 0)
    left_shoulder_max = percentile(series["leftShoulderAngle"], 95, 0)
    right_shoulder_max = percentile(series["rightShoulderAngle"], 95, 0)
    left_hip_min = percentile(series["leftHipAngle"], 5, 180)
    right_hip_min = percentile(series["rightHipAngle"], 5, 180)
    left_hip_max = percentile(series["leftHipAngle"], 95, 0)
    right_hip_max = percentile(series["rightHipAngle"], 95, 0)
    shoulder_width = max(0.03, percentile(series["shoulderWidth"], 50, 0.2))
    y_raise_side = y_raise_working_side(frames)
    y_raise_working = y_raise_side.get(str(y_raise_side.get("side") or "left"), {})
    single_arm_pulldown_side = single_arm_pulldown_working_side(frames)
    single_arm_hammer_row_side = single_arm_hammer_row_working_side(frames)
    single_arm_hammer_row_working = single_arm_hammer_row_side.get(str(single_arm_hammer_row_side.get("side") or "left"), {})
    rear_leg_raise_side = plate_loaded_rear_leg_raise_working_side(frames)
    rear_leg_raise_working = rear_leg_raise_side.get(str(rear_leg_raise_side.get("side") or "left"), {})

    peak_times = [frames[index].time_ms for index in peaks if index < len(frames)]
    rep_intervals = np.diff(peak_times) / 1000 if len(peak_times) > 1 else np.array([])

    measurements = {
        "motionRange": round(signal_range, 1),
        "averagePoseConfidence": round(average_valid([frame.quality for frame in frames]), 3),
        "metricConfidence": summarized_metric_confidences(frames),
        "leftKneeAngleMin": round(left_knee_min, 1),
        "rightKneeAngleMin": round(right_knee_min, 1),
        "kneeAngleAsymmetry": round(abs(left_knee_min - right_knee_min), 1),
        "leftElbowAngleMin": round(left_elbow_min, 1),
        "rightElbowAngleMin": round(right_elbow_min, 1),
        "leftElbowAngleMax": round(left_elbow_max, 1),
        "rightElbowAngleMax": round(right_elbow_max, 1),
        "elbowAngleRange": round(
            max(
                0.0,
                left_elbow_max - left_elbow_min,
                right_elbow_max - right_elbow_min,
            ),
            1,
        ),
        "elbowAngleAsymmetry": round(abs(left_elbow_min - right_elbow_min), 1),
        "leftShoulderAngleMax": round(left_shoulder_max, 1),
        "rightShoulderAngleMax": round(right_shoulder_max, 1),
        "shoulderAngleMax": round(max(left_shoulder_max, right_shoulder_max), 1),
        "shoulderAngleAsymmetry": round(abs(left_shoulder_max - right_shoulder_max), 1),
        "yRaiseWorkingSide": y_raise_side.get("side"),
        "yRaiseWorkingSideConfidence": y_raise_side.get("confidence"),
        "yRaiseWorkingSideDiagnostics": y_raise_side,
        "yRaiseWorkingShoulderAngleMax": y_raise_working.get("topAngle", 0.0),
        "yRaiseWorkingShoulderAngleRange": y_raise_working.get("angleRange", 0.0),
        "yRaiseWorkingWristAboveShoulder": y_raise_working.get("wristAboveShoulder", 0.0),
        "singleArmPulldownWorkingSide": single_arm_pulldown_side.get("side"),
        "singleArmPulldownWorkingSideConfidence": single_arm_pulldown_side.get("confidence"),
        "singleArmPulldownWorkingSideDiagnostics": single_arm_pulldown_side,
        "singleArmHammerRowWorkingSide": single_arm_hammer_row_side.get("side"),
        "singleArmHammerRowWorkingSideConfidence": single_arm_hammer_row_side.get("confidence"),
        "singleArmHammerRowWorkingSideDiagnostics": single_arm_hammer_row_side,
        "singleArmHammerRowWorkingElbowFlexionRange": single_arm_hammer_row_working.get("flexionRange", 0.0),
        "singleArmHammerRowWorkingPeakFlexion": single_arm_hammer_row_working.get("peakFlexion", 0.0),
        "plateLoadedRearLegRaiseWorkingSide": rear_leg_raise_side.get("side"),
        "plateLoadedRearLegRaiseWorkingSideConfidence": rear_leg_raise_side.get("confidence"),
        "plateLoadedRearLegRaiseWorkingSideDiagnostics": rear_leg_raise_side,
        "plateLoadedRearLegRaiseWorkingHipAngleMax": rear_leg_raise_working.get("topHipAngle", 0.0),
        "plateLoadedRearLegRaiseWorkingHipAngleRange": rear_leg_raise_working.get("hipAngleRange", 0.0),
        "leftHipAngleMin": round(left_hip_min, 1),
        "rightHipAngleMin": round(right_hip_min, 1),
        "leftHipAngleMax": round(left_hip_max, 1),
        "rightHipAngleMax": round(right_hip_max, 1),
        "hipAngleMax": round(max(left_hip_max, right_hip_max), 1),
        "hipAngleAsymmetry": round(abs(left_hip_min - right_hip_min), 1),
        "trunkLeanMedian": round(percentile(series["trunkLean"], 50), 1),
        "trunkLeanMax": round(percentile(series["trunkLean"], 95), 1),
        "trunkLeanRange": round(range_value(series["trunkLean"]), 1),
        "torsoSwayRatio": round(range_value(series["torsoX"]) / shoulder_width, 3),
        "torsoSupportSwayRatio": round(
            range_value(series["torsoSupportOffsetX"]) / shoulder_width,
            3,
        ),
        "footMovementRatio": round(
            max(range_value(series["leftAnkleX"]), range_value(series["rightAnkleX"])) / shoulder_width,
            3,
        ),
        "wristStackOffset": round(
            average_valid(series["leftWristStack"] + series["rightWristStack"]),
            3,
        ),
        "elbowHeightMedian": round(
            average_valid(series["leftElbowHeight"] + series["rightElbowHeight"]),
            3,
        ),
        "wristAboveShoulderMax": round(
            max(
                percentile(series["leftWristAboveShoulder"], 95, 0),
                percentile(series["rightWristAboveShoulder"], 95, 0),
            ),
            3,
        ),
        "hipDepthMax": round(percentile(series["hipDepth"], 95), 3),
        "averageRepSeconds": round(float(np.mean(rep_intervals)), 2) if rep_intervals.size else None,
        "repTempoVariation": round(float(np.std(rep_intervals)), 2) if rep_intervals.size > 1 else None,
    }
    if fixed_foot_action(action_type):
        measurements["fixedFootIgnoredForScoring"] = True
    return measurements


def issue(
    code: str,
    severity: str,
    title: str,
    observation: str,
    correction: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "severity": severity,
        "title": title,
        "observation": observation,
        "correction": correction,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def build_stability_issue(stability: dict[str, Any]) -> dict[str, Any] | None:
    summary = stability.get("summary") or {}
    if not summary.get("evaluated"):
        return None
    unstable_segments = [
        item for item in stability.get("judgmentSegments") or []
        if item.get("state") == "unstable"
    ]
    if not unstable_segments:
        return None
    mode = str(stability.get("mode") or "disabled")
    code = "TRUNK_RELATIVE_INSTABILITY" if mode == "relative_coupled" else "TRUNK_ABSOLUTE_INSTABILITY"
    mode_text = "动作允许耦合后的剩余晃动" if mode == "relative_coupled" else "固定支撑基线"
    max_delta = float(summary.get("maxAngleDeltaDeg") or 0.0)
    unstable_ms = int(summary.get("unstableDurationMs") or 0)
    return issue(
        code,
        "yellow",
        "躯干稳定性不足",
        f"躯干相对{mode_text}持续偏移，累计约 {unstable_ms / 1000:.2f} 秒，最大可见角度偏移约 {max_delta:.1f} 度。",
        "降低负重并保持躯干与器械支撑或动作基线的关系固定，只让目标关节完成主要行程。",
        timeRangesMs=[
            [int(item["startTimeMs"]), int(item["endTimeMs"])]
            for item in unstable_segments[:12]
        ],
        confidence=round(average_valid([
            float(item.get("confidence") or 0.0) for item in unstable_segments
        ]), 3),
        measurements={
            "mode": mode,
            "view": stability.get("view"),
            "maxAngleDeltaDeg": round(max_delta, 2),
            "angleThresholdDeg": summary.get("angleThresholdDeg"),
            "unstableDurationMs": unstable_ms,
            "cameraCompensation": summary.get("cameraCompensation"),
        },
    )


def evaluate_rules(
    action_type: str,
    family: str,
    measurements: dict[str, Any],
    capture_quality: str,
    camera_angle: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    strengths: list[str] = []
    metric_confidence_map = measurements.get("metricConfidence") or {}
    fixed_foot_ignored = bool(measurements.get("fixedFootIgnoredForScoring")) or fixed_foot_action(action_type)

    def can_judge(metric: str) -> bool:
        return float(metric_confidence_map.get(metric, 1.0)) >= POSE_CONFIDENCE_FLOOR

    if capture_quality == "insufficient":
        issues.append(issue(
            "INSUFFICIENT_EVIDENCE",
            "red",
            "鐢婚潰璇佹嵁涓嶈冻",
            "鍏抽敭鍏宠妭鍦ㄨ緝澶氱敾闈腑涓嶅彲瑙侊紝褰撳墠瑙嗛涓嶉€傚悎鍋氱粏鑺傚垽鏂€?",
            "閲嶆柊鎷嶆憚瀹屾暣宸ヤ綔缁勶紝淇濇寔鍏ㄨ韩鍜岃礋閲嶈矾寰勫叆闀滐紝骞堕伩鍏嶉伄鎸′笌涓€斿彉鐒︺€?",
        ))
        return issues, strengths

    movement_match = measurements.get("movementMatch") or {}
    if movement_match.get("mismatch"):
        action = ACTION_CATALOG.get(action_type, ACTION_CATALOG["other"])
        expected_label = "lower-body" if movement_match.get("expectedGroup") == "lower_body" else "upper-body"
        detected_label = "lower-body" if movement_match.get("detectedGroup") == "lower_body" else "upper-body"
        issues.append(issue(
            "ACTION_MISMATCH",
            "red",
            "Selected action does not match video motion",
            f"Selected action is {action['name']}; expected {expected_label} motion, but the visible motion looks more like {detected_label}.",
            "Choose the matching exercise or reshoot the full working set with the body and machine path visible.",
        ))
        return issues, strengths

    if action_type == "y_raise" and camera_angle not in {"front", "front_oblique"}:
        issues.append(issue(
            "Y_RAISE_CAMERA_LIMITED",
            "yellow",
            "Y瀛椾晶骞充妇鏈轰綅璇佹嵁鏈夐檺",
            "褰撳墠鏈轰綅鍙互鍒ゆ柇鍗曚晶鎶珮骞呭害銆佽倶閮ㄧǔ瀹氬拰韬綋浠ｅ伩锛屼絾瀵规墜鑷傛槸鍚︾ǔ瀹氭部鑲╄儧闈㈡枩涓婃柟绉诲姩鐨勫垽鏂疆淇″害杈冧綆銆?",
            "浼樺厛浣跨敤姝ｅ墠鏂规垨杞诲井鏂滃墠鏂规満浣嶅鏍歌矾寰勶紱渚ф柟鎴栧悗鏂圭礌鏉愬彧浣滀负骞呭害鍜屾帶鍒惰川閲忕殑杈呭姪璇佹嵁銆?",
        ))

    if action_type == "single_arm_pulldown" and camera_angle not in {"front", "front_oblique", "side_front"}:
        issues.append(issue(
            "SINGLE_ARM_PULLDOWN_CAMERA_LIMITED",
            "yellow",
            "鍗曡噦涓嬫媺鏈轰綅璇佹嵁鏈夐檺",
            "褰撳墠鏈轰綅涓嶅埄浜庡悓鏃剁‘璁ゅ伐浣滀晶鑲橀儴涓嬫媺璺緞銆佽函骞蹭唬鍋垮拰鎵嬫焺鍥炴斁杞ㄨ抗銆?",
            "浼樺厛浣跨敤姝ｅ墠鏂广€佹枩鍓嶆柟鎴栦晶鍓嶆柟鏈轰綅锛岃宸ヤ綔渚ц偐銆佽倶銆佽厱鍜岃函骞插叏绋嬪叆闀溿€?",
        ))

    if action_type == "single_arm_hammer_row" and camera_angle not in {"side", "side_front", "front_oblique"}:
        issues.append(issue(
            "HAMMER_ROW_CAMERA_LIMITED",
            "yellow",
            "Single-arm row camera evidence is limited",
            "This exercise is best judged from a side-front or oblique view so the working elbow path and torso support are visible.",
            "Use a side-front or front-oblique angle that keeps the working shoulder, elbow, wrist, and chest support in frame.",
        ))

    if action_type == "preacher_curl" and camera_angle not in {"side", "side_front", "front_oblique"}:
        issues.append(issue(
            "PREACHER_CURL_CAMERA_LIMITED",
            "yellow",
            "Preacher curl camera evidence is limited",
            "The current angle may hide whether the upper arm stays on the preacher pad through the curl.",
            "Use a side or side-front angle that shows the upper arm, elbow, wrist, and support pad for the full set.",
        ))

    if action_type in {"hack_squat", "hip_thrust", "back_extension", "romanian_deadlift", "plate_loaded_rear_leg_raise"} and camera_angle not in {"side", "side_front", "side_rear"}:
        issues.append(issue(
            "SIDE_VIEW_RECOMMENDED",
            "yellow",
            "渚у悜鏈轰綅璇佹嵁鏇村厖鍒?",
            "褰撳墠鏈轰綅浠嶅彲鍋氬熀纭€鍒ゆ柇锛屼絾杩欎釜鍔ㄤ綔鐨勪富瑕佽绋嬫洿渚濊禆渚у悜瑙嗚涓嬬殑楂嬨€佽啙鍜岃函骞茬浉瀵逛綅缃€?",
            "澶嶆牳鏍囧噯鍔ㄤ綔鏃朵紭鍏堜娇鐢ㄦ渚ф柟鎴栦晶鍓?渚у悗 30-45 搴︽満浣嶏紝骞朵繚璇佽剼銆佽啙銆侀珛銆佽偐鍏ㄧ▼鍏ラ暅銆?",
        ))

    can_judge_lateral_shift = (
        family in {"squat", "hinge"}
        and camera_angle in {"front", "front_oblique", "rear"}
        and not fixed_foot_ignored
        and can_judge("ankleSupport")
    )
    if can_judge_lateral_shift and measurements["torsoSupportSwayRatio"] > 0.32:
        issues.append(issue(
            "TORSO_SWAY",
            "yellow",
            "韬共妯悜绉诲姩鍋忓",
            "鐩稿鍙岃剼鏀拺涓績锛岃函骞插湪鍔ㄤ綔涓殑宸﹀彸浣嶇Щ杈冩槑鏄俱€?",
            "涓嬫鍏堥檷浣庤礋閲嶆垨娆℃暟锛屼繚鎸佽倠楠ㄣ€侀鐩嗗拰鏀拺闈㈢ǔ瀹氬悗鍐嶅姞閲忋€?",
        ))
    elif can_judge_lateral_shift:
        strengths.append("鍔ㄤ綔杩囩▼涓韩浣撲腑蹇冪殑妯悜绋冲畾鎬у熀鏈彲鎺с€?")

    if measurements["averageRepSeconds"] is not None and measurements["averageRepSeconds"] < 0.9:
        issues.append(issue(
            "TEMPO_TOO_FAST",
            "yellow",
            "瀹屾暣閲嶅鑺傚鍋忓揩",
            "鍗曟閲嶅闂撮殧杈冪煭锛岀蹇冨拰鍏抽敭浣嶇疆鐨勬帶鍒剁┖闂存湁闄愩€?",
            "涓嬫璁╀笅鏀鹃樁娈佃嚦灏戜繚鎸佺害 2 绉掞紝鍏抽敭浣嶇疆鍋滅ǔ鍚庡啀瀹屾垚鍥炵▼銆?",
        ))

    if family == "squat":
        if action_type == "hack_squat":
            hip_depth = float(measurements.get("hipDepthMax") or -1.0)
            if measurements["motionRange"] > 10 and hip_depth < -0.035:
                issues.append(issue(
                    "HACK_SQUAT_DEPTH_LIMITED",
                    "yellow",
                    "Hack squat visible depth is limited",
                    "The active-window hip-to-knee depth still looks shallow. Foot and ankle landmarks are ignored for this fixed-foot machine pattern.",
                    "Lower the load and use the chest pad/rails to control the sled until the visible hip-knee relationship reaches the target depth.",
                    measurements={"hipKneeDepth": round(hip_depth, 3)},
                ))
            else:
                strengths.append("Hack squat depth is judged from visible hip-knee depth; foot landmarks are excluded from primary scoring.")

            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 28:
                issues.append(issue(
                    "HACK_SQUAT_SUPPORT_SHIFT",
                    "yellow",
                    "Hack squat trunk support is unstable",
                    "The trunk angle changes too much during the active window. The foot landmarks are ignored, so this warning is based on torso and support-pad control.",
                    "Reduce load, keep chest and shoulders fixed against the support, and control the sled path through the descent and ascent.",
                ))
            else:
                strengths.append("Hack squat trunk support stays stable through the active window.")
            strengths.append("Fixed-foot ankle/heel/toe landmarks are used only for display locking, not scoring.")
            return issues, list(dict.fromkeys(strengths))

        depth_angle = min(measurements["leftKneeAngleMin"], measurements["rightKneeAngleMin"])
        depth_threshold = 132 if action_type == "hack_squat" else 118
        hack_hip_depth_ok = (
            action_type == "hack_squat"
            and float(measurements.get("hipDepthMax") or -1.0) >= -0.035
        )
        if can_judge("kneeAngle") and depth_angle > depth_threshold and measurements["motionRange"] > 10 and not hack_hip_depth_ok:
            depth_issue_code = "HACK_SQUAT_DEPTH_LIMITED" if action_type == "hack_squat" else "SQUAT_DEPTH_LIMITED"
            issues.append(issue(
                depth_issue_code,
                "yellow",
                "褰撳墠鍙娣卞害鍋忔祬",
                "鏈€浣庝綅缃殑鍙楂嬭啙灞堟洸浠嶈緝鏈夐檺锛屽綋鍓嶈绋嬫洿鍍忔祬韫层€?",
                "鍏堢敤鍙帶璐熼噸鍋氬埌鐩爣娣卞害锛屼繚鎸佸叏鑴氭帉鍙楀姏锛屼笉瑕侀潬绐佺劧涓嬪潬鎹㈡繁搴︺€?",
            ))
        else:
            strengths.append("褰撳墠鍙琛岀▼宸茬粡瑕嗙洊鍒版湁鏁堟繁韫插尯闂淬€?")

        if action_type == "hack_squat":
            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 28:
                issues.append(issue(
                    "HACK_SQUAT_SUPPORT_SHIFT",
                    "yellow",
                    "鍝堝厠娣辫共鑳稿灚鏀拺涓嶅绋冲畾",
                    "鍔ㄤ綔杩囩▼涓函骞茶搴﹀彉鍖栬緝澶э紝鍙兘娌℃湁绋冲畾璐翠綇鑳稿灚鎴栧湪搴曢儴鐢ㄨ函骞叉檭鍔ㄦ崲琛岀▼銆?",
                    "涓嬩竴缁勫厛闄嶄綆璐熼噸锛岃兏鍙ｅ拰鑲╅儴绋冲畾闈犱綇鏀拺鍨紝鍙屾墜鎻＄ǔ鎶婃墜锛屾部杞ㄩ亾鎺у埗涓嬭共鍜屾帹璧枫€?",
                ))
            else:
                strengths.append("鍝堝厠娣辫共杩囩▼涓兏鍨敮鎾戞€讳綋绋冲畾锛屾病鏈夋槑鏄捐劚绂诲櫒姊拌建閬撱€?")
        elif can_judge("trunkLean") and measurements["trunkLeanMax"] > 62:
            issues.append(issue(
                "SQUAT_TRUNK_LEAN",
                "yellow",
                "搴曢儴韬共鍓嶅€捐緝澶?",
                "鏈€娣变綅缃函骞叉帴杩戞槑鏄惧墠鍊撅紝璧疯韩鏃跺韬共鏀拺瑕佹眰杈冮珮銆?",
                "闄嶄綆璐熼噸锛屽厛缁冧範楂嬭啙鍚屾涓嬮檷鍜屼腑瓒冲彈鍔涳紝閬垮厤楂嬮儴鍏堜簬鑳稿彛鎶捣銆?",
            ))
        else:
            strengths.append("涓嬮檷涓庤捣韬樁娈电殑韬共瑙掑害娌℃湁鍑虹幇鏄庢樉澶辨帶銆?")

        if action_type == "hack_squat":
            strengths.append("鍝堝厠娣辫共鑴氶儴浼氳骞冲彴鍜屾満鏋堕伄鎸★紝鏈涓嶆妸鑴氳笣鐐规紓绉讳綔涓轰富瑕佹墸鍒嗚瘉鎹€?")
        elif can_judge("ankleSupport") and measurements["footMovementRatio"] > 0.28:
            issues.append(issue(
                "FOOT_PRESSURE_UNSTABLE",
                "yellow",
                "鏀拺鑴氫綅缃彉鍖栬緝澶?",
                "宸ヤ綔缁勪腑鑴氳笣浣嶇疆鍙樺寲鏄庢樉锛屽彲鑳藉瓨鍦ㄨ剼鎺屽帇鍔涜浆绉绘垨绔欎綅绉诲姩銆?",
                "鐢ㄨ交涓€妗ｈ礋閲嶄繚鎸佽剼璺熴€佹媷瓒炬牴鍜屽皬瓒炬牴涓夌偣鍘嬪湴锛屾暣缁勪笉鎸剼銆?",
            ))

        if can_judge("kneeAngle") and camera_angle in {"front", "front_oblique", "rear"} and measurements["kneeAngleAsymmetry"] > 16:
            issues.append(issue(
                "KNEE_ASYMMETRY",
                "yellow",
                "宸﹀彸鑶濆眻鏇插瓨鍦ㄥ彲瑙佸樊寮?",
                "褰撳墠鏈轰綅涓嬪乏鍙宠啙鐨勬渶浣庤搴﹀樊寮傝緝鏄庢樉銆?",
                "涓嬫闄嶄綆璐熼噸锛屾鏌ュ弻鑴氱珯璺濆拰鑴氬皷鏂瑰悜锛屽苟鐢ㄦ鍓嶆柟瑙嗛澶嶆牳鑶濈洊杞ㄨ抗銆?",
            ))

    elif family == "hinge":
        if action_type == "plate_loaded_rear_leg_raise":
            working_confidence = float(measurements.get("plateLoadedRearLegRaiseWorkingSideConfidence") or 0.0)
            top_hip_angle = float(measurements.get("plateLoadedRearLegRaiseWorkingHipAngleMax") or 0.0)
            hip_angle_range = float(measurements.get("plateLoadedRearLegRaiseWorkingHipAngleRange") or 0.0)
            if working_confidence < POSE_CONFIDENCE_FLOOR:
                issues.append(issue(
                    "REAR_LEG_RAISE_EVIDENCE_LIMITED",
                    "yellow",
                    "Working leg evidence is limited",
                    "The working hip, knee, or torso landmarks are not visible enough to strongly judge the rear-leg extension path.",
                    "Record from a side or side-front angle and keep the support pad, working hip, and working knee visible through the full set.",
                    confidence=round(working_confidence, 3),
                ))
            elif top_hip_angle < 145 or hip_angle_range < 22:
                issues.append(issue(
                    "REAR_LEG_RAISE_EXTENSION_LIMITED",
                    "yellow",
                    "Rear-leg hip extension range is limited",
                    "The working leg does not clearly reach the top extension range shown in the standard video.",
                    "Keep the torso supported on the pad and drive the working thigh back/up through the hip, then control the return without swinging the pelvis.",
                    measurements={
                        "topHipAngle": round(float(top_hip_angle), 1),
                        "hipAngleRange": round(float(hip_angle_range), 1),
                    },
                ))
            else:
                strengths.append("Plate-loaded rear leg raise uses working-side hip extension as the primary action; the observed range is within the target zone.")

            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 24:
                issues.append(issue(
                    "REAR_LEG_RAISE_SUPPORT_SHIFT",
                    "yellow",
                    "Torso support shifts during the rear-leg raise",
                    "The torso angle changes too much while the working leg extends, which can turn the movement into a body swing.",
                    "Keep the chest and forearms fixed on the support and make the hip extension come from the working leg rather than trunk motion.",
                ))
            else:
                strengths.append("Torso support stays stable enough for the rear-leg raise standard.")
            return issues, list(dict.fromkeys(strengths))

        if action_type == "hip_thrust":
            top_hip_angle = float(measurements.get("hipAngleMax") or 0.0)
            if can_judge("hipAngle") and top_hip_angle < 150:
                issues.append(issue(
                    "HIP_THRUST_LOCKOUT_INCOMPLETE",
                    "yellow",
                    "鑷€妗ラ《閮ㄤ几楂嬩笉瓒?",
                    "褰撳墠鍙椤堕儴楂嬭娌℃湁绋冲畾杩涘叆浼搁珛閿佸畾鍖洪棿锛岃噣閮ㄩ《宄版敹缂╀笉鍏呭垎銆?",
                    "闄嶄綆璐熼噸锛岄《绔鑲┿€侀珛銆佽啙鎺ヨ繎涓€鏉＄嚎锛岄鐩嗕繚鎸佷腑绔嬪悗鍐嶆帶鍒跺洖钀姐€?",
                ))
            else:
                strengths.append("鑷€妗ラ《閮ㄤ几楂嬪箙搴﹀熀鏈埌浣嶏紝涓昏鍔ㄤ綔鐩爣鎴愮珛銆?")
            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 30:
                issues.append(issue(
                    "HIP_THRUST_TORSO_SHIFT",
                    "yellow",
                    "鑷€妗ヨ繃绋嬩腑韬共鏀拺涓嶇ǔ",
                    "宸ヤ綔缁勪腑鑲╅珛鐩稿瑙掑害娉㈠姩杈冨ぇ锛屽彲鑳藉瓨鍦ㄩ《宄版椂韬綋婊戝姩鎴栧€熻函骞叉憜鍔ㄥ畬鎴愬姩浣溿€?",
                    "鍥哄畾鑲╄儗鏀偣锛屾敹绱ц倠楠ㄥ拰楠ㄧ泦锛屽厛鐢ㄥ彲鎺ц礋閲嶅畬鎴愮ǔ瀹氱殑浼搁珛椤跺嘲銆?",
                ))
            return issues, list(dict.fromkeys(strengths))

        if action_type == "back_extension":
            if measurements["motionRange"] < 20:
                issues.append(issue(
                    "BACK_EXTENSION_RANGE_LIMITED",
                    "yellow",
                    "灞辩緤鎸鸿韩灞堜几骞呭害涓嶈冻",
                    "褰撳墠鍙韬共鎶樺彔鍜屽洖浣嶅箙搴﹀亸灏忥紝鏈舰鎴愭竻妤氱殑涓嬫斁-浼搁珛鍥炰綅杩囩▼銆?",
                    "淇濇寔楂嬮儴鏀偣绋冲畾锛屽悜涓嬫帶鍒舵姌鍙犲埌鍙帶鑼冨洿锛屽啀鐢ㄨ噣鑵垮悗渚у甫鍔ㄨ函骞插洖鍒颁腑绔嬮檮杩戙€?",
                ))
            else:
                strengths.append("灞辩緤鎸鸿韩鐨勮函骞叉姌鍙犱笌浼搁珛鍥炰綅骞呭害鍩烘湰鎴愮珛銆?")
            return issues, list(dict.fromkeys(strengths))

        hip_angle = min(measurements["leftHipAngleMin"], measurements["rightHipAngleMin"])
        knee_angle = min(measurements["leftKneeAngleMin"], measurements["rightKneeAngleMin"])
        if can_judge("hipAngle") and hip_angle > 130:
            issues.append(issue(
                "HINGE_RANGE_LIMITED",
                "yellow",
                "楂嬮儴閾伴摼骞呭害鏈夐檺",
                "鏈€浣庝綅缃殑楂嬮儴鎶樺彔骞呭害浠嶈緝灏忥紝鍚庝晶閾炬病鏈夊厖鍒嗚繘鍏ュ伐浣滃尯闂淬€?",
                "淇濇寔灏忚吙鍩烘湰绋冲畾锛屾兂璞￠珛閮ㄥ悜鍚庢壘澧欙紝鍦ㄨ叞鑳屼綅缃笉鍙樼殑鍓嶆彁涓嬪鍔犺绋嬨€?",
            ))
        else:
            strengths.append("鍔ㄤ綔涓昏鐢遍珛閮ㄥ悗绉诲拰浼搁珛瀹屾垚锛岄珛閾伴摼妯″紡鍩烘湰鎴愮珛銆?")

        if action_type in {"romanian_deadlift", "plate_loaded_romanian_deadlift"}:
            strengths.append("缃楅┈灏间簹纭媺鏍囧噯瑙嗛鍏佽鑶濈洊鏈夌ǔ瀹氬井灞堬紝褰撳墠浼樺厛鎸夐珛涓诲鍜屽櫒姊版憜鑷傝矾寰勫垽鏂€?")
        elif can_judge("kneeAngle") and knee_angle < 112:
            issues.append(issue(
                "HINGE_KNEE_DRIFT",
                "yellow",
                "鑶濆叧鑺傚眻鏇插亸澶?",
                "鍔ㄤ綔鏈€浣庣偣鐨勫眻鑶濆箙搴﹁緝澶э紝楂嬮摪閾鹃€愭笎鎺ヨ繎韫茶捣妯″紡銆?",
                "淇濇寔鑶濈洊寰眻浣嗕笉杩囧害鍓嶇Щ锛屽厛鍚戝悗閫侀珛锛屽啀璁╄礋閲嶈创杩戣吙閮ㄤ笅闄嶃€?",
            ))
        else:
            strengths.append("鑶濈洊淇濇寔鍚堢悊寰眻锛屾病鏈夋槑鏄鹃攣姝绘垨杩囧害涓嬭共銆?")

        if action_type not in {"romanian_deadlift", "plate_loaded_romanian_deadlift"} and can_judge("trunkLean") and measurements["trunkLeanRange"] > 28:
            issues.append(issue(
                "HINGE_TRUNK_CONTROL",
                "yellow",
                "搴曢儴涓庡洖绋嬬殑韬共瑙掑害鍙樺寲杈冨ぇ",
                "宸ヤ綔缁勪腑韬共瑙掑害娉㈠姩杈冨锛屽簳閮ㄦ帶鍒跺彲鑳藉厛浜庣洰鏍囪倢缇ょ柌鍔炽€?",
                "鍏堢缉鐭埌鑳界ǔ瀹氫繚鎸佽儗閮ㄥ欢灞曠殑琛岀▼锛屽啀閫愭澧炲姞娣卞害鎴栭噸閲忋€?",
            ))

    elif family == "press":
        if action_type == "machine_chest_press":
            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 28:
                issues.append(issue(
                    "MACHINE_CHEST_PRESS_SUPPORT_SHIFT",
                    "yellow",
                    "Back support position changes during the press",
                    "The torso angle changes enough that the lifter may be leaving the seat or bench support during the press.",
                    "Keep the head, upper back, and pelvis supported while the elbows and handles complete the press path.",
                ))
            else:
                strengths.append("Back support remains stable enough for the visible machine-press arm path.")
            return issues, list(dict.fromkeys(strengths))

        if can_judge("wristStack") and measurements["wristStackOffset"] > 0.48:
            issues.append(issue(
                "WRIST_STACK",
                "yellow",
                "鎵嬭厱涓庤倶閮ㄦ壙閲嶇嚎鍋忕",
                "鎵嬭厱鐩稿鑲橀儴鐨勬í鍚戝亸绉昏緝澶氾紝璐熼噸娌℃湁鎸佺画钀藉湪杈冪ǔ瀹氱殑鎵块噸绾夸笂銆?",
                "闄嶄綆閲嶉噺锛屾彙绱у櫒姊帮紝璁╂墜鑵曞ぇ浣撳彔鍦ㄨ倶閮ㄤ笂鏂癸紝鍐嶅畬鎴愭帹璧枫€?",
            ))
        else:
            strengths.append("鎵嬭厱涓庤倶閮ㄧ殑鎵块噸鍏崇郴鎬讳綋绋冲畾銆?")

        if (
            not (action_type == "bench_press" and camera_angle == "side")
            and can_judge("elbowAngle")
            and measurements["elbowAngleAsymmetry"] > 18
        ):
            issues.append(issue(
                "PRESS_ASYMMETRY",
                "yellow",
                "宸﹀彸鎺ㄤ妇鑺傚涓嶄竴鑷?",
                "宸﹀彸鑲樺湪鏈€浣庝綅缃殑瑙掑害宸紓杈冩槑鏄撅紝鍙兘瀛樺湪涓€渚у厛瀹屾垚鎴栧厛澶辨帶銆?",
                "涓嬩竴缁勯檷浣庨噸閲忥紝纭繚涓や晶鍚屾椂涓嬫斁銆佸悓鏃跺惎鍔紝涓嶈璁╁己渚ф姠鍏堥攣瀹氥€?",
            ))

        if action_type == "bench_press" and camera_angle not in {"side", "side_front", "side_rear"}:
            issues.append(issue(
                "BENCH_SAFETY_VIEW",
                "yellow",
                "褰撳墠鏈轰綅鏃犳硶瀹屾暣纭鍗ф帹淇濇姢璁剧疆",
                "鐜版湁鐢婚潰閫傚悎鐪嬫帹涓捐妭濂忥紝浣嗚Е鑳哥偣銆佽偐閮ㄤ綅缃拰淇濇姢瑁呯疆涓嶅娓呮銆?",
                "澶ч噸閲忓崸鎺ㄨ浣跨敤淇濇姢鑰呮垨瀹夊叏鏉狅紝骞惰ˉ鎷嶄晶鍓嶆柟鏈轰綅銆?",
            ))

    elif family == "pull":
        if action_type == "single_arm_hammer_row":
            working_confidence = float(measurements.get("singleArmHammerRowWorkingSideConfidence") or 0.0)
            flexion_range = float(measurements.get("singleArmHammerRowWorkingElbowFlexionRange") or 0.0)
            peak_flexion = float(measurements.get("singleArmHammerRowWorkingPeakFlexion") or 0.0)
            if working_confidence < POSE_CONFIDENCE_FLOOR:
                issues.append(issue(
                    "HAMMER_ROW_EVIDENCE_LIMITED",
                    "yellow",
                    "Working-arm evidence is limited",
                    "The working shoulder, elbow, or wrist is not visible enough to strongly judge the single-arm row path.",
                    "Record from a side-front angle and keep the working arm, handle path, and chest support visible.",
                    confidence=round(working_confidence, 3),
                ))
            elif flexion_range < 28 or peak_flexion < 62:
                issues.append(issue(
                    "HAMMER_ROW_RANGE_LIMITED",
                    "yellow",
                    "Single-arm row range is limited",
                    "The working elbow does not show enough pull-and-return range for the standard single-arm Hammer row.",
                    "Start from a controlled stretch, then pull the working elbow back toward the torso/rib line before returning under control.",
                    measurements={
                        "workingElbowFlexionRange": round(float(flexion_range), 1),
                        "workingPeakFlexion": round(float(peak_flexion), 1),
                    },
                ))
            else:
                strengths.append("Single-arm Hammer row uses the working-side elbow cycle as the primary movement; the visible pull range is adequate.")

            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 24:
                issues.append(issue(
                    "HAMMER_ROW_TORSO_COMPENSATION",
                    "yellow",
                    "Torso movement contributes too much",
                    "The torso angle changes noticeably during the pull, which can substitute body motion for the working-side row.",
                    "Keep the chest/support contact fixed and let the elbow path create the pull instead of leaning back.",
                ))
            else:
                strengths.append("Torso support remains stable enough for the single-arm row standard.")
            return issues, list(dict.fromkeys(strengths))

        if action_type == "chest_supported_row":
            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 32:
                issues.append(issue(
                    "CHEST_SUPPORTED_ROW_SUPPORT_SHIFT",
                    "yellow",
                    "Chest support position changes during the row",
                    "The visible torso angle changes enough that support contact may be loosening during the pull.",
                    "Keep the sternum/chest in contact with the pad and let the elbows move the handles instead of lifting the torso.",
                ))
            else:
                strengths.append("The chest-supported row keeps the torso stable enough for the visible elbow path to remain primary.")
            return issues, list(dict.fromkeys(strengths))

        if can_judge("trunkLean") and measurements["trunkLeanRange"] > 20:
            issues.append(issue(
                "PULL_TORSO_COMPENSATION",
                "yellow",
                "鎷夊姩鏈韬共鍙備笌鍋忓",
                "鍚戝績鏈鐨勮函骞茶搴﹀彉鍖栬緝澶э紝鍚庡嚑娆″彲鑳藉湪鐢ㄥ悗浠板畬鎴愭墜鏌勮绋嬨€?",
                "鍥哄畾楠ㄧ泦鍜岃倠楠紝鎶婇噸閲忛檷鍒拌兘闈犺倶閮ㄨ矾绾垮畬鎴愰《宄版敹缂╃殑姘村钩銆?",
            ))
        else:
            strengths.append("鎷夊姩杩囩▼涓函骞茶搴﹀熀鏈浐瀹氾紝娌℃湁鏄庢樉鐢ㄥ悗浠版崲琛岀▼銆?")

        if action_type == "open_elbow_row" and measurements["elbowHeightMedian"] > 0.13:
            issues.append(issue(
                "ELBOW_PATH_LOW",
                "yellow",
                "鑲橀儴璺嚎鍋忎綆",
                "鑲橀儴澶氭暟鏃堕棿浣庝簬鑲╃嚎杈冨锛屽姩浣滃紑濮嬫帴杩戞櫘閫氫綆浣嶅垝鑸广€?",
                "鍑忚交閲嶉噺锛岃鑲橀儴娌垮亸楂樿矾绾垮悜鍚庢墦寮€锛屾墜鏌勬湞涓婅兏鏂瑰悜绉诲姩銆?",
            ))
        elif action_type == "open_elbow_row":
            strengths.append("鑲橀儴淇濇寔鍋忛珮璺嚎锛屽姩浣滅洰鏍囦笌涓婅儗璁粌鍩烘湰涓€鑷淬€?")

    elif family == "isolation_elbow":
        if action_type == "preacher_curl":
            elbow_range = float(measurements.get("elbowAngleRange") or 0.0)
            top_elbow = min(
                float(measurements.get("leftElbowAngleMin") or 180.0),
                float(measurements.get("rightElbowAngleMin") or 180.0),
            )
            extended_elbow = max(
                float(measurements.get("leftElbowAngleMax") or 0.0),
                float(measurements.get("rightElbowAngleMax") or 0.0),
            )
            if can_judge("elbowAngle") and (elbow_range < 35 or top_elbow > 115 or extended_elbow < 125):
                issues.append(issue(
                    "PREACHER_CURL_RANGE_LIMITED",
                    "yellow",
                    "Preacher curl range is limited",
                    "The elbow cycle does not clearly move from an extended preacher-pad start into a controlled curl top.",
                    "Keep the upper arm fixed on the pad, curl through the elbow, and return to a controlled extended position before the next rep.",
                    measurements={
                        "elbowAngleRange": round(float(elbow_range), 1),
                        "topElbowAngle": round(float(top_elbow), 1),
                        "extendedElbowAngle": round(float(extended_elbow), 1),
                    },
                ))
            else:
                strengths.append("Preacher curl elbow flexion and return range are adequate for the standard video pattern.")

            if can_judge("trunkLean") and measurements["trunkLeanRange"] > 16:
                issues.append(issue(
                    "PREACHER_CURL_BODY_SWING",
                    "yellow",
                    "Body swing appears during the curl",
                    "The torso shifts enough to suggest momentum may be helping the handle move.",
                    "Keep the chest and upper arms set on the preacher support and make the elbow flexion drive the handle.",
                ))
            else:
                strengths.append("Torso and upper-arm support stay stable enough for preacher curl scoring.")

        elif can_judge("trunkLean") and measurements["trunkLeanRange"] > 18:
            issues.append(issue(
                "ISOLATION_ELBOW_BODY_SWING",
                "yellow",
                "Body swing appears during the elbow isolation movement",
                "The torso angle changes enough to suggest momentum may be helping the elbow movement.",
                "Reduce load and keep the shoulder/upper-arm position fixed while the elbow moves.",
            ))

    elif family == "isolation_shoulder":
        if action_type == "y_raise":
            y_raise_side_confidence = float(
                measurements.get("yRaiseWorkingSideConfidence")
                or metric_confidence_map.get("shoulderAngle", 1.0)
            )
            y_raise_can_judge = y_raise_side_confidence >= POSE_CONFIDENCE_FLOOR
            y_raise_top_angle = float(
                measurements.get("yRaiseWorkingShoulderAngleMax")
                or measurements["shoulderAngleMax"]
            )
            if y_raise_can_judge and y_raise_top_angle < 108:
                issues.append(issue(
                    "Y_RAISE_TOP_RANGE_INCOMPLETE",
                    "yellow",
                    "Y瀛楅《閮ㄥ箙搴︿笉瓒?",
                    "鏈€楂樹綅缃殑鑲╅儴鎶珮瑙掑害涓嶈冻锛屾墜鑷傛病鏈夌ǔ瀹氳繘鍏ュご閮ㄤ袱渚х殑 Y 瀛楀尯闂淬€?",
                    "闄嶄綆閲嶉噺锛屾部鑲╄儧闈㈡妸鎵嬭噦鎶埌鐣ラ珮浜庤偐骞舵帴杩戝ご閮ㄤ袱渚э紝鍐嶆帶鍒朵笅鏀俱€?",
                ))
            else:
                strengths.append("鎵嬭噦宸茬粡杩涘叆鏄庢樉鐨?Y 瀛楅《閮ㄥ尯闂达紝鑲╅儴鎶珮骞呭害鍩烘湰鎴愮珛銆?")

        if can_judge("trunkLean") and (measurements["trunkLeanRange"] > 18 or measurements["torsoSwayRatio"] > 0.3):
            issues.append(issue(
                "ISOLATION_BODY_SWING",
                "yellow",
                "瀛ょ珛鍔ㄤ綔涓韩浣撴憜鍔ㄥ亸澶?",
                "鍔ㄤ綔鍚庡崐绋嬭函骞插彉鍖栧鍔狅紝鐩爣鑲岀兢鐨勬湁鏁堝紶鍔涜鎯€у垎鎷呫€?",
                "闄嶄綆閲嶉噺锛屼繚鎸侀鐩嗗拰鑲嬮鍥哄畾锛屽湪椤堕儴鐭殏鍋滅ǔ鍚庡啀鎺у埗涓嬫斁銆?",
            ))

    elif family == "core_flexion" and action_type == "machine_crunch":
        if camera_angle not in {"side", "side_front", "side_rear"}:
            issues.append(issue(
                "MACHINE_CRUNCH_CAMERA_LIMITED",
                "yellow",
                "Side-view evidence is recommended for machine crunch",
                "The current view cannot strongly separate rib-cage flexion from a whole-body hip hinge.",
                "Record from the side with shoulders, pelvis, and the machine support visible.",
            ))
        else:
            strengths.append("The side view is suitable for judging visible trunk flexion and pelvic support.")

    elif family == "isolation_hip" and action_type in {"standing_hip_abduction", "seated_hip_abduction"}:
        if action_type == "seated_hip_abduction" and camera_angle in {"side", "side_rear"}:
            issues.append(issue(
                "HIP_ABDUCTION_CAMERA_LIMITED",
                "yellow",
                "Front-oblique evidence is recommended for seated hip abduction",
                "A strict side view hides much of the outward knee path, so range and symmetry confidence are limited.",
                "Record from the front-oblique angle with both knees, the pelvis, and both machine pads visible.",
            ))
        if can_judge("trunkLean") and measurements["trunkLeanRange"] > 20:
            issues.append(issue(
                "HIP_ABDUCTION_TORSO_SWAY",
                "yellow",
                "Torso movement contributes to the hip-abduction path",
                "The torso angle changes enough that body sway may be replacing part of the hip movement.",
                "Keep the pelvis and torso fixed against the support and move the leg or knees outward from the hip.",
            ))
        else:
            strengths.append("Pelvis and torso stability are adequate for the visible hip-abduction path.")

    if not issues:
        strengths.append("褰撳墠宸ヤ綔缁勬病鏈夊嚭鐜版槑鏄鹃渶瑕佺珛鍗冲仠姝㈢殑闂锛屽彲浠ュ湪淇濇寔鍔ㄤ綔璐ㄩ噺鐨勫墠鎻愪笅缁х画璁粌銆?")

    return issues, list(dict.fromkeys(strengths))


def event_time_range(event: dict[str, Any]) -> list[list[int]]:
    return [[int(event["startTimeMs"]), int(event["endTimeMs"])]]


def event_confidence(frames: list[PoseFrame], event: dict[str, Any], metric: str) -> float:
    key_index = int(event.get("poseKeyIndex", 0))
    start = max(0, key_index - 1)
    end = min(len(frames) - 1, key_index + 1)
    values = [metric_confidence(frames[index].landmarks, metric) for index in range(start, end + 1)]
    return round(average_valid(values), 3)


def best_side_event_confidence(
    frames: list[PoseFrame],
    event: dict[str, Any],
    metric: str,
) -> tuple[int, float]:
    chains = BILATERAL_METRIC_CHAINS.get(metric)
    if not chains:
        return 0, event_confidence(frames, event, metric)
    key_index = int(event.get("poseKeyIndex", 0))
    start = max(0, key_index - 1)
    end = min(len(frames) - 1, key_index + 1)
    side_scores: list[float] = []
    for chain in chains:
        values = [landmarks_confidence(frames[index].landmarks, chain) for index in range(start, end + 1)]
        side_scores.append(round(average_valid(values), 3))
    best_index = int(np.argmax(side_scores)) if side_scores else 0
    return best_index, float(side_scores[best_index] if side_scores else 0.0)


def best_side_chain_event_confidence(
    frames: list[PoseFrame],
    event: dict[str, Any],
    chains: list[list[int]],
) -> tuple[int, float]:
    key_index = int(event.get("poseKeyIndex", 0))
    start = max(0, key_index - 1)
    end = min(len(frames) - 1, key_index + 1)
    side_scores: list[float] = []
    for chain in chains:
        values = [landmarks_confidence(frames[index].landmarks, chain) for index in range(start, end + 1)]
        side_scores.append(round(average_valid(values), 3))
    best_index = int(np.argmax(side_scores)) if side_scores else 0
    return best_index, float(side_scores[best_index] if side_scores else 0.0)


def hack_squat_visible_depth(landmarks: list[list[float]], side_index: int) -> float:
    side = "LEFT" if int(side_index) == 0 else "RIGHT"
    hip = getattr(LANDMARK, f"{side}_HIP")
    knee = getattr(LANDMARK, f"{side}_KNEE")
    return float(point(landmarks, hip)[1] - point(landmarks, knee)[1])


def bilateral_angle_for_side(
    landmarks: list[list[float]],
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    side_index: int,
) -> float:
    left_value, right_value, _ = bilateral_angle(landmarks, left, right)
    return left_value if int(side_index) == 0 else right_value


def low_confidence_issue(
    event: dict[str, Any],
    metric: str,
    confidence: float,
) -> dict[str, Any]:
    rep_index = int(event["repIndex"])
    return issue(
        "LOW_CONFIDENCE_EVIDENCE",
        "yellow",
        "鍏抽敭鍏宠妭缃俊搴︿笉瓒?",
        f"绗?{rep_index} 娆″姩浣滃叧閿樁娈电殑 {metric} 璇佹嵁涓嶈冻锛屽綋鍓嶄笉閫傚悎鍋氬己鍒ゆ柇銆?",
        "璇锋寜鎺ㄨ崘鏈轰綅閲嶆媿锛屼繚璇佺浉鍏冲叧鑺傛棤閬尅銆佸叏绋嬪叆闀滃悗鍐嶅鏍搞€?",
        repIndexes=[rep_index],
        stage="keyPosition",
        timeRangesMs=event_time_range(event),
        confidence=confidence,
    )


def evaluate_rep_rules(
    action_type: str,
    family: str,
    frames: list[PoseFrame],
    rep_events: list[dict[str, Any]],
    camera_angle: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for event in rep_events:
        key_index = int(event.get("poseKeyIndex", 0))
        if key_index < 0 or key_index >= len(frames):
            continue
        marks = frames[key_index].landmarks
        rep_index = int(event["repIndex"])

        if family == "squat":
            side_view = camera_angle in {"side", "side_front", "side_rear"}
            visible_side_index, visible_knee_conf = best_side_event_confidence(frames, event, "kneeAngle")
            knee_conf = visible_knee_conf if side_view else event_confidence(frames, event, "kneeAngle")
            hip_knee_side_index, hip_knee_conf = best_side_chain_event_confidence(frames, event, [
                side_landmark_indices("left", ["HIP", "KNEE"]),
                side_landmark_indices("right", ["HIP", "KNEE"]),
            ])
            if action_type == "hack_squat":
                if hip_knee_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "hipKneeDepth", hip_knee_conf))
                    continue
                hip_depth = hack_squat_visible_depth(marks, hip_knee_side_index)
                if hip_depth < -0.075:
                    issues.append(issue(
                        "HACK_SQUAT_DEPTH_LIMITED",
                        "yellow",
                        "Hack squat visible depth is limited",
                        "The key frame still has the visible hip well above the visible-side knee. Foot and ankle landmarks are ignored for this fixed-foot pattern.",
                        "Reduce load and control the sled until the visible hip-knee relationship reaches the target depth.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_knee_conf,
                        measurements={"hipKneeDepth": round(float(hip_depth), 3)},
                    ))
                continue
            if action_type == "hack_squat" and knee_conf < POSE_CONFIDENCE_FLOOR and hip_knee_conf >= POSE_CONFIDENCE_FLOOR:
                hip_depth = hack_squat_visible_depth(marks, hip_knee_side_index)
                if hip_depth < -0.075:
                    issues.append(issue(
                        "HACK_SQUAT_DEPTH_LIMITED",
                        "yellow",
                        "褰撳墠鍙娣卞害鍋忔祬",
                        f"绗?{rep_index} 娆″姩浣滃簳閮ㄩ珛閮ㄤ粛鏄庢樉楂樹簬鍙渚ц啙閮紝鍝堝厠娣辫共琛岀▼涓嶈冻銆?",
                        "涓嬩竴缁勫厛闄嶄綆閲嶉噺锛岃兏鍙ｈ创绋宠兏鍨紝娌垮櫒姊拌建閬撴帶鍒朵笅韫插埌鐩爣娣卞害鍚庡啀鎺ㄨ捣銆?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_knee_conf,
                        measurements={"hipKneeDepth": round(float(hip_depth), 3)},
                    ))
                continue
            if knee_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "kneeAngle", knee_conf))
                continue
            left_knee, right_knee, _ = bilateral_angle(
                marks,
                (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
                (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
            )
            depth_angle = (
                left_knee
                if side_view and visible_side_index == 0
                else right_knee
                if side_view
                else min(left_knee, right_knee)
            )
            depth_threshold = 132 if action_type == "hack_squat" else 118
            hip_depth = hack_squat_visible_depth(marks, visible_side_index) if action_type == "hack_squat" else 0.0
            if np.isfinite(depth_angle) and depth_angle > depth_threshold and not (
                action_type == "hack_squat" and hip_depth >= -0.035
            ):
                depth_issue_code = "HACK_SQUAT_DEPTH_LIMITED" if action_type == "hack_squat" else "SQUAT_DEPTH_LIMITED"
                issues.append(issue(
                    depth_issue_code,
                    "yellow",
                    "褰撳墠鍙娣卞害鍋忔祬",
                    f"绗?{rep_index} 娆″姩浣滃簳閮ㄤ綅缃彲瑙侀珛鑶濆眻鏇蹭笉瓒筹紝褰撳墠鏇村儚娴呰共銆?",
                    "涓嬩竴缁勫厛闄嶄綆閲嶉噺锛屽仛鍒扮洰鏍囨繁搴﹀苟淇濇寔鍏ㄨ剼鎺屽彈鍔涘悗鍐嶅姞閲嶃€?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=knee_conf,
                    measurements={
                        "kneeAngleMin": round(float(depth_angle), 1),
                        **({"hipKneeDepth": round(float(hip_depth), 3)} if action_type == "hack_squat" else {}),
                    },
                ))

            if action_type == "hack_squat":
                continue

            trunk_conf = event_confidence(frames, event, "trunkLean")
            if trunk_conf >= POSE_CONFIDENCE_FLOOR:
                shoulders = midpoint(marks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
                hips = midpoint(marks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
                trunk_vector = shoulders[:2] - hips[:2]
                trunk_lean = math.degrees(math.atan2(abs(float(trunk_vector[0])), max(1e-6, abs(float(trunk_vector[1])))))
                if trunk_lean > 62:
                    issues.append(issue(
                        "SQUAT_TRUNK_LEAN",
                        "yellow",
                        "搴曢儴韬共鍓嶅€捐緝澶?",
                        f"绗?{rep_index} 娆″姩浣滃簳閮ㄤ綅缃函骞插墠鍊炬槑鏄撅紝璧疯韩鏃跺韬共鎺у埗瑕佹眰杈冮珮銆?",
                        "涓嬩竴缁勫厛闄嶉噸锛屼繚鎸佽兏鍙ｅ拰楂嬮儴鍚屾璧疯韩锛岄伩鍏嶉珛閮ㄥ厛鎶€?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=trunk_conf,
                        measurements={"trunkLean": round(float(trunk_lean), 1)},
                    ))

        elif family == "hinge":
            if action_type == "plate_loaded_rear_leg_raise":
                working = plate_loaded_rear_leg_raise_working_side(frames)
                working_side = str(working["side"])
                hip_conf = plate_loaded_rear_leg_raise_event_confidence(frames, event, working_side)
                if hip_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "hipAngle", hip_conf))
                    continue

                start_index = int(event.get("poseStartIndex", key_index))
                end_index = int(event.get("poseEndIndex", key_index))
                start_index = max(0, min(len(frames) - 1, start_index))
                end_index = max(0, min(len(frames) - 1, end_index))
                start_hip = lower_limb_side_hip_angle(frames[start_index].landmarks, working_side)
                key_hip = lower_limb_side_hip_angle(marks, working_side)
                end_hip = lower_limb_side_hip_angle(frames[end_index].landmarks, working_side)
                bottom_hip = min(
                    value for value in [start_hip, end_hip, key_hip]
                    if np.isfinite(value)
                ) if any(np.isfinite(value) for value in [start_hip, end_hip, key_hip]) else float("nan")
                hip_range = key_hip - bottom_hip if np.isfinite(key_hip) and np.isfinite(bottom_hip) else 0.0

                if np.isfinite(key_hip) and (key_hip < 142 or hip_range < 18):
                    issues.append(issue(
                        "REAR_LEG_RAISE_EXTENSION_LIMITED",
                        "yellow",
                        "Rear-leg hip extension range is limited",
                        f"Rep {rep_index} does not show enough working-side hip extension at the top.",
                        "Keep the torso supported and drive the working thigh back/up through the hip, then return under control without swinging the pelvis.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_conf,
                        measurements={
                            "workingSide": working_side,
                            "topHipAngle": round(float(key_hip), 1),
                            "hipAngleRange": round(float(hip_range), 1),
                        },
                    ))
                continue

            if action_type == "back_extension":
                side_view = camera_angle in {"side", "side_front", "side_rear"}
                visible_side_index, visible_hip_conf = best_side_event_confidence(frames, event, "hipAngle")
                hip_conf = visible_hip_conf if side_view else event_confidence(frames, event, "hipAngle")
                if hip_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "hipAngle", hip_conf))
                    continue

                start_index = int(event.get("poseStartIndex", key_index))
                end_index = int(event.get("poseEndIndex", key_index))
                start_index = max(0, min(len(frames) - 1, start_index))
                end_index = max(0, min(len(frames) - 1, end_index))

                def visible_hip_angle(frame_index: int) -> float:
                    frame_marks = frames[frame_index].landmarks
                    left_hip, right_hip, _ = bilateral_angle(
                        frame_marks,
                        (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
                        (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
                    )
                    if side_view:
                        return left_hip if visible_side_index == 0 else right_hip
                    return min(left_hip, right_hip)

                start_hip = visible_hip_angle(start_index)
                bottom_hip = visible_hip_angle(key_index)
                end_hip = visible_hip_angle(end_index)
                top_hip = max(start_hip, end_hip)
                back_extension_range = top_hip - bottom_hip if np.isfinite(top_hip) and np.isfinite(bottom_hip) else 0.0

                if np.isfinite(bottom_hip) and (bottom_hip > 130 or back_extension_range < 22):
                    issues.append(issue(
                        "BACK_EXTENSION_RANGE_LIMITED",
                        "yellow",
                        "Back extension range is limited",
                        f"Rep {rep_index} does not show enough controlled hip folding at the bottom. Visible bottom hip angle is about {bottom_hip:.1f} degrees.",
                        "Keep the hip pad stable, fold from the hip to a controlled bottom range, then return with the glutes and hamstrings instead of moving the arms.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_conf,
                        measurements={
                            "bottomHipAngle": round(float(bottom_hip), 1),
                            "hipAngleRange": round(float(back_extension_range), 1),
                        },
                    ))

                if np.isfinite(end_hip) and end_hip < 145:
                    issues.append(issue(
                        "BACK_EXTENSION_TOP_SHORT",
                        "yellow",
                        "Back extension top return is short",
                        f"Rep {rep_index} returns to only about {end_hip:.1f} degrees at the top instead of reaching a neutral hip position.",
                        "Finish by returning the shoulder-hip-knee line close to neutral; avoid stopping in the lower half of the range.",
                        repIndexes=[rep_index],
                        stage="endPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_conf,
                        measurements={"topHipAngle": round(float(end_hip), 1)},
                    ))
                continue

            side_view = camera_angle in {"side", "side_front", "side_rear"}
            visible_side_index, visible_hip_conf = best_side_event_confidence(frames, event, "hipAngle")
            hip_conf = visible_hip_conf if side_view else event_confidence(frames, event, "hipAngle")
            if hip_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "hipAngle", hip_conf))
                continue
            left_hip, right_hip, _ = bilateral_angle(
                marks,
                (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE),
                (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE),
            )
            hip_angle = (
                left_hip
                if side_view and visible_side_index == 0
                else right_hip
                if side_view
                else min(left_hip, right_hip)
            )
            if action_type == "hip_thrust":
                if np.isfinite(hip_angle) and hip_angle < 150:
                    issues.append(issue(
                        "HIP_THRUST_LOCKOUT_INCOMPLETE",
                        "yellow",
                        "鑷€妗ラ《閮ㄤ几楂嬩笉瓒?",
                        f"绗?{rep_index} 娆″姩浣滈《閮ㄩ珛瑙掔害 {hip_angle:.1f} 搴︼紝娌℃湁绋冲畾杩涘叆浼搁珛閿佸畾鍖洪棿銆?",
                        "椤剁璁╄偐銆侀珛銆佽啙鎺ヨ繎涓€鏉＄嚎锛岄鐩嗕繚鎸佷腑绔嬶紝鐭殏鍋滅ǔ鍚庡啀鎺у埗鍥炶惤銆?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=hip_conf,
                        measurements={"topHipAngle": round(float(hip_angle), 1)},
                    ))
                continue

            if np.isfinite(hip_angle) and hip_angle > 130:
                issues.append(issue(
                    "HINGE_RANGE_LIMITED",
                    "yellow",
                    "楂嬮儴閾伴摼骞呭害鏈夐檺",
                    f"绗?{rep_index} 娆″姩浣滃簳閮ㄩ珛閮ㄦ姌鍙犲箙搴︿笉瓒筹紝鍚庝晶閾捐繘鍏ュ伐浣滃尯闂翠笉鍏呭垎銆?",
                    "淇濇寔灏忚吙鍩烘湰绋冲畾锛屽厛鍚戝悗閫侀珛锛屽啀璁╄礋閲嶈创杩戣吙閮ㄤ笅闄嶃€?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=hip_conf,
                    measurements={"hipAngleMin": round(float(hip_angle), 1)},
                ))

            if action_type in {"romanian_deadlift", "plate_loaded_romanian_deadlift"}:
                visible_knee_index, visible_knee_conf = best_side_event_confidence(frames, event, "kneeAngle")
                knee_conf = visible_knee_conf if side_view else event_confidence(frames, event, "kneeAngle")
                if knee_conf >= POSE_CONFIDENCE_FLOOR:
                    left_knee, right_knee, _ = bilateral_angle(
                        marks,
                        (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
                        (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
                    )
                    knee_angle = (
                        left_knee
                        if side_view and visible_knee_index == 0
                        else right_knee
                        if side_view
                        else min(left_knee, right_knee)
                    )
                    if np.isfinite(knee_angle) and np.isfinite(hip_angle) and knee_angle < 105 and hip_angle > 115:
                        issues.append(issue(
                            "RDL_EXCESSIVE_KNEE_BEND",
                            "yellow",
                            "缃楅┈灏间簹纭媺灞堣啙杩囧",
                            f"绗?{rep_index} 娆″姩浣滃簳閮ㄨ啙瑙掔害 {knee_angle:.1f} 搴︼紝鍔ㄤ綔鏇存帴杩戜笅韫茶€屼笉鏄珛涓诲鍚庣Щ銆?",
                            "淇濇寔鑶濈洊寰眻浣嗙浉瀵圭ǔ瀹氾紝鍏堝悜鍚庨€侀珛锛岃璐熼噸璐磋繎鑵块儴涓嬮檷銆?",
                            repIndexes=[rep_index],
                            stage="keyPosition",
                            timeRangesMs=event_time_range(event),
                            confidence=knee_conf,
                            measurements={"kneeAngleMin": round(float(knee_angle), 1)},
                        ))

        elif family == "press":
            if action_type in {"machine_chest_press", "shoulder_press"}:
                if action_type == "machine_chest_press" and camera_angle in {"front", "front_oblique"}:
                    amplitude = float(event.get("signalAmplitude") or 0.0)
                    if amplitude < 10.0:
                        issues.append(issue(
                            "MACHINE_CHEST_PRESS_RANGE_LIMITED",
                            "yellow",
                            "Machine chest press range is incomplete",
                            f"Rep {rep_index} shows limited visible handle travel from the open position into the press.",
                            "Keep the back supported, press the handles through a controlled full path, and return until the chest is stretched without letting the shoulders roll forward.",
                            repIndexes=[rep_index],
                            stage="keyPosition",
                            timeRangesMs=event_time_range(event),
                            confidence=round(float(event.get("quality") or 0.0), 3),
                            measurements={"normalizedHandleTravel": round(amplitude, 1)},
                        ))
                    continue
                arm_chains = [
                    side_landmark_indices("left", ["SHOULDER", "ELBOW"]),
                    side_landmark_indices("right", ["SHOULDER", "ELBOW"]),
                ]
                side_index, proximal_conf = best_side_chain_event_confidence(frames, event, arm_chains)
                wrist = LANDMARK.LEFT_WRIST if side_index == 0 else LANDMARK.RIGHT_WRIST
                wrist_conf = average_valid([
                    visibility(frames[index].landmarks, wrist)
                    for index in range(max(0, key_index - 1), min(len(frames), key_index + 2))
                ])
                elbow_conf = round(0.80 * proximal_conf + 0.20 * wrist_conf, 3)
                if proximal_conf < POSE_CONFIDENCE_FLOOR or wrist_conf < 0.08:
                    issues.append(low_confidence_issue(event, "visibleElbowAngle", elbow_conf))
                    continue
                start_index = max(0, min(len(frames) - 1, int(event.get("poseStartIndex", key_index))))
                end_index = max(0, min(len(frames) - 1, int(event.get("poseEndIndex", key_index))))
                elbow_chains = (
                    (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
                    (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
                )
                start_elbow = bilateral_angle_for_side(
                    frames[start_index].landmarks, elbow_chains[0], elbow_chains[1], side_index
                )
                key_elbow = bilateral_angle_for_side(marks, elbow_chains[0], elbow_chains[1], side_index)
                end_elbow = bilateral_angle_for_side(
                    frames[end_index].landmarks, elbow_chains[0], elbow_chains[1], side_index
                )
                extended_elbow = max(start_elbow, end_elbow)
                bottom_threshold = 112.0 if action_type == "machine_chest_press" else 115.0
                extension_threshold = 142.0 if action_type == "machine_chest_press" else 145.0
                if (
                    np.isfinite(key_elbow)
                    and np.isfinite(extended_elbow)
                    and (key_elbow > bottom_threshold or extended_elbow < extension_threshold)
                ):
                    code = (
                        "MACHINE_CHEST_PRESS_RANGE_LIMITED"
                        if action_type == "machine_chest_press"
                        else "SHOULDER_PRESS_RANGE_LIMITED"
                    )
                    title = (
                        "Machine chest press range is incomplete"
                        if action_type == "machine_chest_press"
                        else "Shoulder press range is incomplete"
                    )
                    issues.append(issue(
                        code,
                        "yellow",
                        title,
                        f"Rep {rep_index} does not clearly move from the supported bottom position to controlled elbow extension.",
                        "Keep the back on the support, lower until the visible elbow reaches the target range, then press to a controlled top without shrugging or bouncing.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "visibleSide": "left" if side_index == 0 else "right",
                            "bottomElbowAngle": round(float(key_elbow), 1),
                            "extendedElbowAngle": round(float(extended_elbow), 1),
                        },
                    ))
                continue

            if action_type == "bench_press" and camera_angle == "side":
                continue
            elbow_conf = event_confidence(frames, event, "elbowAngle")
            if elbow_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "elbowAngle", elbow_conf))
                continue
            if action_type == "bench_press" and camera_angle not in {"front", "front_oblique"}:
                continue
            side_confidences = metric_side_confidences(frames[key_index].landmarks, "elbowAngle")
            if len(side_confidences) >= 2 and min(side_confidences[:2]) < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "elbowAngle", round(min(side_confidences[:2]), 3)))
                continue
            left_elbow, right_elbow, _ = bilateral_angle(
                marks,
                (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
                (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
            )
            asymmetry = abs(left_elbow - right_elbow)
            if np.isfinite(asymmetry) and asymmetry > 18:
                issues.append(issue(
                    "PRESS_ASYMMETRY",
                    "yellow",
                    "宸﹀彸鎺ㄤ妇鑺傚涓嶄竴鑷?",
                    f"绗?{rep_index} 娆″姩浣滃叧閿綅缃乏鍙宠倶瑙掑樊寮傛槑鏄俱€?",
                    "涓嬩竴缁勯檷浣庨噸閲忥紝纭繚涓や晶鍚屾椂涓嬫斁銆佸悓鏃跺惎鍔ㄣ€?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=elbow_conf,
                    measurements={"elbowAngleAsymmetry": round(float(asymmetry), 1)},
                ))

        elif family == "pull" and action_type in {
            "lat_pulldown",
            "single_arm_pulldown",
            "single_arm_hammer_row",
            "chest_supported_row",
            "t_bar_row",
            "plate_loaded_pulldown",
        }:
            if action_type in {"chest_supported_row", "t_bar_row", "plate_loaded_pulldown"}:
                side_index, elbow_conf = best_side_event_confidence(frames, event, "elbowAngle")
                if elbow_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "visibleElbowAngle", elbow_conf))
                    continue
                start_index = max(0, min(len(frames) - 1, int(event.get("poseStartIndex", key_index))))
                elbow_chains = (
                    (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
                    (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
                )
                start_elbow = bilateral_angle_for_side(
                    frames[start_index].landmarks, elbow_chains[0], elbow_chains[1], side_index
                )
                key_elbow = bilateral_angle_for_side(marks, elbow_chains[0], elbow_chains[1], side_index)
                key_threshold = 110.0 if action_type == "plate_loaded_pulldown" else 115.0
                if np.isfinite(start_elbow) and np.isfinite(key_elbow) and (
                    start_elbow < 132.0 or key_elbow > key_threshold
                ):
                    codes = {
                        "chest_supported_row": "CHEST_SUPPORTED_ROW_RANGE_LIMITED",
                        "t_bar_row": "T_BAR_ROW_RANGE_LIMITED",
                        "plate_loaded_pulldown": "PLATE_LOADED_PULLDOWN_RANGE_LIMITED",
                    }
                    issues.append(issue(
                        codes[action_type],
                        "yellow",
                        "Pull range is incomplete",
                        f"Rep {rep_index} does not clearly move the visible arm from a controlled stretch into the target pull position.",
                        "Start from a controlled reach, drive the elbow through the machine path, and return to the stretch without dropping the handles.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "visibleSide": "left" if side_index == 0 else "right",
                            "startElbowAngle": round(float(start_elbow), 1),
                            "keyElbowAngle": round(float(key_elbow), 1),
                        },
                    ))
                continue

            if action_type == "single_arm_hammer_row":
                working = single_arm_hammer_row_working_side(frames)
                working_side = str(working["side"])
                elbow_conf = single_arm_pulldown_event_confidence(frames, event, working_side)
                if elbow_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "elbowAngle", elbow_conf))
                    continue

                start_index = int(event.get("poseStartIndex", key_index))
                start_index = max(0, min(len(frames) - 1, start_index))
                start_marks = frames[start_index].landmarks
                start_elbow = single_arm_pulldown_side_elbow_angle(start_marks, working_side)
                key_elbow = single_arm_pulldown_side_elbow_angle(marks, working_side)
                if np.isfinite(start_elbow) and np.isfinite(key_elbow) and (start_elbow < 135 or key_elbow > 110):
                    issues.append(issue(
                        "HAMMER_ROW_RANGE_INCOMPLETE",
                        "yellow",
                        "Single-arm row range is incomplete",
                        f"Rep {rep_index} does not clearly move the working elbow from stretch into the row peak.",
                        "Start each rep from a controlled reach, then pull the working elbow back toward the torso before returning.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "workingSide": working_side,
                            "startElbowAngle": round(float(start_elbow), 1),
                            "keyElbowAngle": round(float(key_elbow), 1),
                        },
                    ))
                continue

            if action_type == "single_arm_pulldown":
                working = single_arm_pulldown_working_side(frames)
                working_side = str(working["side"])
                elbow_conf = single_arm_pulldown_event_confidence(frames, event, working_side)
                if elbow_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "elbowAngle", elbow_conf))
                    continue

                start_index = int(event.get("poseStartIndex", key_index))
                start_index = max(0, min(len(frames) - 1, start_index))
                start_marks = frames[start_index].landmarks
                start_elbow = single_arm_pulldown_side_elbow_angle(start_marks, working_side)
                key_elbow = single_arm_pulldown_side_elbow_angle(marks, working_side)
                if np.isfinite(start_elbow) and np.isfinite(key_elbow) and (start_elbow < 135 or key_elbow > 110):
                    issues.append(issue(
                        "SINGLE_ARM_PULLDOWN_RANGE_INCOMPLETE",
                        "yellow",
                        "鍗曡噦涓嬫媺琛岀▼涓嶈冻",
                        f"绗?{rep_index} 娆″姩浣滃伐浣滀晶娌℃湁绋冲畾瀹屾垚浠庤倶瑙掑ぇ浜?135 搴﹀埌灏忎簬 110 搴︾殑涓嬫媺琛岀▼銆?",
                        "椤剁鍏堝厖鍒嗘媺浼革紝闅忓悗璁╁伐浣滀晶鑲橀儴鍚戜笅璐磋繎韬綋渚ф柟锛屼笉瑕佸彧绉诲姩鎵嬭厱鎴栧彧鍋氬崐绋嬨€?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "workingSide": working_side,
                            "startElbowAngle": round(float(start_elbow), 1),
                            "keyElbowAngle": round(float(key_elbow), 1),
                        },
                    ))

                elbow_descent = single_arm_pulldown_elbow_descent(start_marks, marks, working_side)
                if elbow_descent < 0.03:
                    issues.append(issue(
                        "SINGLE_ARM_PULLDOWN_ELBOW_PATH_LIMITED",
                        "yellow",
                        "鍗曡噦涓嬫媺鑲橀儴涓嬫媺璺緞涓嶈冻",
                        f"绗?{rep_index} 娆″姩浣滀粠椤堕儴鍒版敹缂╀綅鏃讹紝宸ヤ綔渚ц倶閮ㄥ悜涓嬬Щ鍔ㄤ笉鏄庢樉銆?",
                        "鍏堜笅鍘嬭偐鑳涳紝鍐嶇敤鑲橀儴鍚戜笅甯﹀姩鎵嬫焺鍒拌韩浣撲晶鏂癸紝閬垮厤鐢ㄨ函骞插悗浠版垨鎵嬭厱浠ｅ伩銆?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={"workingSide": working_side, "elbowDescent": round(float(elbow_descent), 3)},
                    ))

                forearm_tilt = single_arm_pulldown_forearm_tilt(marks, working_side)
                if camera_angle in {"front", "front_oblique"} and forearm_tilt > 2.8:
                    issues.append(issue(
                        "SINGLE_ARM_PULLDOWN_FOREARM_DRIFT",
                        "yellow",
                        "鍗曡噦涓嬫媺灏忚噦鍋忕Щ杩囧ぇ",
                        f"绗?{rep_index} 娆″姩浣滄敹缂╀綅灏忚噦妯悜鍋忕Щ杈冨ぇ锛屾墜鏌勮矾寰勫彲鑳藉亸绂荤洰鏍囨媺绾裤€?",
                        "鎻＄ǔ鎵嬫焺锛岃灏忚噦璺熼殢鎷夌嚎鏂瑰悜锛屼笉瑕佷负浜嗘媺浣庨噸閲忔妸鎵嬭厱鏄庢樉鐢╁悜鍐呭渚с€?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={"workingSide": working_side, "forearmTiltRatio": round(float(forearm_tilt), 3)},
                    ))
                continue

            elbow_conf = event_confidence(frames, event, "elbowAngle")
            if elbow_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "elbowAngle", elbow_conf))
                continue

            start_index = int(event.get("poseStartIndex", key_index))
            start_index = max(0, min(len(frames) - 1, start_index))
            start_marks = frames[start_index].landmarks
            _, _, start_elbow = side_elbow_angles(start_marks)
            _, _, key_elbow = side_elbow_angles(marks)
            if np.isfinite(start_elbow) and np.isfinite(key_elbow) and (start_elbow < 135 or key_elbow > 90):
                issues.append(issue(
                    "LAT_PULLDOWN_RANGE_INCOMPLETE",
                    "yellow",
                    "楂樹綅涓嬫媺鑲樿琛岀▼涓嶈冻",
                    f"绗?{rep_index} 娆″姩浣滄病鏈夌ǔ瀹氬畬鎴愪粠鑲樿澶т簬 135掳 鍒板皬浜?90掳 鐨勪笅鎷夎绋嬨€?",
                    "涓嬩竴缁勫厛闄嶄綆閲嶉噺锛岄《閮ㄥ厖鍒嗘媺浼稿悗鍐嶆妸鑲樺悜涓嬫媺鍒拌韩浣撲袱渚э紝涓嶈鍙仛鍗婄▼銆?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=elbow_conf,
                    measurements={
                        "startElbowAngle": round(float(start_elbow), 1),
                        "keyElbowAngle": round(float(key_elbow), 1),
                    },
                ))

            elbow_descent = lat_pulldown_elbow_descent(start_marks, marks)
            if elbow_descent < 0.035:
                issues.append(issue(
                    "LAT_PULLDOWN_ELBOW_PATH_LIMITED",
                    "yellow",
                    "楂樹綅涓嬫媺鑲橀儴涓嬫媺璺緞涓嶈冻",
                    f"绗?{rep_index} 娆″姩浣滀粠椤堕儴鍒版敹缂╀綅鏃讹紝鑲橀儴鍚戜笅绉诲姩涓嶆槑鏄撅紝鏇村儚鎵嬭厱鎴栬韩浣撳湪浠ｅ伩銆?",
                    "鍏堝仛鑲╄儧涓嬪帇锛屽啀鎯宠薄鐢ㄨ倶鍚戜笅澶瑰埌韬綋涓や晶锛岃鑳岄儴甯﹀姩鎵嬭噦瀹屾垚涓嬫媺銆?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=elbow_conf,
                    measurements={"elbowDescent": round(float(elbow_descent), 3)},
                ))

            forearm_tilt = lat_pulldown_forearm_tilt(marks)
            if forearm_tilt > 1.35:
                issues.append(issue(
                    "LAT_PULLDOWN_FOREARM_DRIFT",
                    "yellow",
                    "楂樹綅涓嬫媺灏忚噦瑙掑害鍋忔í",
                    f"绗?{rep_index} 娆″姩浣滄敹缂╀綅灏忚噦妯悜鍋忕Щ杈冨ぇ锛屽彲鑳藉瓨鍦ㄦ彙璺濇垨鎵嬭厱璺緞鎶㈠姩浣溿€?",
                    "淇濇寔鎻¤窛鍜屾墜鑵曠ǔ瀹氾紝璁╁皬鑷傚敖閲忚窡闅忔妸鎵嬭矾绾匡紝涓嶈涓轰簡鎶婇噸閲忔媺浣庤€屾妸鎵嬭厱鍚戝唴澶栫敥銆?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=elbow_conf,
                    measurements={"forearmTiltRatio": round(float(forearm_tilt), 3)},
                ))

        elif family == "isolation_elbow":
            if action_type == "preacher_curl":
                side_index, elbow_conf = best_side_event_confidence(frames, event, "elbowAngle")
                if elbow_conf < POSE_CONFIDENCE_FLOOR:
                    issues.append(low_confidence_issue(event, "elbowAngle", elbow_conf))
                    continue

                working_side = "left" if side_index == 0 else "right"
                start_index = int(event.get("poseStartIndex", key_index))
                end_index = int(event.get("poseEndIndex", key_index))
                start_index = max(0, min(len(frames) - 1, start_index))
                end_index = max(0, min(len(frames) - 1, end_index))
                start_elbow = single_arm_pulldown_side_elbow_angle(frames[start_index].landmarks, working_side)
                key_elbow = single_arm_pulldown_side_elbow_angle(marks, working_side)
                end_elbow = single_arm_pulldown_side_elbow_angle(frames[end_index].landmarks, working_side)
                extended_elbow = max(
                    value for value in [start_elbow, end_elbow]
                    if np.isfinite(value)
                ) if any(np.isfinite(value) for value in [start_elbow, end_elbow]) else float("nan")
                elbow_range = extended_elbow - key_elbow if np.isfinite(extended_elbow) and np.isfinite(key_elbow) else 0.0

                if np.isfinite(key_elbow) and (key_elbow > 115 or elbow_range < 28 or (np.isfinite(extended_elbow) and extended_elbow < 128)):
                    issues.append(issue(
                        "PREACHER_CURL_RANGE_INCOMPLETE",
                        "yellow",
                        "Preacher curl range is incomplete",
                        f"Rep {rep_index} does not clearly move from an extended preacher start into a controlled curl top.",
                        "Keep the upper arm fixed on the pad, curl through the elbow, then return to a controlled extended position before the next rep.",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "workingSide": working_side,
                            "extendedElbowAngle": round(float(extended_elbow), 1) if np.isfinite(extended_elbow) else None,
                            "keyElbowAngle": round(float(key_elbow), 1),
                            "elbowAngleRange": round(float(elbow_range), 1),
                        },
                    ))
                continue

        elif family == "isolation_shoulder" and action_type == "y_raise":
            working = y_raise_working_side(frames)
            working_side = str(working["side"])

            def y_raise_side_event_confidence(include_wrist: bool = False) -> float:
                start = max(0, key_index - 1)
                end = min(len(frames) - 1, key_index + 1)
                return round(average_valid([
                    y_raise_side_confidence(frames[index].landmarks, working_side, include_wrist=include_wrist)
                    for index in range(start, end + 1)
                ]), 3)

            shoulder_conf = y_raise_side_event_confidence(False)
            elbow_conf = y_raise_side_event_confidence(True)
            if shoulder_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "shoulderAngle", shoulder_conf))
                continue

            start_index = int(event.get("poseStartIndex", key_index))
            start_index = max(0, min(len(frames) - 1, start_index))
            start_marks = frames[start_index].landmarks
            start_shoulder = y_raise_side_shoulder_angle(start_marks, working_side)
            top_shoulder = y_raise_side_shoulder_angle(marks, working_side)
            wrist_above = y_raise_wrist_above_shoulder(marks, working_side)
            outward_ratio = y_raise_wrist_outward_ratio(marks, working_side)
            vertical_outward_ratio = y_raise_vertical_outward_ratio(marks, working_side)
            can_judge_y_path = camera_angle in {"front", "front_oblique", "rear"}

            if np.isfinite(start_shoulder) and start_shoulder > 70:
                issues.append(issue(
                    "Y_RAISE_START_TOO_HIGH",
                    "yellow",
                    "Y瀛椾晶骞充妇璧峰浣嶅亸楂?",
                    f"绗?{rep_index} 娆″姩浣滆捣濮嬩綅鑲╅儴瑙掑害绾?{start_shoulder:.1f}掳锛屾病鏈夊洖鍒颁綆浣嶉噸鏂板缓绔嬪紶鍔涖€?",
                    "姣忔涓嬫斁鍒拌韩浣撲袱渚у亸鍓嶇殑浣庝綅鍚庡啀鍚姩锛屼笉瑕佸湪鍗婄▼浣嶇疆杩炵画寮瑰姩銆?",
                    repIndexes=[rep_index],
                    stage="startPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=shoulder_conf,
                    measurements={
                        "workingSide": working_side,
                        "startShoulderAngle": round(float(start_shoulder), 1),
                    },
                ))

            if (
                np.isfinite(top_shoulder)
                and (top_shoulder < 110 or wrist_above < 0.045)
            ):
                issues.append(issue(
                    "Y_RAISE_TOP_RANGE_INCOMPLETE",
                    "yellow",
                    "Y瀛楅《閮ㄥ箙搴︿笉瓒?",
                    f"绗?{rep_index} 娆″姩浣滈《閮ㄨ偐瑙掔害 {top_shoulder:.1f}掳锛屾墜鑵曢珮浜庤偐鐨勫箙搴︾害 {wrist_above:.3f}銆?",
                    "娌胯偐鑳涢潰鎶婃墜鑷傛姮鍒板ご閮ㄤ袱渚х殑 Y 瀛楅《閮紝椤堕儴鐭殏鍋滅ǔ鍚庡啀涓嬫斁銆?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=shoulder_conf,
                    measurements={
                        "workingSide": working_side,
                        "topShoulderAngle": round(float(top_shoulder), 1),
                        "wristAboveShoulder": round(float(wrist_above), 3),
                    },
                ))

            if can_judge_y_path and outward_ratio < 0.35:
                issues.append(issue(
                    "Y_RAISE_PATH_TOO_NARROW",
                    "yellow",
                    "Y瀛楄矾绾胯繃绐?",
                    f"绗?{rep_index} 娆″姩浣滈《閮ㄦ墜鑵曞灞曡窛绂讳笉瓒筹紝鏇存帴杩戝悜鍓嶄笂鏂逛妇鑰屼笉鏄?Y 瀛楁墦寮€銆?",
                    "璁╂墜鑷備粠韬綋涓や晶鍋忓墠鐨勪綅缃悜澶撮儴涓や晶鎵撳紑锛屼繚鎸?Y 瀛楄矾绾匡紝涓嶈鎶婃墜瀹屽叏骞跺埌姝ｅ墠鏂广€?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=shoulder_conf,
                    measurements={
                        "workingSide": working_side,
                        "wristOutwardRatio": round(float(outward_ratio), 3),
                    },
                ))

            if can_judge_y_path and outward_ratio >= 0.35 and vertical_outward_ratio < 1.0:
                issues.append(issue(
                    "Y_RAISE_PATH_TOO_FLAT",
                    "yellow",
                    "Y瀛楄矾绾胯繃骞?",
                    f"绗?{rep_index} 娆″姩浣滈《閮ㄦ墜鑷傛洿鎺ヨ繎妯悜渚у钩涓撅紝鍨傜洿鎶珮涓庡灞曡窛绂绘瘮渚嬬害 {vertical_outward_ratio:.3f}銆?",
                    "鎶婃墜鑷備粠韬綋涓や晶鍋忓墠鐨勪綆浣嶅悜澶撮儴涓や晶鏂滀笂鏂规姮璧凤紝閬垮厤鍙部妯悜渚у钩涓捐矾绾挎妸鎵嬭噦鐢╁埌楂樹綅銆?",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=shoulder_conf,
                    measurements={
                        "workingSide": working_side,
                        "wristOutwardRatio": round(float(outward_ratio), 3),
                        "verticalOutwardRatio": round(float(vertical_outward_ratio), 3),
                    },
                ))

            if elbow_conf >= POSE_CONFIDENCE_FLOOR:
                top_elbow = y_raise_side_elbow_angle(marks, working_side)
                if np.isfinite(top_elbow) and top_elbow < 135:
                    issues.append(issue(
                        "Y_RAISE_ELBOW_BEND",
                        "yellow",
                        "Y瀛椾晶骞充妇鑲橀儴寮洸鍋忓",
                        f"绗?{rep_index} 娆″姩浣滈《閮ㄨ倶瑙掔害 {top_elbow:.1f}掳锛屾墜鑷傛病鏈変繚鎸佸熀鏈几鐩淬€?",
                        "淇濇寔鑲橀儴寰眻浣嗙ǔ瀹氾紝閬垮厤鎶婂姩浣滃仛鎴愬集涓炬垨鐢ㄦ墜鑵曟妸閲嶉噺鐢╀笂鍘汇€?",
                        repIndexes=[rep_index],
                        stage="keyPosition",
                        timeRangesMs=event_time_range(event),
                        confidence=elbow_conf,
                        measurements={
                            "workingSide": working_side,
                            "topElbowAngle": round(float(top_elbow), 1),
                        },
                    ))

        elif family == "core_flexion" and action_type == "machine_crunch":
            trunk_conf = best_side_chain_event_confidence(frames, event, [[
                LANDMARK.LEFT_SHOULDER,
                LANDMARK.RIGHT_SHOULDER,
                LANDMARK.LEFT_HIP,
                LANDMARK.RIGHT_HIP,
            ]])[1]
            if trunk_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "trunkFlexion", trunk_conf))
                continue
            amplitude = float(event.get("signalAmplitude") or 0.0)
            if amplitude < 12.0:
                issues.append(issue(
                    "MACHINE_CRUNCH_RANGE_LIMITED",
                    "yellow",
                    "Machine crunch range is limited",
                    f"Rep {rep_index} shows only about {amplitude:.1f} degrees of visible trunk flexion.",
                    "Keep the pelvis supported and curl the rib cage toward the pelvis; avoid turning the rep into an arm pull or a whole-body hip hinge.",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=trunk_conf,
                    measurements={"trunkFlexionRange": round(amplitude, 1)},
                ))

        elif family == "isolation_hip" and action_type in {"standing_hip_abduction", "seated_hip_abduction"}:
            leg_conf = best_side_chain_event_confidence(frames, event, [
                side_landmark_indices("left", ["HIP", "KNEE", "ANKLE"]),
                side_landmark_indices("right", ["HIP", "KNEE", "ANKLE"]),
            ])[1]
            if leg_conf < POSE_CONFIDENCE_FLOOR:
                issues.append(low_confidence_issue(event, "hipAbductionPath", leg_conf))
                continue
            if action_type == "seated_hip_abduction" and camera_angle in {"side", "side_rear"}:
                continue
            amplitude = float(event.get("signalAmplitude") or 0.0)
            min_amplitude = 12.0 if action_type == "seated_hip_abduction" else 10.0
            if amplitude < min_amplitude:
                issues.append(issue(
                    "HIP_ABDUCTION_RANGE_LIMITED",
                    "yellow",
                    "Hip abduction range is limited",
                    f"Rep {rep_index} does not show enough outward knee/working-leg travel for the selected hip-abduction action.",
                    "Keep the pelvis fixed, move through the hip, pause briefly at the open position, and return under control without swinging the torso.",
                    repIndexes=[rep_index],
                    stage="keyPosition",
                    timeRangesMs=event_time_range(event),
                    confidence=leg_conf,
                    measurements={"normalizedAbductionRange": round(amplitude, 1)},
                ))

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for item in issues:
        key = (item["code"], tuple(item.get("repIndexes") or []))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def score_result(issues: list[dict[str, Any]], confidence: float) -> tuple[int, str]:
    if any(item["code"] == "INSUFFICIENT_EVIDENCE" for item in issues):
        return 0, "insufficient"
    if any(item["code"] == "ACTION_MISMATCH" for item in issues):
        return 0, "red"
    deduction = sum(28 if item["severity"] == "red" else 10 for item in issues)
    confidence_penalty = max(0, int((0.75 - confidence) * 30))
    score = max(35, min(98, 95 - deduction - confidence_penalty))
    if any(item["severity"] == "red" for item in issues):
        return score, "red"
    if issues:
        return score, "yellow"
    return score, "green"


def draw_pose_landmarks(
    frame: np.ndarray,
    landmarks: list[list[float]],
    *,
    landmark_color: tuple[int, int, int],
    connection_color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    visible_points: dict[int, tuple[int, int]] = {}
    for index, item in enumerate(landmarks):
        if len(item) < 4 or float(item[3]) < 0.2:
            continue
        x = int(round(float(item[0]) * width))
        y = int(round(float(item[1]) * height))
        if 0 <= x < width and 0 <= y < height:
            visible_points[index] = (x, y)
    for start, end in POSE_CONNECTIONS:
        if start in visible_points and end in visible_points:
            cv2.line(frame, visible_points[start], visible_points[end], connection_color, 2, cv2.LINE_AA)
    for position in visible_points.values():
        cv2.circle(frame, position, 2, landmark_color, 2, cv2.LINE_AA)


def even_dimension(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def resize_video_frame(frame: np.ndarray, max_dimension: int = 960) -> np.ndarray:
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest <= max_dimension:
        next_width = even_dimension(width)
        next_height = even_dimension(height)
        if next_width == width and next_height == height:
            return frame
        return cv2.resize(frame, (next_width, next_height), interpolation=cv2.INTER_AREA)
    scale = max_dimension / largest
    next_width = even_dimension(round(width * scale))
    next_height = even_dimension(round(height * scale))
    return cv2.resize(frame, (next_width, next_height), interpolation=cv2.INTER_AREA)


def elbow_angle_text(landmarks: list[list[float]]) -> str:
    left_elbow, right_elbow, _ = bilateral_angle(
        landmarks,
        (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
        (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
    )
    left = f"{left_elbow:.0f}" if np.isfinite(left_elbow) else "-"
    right = f"{right_elbow:.0f}" if np.isfinite(right_elbow) else "-"
    return f"ELBOW L/R {left}/{right} deg"


ISSUE_OVERLAY_TEXT = {
    "TRUNK_ABSOLUTE_INSTABILITY": "FIX: trunk moved from fixed baseline",
    "TRUNK_RELATIVE_INSTABILITY": "FIX: extra trunk sway exceeded allowance",
    "LAT_PULLDOWN_RANGE_INCOMPLETE": "FIX: pull through full elbow range",
    "LAT_PULLDOWN_ELBOW_PATH_LIMITED": "FIX: drive elbows down",
    "LAT_PULLDOWN_FOREARM_DRIFT": "FIX: keep forearms aligned",
    "SINGLE_ARM_PULLDOWN_RANGE_INCOMPLETE": "FIX: finish the one-arm pull",
    "SINGLE_ARM_PULLDOWN_ELBOW_PATH_LIMITED": "FIX: drive the working elbow down",
    "SINGLE_ARM_PULLDOWN_FOREARM_DRIFT": "FIX: keep the handle path aligned",
    "SINGLE_ARM_PULLDOWN_CAMERA_LIMITED": "FIX: show the working side",
    "HACK_SQUAT_DEPTH_LIMITED": "FIX: reach hack squat depth",
    "HACK_SQUAT_SUPPORT_SHIFT": "FIX: stay on the support pad",
    "HIP_THRUST_LOCKOUT_INCOMPLETE": "FIX: lock out the hips",
    "HIP_THRUST_TORSO_SHIFT": "FIX: stabilize the shoulder support",
    "BACK_EXTENSION_TOP_SHORT": "FIX: return to neutral top",
    "BACK_EXTENSION_RANGE_LIMITED": "FIX: control the full extension range",
    "RDL_EXCESSIVE_KNEE_BEND": "FIX: hinge instead of squatting",
    "SIDE_VIEW_RECOMMENDED": "CHECK: side view recommended",
    "Y_RAISE_TOP_RANGE_INCOMPLETE": "FIX: reach the high Y position",
    "Y_RAISE_START_TOO_HIGH": "FIX: lower to the start position",
    "Y_RAISE_PATH_TOO_NARROW": "FIX: open arms into a Y path",
    "Y_RAISE_PATH_TOO_FLAT": "FIX: lift on a diagonal Y path",
    "Y_RAISE_ELBOW_BEND": "FIX: keep elbows softly straight",
    "Y_RAISE_CAMERA_LIMITED": "FIX: use front or front-oblique view",
    "Y_RAISE_ASYMMETRY": "FIX: lift both sides evenly",
    "PULL_TORSO_COMPENSATION": "FIX: reduce torso swing",
    "HINGE_RANGE_LIMITED": "FIX: hinge deeper with control",
    "HINGE_KNEE_DRIFT": "FIX: send hips back first",
    "HINGE_TRUNK_CONTROL": "FIX: keep trunk angle controlled",
    "SQUAT_DEPTH_LIMITED": "FIX: reach target depth",
    "SQUAT_TRUNK_LEAN": "FIX: chest and hips rise together",
    "WRIST_STACK": "FIX: stack wrist over elbow",
    "PRESS_ASYMMETRY": "FIX: press both sides evenly",
    "LOW_CONFIDENCE_EVIDENCE": "CHECK: low landmark confidence",
    "COUNT_UNSTABLE": "CHECK: rep count unstable",
    "TARGET_UNCERTAIN": "CHECK: target tracking unstable",
}


def issue_overlay_text(issue_item: dict[str, Any]) -> str:
    code = str(issue_item.get("code") or "")
    if code in ISSUE_OVERLAY_TEXT:
        return ISSUE_OVERLAY_TEXT[code]
    title = str(issue_item.get("title") or code or "review technique").strip()
    return f"CHECK: {title[:46]}"


def feedback_for_time(
    time_ms: int,
    rep_events: Iterable[dict[str, Any]] | None,
    issues: Iterable[dict[str, Any]] | None,
    strengths: Iterable[str] | None,
) -> tuple[str, tuple[int, int, int]]:
    for item in issues or []:
        for start, end in item.get("timeRangesMs") or []:
            if int(start) <= time_ms <= int(end):
                return issue_overlay_text(item), (80, 190, 255)

    for event in rep_events or []:
        start = int(event.get("startTimeMs") or 0)
        key = int(event.get("keyTimeMs") or start)
        end = int(event.get("endTimeMs") or key)
        if start <= time_ms <= end:
            rep_index = int(event.get("repIndex") or 0)
            if abs(time_ms - key) <= 260:
                return f"CHECK: rep {rep_index} key position", (90, 230, 255)
            return f"REP {rep_index}: controlled path", (140, 255, 150)

    first_strength = next(iter(str(item) for item in strengths or [] if str(item).strip()), "")
    if first_strength:
        return "OK: no active issue in this frame", (140, 255, 150)
    return "REVIEW: follow the highlighted skeleton", (230, 240, 255)


def movement_phase_overlay_text(
    time_ms: int,
    phase_judgments: Iterable[dict[str, Any]] | None,
) -> tuple[str, tuple[int, int, int]]:
    items = list(phase_judgments or [])
    if not items:
        return "PHASE: WAITING", (255, 255, 255)
    item = min(items, key=lambda value: abs(int(value.get("timeMs") or 0) - time_ms))
    phase = str(item.get("phase") or "between_reps")
    labels = {
        "to_key": "CONCENTRIC",
        "key": "TURNING POINT",
        "return": "ECCENTRIC",
        "between_reps": "READY",
    }
    rep_index = int(item.get("repIndex") or 0)
    range_status = str(item.get("rangeStatus") or "")
    prefix = f"REP {rep_index} | " if rep_index else ""
    suffix = " | RANGE LIMITED" if range_status == "insufficient" else ""
    color = (80, 230, 255) if range_status == "insufficient" else (255, 255, 255)
    return f"{prefix}{labels.get(phase, phase.upper())}{suffix}", color


def stability_overlay_text(judgment: dict[str, Any] | None) -> tuple[str, tuple[int, int, int]]:
    if not judgment:
        return "TRUNK: NO FRAME JUDGMENT", (185, 195, 205)
    state = str(judgment.get("state") or "unknown")
    colors = {
        "stable": (140, 255, 150),
        "watch": (90, 230, 255),
        "unstable": (80, 105, 255),
        "recovering": (120, 220, 255),
        "unknown": (185, 195, 205),
        "not_evaluated": (185, 195, 205),
    }
    labels = {
        "stable": "STABLE",
        "watch": "WATCH",
        "unstable": "UNSTABLE",
        "recovering": "RECOVERING",
        "unknown": "NO EVIDENCE",
        "not_evaluated": "NOT SCORED",
    }
    text = f"TRUNK: {labels.get(state, state.upper())}"
    return text, colors.get(state, (185, 195, 205))


def draw_trunk_baseline_overlay(
    image: np.ndarray,
    landmarks: list[list[float]],
    judgment: dict[str, Any] | None,
) -> None:
    if not judgment or judgment.get("state") in {"unknown", "not_evaluated"}:
        return
    height, width = image.shape[:2]
    shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
    hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
    start = (int(round(float(hips[0]) * width)), int(round(float(hips[1]) * height)))
    current = (int(round(float(shoulders[0]) * width)), int(round(float(shoulders[1]) * height)))
    trunk_pixels = max(18.0, math.hypot(current[0] - start[0], current[1] - start[1]))
    features = judgment.get("features") or {}
    baseline_screen_angle = float(judgment.get("baselineAngleDeg") or 0.0)
    if features.get("cameraCompensated"):
        baseline_screen_angle += float(features.get("cameraRotationDeg") or 0.0)
    baseline_angle = math.radians(baseline_screen_angle)
    baseline = (
        int(round(start[0] + math.sin(baseline_angle) * trunk_pixels)),
        int(round(start[1] - math.cos(baseline_angle) * trunk_pixels)),
    )
    state = str(judgment.get("state") or "stable")
    current_color = {
        "stable": (120, 255, 140),
        "watch": (70, 220, 255),
        "unstable": (70, 90, 255),
        "recovering": (90, 205, 255),
    }.get(state, (190, 200, 210))
    cv2.line(image, start, baseline, (255, 210, 70), 2, cv2.LINE_AA)
    cv2.line(image, start, current, current_color, 4, cv2.LINE_AA)
    cv2.circle(image, start, 5, (245, 245, 245), -1, cv2.LINE_AA)


def side_elbow_angles(landmarks: list[list[float]]) -> tuple[float, float, float]:
    return bilateral_angle(
        landmarks,
        (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
        (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
    )


def side_shoulder_angles(landmarks: list[list[float]]) -> tuple[float, float, float]:
    return bilateral_angle(
        landmarks,
        (LANDMARK.LEFT_HIP, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW),
        (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW),
    )


def y_raise_wrist_above_shoulder(landmarks: list[list[float]], side: str | None = None) -> float:
    if side is not None:
        prefix = y_raise_side_prefix(side)
        shoulder = getattr(LANDMARK, f"{prefix}_SHOULDER")
        wrist = getattr(LANDMARK, f"{prefix}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            return 0.0
        return float(landmarks[int(shoulder)][1]) - float(landmarks[int(wrist)][1])

    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            continue
        values.append(float(landmarks[int(shoulder)][1]) - float(landmarks[int(wrist)][1]))
    return average_valid(values, 0.0)


def y_raise_wrist_above_shoulder_score(frames: list[PoseFrame], side: str) -> float:
    values = [
        y_raise_wrist_above_shoulder(frame.landmarks, side)
        for frame in frames
        if y_raise_side_confidence(frame.landmarks, side, include_wrist=True) >= 0.35
    ]
    return percentile(values, 80, 0.0) if values else 0.0


def y_raise_wrist_outward_ratio(landmarks: list[list[float]], side: str | None = None) -> float:
    shoulder_width = max(
        0.03,
        float(np.linalg.norm(
            point(landmarks, LANDMARK.LEFT_SHOULDER)[:2]
            - point(landmarks, LANDMARK.RIGHT_SHOULDER)[:2]
        )),
    )
    if side is not None:
        prefix = y_raise_side_prefix(side)
        shoulder = getattr(LANDMARK, f"{prefix}_SHOULDER")
        wrist = getattr(LANDMARK, f"{prefix}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            return 0.0
        return abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(shoulder)][0])) / shoulder_width

    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            continue
        values.append(abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(shoulder)][0])) / shoulder_width)
    return average_valid(values, 0.0)


def y_raise_vertical_outward_ratio(landmarks: list[list[float]], side: str | None = None) -> float:
    if side is not None:
        prefix = y_raise_side_prefix(side)
        shoulder = getattr(LANDMARK, f"{prefix}_SHOULDER")
        wrist = getattr(LANDMARK, f"{prefix}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            return 0.0
        upward = float(landmarks[int(shoulder)][1]) - float(landmarks[int(wrist)][1])
        outward = abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(shoulder)][0]))
        return upward / max(0.03, outward) if upward > 0 else 0.0

    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        if landmarks_confidence(landmarks, [shoulder, wrist]) < 0.35:
            continue
        upward = float(landmarks[int(shoulder)][1]) - float(landmarks[int(wrist)][1])
        outward = abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(shoulder)][0]))
        if upward <= 0:
            values.append(0.0)
            continue
        values.append(upward / max(0.03, outward))
    return average_valid(values, 0.0)


def lat_pulldown_elbow_descent(start_marks: list[list[float]], key_marks: list[list[float]]) -> float:
    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        if landmarks_confidence(start_marks, [shoulder, elbow, wrist]) < 0.35:
            continue
        if landmarks_confidence(key_marks, [shoulder, elbow, wrist]) < 0.35:
            continue
        values.append(float(key_marks[int(elbow)][1]) - float(start_marks[int(elbow)][1]))
    return average_valid(values, 0.0)


def lat_pulldown_forearm_tilt(landmarks: list[list[float]]) -> float:
    values: list[float] = []
    for side in ("LEFT", "RIGHT"):
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        if landmarks_confidence(landmarks, [elbow, wrist]) < 0.35:
            continue
        dx = abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(elbow)][0]))
        dy = abs(float(landmarks[int(wrist)][1]) - float(landmarks[int(elbow)][1]))
        values.append(dx / max(0.015, dy))
    return average_valid(values, 0.0)


def single_arm_pulldown_event_confidence(
    frames: list[PoseFrame],
    event: dict[str, Any],
    side: str,
) -> float:
    key_index = int(event.get("poseKeyIndex", 0))
    start = max(0, key_index - 1)
    end = min(len(frames) - 1, key_index + 1)
    return round(average_valid([
        upper_limb_side_confidence(frames[index].landmarks, side)
        for index in range(start, end + 1)
    ]), 3)


def plate_loaded_rear_leg_raise_event_confidence(
    frames: list[PoseFrame],
    event: dict[str, Any],
    side: str,
) -> float:
    key_index = int(event.get("poseKeyIndex", 0))
    start = max(0, key_index - 1)
    end = min(len(frames) - 1, key_index + 1)
    return round(average_valid([
        lower_limb_side_confidence(frames[index].landmarks, side)
        for index in range(start, end + 1)
    ]), 3)


def single_arm_pulldown_elbow_descent(
    start_marks: list[list[float]],
    key_marks: list[list[float]],
    side: str,
) -> float:
    prefix = upper_limb_side_prefix(side)
    elbow = getattr(LANDMARK, f"{prefix}_ELBOW")
    chain = [
        getattr(LANDMARK, f"{prefix}_SHOULDER"),
        elbow,
        getattr(LANDMARK, f"{prefix}_WRIST"),
    ]
    if landmarks_confidence(start_marks, chain) < 0.35:
        return 0.0
    if landmarks_confidence(key_marks, chain) < 0.35:
        return 0.0
    return float(key_marks[int(elbow)][1]) - float(start_marks[int(elbow)][1])


def single_arm_pulldown_forearm_tilt(landmarks: list[list[float]], side: str) -> float:
    prefix = upper_limb_side_prefix(side)
    elbow = getattr(LANDMARK, f"{prefix}_ELBOW")
    wrist = getattr(LANDMARK, f"{prefix}_WRIST")
    if landmarks_confidence(landmarks, [elbow, wrist]) < 0.35:
        return 0.0
    dx = abs(float(landmarks[int(wrist)][0]) - float(landmarks[int(elbow)][0]))
    dy = abs(float(landmarks[int(wrist)][1]) - float(landmarks[int(elbow)][1]))
    return dx / max(0.015, dy)


def open_browser_video_writer(
    destination: Path,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    last_error = "no codec attempted"
    for codec in ("mp4v",):
        try:
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
            last_error = f"{codec} writer did not open"
        except Exception as error:
            last_error = f"{codec}: {type(error).__name__}: {error}"
    raise RuntimeError(f"cannot open annotated video writer ({last_error})")


def transcode_browser_video(source: Path, destination: Path) -> None:
    ffmpeg = str(os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "").strip()
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe() or "").strip()
        except (ImportError, RuntimeError, OSError):
            ffmpeg = ""
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to produce a browser-compatible annotated video; "
            "install imageio-ffmpeg in ANALYZER_PYTHON or set FFMPEG_BINARY"
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size <= 0:
        message = (result.stderr or result.stdout or "unknown ffmpeg error").strip()[:500]
        raise RuntimeError(f"annotated video H.264 transcoding failed: {message}")


def render_pose_overlay_video(
    video_path: Path,
    frames: Iterable[Any],
    destination: Path,
    *,
    label: str,
    output_fps: float,
    landmark_color: tuple[int, int, int],
    connection_color: tuple[int, int, int],
    rep_events: Iterable[dict[str, Any]] | None = None,
    issues: Iterable[dict[str, Any]] | None = None,
    strengths: Iterable[str] | None = None,
    frame_judgments: Iterable[dict[str, Any]] | None = None,
    phase_judgments: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pose_frames = sorted(
        [frame for frame in frames if getattr(frame, "landmarks", None) is not None],
        key=lambda item: int(getattr(item, "frame_index", 0)),
    )
    if not pose_frames:
        raise RuntimeError("no pose frames available for annotated video")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot open video for annotated render")

    destination.parent.mkdir(parents=True, exist_ok=True)
    intermediate = destination.with_name(f"{destination.stem}.intermediate.mp4")
    writer: cv2.VideoWriter | None = None
    writer_codec: str | None = None
    written = 0
    ordered_judgments = sorted(
        list(frame_judgments or []),
        key=lambda item: int(item.get("timeMs") or 0),
    )
    judgment_index = 0
    try:
        current_frame_index = -1
        for pose_frame in pose_frames:
            frame_index = int(getattr(pose_frame, "frame_index", 0))
            if frame_index < current_frame_index:
                continue
            ok = True
            while current_frame_index < frame_index:
                ok = capture.grab()
                current_frame_index += 1
                if not ok:
                    break
            ok, image = capture.retrieve() if ok else (False, None)
            if not ok or image is None:
                continue

            draw_pose_landmarks(
                image,
                pose_frame.landmarks,
                landmark_color=landmark_color,
                connection_color=connection_color,
            )
            time_ms = int(getattr(pose_frame, "time_ms", 0) or 0)
            while (
                judgment_index + 1 < len(ordered_judgments)
                and abs(int(ordered_judgments[judgment_index + 1].get("timeMs") or 0) - time_ms)
                <= abs(int(ordered_judgments[judgment_index].get("timeMs") or 0) - time_ms)
            ):
                judgment_index += 1
            frame_judgment = ordered_judgments[judgment_index] if ordered_judgments else None
            draw_trunk_baseline_overlay(image, pose_frame.landmarks, frame_judgment)
            height, width = image.shape[:2]
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, max(112, int(height * 0.14))), (10, 16, 12), -1)
            image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
            text = f"{label} | {time_ms / 1000:.1f}s | SKELETON TRACKING"
            feedback_text, feedback_color = feedback_for_time(time_ms, rep_events, issues, strengths)
            stability_text, stability_color = stability_overlay_text(frame_judgment)
            phase_text, phase_color = movement_phase_overlay_text(time_ms, phase_judgments)
            cv2.putText(
                image,
                text[:120],
                (16, max(34, int(height * 0.045))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.58, width / 1250),
                (235, 255, 110),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                stability_text[:116],
                (16, max(58, int(height * 0.068))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.54, width / 1380),
                stability_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                phase_text[:96],
                (16, max(84, int(height * 0.102))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.52, width / 1420),
                phase_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                feedback_text[:96],
                (16, max(106, int(height * 0.128))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.48, width / 1500),
                feedback_color,
                2,
                cv2.LINE_AA,
            )

            image = resize_video_frame(image)
            if writer is None:
                output_height, output_width = image.shape[:2]
                writer, writer_codec = open_browser_video_writer(
                    intermediate,
                    output_fps,
                    (output_width, output_height),
                )
            writer.write(image)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError("annotated video writer produced no frames")

    try:
        transcode_browser_video(intermediate, destination)
    finally:
        intermediate.unlink(missing_ok=True)

    return {
        "filename": destination.name,
        "frames": written,
        "fps": round(max(1.0, float(output_fps)), 2),
        "codec": "h264",
        "sourceCodec": writer_codec or "unknown",
        "pixelFormat": "yuv420p",
        "fastStart": True,
        "browserOptimized": True,
    }


def draw_evidence_frame(
    video_path: Path,
    pose_frame: PoseFrame,
    destination: Path,
    label: str,
) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, pose_frame.frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot extract frame {pose_frame.frame_index}")

    height, width = frame.shape[:2]
    draw_pose_landmarks(
        frame,
        pose_frame.landmarks,
        landmark_color=(80, 255, 190),
        connection_color=(54, 214, 255),
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, max(44, int(height * 0.055))), (10, 16, 12), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    cv2.putText(
        frame,
        label,
        (16, max(30, int(height * 0.037))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.65, width / 900),
        (217, 255, 67),
        2,
        cv2.LINE_AA,
    )

    max_width = 760
    if width > max_width:
        ratio = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * ratio)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return frame.shape[1], frame.shape[0]


def make_contact_sheet(images: list[Path], destination: Path) -> None:
    loaded = [cv2.imread(str(path)) for path in images]
    loaded = [image for image in loaded if image is not None]
    if not loaded:
        return
    cell_width = min(620, max(image.shape[1] for image in loaded))
    cell_height = min(720, max(image.shape[0] for image in loaded))
    canvas = np.full((cell_height * 2, cell_width * 2, 3), (18, 21, 17), dtype=np.uint8)
    for index, image in enumerate(loaded[:4]):
        ratio = min(cell_width / image.shape[1], cell_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, 2)
        x = column * cell_width + (cell_width - resized.shape[1]) // 2
        y = row * cell_height + (cell_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def pose_backend_order(preferred: str | None) -> list[str]:
    value = str(preferred or DEFAULT_POSE_BACKEND).strip().lower()
    if value in {"mediapipe", "mp"}:
        return ["mediapipe"]
    if value in {"rtmlib", "rtmpose-onnx", "onnx"}:
        return ["rtmlib", "mmpose", "mediapipe"]
    if value in {"mmpose", "rtmpose"}:
        return ["mmpose", "rtmlib", "mediapipe"]
    if value in {"auto", "gpu"}:
        return ["rtmlib", "mmpose", "mediapipe"]
    return ["rtmlib", "mmpose", "mediapipe"]


def extract_mediapipe_pose_frames(
    video_path: Path,
    family: str,
    fps: float,
    step: int,
    action_type: str = "other",
    target_roi: list[float] | None = None,
) -> PoseBackendResult:
    del target_roi
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe backend is not installed in this RTMLib-only deployment"
        ) from error
    pose_api = mp.solutions.pose
    capture = cv2.VideoCapture(str(video_path))
    pose_frames: list[PoseFrame] = []
    segmentation_threshold = bounded_float_config("MEDIAPIPE_SEGMENTATION_LANDMARK_THRESHOLD", 0.35, 0.05, 0.95)
    yolo_enabled = yolo_person_segmentation_filter_enabled(action_type, family)
    yolo_threshold = bounded_float_config("YOLO_PERSON_LANDMARK_THRESHOLD", 0.55, 0.05, 0.95)
    yolo_confidence = bounded_float_config("YOLO_PERSON_CONFIDENCE", 0.25, 0.05, 0.95)
    yolo_mask_threshold = bounded_float_config("YOLO_PERSON_MASK_THRESHOLD", 0.50, 0.05, 0.95)
    try:
        yolo_image_size = int(os.environ.get("YOLO_PERSON_IMAGE_SIZE") or "640")
    except ValueError:
        yolo_image_size = 640
    yolo_image_size = max(320, min(1280, yolo_image_size))
    yolo_model, yolo_model_diagnostics = (
        load_yolo_person_segmentation_model()
        if yolo_enabled
        else (None, {"loaded": False, "model": None})
    )
    segmentation_diagnostics: dict[str, Any] = {
        "mediapipeSegmentationFilter": segmentation_filter_enabled(action_type, family),
        "mediapipeSegmentationThreshold": round(segmentation_threshold, 3),
        "mediapipeSegmentationFrames": 0,
        "mediapipeSegmentationFilteredLandmarks": 0,
        "mediapipeSegmentationFilteredFrames": 0,
        "mediapipeSegmentationExamples": [],
        "yoloPersonSegmentationFilter": yolo_enabled,
        "yoloPersonSegmentationLoaded": bool(yolo_model_diagnostics.get("loaded")),
        "yoloPersonSegmentationModel": yolo_model_diagnostics.get("model"),
        "yoloPersonSegmentationError": yolo_model_diagnostics.get("error"),
        "yoloPersonSegmentationThreshold": round(yolo_threshold, 3),
        "yoloPersonSegmentationConfidence": round(yolo_confidence, 3),
        "yoloPersonSegmentationMaskThreshold": round(yolo_mask_threshold, 3),
        "yoloPersonSegmentationImageSize": yolo_image_size,
        "yoloPersonSegmentationFrames": 0,
        "yoloPersonSegmentationDetectedFrames": 0,
        "yoloPersonSegmentationFilteredLandmarks": 0,
        "yoloPersonSegmentationFilteredFrames": 0,
        "yoloPersonSegmentationExamples": [],
    }
    with pose_api.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as estimator:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estimator.process(rgb)
            if result.pose_landmarks:
                landmarks = [
                    [float(item.x), float(item.y), float(item.z), float(item.visibility)]
                    for item in result.pose_landmarks.landmark
                ]
                mask_scores, mask_diagnostics = apply_segmentation_mask_filter(
                    landmarks,
                    getattr(result, "segmentation_mask", None),
                    action_type=action_type,
                    family=family,
                    threshold=segmentation_threshold,
                )
                if mask_diagnostics.get("enabled"):
                    segmentation_diagnostics["mediapipeSegmentationFrames"] += 1
                    filtered_count = int(mask_diagnostics.get("filtered") or 0)
                    segmentation_diagnostics["mediapipeSegmentationFilteredLandmarks"] += filtered_count
                    if filtered_count:
                        segmentation_diagnostics["mediapipeSegmentationFilteredFrames"] += 1
                        examples = segmentation_diagnostics["mediapipeSegmentationExamples"]
                        if len(examples) < 8:
                            examples.append({
                                "frameIndex": frame_index,
                                "timeMs": int(frame_index / fps * 1000),
                                "filtered": mask_diagnostics.get("filteredLandmarks") or [],
                            })
                if yolo_model is not None and yolo_enabled:
                    segmentation_diagnostics["yoloPersonSegmentationFrames"] += 1
                    person_mask, yolo_detection = yolo_person_mask_for_frame(
                        yolo_model,
                        frame,
                        landmarks,
                        confidence_threshold=yolo_confidence,
                        image_size=yolo_image_size,
                        mask_threshold=yolo_mask_threshold,
                    )
                    if yolo_detection.get("detected"):
                        segmentation_diagnostics["yoloPersonSegmentationDetectedFrames"] += 1
                    yolo_scores, yolo_filter_diagnostics = apply_yolo_person_mask_filter(
                        landmarks,
                        person_mask,
                        action_type=action_type,
                        family=family,
                        threshold=yolo_threshold,
                    )
                    mask_scores = combine_landmark_mask_scores(mask_scores, yolo_scores)
                    if yolo_filter_diagnostics.get("enabled"):
                        yolo_filtered_count = int(yolo_filter_diagnostics.get("filtered") or 0)
                        segmentation_diagnostics["yoloPersonSegmentationFilteredLandmarks"] += yolo_filtered_count
                        if yolo_filtered_count:
                            segmentation_diagnostics["yoloPersonSegmentationFilteredFrames"] += 1
                            examples = segmentation_diagnostics["yoloPersonSegmentationExamples"]
                            if len(examples) < 8:
                                examples.append({
                                    "frameIndex": frame_index,
                                    "timeMs": int(frame_index / fps * 1000),
                                    "detection": yolo_detection,
                                    "filtered": yolo_filter_diagnostics.get("filteredLandmarks") or [],
                                })
                quality = action_frame_quality(landmarks, family, action_type)
                pose_frames.append(PoseFrame(
                    frame_index=frame_index,
                    time_ms=int(frame_index / fps * 1000),
                    landmarks=landmarks,
                    signal=movement_signal(landmarks, family),
                    quality=quality,
                    landmark_mask_scores=mask_scores,
                ))
            frame_index += 1
    capture.release()
    return PoseBackendResult(pose_frames, segmentation_diagnostics)


def coco_to_mediapipe_landmarks(
    keypoints: Any,
    scores: Any,
    width: int,
    height: int,
) -> list[list[float]]:
    landmarks = [[0.5, 0.5, 0.0, 0.0] for _ in range(33)]
    mapping = {
        0: LANDMARK.NOSE,
        5: LANDMARK.LEFT_SHOULDER,
        6: LANDMARK.RIGHT_SHOULDER,
        7: LANDMARK.LEFT_ELBOW,
        8: LANDMARK.RIGHT_ELBOW,
        9: LANDMARK.LEFT_WRIST,
        10: LANDMARK.RIGHT_WRIST,
        11: LANDMARK.LEFT_HIP,
        12: LANDMARK.RIGHT_HIP,
        13: LANDMARK.LEFT_KNEE,
        14: LANDMARK.RIGHT_KNEE,
        15: LANDMARK.LEFT_ANKLE,
        16: LANDMARK.RIGHT_ANKLE,
    }
    points = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(scores, dtype=float)
    for source_index, target_index in mapping.items():
        if source_index >= len(points):
            continue
        x, y = points[source_index][:2]
        score = confidence[source_index] if source_index < len(confidence) else 0.0
        landmarks[int(target_index)] = [
            float(x) / max(1, width),
            float(y) / max(1, height),
            0.0,
            float(score),
        ]
    return landmarks


def pose_instances_from_arrays(keypoints: Any, scores: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(scores, dtype=float)
    if points.size == 0 or confidence.size == 0:
        return []
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    if confidence.ndim == 1:
        confidence = np.expand_dims(confidence, axis=0)
    count = min(len(points), len(confidence))
    return [(points[index], confidence[index]) for index in range(count)]


def torso_horizontal_score(landmarks: list[list[float]]) -> float:
    shoulders = midpoint(landmarks, LANDMARK.LEFT_SHOULDER, LANDMARK.RIGHT_SHOULDER)
    hips = midpoint(landmarks, LANDMARK.LEFT_HIP, LANDMARK.RIGHT_HIP)
    dx = abs(float(shoulders[0] - hips[0]))
    dy = abs(float(shoulders[1] - hips[1]))
    return clamp((dx - 0.65 * dy) / 0.32)


def roi_score(center: tuple[float, float], bbox: list[float], target_roi: list[float] | None) -> float:
    if not target_roi:
        return 0.0
    inside = target_roi[0] <= center[0] <= target_roi[2] and target_roi[1] <= center[1] <= target_roi[3]
    overlap = bbox_iou(bbox, target_roi)
    return (0.35 if inside else -0.25) + 0.25 * overlap


def target_base_score(
    landmarks: list[list[float]],
    family: str,
    target_roi: list[float] | None,
) -> tuple[float, dict[str, float], list[float], tuple[float, float]]:
    chain = CHAIN_BY_FAMILY.get(family, CHAIN_BY_FAMILY["general"])
    quality = frame_quality(landmarks, family)
    bbox = landmark_bbox(landmarks, chain)
    center = bbox_center(bbox)
    area_score = clamp(bbox_area(bbox) * 8.0)
    lower_screen_score = clamp((center[1] - 0.28) / 0.45)
    horizontal_score = torso_horizontal_score(landmarks)
    roi_bonus = roi_score(center, bbox, target_roi)
    if family == "press":
        score = (
            0.40 * quality
            + 0.30 * horizontal_score
            + 0.15 * lower_screen_score
            + 0.15 * area_score
            + roi_bonus
        )
    else:
        score = 0.70 * quality + 0.15 * area_score + 0.15 * lower_screen_score + roi_bonus
    features = {
        "quality": round(float(quality), 3),
        "horizontal": round(float(horizontal_score), 3),
        "area": round(float(area_score), 3),
        "lowerScreen": round(float(lower_screen_score), 3),
    }
    return float(score), features, bbox, center


def is_far_smaller_candidate(candidate: dict[str, Any], tracker: TargetTracker, target_roi: list[float] | None) -> bool:
    if target_roi or tracker.center is None or tracker.bbox is None:
        return False
    previous_area = bbox_area(tracker.bbox)
    candidate_area = bbox_area(candidate["bbox"])
    if previous_area <= 1e-8:
        return False
    center_distance = float(np.linalg.norm(np.array(candidate["center"]) - np.array(tracker.center)))
    return bool(
        center_distance > 0.30
        and candidate_area < previous_area * 0.45
        and float(candidate["baseScore"]) < 0.82
    )


def dominant_target_override(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_roi: list[float] | None,
) -> dict[str, Any]:
    if target_roi or len(candidates) < 2:
        return selected
    dominant = max(candidates, key=lambda item: item["baseScore"])
    if dominant is selected:
        return selected
    selected_area = bbox_area(selected["bbox"])
    dominant_area = bbox_area(dominant["bbox"])
    if (
        dominant_area >= max(0.025, selected_area * 3.0)
        and float(dominant["baseScore"]) >= float(selected["baseScore"]) + 0.14
    ):
        return dominant
    return selected


def select_target_instance(
    instances: list[tuple[Any, Any]],
    width: int,
    height: int,
    family: str,
    tracker: TargetTracker,
    target_roi: list[float] | None = None,
) -> tuple[Any, Any, dict[str, Any]] | None:
    if not instances:
        return None
    tracker.frame_count += 1
    if len(instances) > 1:
        tracker.multi_person_frames += 1

    candidates: list[dict[str, Any]] = []
    for index, (points, confidence) in enumerate(instances):
        landmarks = coco_to_mediapipe_landmarks(points, confidence, width, height)
        base_score, features, bbox, center = target_base_score(landmarks, family, target_roi)
        tracking_score = 0.0
        if tracker.center is not None:
            distance = float(np.linalg.norm(np.array(center) - np.array(tracker.center)))
            tracking_score = clamp(1.0 - distance / 0.32)
            if tracker.bbox is not None:
                tracking_score = max(tracking_score, bbox_iou(bbox, tracker.bbox))
        combined_score = 0.72 * base_score + 0.28 * tracking_score if tracker.center is not None else base_score
        candidates.append({
            "index": index,
            "points": points,
            "confidence": confidence,
            "landmarks": landmarks,
            "bbox": bbox,
            "center": center,
            "baseScore": base_score,
            "trackingScore": tracking_score,
            "combinedScore": combined_score,
            "features": features,
        })

    candidates.sort(key=lambda item: item["combinedScore"], reverse=True)
    selected = dominant_target_override(candidates[0], candidates, target_roi)
    if is_far_smaller_candidate(selected, tracker, target_roi):
        tracker.target_lost_count += 1
        tracker.rejected_distractor_count += 1
        tracker.lock_confidences.append(0.0)
        return None
    runner_up_scores = [item["combinedScore"] for item in candidates if item is not selected]
    second_score = max(runner_up_scores) if runner_up_scores else 0.0
    margin = float(selected["combinedScore"] - second_score)
    previous_center = tracker.center
    if previous_center is not None:
        center_distance = float(np.linalg.norm(np.array(selected["center"]) - np.array(previous_center)))
        if center_distance > 0.28 and len(instances) > 1:
            tracker.target_switch_count += 1
    tracker.center = selected["center"]
    tracker.bbox = selected["bbox"]
    tracker.selected_index = int(selected["index"])
    lock_confidence = clamp(0.42 + margin + 0.18 * selected["trackingScore"] + 0.12 * selected["baseScore"])
    tracker.lock_confidences.append(lock_confidence)
    return selected["points"], selected["confidence"], {
        "targetId": int(selected["index"]),
        "candidateCount": len(instances),
        "bbox": [round(float(value), 4) for value in selected["bbox"]],
        "lockConfidence": round(float(lock_confidence), 3),
        "score": round(float(selected["combinedScore"]), 3),
        "features": selected["features"],
    }


def target_tracker_diagnostics(tracker: TargetTracker) -> dict[str, Any]:
    average_lock = average_valid(tracker.lock_confidences)
    return {
        "targetPersonId": tracker.selected_index,
        "targetSwitchCount": tracker.target_switch_count,
        "targetLostCount": tracker.target_lost_count,
        "rejectedDistractorCount": tracker.rejected_distractor_count,
        "targetLockConfidence": round(float(average_lock), 3),
        "multiPersonFrames": tracker.multi_person_frames,
    }


def extract_mmpose_pose_frames(
    video_path: Path,
    family: str,
    fps: float,
    step: int,
    action_type: str = "other",
    target_roi: list[float] | None = None,
) -> PoseBackendResult:
    try:
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except Exception as error:
        raise RuntimeError(f"MMPose unavailable: {error}") from error

    pose_config = os.environ.get("MMPOSE_CONFIG") or "human"
    checkpoint = os.environ.get("MMPOSE_CHECKPOINT") or None
    device = os.environ.get("MMPOSE_DEVICE") or "cuda:0"
    kwargs: dict[str, Any] = {"pose2d": pose_config, "device": device}
    if checkpoint:
        kwargs["pose2d_weights"] = checkpoint
    inferencer = MMPoseInferencer(**kwargs)

    capture = cv2.VideoCapture(str(video_path))
    pose_frames: list[PoseFrame] = []
    tracker = TargetTracker()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        height, width = frame.shape[:2]
        result = next(inferencer(frame, show=False, return_vis=False))
        predictions = result.get("predictions") or []
        people = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        if people:
            instances = [
                (person.get("keypoints") or [], person.get("keypoint_scores") or [])
                for person in people
            ]
            selected = select_target_instance(instances, width, height, family, tracker, target_roi)
            if not selected:
                frame_index += 1
                continue
            person_points, person_scores, target_info = selected
            landmarks = coco_to_mediapipe_landmarks(person_points, person_scores, width, height)
            quality = action_frame_quality(landmarks, family, action_type)
            pose_frames.append(PoseFrame(
                frame_index=frame_index,
                time_ms=int(frame_index / fps * 1000),
                landmarks=landmarks,
                signal=movement_signal(landmarks, family),
                quality=quality,
                target_id=target_info["targetId"],
                candidate_count=target_info["candidateCount"],
                target_lock_confidence=target_info["lockConfidence"],
                person_bbox=target_info["bbox"],
            ))
        frame_index += 1
    capture.release()
    return PoseBackendResult(pose_frames, target_tracker_diagnostics(tracker))


def parse_size_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    normalized = raw.lower().replace(",", "x").replace("*", "x")
    parts = [item.strip() for item in normalized.split("x") if item.strip()]
    if len(parts) != 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return default


def best_pose_instance(keypoints: Any, scores: Any) -> tuple[Any, Any] | None:
    points = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(scores, dtype=float)
    if points.size == 0 or confidence.size == 0:
        return None
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    if confidence.ndim == 1:
        confidence = np.expand_dims(confidence, axis=0)
    if not len(points):
        return None
    person_index = int(np.argmax(np.mean(confidence, axis=1)))
    return points[person_index], confidence[person_index]


def prepare_onnxruntime_cuda_dlls() -> None:
    if os.name != "nt":
        return
    try:
        import torch  # type: ignore
    except Exception:
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.exists():
        return
    path_value = str(torch_lib)
    os.environ["PATH"] = f"{path_value}{os.pathsep}{os.environ.get('PATH', '')}"
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory:
        add_dll_directory(path_value)


def extract_rtmlib_pose_frames(
    video_path: Path,
    family: str,
    fps: float,
    step: int,
    action_type: str = "other",
    target_roi: list[float] | None = None,
) -> PoseBackendResult:
    pose_model = os.environ.get("RTMLIB_POSE_MODEL") or ""
    det_model = os.environ.get("RTMLIB_DET_MODEL") or ""
    if not pose_model:
        raise RuntimeError("RTMLib requires local RTMLIB_POSE_MODEL; upstream model downloads are disabled")

    pose_path = Path(pose_model).expanduser()
    if not pose_path.exists():
        raise RuntimeError(f"RTMLib pose model not found: {pose_path}")

    backend = os.environ.get("RTMLIB_BACKEND") or "onnxruntime"
    device = os.environ.get("RTMLIB_DEVICE") or "cuda"
    mode = os.environ.get("RTMLIB_MODE") or "balanced"
    pose_input_size = parse_size_env("RTMLIB_POSE_INPUT_SIZE", (192, 256))
    det_input_size = parse_size_env("RTMLIB_DET_INPUT_SIZE", (640, 640))
    is_one_stage = "rtmo" in pose_path.name.lower() or os.environ.get("RTMLIB_ONE_STAGE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if backend == "onnxruntime" and "cuda" in device.lower():
        prepare_onnxruntime_cuda_dlls()

    try:
        if is_one_stage:
            from rtmlib import RTMO  # type: ignore

            estimator = RTMO(
                str(pose_path),
                model_input_size=parse_size_env("RTMLIB_POSE_INPUT_SIZE", (640, 640)),
                backend=backend,
                device=device,
            )
        else:
            if not det_model:
                raise RuntimeError(
                    "RTMLib top-down mode requires local RTMLIB_DET_MODEL; upstream model downloads are disabled"
                )
            det_path = Path(det_model).expanduser()
            if not det_path.exists():
                raise RuntimeError(f"RTMLib detector model not found: {det_path}")
            from rtmlib import Body  # type: ignore

            estimator = Body(
                det=str(det_path),
                det_input_size=det_input_size,
                pose=str(pose_path),
                pose_input_size=pose_input_size,
                mode=mode,
                backend=backend,
                device=device,
            )
    except Exception as error:
        raise RuntimeError(f"RTMLib unavailable: {error}") from error

    capture = cv2.VideoCapture(str(video_path))
    pose_frames: list[PoseFrame] = []
    tracker = TargetTracker()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        height, width = frame.shape[:2]
        keypoints, scores = estimator(frame)
        instances = pose_instances_from_arrays(keypoints, scores)
        selected = select_target_instance(instances, width, height, family, tracker, target_roi)
        if selected:
            person_points, person_scores, target_info = selected
            landmarks = coco_to_mediapipe_landmarks(person_points, person_scores, width, height)
            quality = action_frame_quality(landmarks, family, action_type)
            pose_frames.append(PoseFrame(
                frame_index=frame_index,
                time_ms=int(frame_index / fps * 1000),
                landmarks=landmarks,
                signal=movement_signal(landmarks, family),
                quality=quality,
                target_id=target_info["targetId"],
                candidate_count=target_info["candidateCount"],
                target_lock_confidence=target_info["lockConfidence"],
                person_bbox=target_info["bbox"],
            ))
        frame_index += 1
    capture.release()
    return PoseBackendResult(pose_frames, target_tracker_diagnostics(tracker))


PoseExtractor = Callable[..., list[PoseFrame] | PoseBackendResult]


def estimate_pose_frames(
    video_path: Path,
    family: str,
    fps: float,
    step: int,
    preferred_backend: str | None = None,
    action_type: str = "other",
    target_roi: list[float] | None = None,
    extractors: dict[str, PoseExtractor] | None = None,
    strict_backend: bool = False,
) -> PoseBackendResult:
    requested = str(preferred_backend or os.environ.get("POSE_BACKEND") or DEFAULT_POSE_BACKEND).strip().lower()
    available_extractors = extractors or {
        "rtmlib": extract_rtmlib_pose_frames,
        "mmpose": extract_mmpose_pose_frames,
        "mediapipe": extract_mediapipe_pose_frames,
    }
    diagnostics: dict[str, Any] = {
        "requestedPoseBackend": requested,
        "poseBackend": None,
        "poseBackendFallback": None,
    }
    errors: list[str] = []
    strict_backend = strict_backend or truthy_config(os.environ.get("POSE_BACKEND_STRICT"))
    backend_candidates = [requested] if strict_backend else pose_backend_order(requested)
    for backend in backend_candidates:
        extractor = available_extractors.get(backend)
        if not extractor:
            continue
        try:
            extracted = extractor(video_path, family, fps, step, action_type, target_roi)
            if isinstance(extracted, PoseBackendResult):
                frames = extracted.frames
                extra_diagnostics = extracted.diagnostics
            else:
                frames = extracted
                extra_diagnostics = {}
            if frames or backend == "mediapipe":
                diagnostics["poseBackend"] = backend
                if errors:
                    diagnostics["poseBackendFallback"] = "; ".join(errors)[-500:]
                diagnostics.update(extra_diagnostics)
                return PoseBackendResult(frames=frames, diagnostics=diagnostics)
            errors.append(f"{backend} returned no pose frames")
        except Exception as error:
            errors.append(f"{backend}: {error}")
    diagnostics["poseBackend"] = "none"
    diagnostics["poseBackendFallback"] = "; ".join(errors)[-500:] if errors else "No pose backend attempted"
    return PoseBackendResult(frames=[], diagnostics=diagnostics)


def analyze_video(payload: dict[str, Any]) -> dict[str, Any]:
    video_path = Path(payload["videoPath"]).resolve()
    output_dir = Path(payload["outputDir"]).resolve()
    requested_action_type = str(payload.get("actionType") or "auto_detect").strip()
    auto_action_requested = requested_action_type.lower() in AUTO_ACTION_TYPES
    action_type = "other" if auto_action_requested else requested_action_type
    action = ACTION_CATALOG.get(action_type, ACTION_CATALOG["other"])
    family = action["family"]
    camera_angle = str(payload.get("cameraAngle") or "unknown")
    target_roi = normalize_target_roi(payload.get("targetRoi"))
    auto_action_detection: dict[str, Any] = {
        "enabled": auto_action_requested,
        "requestedActionType": requested_action_type or "auto_detect",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = [
        calculation_log(
            "input",
            "璇诲彇鍒嗘瀽浠诲姟",
            "宸茶鍙栬棰戣矾寰勩€佸姩浣滅被鍨嬨€佹媿鎽勮搴﹀拰鍙€?ROI銆?",
            {
                "actionType": action_type,
                "actionName": action["name"],
                "family": family,
                "cameraAngle": camera_angle,
                "targetRoi": target_roi,
                "autoActionDetection": auto_action_detection,
            },
        )
    ]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("瑙嗛鏃犳硶鎵撳紑锛岃纭鏂囦欢鏍煎紡瀹屾暣銆?")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("瑙嗛鍏冩暟鎹棤鏁堬紝鏃犳硶寮€濮嬪垎鏋愩€?")

    configured_sample_fps = bounded_float_config("ANALYSIS_SAMPLE_FPS", 10.0, 4.0, 15.0)
    sample_fps = min(configured_sample_fps, max(4.0, fps))
    step = max(1, int(round(fps / sample_fps)))
    actual_sample_fps = fps / step
    render_mode = annotated_video_mode()
    capture.release()
    logs.append(calculation_log(
        "video",
        "璇诲彇瑙嗛鍩虹淇℃伅",
        f"瑙嗛 {round(duration, 2)} 绉掞紝鍘熷甯х巼 {round(fps, 2)} fps锛屾寜绾?{round(actual_sample_fps, 2)} fps 鎶芥牱鍒嗘瀽銆?",
        {
            "durationSeconds": round(duration, 2),
            "fps": round(fps, 2),
            "frameCount": frame_count,
            "width": width,
            "height": height,
            "sampleFps": round(actual_sample_fps, 2),
            "sampleStep": step,
            "configuredSampleFps": configured_sample_fps,
            "annotatedVideoMode": render_mode,
        },
    ))

    backend = estimate_pose_frames(
        video_path,
        family,
        fps,
        step,
        preferred_backend=payload.get("poseBackend"),
        action_type=action_type,
        target_roi=target_roi,
        strict_backend=truthy_config(payload.get("strictPoseBackend")),
    )
    raw_pose_frames = backend.frames
    pose_landmark_priors = normalize_pose_landmark_priors(
        payload.get("poseLandmarkPriors")
        or payload.get("glmLandmarkPriors")
        or payload.get("visionLandmarkPriors")
    )
    pose_frames = smooth_low_confidence_landmarks(
        raw_pose_frames,
        POSE_CONFIDENCE_FLOOR,
        family,
        action_type,
        pose_landmark_priors,
    )
    if auto_action_requested and pose_frames:
        auto_action_detection = infer_action_type_from_frames(pose_frames, actual_sample_fps)
        action_type = str(auto_action_detection.get("actionType") or "other")
        action = ACTION_CATALOG.get(action_type, ACTION_CATALOG["other"])
        family = action["family"]
        pose_frames = smooth_low_confidence_landmarks(
            raw_pose_frames,
            POSE_CONFIDENCE_FLOOR,
            family,
            action_type,
            pose_landmark_priors,
        )
        logs.append(calculation_log(
            "auto_action",
            "鑷姩璇嗗埆鍔ㄤ綔绫诲瀷",
            f"鏍规嵁楠ㄩ杩愬姩鐗瑰緛閫夋嫨 {action['name']}锛屽悗缁鍒欐寜璇ュ姩浣滄墽琛屻€?",
            {
                "requestedActionType": requested_action_type or "auto_detect",
                "selectedActionType": action_type,
                "selectedActionName": action["name"],
                "selectedFamily": family,
                "confidence": auto_action_detection.get("confidence"),
                "features": auto_action_detection.get("features"),
                "signature": auto_action_detection.get("signature"),
            },
            "warning" if float(auto_action_detection.get("confidence") or 0.0) < 0.12 else "done",
        ))
    for frame in pose_frames:
        frame.signal = movement_signal(frame.landmarks, family)
        frame.quality = action_frame_quality(frame.landmarks, family, action_type)
    full_pose_frames = list(pose_frames)

    expected_samples = max(1, int(duration * actual_sample_fps))
    pose_coverage = min(1.0, len(pose_frames) / expected_samples)
    average_quality = average_valid([item.quality for item in pose_frames])
    capture_quality = "good"
    if pose_coverage < 0.45 or average_quality < 0.48 or len(pose_frames) < 8:
        capture_quality = "insufficient"
    elif pose_coverage < 0.72 or average_quality < 0.65:
        capture_quality = "limited"
    logs.append(calculation_log(
        "pose",
        "鎻愬彇浜轰綋鍏抽敭鐐?",
        f"濮挎€佸悗绔繑鍥?{len(raw_pose_frames)} 涓Э鎬佸抚锛岃鐩栫巼 {round(pose_coverage * 100, 1)}%锛屽钩鍧囧叧閿偣璐ㄩ噺 {round(average_quality, 3)}銆?",
        {
            "poseBackend": backend.diagnostics.get("poseBackend"),
            "poseBackendFallback": backend.diagnostics.get("poseBackendFallback"),
            "rawPoseFrames": len(raw_pose_frames),
            "smoothedPoseFrames": len(pose_frames),
            "expectedSamples": expected_samples,
            "poseCoverage": round(pose_coverage, 3),
            "averageQuality": round(average_quality, 3),
            "captureQuality": capture_quality,
            "targetLockConfidence": backend.diagnostics.get("targetLockConfidence"),
            "targetSwitchCount": backend.diagnostics.get("targetSwitchCount"),
            "multiPersonFrames": backend.diagnostics.get("multiPersonFrames"),
            "poseLandmarkPriors": {
                "count": len(pose_landmark_priors),
                "source": "payload_glm_or_manual",
            },
        },
        "warning" if capture_quality != "good" else "done",
    ))

    pose_engine_comparison: dict[str, Any] | None = None
    secondary_pose_frames_for_artifact: list[Any] | None = None
    if pose_engine_compare_enabled(payload):
        primary_backend = backend.diagnostics.get("poseBackend")
        compare_started = time.time()
        logs.append(calculation_log(
            "pose_compare_start",
            "Start secondary pose engine comparison",
            "Running an optional MediaPipe 33-landmark pass for diagnostics only.",
            {
                "primaryBackend": primary_backend,
                "secondaryBackend": POSE_COMPARE_SECONDARY_BACKEND,
                "primaryPoseFrames": len(pose_frames),
                "poseCoverage": round(pose_coverage, 3),
                "averageQuality": round(average_quality, 3),
            },
        ))
        try:
            from pose_compare import build_pose_engine_comparison

            pose_engine_comparison = build_pose_engine_comparison(
                video_path=video_path,
                primary_frames=pose_frames,
                primary_backend=str(primary_backend or "unknown"),
                primary_pose_coverage=pose_coverage,
                primary_average_confidence=average_quality,
                fps=fps,
                frame_count=frame_count,
                step=step,
            )
            logs.append(calculation_log(
                "pose_compare_secondary_done",
                "Secondary pose engine finished",
                "The diagnostic MediaPipe comparison completed without changing the primary analysis result.",
                {
                    "secondary": pose_engine_comparison.get("secondary"),
                    "runtimeMs": pose_engine_comparison.get("runtimeMs"),
                },
            ))
            logs.append(calculation_log(
                "pose_compare_summary",
                "Pose engine comparison summarized",
                f"Recommendation: {pose_engine_comparison.get('recommendation')}.",
                {
                    "primary": pose_engine_comparison.get("primary"),
                    "secondary": pose_engine_comparison.get("secondary"),
                    "topDivergentJoints": pose_engine_comparison.get("topDivergentJoints"),
                    "recommendation": pose_engine_comparison.get("recommendation"),
                },
                "warning" if pose_engine_comparison.get("recommendation") == "needs_manual_review" else "done",
            ))
        except Exception as error:
            runtime_ms = int(round((time.time() - compare_started) * 1000))
            pose_engine_comparison = failed_pose_engine_comparison_payload(
                primary_backend=str(primary_backend or "unknown"),
                primary_pose_coverage=pose_coverage,
                primary_average_confidence=average_quality,
                primary_frame_count=len(pose_frames),
                runtime_ms=runtime_ms,
                error=error,
            )
            logs.append(calculation_log(
                "pose_compare_summary",
                "Secondary pose engine failed",
                "The optional comparison failed, so the primary analysis result is kept unchanged.",
                {
                    "runtimeMs": runtime_ms,
                    "errorType": type(error).__name__,
                    "error": str(error)[:500],
                    "recommendation": "keep_primary",
                },
                "warning",
            ))

    if not pose_frames:
        pose_fusion = build_pose_fusion(
            comparison=pose_engine_comparison,
            backend_diagnostics=backend.diagnostics,
            rep_diagnostics=None,
            primary_rep_count=0,
            secondary_rule_summary=None,
            capture_quality="insufficient",
            action_type=action_type,
            family=family,
        )
        logs.append(calculation_log(
            "quality",
            "璇佹嵁涓嶈冻锛屽仠姝㈠姩浣滆绠?",
            "娌℃湁鍙敤濮挎€佸抚锛屾棤娉曠户缁绠楀姩浣滀俊鍙枫€佹鏁板拰璇勫垎銆?",
            {
                "poseBackend": backend.diagnostics.get("poseBackend"),
                "poseCoverage": 0,
                "reason": "no_pose_frames",
            },
            "error",
        ))
        return {
            "actionType": action_type,
            "actionName": action["name"],
            "bodyPart": action["bodyPart"],
            "family": family,
            "captureQuality": "insufficient",
            "confidence": 0,
            "repCount": 0,
            "repCountSource": "xiaoyuCoach",
            "overallScore": 0,
            "safetyLevel": "red",
            "issues": [issue(
                "INSUFFICIENT_EVIDENCE",
                "red",
                "娌℃湁璇嗗埆鍒板畬鏁磋缁冭€?",
                "瑙嗛涓病鏈夋寔缁彲瑙佺殑瀹屾暣浜轰綋鍔ㄤ綔璇佹嵁銆?",
                "閲嶆柊鎷嶆憚瀹屾暣宸ヤ綔缁勶紝淇濊瘉鍏ㄨ韩鍏ラ暅銆佸厜绾垮厖瓒充笖娌℃湁鏄庢樉閬尅銆?",
            )],
            "strengths": [],
            "measurements": {},
            "frameJudgments": [],
            "judgmentSegments": [],
            "stabilityProfile": stability_profile(action_type),
            "keyframes": [],
            "cameraAdvice": CAMERA_GUIDANCE[family],
            "metadata": {
                "durationSeconds": round(duration, 2),
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "orientation": "portrait" if height >= width else "landscape",
                "poseCoverage": 0,
                "poseBackend": backend.diagnostics.get("poseBackend"),
            },
            "diagnostics": {
                **backend.diagnostics,
                "selectedFamily": family,
                "detectedFamily": "unknown",
                "detectedGroup": "unknown",
                "autoActionDetection": auto_action_detection,
                "poseFusion": pose_fusion,
                "lowConfidenceWindows": [],
                **({"poseEngineComparison": pose_engine_comparison} if pose_engine_comparison else {}),
            },
            "calculationLogs": logs,
            "analysisVersion": "local-pose-v4",
            "ruleVersion": f"{family}-rules-v4",
        }

    raw_signal_full, signal_source = motion_signal_series(pose_frames, family, action_type, camera_angle)
    full_pose_frame_count = len(pose_frames)
    active_start, active_end, active_window = select_active_training_window(
        pose_frames,
        raw_signal_full,
        family,
        action_type,
        actual_sample_fps,
    )
    logs.append(calculation_log(
        "window",
        "閫夋嫨鏈夋晥璁粌鐗囨",
        f"鏈夋晥鐗囨浠?{active_window.get('startTimeMs', 0)}ms 鍒?{active_window.get('endTimeMs', 0)}ms锛屽師鍥狅細{active_window.get('reason', 'unknown')}銆?",
        {
            "fullPoseFrames": full_pose_frame_count,
            "activeStartPoseIndex": active_start,
            "activeEndPoseIndex": active_end,
            "activePoseFrames": max(0, active_end - active_start + 1),
            "activeTrainingWindow": active_window,
        },
        "warning" if float(active_window.get("confidence") or 1.0) < 0.22 else "done",
    ))
    pose_frames = pose_frames[active_start : active_end + 1]
    raw_signal = raw_signal_full[active_start : active_end + 1]
    for index, frame in enumerate(pose_frames):
        frame.signal = float(raw_signal[index])
    smoothed = smooth_signal(raw_signal, max(3, int(actual_sample_fps * 0.25) | 1))
    raw_rep_events: list[dict[str, Any]] = []
    if action_type == "machine_chest_press":
        peaks, signal_range = detect_repetitions(
            smoothed,
            actual_sample_fps,
            prominence_ratio=0.08,
            min_prominence=1.0,
            distance_seconds=0.75,
            prominence_window_seconds=1.6,
        )
    elif action_type in {"lat_pulldown", "single_arm_pulldown", "single_arm_hammer_row"}:
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        flexed_thresholds = {
            "lat_pulldown": 90.0,
            "single_arm_pulldown": 110.0,
            "single_arm_hammer_row": 110.0,
        }
        counter_rules = {
            "lat_pulldown": "elbow_angle_gt_135_to_lt_90",
            "single_arm_pulldown": "single_arm_elbow_angle_gt_135_to_lt_110",
            "single_arm_hammer_row": "single_arm_row_elbow_angle_gt_135_to_lt_110",
        }
        raw_rep_events = segment_lat_pulldown_repetitions(
            pose_frames,
            smoothed,
            actual_sample_fps,
            flexed_elbow_angle=flexed_thresholds.get(action_type, 90.0),
            counter_rule=counter_rules.get(action_type, "elbow_angle_gt_135_to_lt_90"),
        )
        peaks = [int(event["poseKeyIndex"]) for event in raw_rep_events]
    elif action_type == "preacher_curl":
        peaks, signal_range = detect_repetitions(
            smoothed,
            actual_sample_fps,
            prominence_ratio=0.14,
            min_prominence=2.0,
            distance_seconds=0.65,
        )
        raw_rep_events = segment_repetitions(pose_frames, smoothed, peaks, actual_sample_fps, "isolation_elbow")
        for event in raw_rep_events:
            event["counterRule"] = "preacher_elbow_flexion_peak_cycle"
    elif action_type == "y_raise":
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        raw_rep_events = segment_y_raise_repetitions(pose_frames, smoothed, actual_sample_fps)
        peaks = [int(event["poseKeyIndex"]) for event in raw_rep_events]
    elif action_type in {"hip_thrust", "plate_loaded_rear_leg_raise"}:
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        raw_rep_events = segment_hip_thrust_repetitions(pose_frames, smoothed, actual_sample_fps)
        if action_type == "plate_loaded_rear_leg_raise":
            for event in raw_rep_events:
                event["counterRule"] = "working_hip_extension_bottom_top_bottom"
        peaks = [int(event["poseKeyIndex"]) for event in raw_rep_events]
    elif action_type == "back_extension":
        peaks, signal_range = detect_repetitions(
            smoothed,
            actual_sample_fps,
            prominence_ratio=0.14,
            min_prominence=2.0,
            distance_seconds=0.65,
        )
        raw_rep_events = segment_repetitions(pose_frames, smoothed, peaks, actual_sample_fps, "hinge")
    elif action_type == "plate_loaded_romanian_deadlift":
        peaks, signal_range = detect_repetitions(
            smoothed,
            actual_sample_fps,
            prominence_ratio=0.14,
            min_prominence=2.0,
            distance_seconds=0.65,
        )
        raw_rep_events = segment_repetitions(pose_frames, smoothed, peaks, actual_sample_fps, "hinge")
        for event in raw_rep_events:
            event["counterRule"] = "plate_loaded_rdl_hip_flexion_top_bottom_top"
    elif family == "hinge":
        raw_rep_events, smoothed = segment_hinge_repetitions(pose_frames, actual_sample_fps)
        raw_signal = smoothed
        signal_source = "hinge_shoulder_hip_line_top_bottom_top"
        signal_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size else 0.0
        peaks = [int(event["poseKeyIndex"]) for event in raw_rep_events]
    else:
        peaks, signal_range = detect_repetitions(smoothed, actual_sample_fps)
    signature = movement_signature(pose_frames, actual_sample_fps, preferred_family=family)
    logs.append(calculation_log(
        "signal",
        "璁＄畻鍔ㄤ綔淇″彿鍜屽姩浣滄棌鍖归厤",
        f"浣跨敤 {signal_source} 淇″彿锛屼俊鍙峰箙搴?{round(signal_range, 2)}锛涚畻娉曟渶鍍?{signature.get('detectedFamily')}锛岀疆淇″害 {signature.get('confidence')}銆?",
        {
            "signalSource": signal_source,
            "signalRange": round(signal_range, 3),
            "rawSignal": numeric_summary(raw_signal),
            "smoothedSignal": numeric_summary(smoothed),
            "peakPoseIndexes": peaks,
            "peakTimesMs": [pose_frames[int(index)].time_ms for index in peaks if 0 <= int(index) < len(pose_frames)],
            "movementSignature": signature,
        },
        "warning" if signature.get("detectedFamily") != family and float(signature.get("confidence") or 0) < 0.45 else "done",
    ))
    if action_type not in {
        "lat_pulldown",
        "single_arm_pulldown",
        "single_arm_hammer_row",
        "preacher_curl",
        "y_raise",
    } and family != "hinge":
        raw_rep_events = segment_repetitions(pose_frames, smoothed, peaks, actual_sample_fps, signature["detectedFamily"])
    rep_events, rep_diagnostics = validate_rep_events(raw_rep_events, smoothed, actual_sample_fps, action_type)
    if action_type == "lat_pulldown":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "elbow_angle_gt_135_to_lt_90",
            "extendedElbowAngleThreshold": 135,
            "flexedElbowAngleThreshold": 90,
            "extendedFlexionSignalThreshold": 45,
            "flexedFlexionSignalThreshold": 90,
        }
    elif action_type == "single_arm_pulldown":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "single_arm_elbow_angle_gt_135_to_lt_110",
            "extendedElbowAngleThreshold": 135,
            "flexedElbowAngleThreshold": 110,
            "extendedFlexionSignalThreshold": 45,
            "flexedFlexionSignalThreshold": 70,
        }
    elif action_type == "hip_thrust":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "hip_angle_bottom_top_bottom",
            "bottomToTopSignal": "hip_extension_angle",
            "topHipAngleTarget": 150,
        }
    elif action_type == "plate_loaded_rear_leg_raise":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "working_hip_extension_bottom_top_bottom",
            "bottomToTopSignal": "working_side_hip_extension_angle",
            "topHipAngleTarget": 145,
            "minHipAngleRange": 22,
        }
    elif action_type == "single_arm_hammer_row":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "single_arm_row_elbow_angle_gt_135_to_lt_110",
            "extendedElbowAngleThreshold": 135,
            "flexedElbowAngleThreshold": 110,
        }
    elif action_type == "chest_supported_row":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "visible_arm_elbow_angle_gt_135_to_lt_112",
            "extendedElbowAngleThreshold": 135,
            "flexedElbowAngleThreshold": 112,
        }
    elif action_type == "plate_loaded_pulldown":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "visible_arm_elbow_angle_gt_135_to_lt_110",
            "extendedElbowAngleThreshold": 135,
            "flexedElbowAngleThreshold": 110,
        }
    elif action_type == "preacher_curl":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "preacher_elbow_flexion_peak_cycle",
            "primarySignal": "elbow_flexion_angle",
            "targetTopElbowAngle": 115,
            "targetExtendedElbowAngle": 128,
        }
    elif action_type == "back_extension":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "back_extension_hip_flexion_top_bottom_top",
            "primarySignal": "hip_flexion_angle",
        }
    elif action_type == "plate_loaded_romanian_deadlift":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "plate_loaded_rdl_trunk_hinge_top_bottom_top",
            "primarySignal": "trunk_hinge_angle",
        }
    elif action_type == "y_raise":
        rep_diagnostics = {
            **rep_diagnostics,
            "counterRule": "shoulder_angle_lt_60_to_gt_110_y_top",
            "lowShoulderAngleThreshold": 60,
            "topShoulderAngleThreshold": 110,
        }
    secondary_rule_summary: dict[str, Any] | None = None
    if pose_engine_comparison and not pose_engine_comparison.get("error"):
        try:
            from pose_compare import extract_secondary_pose_frames

            secondary_pose_frames_for_artifact = extract_secondary_pose_frames(video_path, fps, step)
            secondary_rule_summary = secondary_action_rule_summary(
                secondary_pose_frames_for_artifact,
                family=family,
                action_type=action_type,
                camera_angle=camera_angle,
                sample_fps=actual_sample_fps,
            )
            pose_engine_comparison["secondaryRuleSummary"] = secondary_rule_summary
        except Exception as error:
            secondary_rule_summary = {
                "repCount": 0,
                "repEvents": [],
                "countUnstable": True,
                "error": f"{type(error).__name__}: {str(error)[:240]}",
            }
            pose_engine_comparison["secondaryRuleSummary"] = secondary_rule_summary
    pose_fusion = build_pose_fusion(
        comparison=pose_engine_comparison,
        backend_diagnostics=backend.diagnostics,
        rep_diagnostics=rep_diagnostics,
        primary_rep_count=len(rep_events),
        secondary_rule_summary=secondary_rule_summary,
        capture_quality=capture_quality,
        action_type=action_type,
        family=family,
    )
    primary_public_rep_events = public_rep_events(rep_events, "xiaoyuCoach")
    secondary_public_rep_events = public_rep_events(
        (secondary_rule_summary or {}).get("repEvents") or [],
        "motionTracker",
    )
    output_rep_events = (
        secondary_public_rep_events
        if pose_fusion.get("repCountSource") == "motionTracker" and secondary_public_rep_events
        else primary_public_rep_events
    )
    output_rep_count = int(pose_fusion.get("fusedRepCount") or len(output_rep_events))
    phase_source_frames = (
        secondary_pose_frames_for_artifact
        if pose_fusion.get("repCountSource") == "motionTracker" and secondary_pose_frames_for_artifact
        else full_pose_frames
    )
    phase_judgments = build_movement_phase_judgments(
        phase_source_frames,
        output_rep_events,
        actual_sample_fps,
    )
    fixed_foot_support: dict[str, Any] = {"enabled": False, "ignoredForScoring": False}
    xiaoyu_render_frames: list[PoseFrame] = list(full_pose_frames)
    fixed_display_window: dict[str, Any] | None = None
    if fixed_foot_action(action_type):
        pad_ms = 650
        if rep_events:
            display_start_ms = max(
                int(active_window.get("startTimeMs") or pose_frames[0].time_ms),
                min(int(event.get("startTimeMs") or pose_frames[0].time_ms) for event in rep_events) - pad_ms,
            )
            display_end_ms = min(
                int(active_window.get("endTimeMs") or pose_frames[-1].time_ms),
                max(int(event.get("endTimeMs") or pose_frames[-1].time_ms) for event in rep_events) + pad_ms,
            )
            window_source = "rep_events"
        else:
            display_start_ms = int(active_window.get("startTimeMs") or pose_frames[0].time_ms)
            display_end_ms = int(active_window.get("endTimeMs") or pose_frames[-1].time_ms)
            window_source = "active_window"
        support_source_frames = [
            frame for frame in pose_frames
            if display_start_ms <= int(frame.time_ms) <= display_end_ms
        ] or pose_frames
        fixed_foot_support = build_fixed_foot_support(support_source_frames, action_type)
        xiaoyu_render_source_frames = [
            frame for frame in full_pose_frames
            if display_start_ms <= int(frame.time_ms) <= display_end_ms
        ] or support_source_frames
        xiaoyu_render_frames = apply_fixed_foot_display_lock(
            xiaoyu_render_source_frames,
            action_type,
            fixed_foot_support,
        )
        fixed_display_window = {
            "source": window_source,
            "startTimeMs": display_start_ms,
            "endTimeMs": display_end_ms,
            "renderPoseFrames": len(xiaoyu_render_frames),
        }
        logs.append(calculation_log(
            "fixed_foot_support",
            "Fixed-foot support handling",
            "Fixed-foot action: foot landmarks are ignored for scoring; display locks reliable ankles and hides heel/toe details.",
            {
                "support": fixed_foot_support,
                "displayWindow": fixed_display_window,
            },
            "warning" if int(fixed_foot_support.get("anchorCount") or 0) == 0 else "done",
        ))
    xiaoyu_render_frames = recover_short_display_landmark_gaps(xiaoyu_render_frames)
    xiaoyu_render_frames = apply_display_landmark_suppression(
        xiaoyu_render_frames,
        action_type,
        family,
    )
    valid_peaks = [int(event["poseKeyIndex"]) for event in rep_events]
    stage_indices = select_stages_from_rep_events(smoothed, rep_events, actual_sample_fps) if rep_events else select_stages(smoothed, peaks, actual_sample_fps)
    logs.append(calculation_log(
        "reps",
        "鍒囧垎閲嶅娆℃暟",
        f"鍊欓€夊嘲鍊?{len(peaks)} 涓紝鍘熷 rep {len(raw_rep_events)} 涓紝鏈€缁堟湁鏁?rep {len(rep_events)} 涓€?",
        {
            "rawPeakCount": len(peaks),
            "rawRepCount": len(raw_rep_events),
            "validRepCount": len(rep_events),
            "repDiagnostics": rep_diagnostics,
            "repEvents": [
                {
                    "repIndex": event.get("repIndex"),
                    "startTimeMs": event.get("startTimeMs"),
                    "keyTimeMs": event.get("keyTimeMs"),
                    "endTimeMs": event.get("endTimeMs"),
                    "quality": event.get("quality"),
                    "startSignal": event.get("startSignal"),
                    "keySignal": event.get("keySignal"),
                    "endSignal": event.get("endSignal"),
                    "signalAmplitude": event.get("signalAmplitude"),
                    "durationSeconds": event.get("durationSeconds"),
                    "descentSeconds": event.get("descentSeconds"),
                    "ascentSeconds": event.get("ascentSeconds"),
                }
                for event in rep_events[:20]
            ],
        },
        "warning" if rep_diagnostics.get("countUnstable") else "done",
    ))
    logs.append(calculation_log(
        "pose_fusion",
        "铻嶅悎涓ゅ楠ㄩ寮曟搸鍒ゆ柇",
        f"铻嶅悎寤鸿閫夋嫨 {pose_fusion.get('selectedEngine')}锛屾鏁版潵婧?{pose_fusion.get('repCountSource')}锛岃瘎鍒嗘潵婧?{pose_fusion.get('scoreSource')}銆?",
        pose_fusion,
        "warning" if pose_fusion.get("selectedEngine") in {"hybridReview", "needsReview"} else "done",
    ))
    measurements = summarized_measurements(pose_frames, signal_range, valid_peaks, action_type)
    measurements["poseCoverage"] = round(pose_coverage, 3)
    measurements["activeTrainingWindow"] = active_window
    measurements["fixedFootSupport"] = fixed_foot_support
    if fixed_display_window is not None:
        measurements["fixedFootDisplayWindow"] = fixed_display_window
    measurements["movementMatch"] = movement_match_profile(pose_frames, family, actual_sample_fps)
    measurements["movementSignature"] = signature
    measurements["repSegmentation"] = {
        **rep_diagnostics,
        "signalSource": signal_source,
        "activeWindowTrimmedStartFrames": active_window.get("trimmedStartFrames", 0),
        "activeWindowTrimmedEndFrames": active_window.get("trimmedEndFrames", 0),
    }
    selected_stability_profile = stability_profile(action_type)
    should_estimate_camera_motion = (
        camera_angle in set(selected_stability_profile.get("supportedViews") or [])
        and selected_stability_profile.get("mode") not in {"disabled", "primary_trunk_motion"}
    )
    camera_motion = (
        estimate_camera_motion(video_path, pose_frames)
        if should_estimate_camera_motion
        else {
            "frames": [],
            "summary": {
                "method": "background_optical_flow_ransac",
                "status": "not_required",
                "availableFrames": 0,
                "frameCount": len(pose_frames),
                "coverage": 0.0,
                "reason": "stability_mode_or_view_not_scored",
            },
        }
    )
    stability = build_stability_analysis(
        pose_frames,
        action_type,
        camera_angle,
        actual_sample_fps,
        movement_values=smoothed,
        rep_events=rep_events,
        camera_motion=camera_motion,
    )
    measurements["stability"] = stability.get("summary") or {}
    logs.append(calculation_log(
        "camera_motion",
        "估计相机运动并补偿躯干轨迹",
        "使用人物区域外的背景特征估计相机平移、旋转和缩放；背景证据不足时自动退回未补偿判断并在结果中标明。",
        camera_motion.get("summary") or {},
        "done" if (stability.get("summary") or {}).get("cameraCompensation", {}).get("applied") else "warning",
    ))
    logs.append(calculation_log(
        "measurements",
        "姹囨€昏搴︺€佸箙搴﹀拰缃俊搴?",
        "宸茶绠楀叧鑺傝搴︺€佹椿鍔ㄥ箙搴︺€佸乏鍙冲樊寮傘€佸叧閿寚鏍囩疆淇″害鍜屽姩浣滄棌鍖归厤缁撴灉銆?",
        {
            "angles": measurements.get("angles"),
            "ranges": measurements.get("ranges"),
            "asymmetry": measurements.get("asymmetry"),
            "metricConfidence": measurements.get("metricConfidence"),
            "movementMatch": measurements.get("movementMatch"),
            "repSegmentation": measurements.get("repSegmentation"),
            "stability": measurements.get("stability"),
        },
    ))

    issues, strengths = evaluate_rules(
        action_type,
        family,
        measurements,
        capture_quality,
        camera_angle,
    )
    issues = [item for item in issues if item.get("code") not in LEGACY_STABILITY_ISSUE_CODES]
    stability_issue = build_stability_issue(stability)
    if stability_issue is not None:
        issues.append(stability_issue)
    elif (stability.get("summary") or {}).get("evaluated"):
        strengths.append("逐帧稳定性状态机未发现达到持续时间阈值的躯干晃动。")
    logs.append(calculation_log(
        "stability",
        "逐帧躯干稳定性判断完成",
        "稳定性引擎已按动作配置和拍摄视角输出逐帧状态与连续时间区间。",
        {
            "profile": stability.get("profile"),
            "summary": stability.get("summary"),
            "segments": stability.get("judgmentSegments"),
            "legacyIssueCodesSuppressed": sorted(LEGACY_STABILITY_ISSUE_CODES),
        },
        "warning" if stability_issue is not None else "done",
    ))
    if rep_diagnostics.get("countUnstable"):
        issues.append(issue(
            "COUNT_UNSTABLE",
            "yellow",
            "Rep count evidence unstable",
            "The rep counter rejected several candidate peaks because the bench-press motion signal was inconsistent.",
            "Retest with a clearer side-front camera angle or manually lock the target region before trusting the rep count.",
        ))
    if (
        float(backend.diagnostics.get("targetLockConfidence") or 1.0) < 0.55
        or int(backend.diagnostics.get("targetSwitchCount") or 0) > 0
    ):
        issues.append(issue(
            "TARGET_UNCERTAIN",
            "yellow",
            "Target person lock is uncertain",
            "The pose backend saw multiple people or an unstable target track, so detailed per-rep judgments may be unreliable.",
            "Use a tighter crop/target region or reshoot with fewer people in frame.",
        ))
    if not any(item["code"] in {"ACTION_MISMATCH", "INSUFFICIENT_EVIDENCE", "COUNT_UNSTABLE", "TARGET_UNCERTAIN"} for item in issues):
        rep_issues = evaluate_rep_rules(action_type, family, pose_frames, rep_events, camera_angle)
        rep_codes = {item["code"] for item in rep_issues}
        issues = [item for item in issues if item["code"] not in rep_codes] + rep_issues
        issues = [item for item in issues if item.get("code") not in LEGACY_STABILITY_ISSUE_CODES]
        if stability_issue is not None and not any(item.get("code") == stability_issue.get("code") for item in issues):
            issues.append(stability_issue)
    analysis_average_quality = average_valid([item.quality for item in pose_frames])
    confidence = max(0.0, min(0.98, analysis_average_quality * min(1.0, pose_coverage / 0.8)))
    overall_score, safety_level = score_result(issues, confidence)
    logs.append(calculation_log(
        "rules",
        "鎵ц鍔ㄤ綔瑙勫垯鍜岃瘎鍒?",
        f"瑙勫垯鍙戠幇 {len(issues)} 涓棶棰橈紝缁煎悎缃俊搴?{round(confidence, 3)}锛屾渶缁堝垎鏁?{overall_score}锛屽畨鍏ㄧ瓑绾?{safety_level}銆?",
        {
            "issueCodes": [item.get("code") for item in issues],
            "issues": issues,
            "strengths": strengths,
            "analysisAverageQuality": round(analysis_average_quality, 3),
            "confidence": round(confidence, 3),
            "overallScore": overall_score,
            "safetyLevel": safety_level,
            "scoreInputs": {
                "issueCount": len(issues),
                "captureQuality": capture_quality,
                "poseCoverage": round(pose_coverage, 3),
                "averageQuality": round(average_quality, 3),
            },
        },
        "warning" if issues else "done",
    ))

    keyframes = []
    image_paths = []
    for index, pose_index in enumerate(stage_indices[:4]):
        pose_index = max(0, min(len(pose_frames) - 1, int(pose_index)))
        selected = pose_frames[pose_index]
        display_selected = (
            apply_fixed_foot_display_lock([selected], action_type, fixed_foot_support)[0]
            if fixed_foot_action(action_type)
            else selected
        )
        display_selected = apply_display_landmark_suppression([display_selected], action_type, family)[0]
        image_name = f"stage_{index + 1}.jpg"
        image_path = output_dir / image_name
        draw_evidence_frame(
            video_path,
            display_selected,
            image_path,
            f"STAGE {index + 1}",
        )
        image_paths.append(image_path)
        keyframes.append({
            "stage": action["stages"][index],
            "timeMs": selected.time_ms,
            "frameIndex": selected.frame_index,
            "image": image_name,
            "quality": round(selected.quality, 3),
        })

    contact_sheet = output_dir / "contact_sheet.jpg"
    make_contact_sheet(image_paths, contact_sheet)
    annotated_videos: dict[str, dict[str, Any]] = {}
    annotated_video_warnings: list[dict[str, str]] = []

    # The visible evidence must follow the selected pose engine. Rep counting may
    # legitimately use another signal, but that must not silently replace the
    # skeleton video shown to the user.
    selected_annotated_key = (
        "motionTracker"
        if pose_fusion.get("selectedEngine") == "motionTracker"
        else "xiaoyuCoach"
    )
    render_xiaoyu = render_mode == "all" or (render_mode == "selected" and selected_annotated_key == "xiaoyuCoach")
    render_motion_tracker = (
        pose_engine_comparison
        and (render_mode == "all" or (render_mode == "selected" and selected_annotated_key == "motionTracker"))
    )

    def render_xiaoyu_video() -> None:
        annotated_videos["xiaoyuCoach"] = render_pose_overlay_video(
            video_path,
            xiaoyu_render_frames,
            output_dir / "pose_xiaoyu_coach.mp4",
            label=f"XIAOYU-COACH {backend.diagnostics.get('poseBackend') or 'primary'}",
            output_fps=actual_sample_fps,
            landmark_color=(80, 255, 190),
            connection_color=(54, 214, 255),
            rep_events=primary_public_rep_events,
            issues=issues,
            strengths=strengths,
            frame_judgments=stability.get("frameJudgments"),
            phase_judgments=phase_judgments,
        )
        annotated_videos["xiaoyuCoach"]["repCount"] = len(primary_public_rep_events)
        annotated_videos["xiaoyuCoach"]["repCountSource"] = "xiaoyuCoach"

    def render_motion_tracker_video() -> None:
        nonlocal secondary_pose_frames_for_artifact
        if secondary_pose_frames_for_artifact is None:
            from pose_compare import extract_secondary_pose_frames

            secondary_pose_frames_for_artifact = extract_secondary_pose_frames(video_path, fps, step)
        annotated_videos["motionTracker"] = render_pose_overlay_video(
            video_path,
            secondary_pose_frames_for_artifact,
            output_dir / "pose_motion_tracker.mp4",
            label="MOTION-TRACKER MEDIAPIPE",
            output_fps=actual_sample_fps,
            landmark_color=(255, 120, 230),
            connection_color=(255, 180, 80),
            rep_events=secondary_public_rep_events or output_rep_events,
            issues=issues,
            strengths=strengths,
            frame_judgments=stability.get("frameJudgments"),
            phase_judgments=phase_judgments,
        )
        annotated_videos["motionTracker"]["repCount"] = len(secondary_public_rep_events)
        annotated_videos["motionTracker"]["repCountSource"] = "motionTracker"

    if render_xiaoyu:
        try:
            render_xiaoyu_video()
        except Exception as error:
            annotated_video_warnings.append({
                "engine": "xiaoyuCoach",
                "error": f"{type(error).__name__}: {str(error)[:240]}",
            })

    if render_motion_tracker:
        try:
            render_motion_tracker_video()
        except Exception as error:
            annotated_video_warnings.append({
                "engine": "motionTracker",
                "error": f"{type(error).__name__}: {str(error)[:240]}",
            })
            if render_mode == "selected" and "xiaoyuCoach" not in annotated_videos:
                try:
                    render_xiaoyu_video()
                except Exception as fallback_error:
                    annotated_video_warnings.append({
                        "engine": "xiaoyuCoach",
                        "error": f"{type(fallback_error).__name__}: {str(fallback_error)[:240]}",
                    })

    if pose_engine_comparison is not None:
        pose_engine_comparison["annotatedVideos"] = annotated_videos
    logs.append(calculation_log(
        "annotated_videos",
        "Generate pose overlay videos",
        "Rendered annotated skeleton videos for user review.",
        {
            "annotatedVideos": annotated_videos,
            "warnings": annotated_video_warnings,
            "sampleFps": round(actual_sample_fps, 2),
            "mode": render_mode,
            "selected": selected_annotated_key,
        },
        "warning" if annotated_video_warnings else "done",
    ))
    logs.append(calculation_log(
        "artifacts",
        "鐢熸垚璇佹嵁鍥剧墖",
        f"宸茬敓鎴?{len(keyframes)} 寮犻樁娈靛叧閿抚鍜?1 寮犲洓瀹牸鎷煎浘銆?",
        {
            "keyframes": keyframes,
            "contactSheet": "contact_sheet.jpg",
            "annotatedVideos": annotated_videos,
            "stagePoseIndexes": [int(index) for index in stage_indices[:4]],
            "stageSignals": [
                {
                    "poseIndex": int(max(0, min(len(pose_frames) - 1, int(pose_index)))),
                    "timeMs": pose_frames[int(max(0, min(len(pose_frames) - 1, int(pose_index))))].time_ms,
                    "frameIndex": pose_frames[int(max(0, min(len(pose_frames) - 1, int(pose_index))))].frame_index,
                    "signal": round(float(smoothed[int(max(0, min(len(pose_frames) - 1, int(pose_index))))]), 3),
                }
                for pose_index in stage_indices[:4]
            ],
            "outputDir": output_dir,
        },
    ))

    return {
        "actionType": action_type,
        "actionName": action["name"],
        "bodyPart": action["bodyPart"],
        "family": family,
        "captureQuality": capture_quality,
        "confidence": round(confidence, 3),
        "repCount": output_rep_count,
        "repCountSource": pose_fusion.get("repCountSource"),
        "overallScore": overall_score,
        "safetyLevel": safety_level,
        "issues": issues,
        "strengths": strengths,
        "measurements": measurements,
        "repEvents": output_rep_events,
        "phaseJudgments": phase_judgments,
        "frameJudgments": stability.get("frameJudgments") or [],
        "judgmentSegments": stability.get("judgmentSegments") or [],
        "stabilityProfile": stability.get("profile") or stability_profile(action_type),
        "keyframes": keyframes,
        "contactSheet": "contact_sheet.jpg",
        "annotatedVideos": annotated_videos,
        "cameraAdvice": CAMERA_GUIDANCE[family],
        "metadata": {
            "durationSeconds": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "orientation": "portrait" if height >= width else "landscape",
            "poseCoverage": round(pose_coverage, 3),
            "sampleFps": round(actual_sample_fps, 2),
            "poseBackend": backend.diagnostics.get("poseBackend"),
            "analysisPoseFrames": len(pose_frames),
            "fullPoseFrames": full_pose_frame_count,
        },
        "diagnostics": {
            **backend.diagnostics,
            "selectedFamily": family,
            "detectedFamily": signature.get("detectedFamily"),
            "detectedGroup": signature.get("detectedGroup"),
            "movementSignature": signature,
            "activeTrainingWindow": active_window,
            "repSegmentation": measurements["repSegmentation"],
            "stability": stability.get("summary") or {},
            "targetRoi": target_roi,
            "autoActionDetection": auto_action_detection,
            "poseFusion": pose_fusion,
            "lowConfidenceWindows": low_confidence_windows(pose_frames),
            **({"poseEngineComparison": pose_engine_comparison} if pose_engine_comparison else {}),
        },
        "calculationLogs": logs,
        "analysisVersion": "local-pose-v4",
        "ruleVersion": f"{family}-rules-v4",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: analyze_video.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    try:
        result = analyze_video(payload)
        output_path.write_text(
            json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0
    except Exception as error:
        output_path.write_text(
            json.dumps(
                {"ok": False, "error": {"message": str(error), "type": type(error).__name__}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
