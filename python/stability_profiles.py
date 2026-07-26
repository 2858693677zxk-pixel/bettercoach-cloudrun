"""Per-action trunk stability policy for BetterCoach video analysis.

The policy is intentionally separate from pose measurements.  It decides when
trunk motion is a fault, when it is allowed to couple with the primary action,
and when trunk motion is itself the exercise.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_STABILITY_VIEWS = ("side", "rear")


def _profile(
    mode: str,
    *,
    angle_threshold: float = 4.5,
    tilt_threshold: float = 4.0,
    minimum_duration_ms: int = 320,
    recovery_duration_ms: int = 240,
    max_coupled_angle: float = 0.0,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "supportedViews": list(SUPPORTED_STABILITY_VIEWS),
        "angleThresholdDeg": angle_threshold,
        "tiltThresholdDeg": tilt_threshold,
        "minimumDurationMs": minimum_duration_ms,
        "recoveryDurationMs": recovery_duration_ms,
        "maxCoupledAngleDeg": max_coupled_angle,
    }


STABILITY_PROFILES: dict[str, dict[str, Any]] = {
    # Fixed or pad-supported machines.  Trunk movement is judged against a
    # stable baseline, not against synchronization with the working joint.
    "machine_chest_press": _profile("absolute_fixed", angle_threshold=4.0),
    "standing_hip_abduction": _profile("absolute_fixed", angle_threshold=4.5),
    "seated_hip_abduction": _profile("absolute_fixed", angle_threshold=4.0),
    "chest_supported_row": _profile("absolute_fixed", angle_threshold=4.0),
    "plate_loaded_pulldown": _profile("absolute_fixed", angle_threshold=4.5),
    "hack_squat": _profile("absolute_fixed", angle_threshold=4.5),
    "plate_loaded_rear_leg_raise": _profile("absolute_fixed", angle_threshold=4.5),
    "preacher_curl": _profile("absolute_fixed", angle_threshold=4.0),
    "single_arm_hammer_row": _profile("absolute_fixed", angle_threshold=4.0),
    "leg_extension": _profile("absolute_fixed", angle_threshold=4.0),
    "leg_curl": _profile("absolute_fixed", angle_threshold=4.0),

    # The bench is a support even though the implement is not fixed.
    "bench_press": _profile("absolute_supported", angle_threshold=5.0),
    "dumbbell_press": _profile("absolute_supported", angle_threshold=5.0),
    # Seated cable pulldowns use the seat/thigh pad as support. Arm motion
    # decides phase and range; shoulder-to-hip rotation is judged separately
    # against the starting trunk baseline.
    "lat_pulldown": _profile("absolute_supported", angle_threshold=5.5),

    # Trunk motion may couple with the primary action.  The engine removes a
    # bounded learned coupling but still rejects large excursions and residual
    # sway.  Rear view always judges lateral stability against an absolute
    # baseline because sagittal coupling is hidden there.
    "t_bar_row": _profile("relative_coupled", angle_threshold=6.0, max_coupled_angle=12.0),
    "single_arm_pulldown": _profile("relative_coupled", angle_threshold=5.5, max_coupled_angle=9.0),
    "barbell_squat": _profile("relative_coupled", angle_threshold=6.5, max_coupled_angle=20.0),
    "goblet_squat": _profile("relative_coupled", angle_threshold=6.5, max_coupled_angle=18.0),
    "shoulder_press": _profile("relative_coupled", angle_threshold=5.0, max_coupled_angle=8.0),
    "push_up": _profile("relative_coupled", angle_threshold=5.5, max_coupled_angle=8.0),
    "dip": _profile("relative_coupled", angle_threshold=5.5, max_coupled_angle=10.0),
    "row": _profile("relative_coupled", angle_threshold=6.0, max_coupled_angle=12.0),
    "open_elbow_row": _profile("relative_coupled", angle_threshold=6.0, max_coupled_angle=12.0),
    "pull_up": _profile("relative_coupled", angle_threshold=6.0, max_coupled_angle=10.0),
    "face_pull": _profile("relative_coupled", angle_threshold=5.0, max_coupled_angle=8.0),
    "lateral_raise": _profile("relative_coupled", angle_threshold=4.5, max_coupled_angle=7.0),
    "y_raise": _profile("relative_coupled", angle_threshold=4.5, max_coupled_angle=7.0),
    "fly": _profile("relative_coupled", angle_threshold=5.0, max_coupled_angle=8.0),
    "biceps_curl": _profile("relative_coupled", angle_threshold=4.5, max_coupled_angle=7.0),
    "triceps_extension": _profile("relative_coupled", angle_threshold=4.5, max_coupled_angle=7.0),

    # These actions intentionally move the trunk.  A separate action-specific
    # path rule may score that motion, but the generic stability engine must not
    # call the motion itself instability.
    "machine_crunch": _profile("primary_trunk_motion"),
    "plate_loaded_romanian_deadlift": _profile("primary_trunk_motion"),
    "hip_thrust": _profile("primary_trunk_motion"),
    "back_extension": _profile("primary_trunk_motion"),
    "deadlift": _profile("primary_trunk_motion"),
    "romanian_deadlift": _profile("primary_trunk_motion"),
}


DEFAULT_STABILITY_PROFILE = {
    **_profile("disabled"),
    "reason": "action_profile_not_configured",
}


def stability_profile(action_type: str) -> dict[str, Any]:
    """Return a mutable copy so callers can add runtime thresholds safely."""

    return deepcopy(STABILITY_PROFILES.get(action_type, DEFAULT_STABILITY_PROFILE))
