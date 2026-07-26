import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_video import (
    ACTION_CATALOG,
    LANDMARK,
    PoseFrame,
    TargetTracker,
    action_frame_quality,
    apply_display_landmark_suppression,
    apply_fixed_foot_display_lock,
    apply_segmentation_mask_filter,
    apply_yolo_person_mask_filter,
    build_fixed_foot_support,
    build_movement_phase_judgments,
    build_stability_analysis,
    build_stability_issue,
    calculation_log,
    estimate_pose_frames,
    evaluate_rules,
    evaluate_rep_rules,
    infer_action_type_from_frames,
    metric_confidence,
    normalize_pose_landmark_priors,
    recover_short_display_landmark_gaps,
    motion_signal_series,
    movement_match_profile,
    numeric_summary,
    segment_hinge_repetitions,
    segment_hip_thrust_repetitions,
    segment_lat_pulldown_repetitions,
    segment_repetitions,
    segment_y_raise_repetitions,
    select_active_training_window,
    select_stages_from_rep_events,
    select_target_instance,
    smooth_low_confidence_landmarks,
    transcode_browser_video,
    validate_rep_events,
)


class BrowserVideoTranscodeTests(unittest.TestCase):
    @patch("analyze_video.subprocess.run")
    @patch("analyze_video.shutil.which", return_value="ffmpeg")
    def test_transcode_uses_wechat_compatible_h264_settings(self, _which, run):
        def complete(command, **_kwargs):
            Path(command[-1]).write_bytes(b"h264-video")
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        run.side_effect = complete
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            destination = Path(directory) / "output.mp4"
            source.write_bytes(b"mp4v-video")

            transcode_browser_video(source, destination)

        command = run.call_args.args[0]
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[-1], str(destination))


def landmarks_with_elbow_angle(angle_degrees):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")

        marks[int(shoulder)] = [x, 0.35, 0.0, 0.95]
        marks[int(elbow)] = [x, 0.55, 0.0, 0.95]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [x + 0.16 * math.sin(theta), 0.55 - 0.16 * math.cos(theta), 0.0, 0.95]

        # Lower body stays nearly fixed, so a selected squat should not pass.
        marks[int(hip)] = [x, 0.65, 0.0, 0.95]
        marks[int(knee)] = [x, 0.82, 0.0, 0.95]
        marks[int(ankle)] = [x, 0.98, 0.0, 0.95]

    return marks


def landmarks_with_elbow_angles(left_angle_degrees, right_angle_degrees, right_confidence=0.95):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x, angle_degrees, confidence in (
        ("LEFT", 0.35, left_angle_degrees, 0.95),
        ("RIGHT", 0.65, right_angle_degrees, right_confidence),
    ):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")

        marks[int(shoulder)] = [x, 0.35, 0.0, confidence]
        marks[int(elbow)] = [x, 0.55, 0.0, confidence]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [x + 0.16 * math.sin(theta), 0.55 - 0.16 * math.cos(theta), 0.0, confidence]
        marks[int(hip)] = [x, 0.65, 0.0, 0.95]
        marks[int(knee)] = [x, 0.82, 0.0, 0.95]
        marks[int(ankle)] = [x, 0.98, 0.0, 0.95]

    return marks


def landmarks_with_lat_pulldown_shape(angle_degrees, elbow_y):
    marks = landmarks_with_elbow_angle(angle_degrees)
    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        marks[int(shoulder)] = [x, 0.42, 0.0, 0.95]
        marks[int(elbow)] = [x, elbow_y, 0.0, 0.95]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [x + 0.16 * math.sin(theta), elbow_y - 0.16 * math.cos(theta), 0.0, 0.95]
    return marks


def landmarks_with_single_arm_pulldown_shape(
    working_angle_degrees,
    working_elbow_y,
    working_side="RIGHT",
    inactive_angle_degrees=165,
):
    marks = landmarks_with_elbow_angles(inactive_angle_degrees, inactive_angle_degrees)
    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        angle_degrees = working_angle_degrees if side == working_side else inactive_angle_degrees
        elbow_y = working_elbow_y if side == working_side else 0.50
        confidence = 0.95 if side == working_side else 0.35
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        marks[int(shoulder)] = [x, 0.42, 0.0, confidence]
        marks[int(elbow)] = [x, elbow_y, 0.0, confidence]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [x + 0.16 * math.sin(theta), elbow_y - 0.16 * math.cos(theta), 0.0, confidence]
    return marks


def landmarks_with_hip_abduction(knee_distance):
    marks = landmarks_with_elbow_angle(150)
    for side, direction in (("LEFT", -1), ("RIGHT", 1)):
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        hip_x = 0.46 if side == "LEFT" else 0.54
        knee_x = 0.5 + direction * knee_distance / 2.0
        marks[int(hip)] = [hip_x, 0.64, 0.0, 0.95]
        marks[int(knee)] = [knee_x, 0.80, 0.0, 0.95]
        marks[int(ankle)] = [knee_x, 0.96, 0.0, 0.95]
    return marks


def landmarks_with_single_arm_hammer_row_shape(
    working_angle_degrees,
    working_side="RIGHT",
    inactive_angle_degrees=165,
):
    marks = landmarks_with_elbow_angles(inactive_angle_degrees, inactive_angle_degrees)
    for side, x, direction in (("LEFT", 0.36, -1), ("RIGHT", 0.64, 1)):
        angle_degrees = working_angle_degrees if side == working_side else inactive_angle_degrees
        confidence = 0.95 if side == working_side else 0.30
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        hip = getattr(LANDMARK, f"{side}_HIP")

        marks[int(shoulder)] = [x, 0.42, 0.0, confidence]
        marks[int(hip)] = [x, 0.66, 0.0, 0.95]
        elbow_shift = max(0.0, 180 - angle_degrees) / 120.0 * 0.06
        marks[int(elbow)] = [x + direction * elbow_shift, 0.55, 0.0, confidence]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [
            marks[int(elbow)][0] + direction * 0.16 * math.sin(theta),
            0.55 - 0.16 * math.cos(theta),
            0.0,
            confidence,
        ]
    return marks


def landmarks_with_plate_loaded_rear_leg_raise_shape(
    working_hip_angle_degrees,
    working_side="RIGHT",
    inactive_hip_angle_degrees=118,
):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]
    for side, x, direction in (("LEFT", 0.38, -1), ("RIGHT", 0.62, 1)):
        angle_degrees = working_hip_angle_degrees if side == working_side else inactive_hip_angle_degrees
        confidence = 0.95 if side == working_side else 0.35
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")

        marks[int(shoulder)] = [x, 0.36, 0.0, confidence]
        marks[int(hip)] = [x, 0.62, 0.0, confidence]
        theta = math.radians(angle_degrees)
        marks[int(knee)] = [x + direction * 0.22 * math.sin(theta), 0.62 - 0.22 * math.cos(theta), 0.0, confidence]
        marks[int(ankle)] = [marks[int(knee)][0] + direction * 0.04, marks[int(knee)][1] + 0.18, 0.0, confidence]
        marks[int(elbow)] = [x, 0.47, 0.0, 0.95]
        marks[int(wrist)] = [x, 0.57, 0.0, 0.95]
    return marks


def landmarks_with_row_shape(angle_degrees, elbow_shift):
    marks = landmarks_with_elbow_angle(angle_degrees)
    for side, base_x, direction in (("LEFT", 0.35, -1), ("RIGHT", 0.65, 1)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        x = base_x + direction * elbow_shift
        marks[int(shoulder)] = [base_x, 0.36, 0.0, 0.95]
        marks[int(elbow)] = [x, 0.55, 0.0, 0.95]
        theta = math.radians(angle_degrees)
        marks[int(wrist)] = [x + direction * 0.16 * math.sin(theta), 0.55 - 0.16 * math.cos(theta), 0.0, 0.95]
    return marks


def landmarks_with_y_raise_shape(
    shoulder_angle_degrees,
    elbow_angle_degrees=170,
    working_side=None,
    inactive_shoulder_angle_degrees=35,
):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x, direction in (("LEFT", 0.38, -1), ("RIGHT", 0.62, 1)):
        side_angle = shoulder_angle_degrees
        if working_side and side.lower() != str(working_side).lower():
            side_angle = inactive_shoulder_angle_degrees
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")

        marks[int(shoulder)] = [x, 0.42, 0.0, 0.95]
        marks[int(hip)] = [x, 0.70, 0.0, 0.95]
        marks[int(knee)] = [x, 0.86, 0.0, 0.95]
        marks[int(ankle)] = [x, 0.98, 0.0, 0.95]

        shoulder_theta = math.radians(side_angle)
        upper_len = 0.18
        elbow_x = x + direction * upper_len * math.sin(shoulder_theta)
        elbow_y = 0.42 + upper_len * math.cos(shoulder_theta)
        marks[int(elbow)] = [elbow_x, elbow_y, 0.0, 0.95]

        elbow_theta = shoulder_theta + math.radians(180 - elbow_angle_degrees)
        forearm_len = 0.16
        marks[int(wrist)] = [
            elbow_x + direction * forearm_len * math.sin(elbow_theta),
            elbow_y + forearm_len * math.cos(elbow_theta),
            0.0,
            0.95,
        ]

    return marks


def coco_candidate(kind, confidence):
    points = np.zeros((17, 2), dtype=float)
    scores = np.full((17,), confidence, dtype=float)
    if kind == "lying":
        coords = {
            5: (62, 55), 6: (61, 57), 7: (70, 44), 8: (69, 46),
            9: (72, 30), 10: (72, 32), 11: (32, 61), 12: (31, 63),
            13: (20, 78), 14: (18, 79), 15: (10, 92), 16: (8, 93),
        }
    else:
        coords = {
            5: (25, 18), 6: (30, 18), 7: (24, 36), 8: (31, 36),
            9: (23, 54), 10: (32, 54), 11: (26, 55), 12: (30, 55),
            13: (25, 76), 14: (31, 76), 15: (25, 96), 16: (31, 96),
        }
    for index, value in coords.items():
        points[index] = value
    return points, scores


def landmarks_with_knee_angle(angle_degrees, confidence=0.95):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")

        marks[int(shoulder)] = [x, 0.35, 0.0, 0.95]
        marks[int(hip)] = [x, 0.58, 0.0, 0.95]
        marks[int(knee)] = [x, 0.76, 0.0, 0.95]
        theta = math.radians(angle_degrees)
        marks[int(ankle)] = [x + 0.17 * math.sin(theta), 0.76 - 0.17 * math.cos(theta), 0.0, confidence]
        marks[int(elbow)] = [x, 0.48, 0.0, 0.95]
        marks[int(wrist)] = [x, 0.62, 0.0, 0.95]

    return marks


def landmarks_with_hip_angle(angle_degrees):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")

        marks[int(shoulder)] = [x, 0.35, 0.0, 0.95]
        marks[int(hip)] = [x, 0.62, 0.0, 0.95]
        theta = math.radians(angle_degrees)
        marks[int(knee)] = [x + 0.22 * math.sin(theta), 0.62 - 0.22 * math.cos(theta), 0.0, 0.95]
        marks[int(ankle)] = [marks[int(knee)][0], min(0.98, marks[int(knee)][1] + 0.18), 0.0, 0.95]
        marks[int(elbow)] = [x, 0.48, 0.0, 0.95]
        marks[int(wrist)] = [x, 0.62, 0.0, 0.95]

    return marks


def landmarks_with_hip_and_knee_angles(hip_angle_degrees, knee_angle_degrees):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]

    for side, x in (("LEFT", 0.35), ("RIGHT", 0.65)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")

        marks[int(shoulder)] = [x, 0.35, 0.0, 0.95]
        marks[int(hip)] = [x, 0.62, 0.0, 0.95]
        hip_theta = math.radians(hip_angle_degrees)
        marks[int(knee)] = [x + 0.22 * math.sin(hip_theta), 0.62 - 0.22 * math.cos(hip_theta), 0.0, 0.95]
        knee_theta = math.radians(knee_angle_degrees)
        marks[int(ankle)] = [
            marks[int(knee)][0] + 0.18 * math.sin(knee_theta),
            marks[int(knee)][1] - 0.18 * math.cos(knee_theta),
            0.0,
            0.95,
        ]
        marks[int(elbow)] = [x, 0.48, 0.0, 0.95]
        marks[int(wrist)] = [x, 0.62, 0.0, 0.95]

    return marks


def landmarks_with_trunk_lean(lean_degrees):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]
    theta = math.radians(lean_degrees)
    hip_mid = np.array([0.5, 0.62])
    shoulder_mid = hip_mid + np.array([0.24 * math.sin(theta), -0.24 * math.cos(theta)])

    for side, offset in (("LEFT", -0.06), ("RIGHT", 0.06)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        hip = getattr(LANDMARK, f"{side}_HIP")
        knee = getattr(LANDMARK, f"{side}_KNEE")
        ankle = getattr(LANDMARK, f"{side}_ANKLE")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")

        marks[int(shoulder)] = [float(shoulder_mid[0] + offset), float(shoulder_mid[1]), 0.0, 0.95]
        marks[int(hip)] = [float(hip_mid[0] + offset), float(hip_mid[1]), 0.0, 0.95]
        marks[int(knee)] = [float(hip_mid[0] + offset * 0.8), 0.80, 0.0, 0.95]
        marks[int(ankle)] = [float(hip_mid[0] + offset * 0.8), 0.96, 0.0, 0.95]
        marks[int(elbow)] = [float(shoulder_mid[0] + offset + 0.02), float(shoulder_mid[1] + 0.14), 0.0, 0.95]
        marks[int(wrist)] = [float(shoulder_mid[0] + offset + 0.03), float(shoulder_mid[1] + 0.28), 0.0, 0.95]

    return marks


def camera_rotated_landmarks(landmarks, rotation_degrees, canvas_size=1000):
    theta = math.radians(rotation_degrees)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    center = canvas_size / 2.0
    forward = np.asarray([
        [cosine, -sine, center - cosine * center + sine * center],
        [sine, cosine, center - sine * center - cosine * center],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inverse = np.linalg.inv(forward)[:2]
    rotated = []
    for item in landmarks:
        values = list(item)
        pixel = np.asarray([values[0] * canvas_size, values[1] * canvas_size, 1.0], dtype=float)
        transformed = forward @ pixel
        values[0] = float(transformed[0] / canvas_size)
        values[1] = float(transformed[1] / canvas_size)
        rotated.append(values)
    return rotated, inverse.tolist()


def set_side_confidence(marks, side, confidence):
    for name in ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE"):
        index = getattr(LANDMARK, f"{side}_{name}")
        marks[int(index)][3] = confidence
    return marks


def landmarks_with_back_extension_and_arm_motion(lean_degrees, elbow_angle_degrees):
    marks = landmarks_with_trunk_lean(lean_degrees)
    for side, direction in (("LEFT", -1), ("RIGHT", 1)):
        shoulder = getattr(LANDMARK, f"{side}_SHOULDER")
        elbow = getattr(LANDMARK, f"{side}_ELBOW")
        wrist = getattr(LANDMARK, f"{side}_WRIST")
        shoulder_x, shoulder_y = marks[int(shoulder)][0], marks[int(shoulder)][1]
        marks[int(elbow)] = [shoulder_x + direction * 0.05, shoulder_y + 0.12, 0.0, 0.95]
        theta = math.radians(elbow_angle_degrees)
        marks[int(wrist)] = [
            marks[int(elbow)][0] + direction * 0.14 * math.sin(theta),
            marks[int(elbow)][1] - 0.14 * math.cos(theta),
            0.0,
            0.95,
        ]
    return marks


def pose_frame(index, landmarks, signal=0.0):
    return PoseFrame(
        frame_index=index,
        time_ms=index * 120,
        landmarks=landmarks,
        signal=signal,
        quality=0.95,
    )


class NewEquipmentActionTest(unittest.TestCase):
    def test_new_equipment_actions_are_registered(self):
        expected = {
            "machine_chest_press",
            "machine_crunch",
            "standing_hip_abduction",
            "seated_hip_abduction",
            "chest_supported_row",
            "t_bar_row",
            "plate_loaded_pulldown",
            "plate_loaded_romanian_deadlift",
        }

        self.assertTrue(expected.issubset(ACTION_CATALOG))

    def test_seated_hip_abduction_signal_tracks_knee_separation(self):
        frames = [
            pose_frame(index, landmarks_with_hip_abduction(distance))
            for index, distance in enumerate([0.10, 0.22, 0.38, 0.22, 0.10])
        ]

        signal, source = motion_signal_series(
            frames, "isolation_hip", "seated_hip_abduction", "front_oblique"
        )

        self.assertEqual(source, "seated_hip_abduction_normalized_knee_separation")
        self.assertGreater(float(signal[2] - signal[0]), 50.0)

    def test_machine_crunch_signal_tracks_trunk_flexion(self):
        frames = [
            pose_frame(index, landmarks_with_trunk_lean(lean))
            for index, lean in enumerate([8, 24, 48, 24, 8])
        ]

        signal, source = motion_signal_series(frames, "core_flexion", "machine_crunch", "side")

        self.assertEqual(source, "machine_crunch_trunk_flexion_angle")
        self.assertGreater(float(signal[2] - signal[0]), 30.0)

    def test_plate_loaded_rdl_uses_hip_flexion_signal(self):
        frames = [
            pose_frame(index, landmarks_with_trunk_lean(lean))
            for index, lean in enumerate([8, 28, 52, 28, 8])
        ]

        signal, source = motion_signal_series(
            frames, "hinge", "plate_loaded_romanian_deadlift", "side"
        )

        self.assertEqual(source, "plate_loaded_rdl_trunk_hinge_angle")
        self.assertGreater(float(signal[2] - signal[0]), 20.0)


class MovementMatchProfileTest(unittest.TestCase):
    def test_flags_upper_body_video_when_squat_was_selected(self):
        frames = []
        for index in range(36):
            elbow_angle = 65 + 95 * abs(math.sin(index / 35 * math.pi * 3))
            frames.append(PoseFrame(
                frame_index=index,
                time_ms=index * 120,
                landmarks=landmarks_with_elbow_angle(elbow_angle),
                signal=0.0,
                quality=0.95,
            ))

        profile = movement_match_profile(frames, "squat", 8.0)

        self.assertTrue(profile["mismatch"])
        self.assertEqual(profile["expectedGroup"], "lower_body")
        self.assertEqual(profile["detectedGroup"], "upper_body")
        self.assertGreater(profile["upperBodyRange"], profile["lowerBodyRange"])

    def test_flags_hinge_video_when_lat_pulldown_was_selected(self):
        frames = []
        for index in range(36):
            hip_angle = 65 + 95 * abs(math.sin(index / 35 * math.pi * 3))
            frames.append(pose_frame(index, landmarks_with_hip_angle(hip_angle)))

        profile = movement_match_profile(frames, "pull", 8.0)

        self.assertTrue(profile["mismatch"])
        self.assertEqual(profile["expectedGroup"], "upper_body")
        self.assertEqual(profile["detectedGroup"], "lower_body")
        self.assertIn(profile["detectedFamily"], {"hinge", "squat", "isolation_knee"})

    def test_back_extension_folded_arms_do_not_create_action_mismatch(self):
        frames = [
            pose_frame(index, landmarks_with_back_extension_and_arm_motion(lean, elbow))
            for index, (lean, elbow) in enumerate([
                (18, 160), (32, 120), (48, 70), (62, 35), (50, 85), (34, 130), (20, 165),
                (19, 158), (36, 112), (52, 62), (64, 32), (47, 88), (30, 135), (18, 168),
            ])
        ]

        profile = movement_match_profile(frames, "hinge", 8.0)

        self.assertFalse(profile["mismatch"])
        self.assertEqual(profile["expectedFamily"], "hinge")

    def test_auto_action_detection_separates_lat_pulldown_from_row(self):
        lat_frames = [
            pose_frame(index, landmarks_with_lat_pulldown_shape(angle, elbow_y))
            for index, (angle, elbow_y) in enumerate([
                (160, 0.29), (135, 0.34), (95, 0.46), (80, 0.58),
                (112, 0.48), (145, 0.34), (160, 0.29),
            ] * 3)
        ]
        row_frames = [
            pose_frame(index, landmarks_with_row_shape(angle, shift))
            for index, (angle, shift) in enumerate([
                (155, 0.00), (130, 0.02), (108, 0.05), (100, 0.07),
                (118, 0.05), (140, 0.02), (155, 0.00),
            ] * 3)
        ]

        lat = infer_action_type_from_frames(lat_frames, 8.0)
        row = infer_action_type_from_frames(row_frames, 8.0)

        self.assertEqual(lat["actionType"], "lat_pulldown")
        self.assertEqual(row["actionType"], "row")

    def test_auto_action_detection_identifies_high_y_raise(self):
        frames = [
            pose_frame(index, landmarks_with_y_raise_shape(angle, working_side="LEFT"))
            for index, angle in enumerate([
                35, 48, 78, 112, 132, 118, 82, 50, 35,
                38, 62, 95, 125, 136, 104, 70, 42, 35,
            ])
        ]

        result = infer_action_type_from_frames(frames, 8.0)

        self.assertEqual(result["actionType"], "y_raise")
        self.assertEqual(result["family"], "isolation_shoulder")


class RuleEvaluationTest(unittest.TestCase):
    def test_y_raise_marks_non_front_camera_as_limited_evidence(self):
        issues, _ = evaluate_rules(
            "y_raise",
            "isolation_shoulder",
            {
                "metricConfidence": {},
                "movementMatch": {"mismatch": False},
                "averageRepSeconds": None,
                "shoulderAngleMax": 132,
                "yRaiseWorkingSideConfidence": 0.95,
                "yRaiseWorkingShoulderAngleMax": 132,
                "trunkLeanRange": 4,
                "torsoSwayRatio": 0.05,
            },
            "good",
            "rear",
        )

        self.assertTrue(any(item["code"] == "Y_RAISE_CAMERA_LIMITED" for item in issues))
        self.assertFalse(any(item["code"] == "INSUFFICIENT_EVIDENCE" for item in issues))

    def test_hack_squat_does_not_fail_on_machine_occluded_foot_markers(self):
        issues, strengths = evaluate_rules(
            "hack_squat",
            "squat",
            {
                "metricConfidence": {"kneeAngle": 0.9, "trunkLean": 0.9, "ankleSupport": 0.9},
                "movementMatch": {"mismatch": False},
                "averageRepSeconds": None,
                "leftKneeAngleMin": 124,
                "rightKneeAngleMin": 130,
                "motionRange": 42,
                "hipDepthMax": 0.01,
                "trunkLeanRange": 8,
                "trunkLeanMax": 42,
                "torsoSupportSwayRatio": 0.1,
                "footMovementRatio": 0.7,
                "kneeAngleAsymmetry": 2,
            },
            "good",
            "side",
        )

        self.assertFalse(any(item["code"] == "FOOT_PRESSURE_UNSTABLE" for item in issues))
        self.assertTrue(any("foot" in item.lower() for item in strengths))


class StabilityEngineTest(unittest.TestCase):
    def test_fixed_machine_uses_absolute_baseline_and_sustained_duration(self):
        angles = [0] * 8 + [9] * 6 + [0] * 6
        frames = [pose_frame(index, landmarks_with_trunk_lean(angle)) for index, angle in enumerate(angles)]

        result = build_stability_analysis(
            frames,
            "machine_chest_press",
            "side",
            1000 / 120,
            movement_values=np.zeros(len(frames)),
        )
        problem = build_stability_issue(result)

        self.assertEqual(result["mode"], "absolute_fixed")
        self.assertTrue(result["summary"]["evaluated"])
        self.assertTrue(any(item["state"] == "unstable" for item in result["frameJudgments"]))
        self.assertEqual(problem["code"], "TRUNK_ABSOLUTE_INSTABILITY")
        self.assertTrue(problem["timeRangesMs"])

    def test_single_pose_spike_does_not_become_instability_issue(self):
        angles = [0] * 8 + [11] + [0] * 8
        frames = [pose_frame(index, landmarks_with_trunk_lean(angle)) for index, angle in enumerate(angles)]

        result = build_stability_analysis(
            frames,
            "machine_chest_press",
            "side",
            1000 / 120,
            movement_values=np.zeros(len(frames)),
        )

        self.assertFalse(any(item["state"] == "unstable" for item in result["frameJudgments"]))
        self.assertIsNone(build_stability_issue(result))

    def test_primary_trunk_action_is_explicitly_not_scored(self):
        frames = [
            pose_frame(index, landmarks_with_trunk_lean(angle))
            for index, angle in enumerate([0, 8, 18, 28, 36, 28, 18, 8, 0])
        ]

        result = build_stability_analysis(
            frames,
            "machine_crunch",
            "side",
            1000 / 120,
            movement_values=np.arange(len(frames)),
        )

        self.assertEqual(result["mode"], "primary_trunk_motion")
        self.assertFalse(result["summary"]["evaluated"])
        self.assertTrue(all(item["state"] == "not_evaluated" for item in result["frameJudgments"]))
        self.assertIsNone(build_stability_issue(result))

    def test_oblique_view_is_not_used_for_strong_stability_conclusion(self):
        frames = [pose_frame(index, landmarks_with_trunk_lean(12)) for index in range(10)]

        result = build_stability_analysis(
            frames,
            "machine_chest_press",
            "side_front",
            1000 / 120,
            movement_values=np.zeros(len(frames)),
        )

        self.assertFalse(result["summary"]["evaluated"])
        self.assertEqual(result["summary"]["reason"], "STABILITY_VIEW_UNSUPPORTED")
        self.assertTrue(all(item["state"] == "unknown" for item in result["frameJudgments"]))

    def test_relative_mode_allows_bounded_trunk_coupling(self):
        movement = np.asarray([0, 1, 2, 3, 4, 3, 2, 1, 0] * 2, dtype=float)
        angles = movement * 3.0
        frames = [pose_frame(index, landmarks_with_trunk_lean(angle)) for index, angle in enumerate(angles)]

        result = build_stability_analysis(
            frames,
            "barbell_squat",
            "side",
            1000 / 120,
            movement_values=movement,
        )

        self.assertEqual(result["mode"], "relative_coupled")
        self.assertFalse(any(item["state"] == "unstable" for item in result["frameJudgments"]))
        self.assertIsNone(build_stability_issue(result))

    def test_camera_rotation_is_removed_before_absolute_stability_scoring(self):
        camera_angles = [0] * 8 + [9] * 6 + [0] * 6
        frames = []
        motion_frames = []
        for index, camera_angle in enumerate(camera_angles):
            landmarks, inverse = camera_rotated_landmarks(landmarks_with_trunk_lean(0), camera_angle)
            frames.append(pose_frame(index, landmarks))
            motion_frames.append({
                "frameIndex": index,
                "timeMs": index * 120,
                "available": True,
                "confidence": 0.9,
                "canvasWidth": 1000,
                "canvasHeight": 1000,
                "inverseAffine": inverse,
                "cumulativeRotationDeg": camera_angle,
                "cumulativeTranslationX": 0.0,
                "cumulativeTranslationY": 0.0,
            })

        raw_result = build_stability_analysis(
            frames,
            "machine_chest_press",
            "side",
            1000 / 120,
            movement_values=np.zeros(len(frames)),
        )
        compensated_result = build_stability_analysis(
            frames,
            "machine_chest_press",
            "side",
            1000 / 120,
            movement_values=np.zeros(len(frames)),
            camera_motion={
                "frames": motion_frames,
                "summary": {"method": "test_affine", "status": "ready", "coverage": 1.0},
            },
        )

        self.assertTrue(any(item["state"] == "unstable" for item in raw_result["frameJudgments"]))
        self.assertFalse(any(item["state"] == "unstable" for item in compensated_result["frameJudgments"]))
        self.assertTrue(compensated_result["summary"]["cameraCompensation"]["applied"])
        self.assertTrue(all(item["features"]["cameraCompensated"] for item in compensated_result["frameJudgments"]))


class CalculationLogTest(unittest.TestCase):
    def test_numeric_summary_and_log_are_json_safe(self):
        summary = numeric_summary(np.array([1.0, 2.0, np.nan, 4.0]))
        entry = calculation_log("signal", "娴嬭瘯鏃ュ織", "鐢ㄤ簬楠岃瘉鏃ュ織鍙簭鍒楀寲", {"summary": summary})

        self.assertEqual(summary["count"], 3)
        self.assertEqual(entry["stage"], "signal")
        self.assertEqual(entry["details"]["summary"]["max"], 4.0)


class PoseQualityRepairTest(unittest.TestCase):
    def test_single_arm_pulldown_quality_ignores_occluded_legs(self):
        marks = landmarks_with_single_arm_pulldown_shape(104, 0.56, working_side="RIGHT")
        for side in ("LEFT", "RIGHT"):
            for name in ("HIP", "KNEE", "ANKLE"):
                marks[int(getattr(LANDMARK, f"{side}_{name}"))][3] = 0.05

        quality = action_frame_quality(marks, "pull", "single_arm_pulldown")

        self.assertGreater(quality, 0.75)

    def test_hack_squat_quality_ignores_occluded_ankles(self):
        marks = landmarks_with_knee_angle(96)
        for side in ("LEFT", "RIGHT"):
            marks[int(getattr(LANDMARK, f"{side}_ANKLE"))][3] = 0.05

        quality = action_frame_quality(marks, "squat", "hack_squat")

        self.assertGreater(quality, 0.75)

    def test_fixed_foot_display_locks_ankle_and_hides_foot_detail(self):
        frames = []
        for index in range(8):
            marks = landmarks_with_knee_angle(100)
            for side, x in (("LEFT", 0.36), ("RIGHT", 0.64)):
                ankle = int(getattr(LANDMARK, f"{side}_ANKLE"))
                heel = int(getattr(LANDMARK, f"{side}_HEEL"))
                toe = int(getattr(LANDMARK, f"{side}_FOOT_INDEX"))
                marks[ankle] = [x + 0.006 * ((index % 3) - 1), 0.88, 0.0, 0.92]
                marks[heel] = [x - 0.24, 0.92, 0.0, 0.95]
                marks[toe] = [x + 0.24, 0.92, 0.0, 0.95]
            frames.append(pose_frame(index, marks))

        support = build_fixed_foot_support(frames, "hack_squat")
        display = apply_fixed_foot_display_lock(frames, "hack_squat", support)
        left_ankle = int(LANDMARK.LEFT_ANKLE)
        left_heel = int(LANDMARK.LEFT_HEEL)
        left_toe = int(LANDMARK.LEFT_FOOT_INDEX)

        self.assertEqual(support["mode"], "lock_ankle_hide_foot_detail")
        self.assertAlmostEqual(display[0].landmarks[left_ankle][0], display[-1].landmarks[left_ankle][0])
        self.assertEqual(display[0].landmarks[left_heel][3], 0.0)
        self.assertEqual(display[0].landmarks[left_toe][3], 0.0)

    def test_hip_thrust_display_hides_knees_and_distal_hands(self):
        marks = landmarks_with_hip_angle(165)
        for name in (
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_INDEX",
            "RIGHT_INDEX",
        ):
            marks[int(getattr(LANDMARK, name))][3] = 0.95
        frame = pose_frame(0, marks)

        display = apply_fixed_foot_display_lock(
            [frame],
            "hip_thrust",
            {"enabled": True, "mode": "hide_unreliable_foot", "anchors": {}},
        )[0].landmarks

        self.assertLess(display[int(LANDMARK.LEFT_KNEE)][3], 0.1)
        self.assertLess(display[int(LANDMARK.RIGHT_KNEE)][3], 0.1)
        self.assertLess(display[int(LANDMARK.LEFT_ANKLE)][3], 0.1)
        self.assertLess(display[int(LANDMARK.RIGHT_WRIST)][3], 0.1)
        self.assertGreater(display[int(LANDMARK.LEFT_HIP)][3], 0.9)
        self.assertGreater(display[int(LANDMARK.RIGHT_SHOULDER)][3], 0.9)

    def test_segmentation_mask_filter_rejects_lower_limb_points_outside_person_mask(self):
        marks = landmarks_with_hip_angle(165)
        right_hip = int(LANDMARK.RIGHT_HIP)
        right_knee = int(LANDMARK.RIGHT_KNEE)
        right_ankle = int(LANDMARK.RIGHT_ANKLE)
        mask = np.zeros((100, 100), dtype=np.float32)
        hip_x = int(round(marks[right_hip][0] * 99))
        hip_y = int(round(marks[right_hip][1] * 99))
        mask[max(0, hip_y - 3): hip_y + 4, max(0, hip_x - 3): hip_x + 4] = 1.0

        mask_scores, diagnostics = apply_segmentation_mask_filter(
            marks,
            mask,
            action_type="hip_thrust",
            family="hinge",
            threshold=0.35,
        )

        self.assertIsNotNone(mask_scores)
        self.assertGreaterEqual(diagnostics["filtered"], 2)
        self.assertLess(marks[right_knee][3], 0.1)
        self.assertLess(marks[right_ankle][3], 0.1)
        self.assertGreater(marks[right_hip][3], 0.9)

    def test_yolo_person_mask_filter_rejects_lower_limb_points_outside_person_mask(self):
        previous = os.environ.get("YOLO_PERSON_SEGMENTATION_FILTER")
        os.environ["YOLO_PERSON_SEGMENTATION_FILTER"] = "1"
        try:
            marks = landmarks_with_hip_angle(165)
            right_hip = int(LANDMARK.RIGHT_HIP)
            right_knee = int(LANDMARK.RIGHT_KNEE)
            right_ankle = int(LANDMARK.RIGHT_ANKLE)
            mask = np.zeros((120, 120), dtype=np.float32)
            hip_x = int(round(marks[right_hip][0] * 119))
            hip_y = int(round(marks[right_hip][1] * 119))
            mask[max(0, hip_y - 4): hip_y + 5, max(0, hip_x - 4): hip_x + 5] = 1.0

            mask_scores, diagnostics = apply_yolo_person_mask_filter(
                marks,
                mask,
                action_type="romanian_deadlift",
                family="hinge",
                threshold=0.55,
            )

            self.assertIsNotNone(mask_scores)
            self.assertGreaterEqual(diagnostics["filtered"], 2)
            self.assertLess(marks[right_knee][3], 0.1)
            self.assertLess(marks[right_ankle][3], 0.1)
            self.assertGreater(marks[right_hip][3], 0.9)
        finally:
            if previous is None:
                os.environ.pop("YOLO_PERSON_SEGMENTATION_FILTER", None)
            else:
                os.environ["YOLO_PERSON_SEGMENTATION_FILTER"] = previous

    def test_hinge_display_hides_foot_detail_without_hiding_ankle(self):
        marks = landmarks_with_hip_angle(165)
        for name in ("LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
            marks[int(getattr(LANDMARK, name))][3] = 0.95
        marks[int(LANDMARK.LEFT_ANKLE)][3] = 0.95
        marks[int(LANDMARK.LEFT_HIP)][3] = 0.95

        display = apply_display_landmark_suppression(
            [pose_frame(0, marks)],
            "romanian_deadlift",
            "hinge",
        )[0].landmarks

        self.assertEqual(display[int(LANDMARK.LEFT_HEEL)][3], 0.0)
        self.assertEqual(display[int(LANDMARK.RIGHT_FOOT_INDEX)][3], 0.0)
        self.assertGreater(display[int(LANDMARK.LEFT_ANKLE)][3], 0.9)
        self.assertGreater(display[int(LANDMARK.LEFT_HIP)][3], 0.9)

    def test_glm_landmark_prior_repositions_low_confidence_point(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angle(150)),
            pose_frame(1, landmarks_with_elbow_angle(120)),
            pose_frame(2, landmarks_with_elbow_angle(90)),
        ]
        frames[1].landmarks[int(LANDMARK.LEFT_WRIST)] = [0.88, 0.88, 0.0, 0.1]
        priors = normalize_pose_landmark_priors([{
            "source": "glm",
            "frameIndex": 1,
            "landmarks": {
                "left_wrist": [0.42, 0.44, 0.0, 0.93],
            },
        }])

        repaired = smooth_low_confidence_landmarks(frames, 0.55, "pull", "lat_pulldown", priors)

        self.assertAlmostEqual(repaired[1].landmarks[int(LANDMARK.LEFT_WRIST)][0], 0.42)
        self.assertAlmostEqual(repaired[1].landmarks[int(LANDMARK.LEFT_WRIST)][1], 0.44)
        self.assertGreaterEqual(repaired[1].landmarks[int(LANDMARK.LEFT_WRIST)][3], 0.93)

class RepSegmentationTest(unittest.TestCase):
    def test_lat_pulldown_signal_uses_elbow_flexion_angle(self):
        angles = [165, 120, 70, 120, 165]
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(angle))
            for index, angle in enumerate(angles)
        ]

        signal, source = motion_signal_series(frames, "pull", "lat_pulldown", "front_oblique")

        self.assertEqual(source, "lat_pulldown_elbow_flexion_angle")
        np.testing.assert_allclose(signal, [180 - value for value in angles], atol=0.01)

    def test_y_raise_signal_uses_shoulder_elevation_angle(self):
        angles = [35, 75, 125, 75, 35]
        frames = [
            pose_frame(index, landmarks_with_y_raise_shape(angle, working_side="LEFT"))
            for index, angle in enumerate(angles)
        ]

        signal, source = motion_signal_series(frames, "isolation_shoulder", "y_raise", "front")

        self.assertEqual(source, "y_raise_working_side_shoulder_angle")
        np.testing.assert_allclose(signal, angles, atol=0.01)

    def test_lat_pulldown_counts_gt_135_to_lt_90_elbow_cycles(self):
        angles = [
            160, 158, 150, 132, 112, 88, 82, 104, 128, 146, 158,
            160, 150, 130, 108, 86, 80, 101, 126, 145, 160,
        ]
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(angle))
            for index, angle in enumerate(angles)
        ]
        signal, _ = motion_signal_series(frames, "pull", "lat_pulldown", "front_oblique")

        events = segment_lat_pulldown_repetitions(frames, signal, 8.0)
        valid, diagnostics = validate_rep_events(events, signal, 8.0, "lat_pulldown")

        self.assertEqual(len(valid), 2)
        self.assertTrue(all(item.get("counterRule") == "elbow_angle_gt_135_to_lt_90" for item in valid))
        self.assertEqual(diagnostics["validRepCount"], 2)

    def test_lat_pulldown_partial_rep_still_switches_to_eccentric(self):
        angles = [160, 154, 142, 128, 114, 104, 108, 120, 138, 154, 162]
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(angle))
            for index, angle in enumerate(angles)
        ]
        signal, _ = motion_signal_series(frames, "pull", "lat_pulldown", "side")

        events = segment_lat_pulldown_repetitions(frames, signal, 8.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["rangeStatus"], "insufficient")
        self.assertEqual(events[0]["phaseRule"], "elbow_angle_direction_reversal")
        timeline = build_movement_phase_judgments(frames, events, 8.0)
        after_turn = next(item for item in timeline if item["timeMs"] > events[0]["keyTimeMs"])
        self.assertEqual(after_turn["phase"], "return")
        self.assertEqual(after_turn["rangeStatus"], "insufficient")

    def test_single_arm_pulldown_counts_video_calibrated_bottom_position(self):
        frames = [
            pose_frame(index, landmarks_with_single_arm_pulldown_shape(angle, elbow_y))
            for index, (angle, elbow_y) in enumerate([
                (160, 0.30), (150, 0.34), (130, 0.42), (112, 0.50),
                (104, 0.56), (118, 0.48), (140, 0.38), (160, 0.30),
            ])
        ]
        signal, source = motion_signal_series(frames, "pull", "single_arm_pulldown", "side_front")

        events = segment_lat_pulldown_repetitions(
            frames,
            signal,
            8.0,
            flexed_elbow_angle=110.0,
            counter_rule="single_arm_elbow_angle_gt_135_to_lt_110",
        )
        valid, diagnostics = validate_rep_events(events, signal, 8.0, "single_arm_pulldown")

        self.assertEqual(source, "single_arm_pulldown_working_side_elbow_flexion_angle")
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["counterRule"], "single_arm_elbow_angle_gt_135_to_lt_110")
        self.assertEqual(diagnostics["validRepCount"], 1)

    def test_single_arm_hammer_row_signal_uses_working_side_elbow_flexion(self):
        frames = [
            pose_frame(index, landmarks_with_single_arm_hammer_row_shape(angle, working_side="RIGHT"))
            for index, angle in enumerate([160, 132, 98, 124, 160])
        ]

        signal, source = motion_signal_series(frames, "pull", "single_arm_hammer_row", "side_front")

        self.assertEqual(source, "single_arm_hammer_row_working_side_elbow_flexion_angle")
        self.assertGreater(float(np.max(signal) - np.min(signal)), 45)

    def test_plate_loaded_rear_leg_raise_signal_uses_working_hip_extension(self):
        frames = [
            pose_frame(index, landmarks_with_plate_loaded_rear_leg_raise_shape(angle, working_side="RIGHT"))
            for index, angle in enumerate([112, 126, 150, 128, 112])
        ]

        signal, source = motion_signal_series(frames, "hinge", "plate_loaded_rear_leg_raise", "side")

        self.assertEqual(source, "plate_loaded_rear_leg_raise_working_hip_extension_angle")
        self.assertGreater(float(np.max(signal) - np.min(signal)), 35)

    def test_preacher_curl_signal_uses_elbow_flexion_angle(self):
        angles = [150, 112, 72, 110, 150]
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(angle))
            for index, angle in enumerate(angles)
        ]

        signal, source = motion_signal_series(frames, "isolation_elbow", "preacher_curl", "side")

        self.assertEqual(source, "preacher_curl_elbow_flexion_angle")
        np.testing.assert_allclose(signal, [180 - value for value in angles], atol=0.01)

    def test_y_raise_counts_low_to_high_y_cycles(self):
        angles = [
            35, 42, 68, 96, 122, 132, 118, 82, 55, 36,
            38, 48, 74, 104, 124, 136, 116, 78, 52, 35,
        ]
        frames = [
            pose_frame(index, landmarks_with_y_raise_shape(angle, working_side="LEFT"))
            for index, angle in enumerate(angles)
        ]
        signal, _ = motion_signal_series(frames, "isolation_shoulder", "y_raise", "front")

        events = segment_y_raise_repetitions(frames, signal, 8.0)
        valid, diagnostics = validate_rep_events(events, signal, 8.0, "y_raise")

        self.assertEqual(len(valid), 2)
        self.assertTrue(all(item.get("counterRule") == "shoulder_angle_lt_60_to_gt_110_y_top" for item in valid))
        self.assertEqual(diagnostics["validRepCount"], 2)

    def test_hinge_counts_torso_lean_top_bottom_top_cycles(self):
        values = [
            8, 10, 18, 35, 55, 72, 82, 70, 48, 24, 10,
            9, 15, 34, 56, 74, 83, 68, 42, 20, 9,
            10, 16, 36, 58, 76, 84, 70, 46, 22, 10,
        ]
        frames = [
            pose_frame(index, landmarks_with_trunk_lean(90 - value))
            for index, value in enumerate(values)
        ]

        events, signal = segment_hinge_repetitions(frames, 8.0)
        valid, diagnostics = validate_rep_events(events, signal, 8.0, "deadlift")

        self.assertEqual(len(valid), 3)
        self.assertEqual(valid[0]["counterRule"], "shoulder_hip_line_top_bottom_top")
        self.assertEqual(diagnostics["validRepCount"], 3)

    def test_hip_thrust_allows_slow_controlled_rep(self):
        signal = np.array([
            120, 122, 128, 138, 150, 162, 172, 176, 174, 166,
            154, 142, 130, 122, 120,
        ], dtype=float)
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 7,
            "poseEndIndex": 14,
            "durationSeconds": 7.0,
            "signalAmplitude": 56.0,
            "counterRule": "hip_angle_bottom_top_bottom",
        }]
        valid, diagnostics = validate_rep_events(events, signal, 2.0, "hip_thrust")

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["counterRule"], "hip_angle_bottom_top_bottom")
        self.assertEqual(diagnostics["validRepCount"], 1)

    def test_segments_three_reps_from_three_peaks(self):
        frames = [
            pose_frame(index, landmarks_with_knee_angle(90), signal=value)
            for index, value in enumerate([0, 20, 45, 18, 0, 22, 50, 21, 0, 19, 47, 18, 0])
        ]
        signal = np.array([frame.signal for frame in frames], dtype=float)

        events = segment_repetitions(frames, signal, [2, 6, 10], 8.0, "squat")

        self.assertEqual([item["repIndex"] for item in events], [1, 2, 3])
        self.assertEqual(events[1]["keyFrameIndex"], 6)
        self.assertEqual(events[1]["signalAmplitude"], 50)
        self.assertEqual(events[1]["durationSeconds"], 0.5)

    def test_stage_selection_prefers_full_amplitude_rep_over_clear_partial_rep(self):
        signal = np.array([0, 8, 0, 0, 50, 0], dtype=float)
        events = [
            {
                "repIndex": 1,
                "poseStartIndex": 0,
                "poseKeyIndex": 1,
                "poseEndIndex": 2,
                "quality": 0.99,
                "signalAmplitude": 8,
                "durationSeconds": 0.25,
            },
            {
                "repIndex": 2,
                "poseStartIndex": 3,
                "poseKeyIndex": 4,
                "poseEndIndex": 5,
                "quality": 0.8,
                "signalAmplitude": 50,
                "durationSeconds": 0.25,
            },
        ]

        stages = select_stages_from_rep_events(signal, events, 8.0)

        self.assertEqual(stages[2], 4)

    def test_filters_short_low_amplitude_bench_candidates(self):
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(90), signal=value)
            for index, value in enumerate([0, 12, 28, 12, 0, 1, 3, 1, 0])
        ]
        signal = np.array([frame.signal for frame in frames], dtype=float)
        events = segment_repetitions(frames, signal, [2, 6], 8.0, "press")

        valid, diagnostics = validate_rep_events(events, signal, 8.0, "bench_press")

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["poseKeyIndex"], 2)
        self.assertEqual(diagnostics["rejectedRepCount"], 1)

    def test_filters_too_long_generic_rep_candidates(self):
        frames = [
            pose_frame(index, landmarks_with_elbow_angle(90), signal=value)
            for index, value in enumerate([0, 20, 45, 20, 0] + [0] * 60 + [0, 22, 50, 22, 0])
        ]
        signal = np.array([frame.signal for frame in frames], dtype=float)
        events = [
            {
                "repIndex": 1,
                "poseStartIndex": 0,
                "poseKeyIndex": 2,
                "poseEndIndex": 4,
                "durationSeconds": 0.5,
                "signalAmplitude": 45,
            },
            {
                "repIndex": 2,
                "poseStartIndex": 4,
                "poseKeyIndex": 67,
                "poseEndIndex": 69,
                "durationSeconds": 8.1,
                "signalAmplitude": 50,
            },
        ]

        valid, diagnostics = validate_rep_events(events, signal, 8.0, "lat_pulldown")

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["repIndex"], 1)
        self.assertEqual(diagnostics["rejectedReps"][0]["reason"], "too_long")


class ActiveTrainingWindowTest(unittest.TestCase):
    def test_lat_pulldown_window_uses_elbow_flexion_motion(self):
        frames = []
        for index in range(20):
            frames.append(pose_frame(index, landmarks_with_elbow_angle(168)))
        for offset in range(32):
            flexion = 18 + 92 * abs(math.sin(offset / 31 * math.pi * 2))
            frames.append(pose_frame(20 + offset, landmarks_with_elbow_angle(180 - flexion)))
        for index in range(52, 72):
            frames.append(pose_frame(index, landmarks_with_elbow_angle(168)))

        signal, _ = motion_signal_series(frames, "pull", "lat_pulldown", "front_oblique")
        start, end, diagnostics = select_active_training_window(
            frames,
            signal,
            "pull",
            "lat_pulldown",
            8.0,
        )

        self.assertEqual(diagnostics["reason"], "lat_pulldown_all_elbow_turning_points")
        self.assertEqual(diagnostics["turningPointCount"], 2)
        self.assertGreater(start, 0)
        self.assertLess(end, len(frames) - 1)

    def test_trims_setup_and_getting_up_after_main_hinge_set(self):
        frames = []
        signal = []
        for index in range(12):
            frames.append(pose_frame(index, landmarks_with_trunk_lean(5), signal=2.0))
            signal.append(2.0)
        for index in range(12, 44):
            value = 24.0 + 18.0 * abs(math.sin((index - 12) / 31 * math.pi * 4))
            frames.append(pose_frame(index, landmarks_with_trunk_lean(56), signal=value))
            signal.append(value)
        for index in range(44, 58):
            frames.append(pose_frame(index, landmarks_with_trunk_lean(7), signal=5.0))
            signal.append(5.0)

        start, end, diagnostics = select_active_training_window(
            frames,
            np.asarray(signal, dtype=float),
            "hinge",
            "deadlift",
            8.0,
        )

        self.assertGreater(start, 0)
        self.assertLess(end, len(frames) - 1)
        self.assertGreater(diagnostics["trimmedEndFrames"], 0)
        self.assertEqual(diagnostics["reason"], "trunk_lean_posture")
        self.assertLessEqual(frames[start].time_ms, frames[12].time_ms)
        self.assertGreaterEqual(frames[end].time_ms, frames[43].time_ms)

    def test_trims_closeup_tail_for_bench_when_motion_spikes_after_set(self):
        frames = []
        signal = []
        for index in range(40):
            frame = pose_frame(index, landmarks_with_elbow_angle(160), signal=3.0)
            frame.person_bbox = [0.28, 0.28, 0.68, 0.78]
            frames.append(frame)
            signal.append(3.0)
        for index in range(40, 52):
            frame = pose_frame(index, landmarks_with_elbow_angle(60), signal=70.0 + index)
            frame.person_bbox = [0.0, 0.0, 1.0, 1.0]
            frames.append(frame)
            signal.append(70.0 + index)

        start, end, diagnostics = select_active_training_window(
            frames,
            np.asarray(signal, dtype=float),
            "press",
            "bench_press",
            8.0,
        )

        self.assertEqual(start, 0)
        self.assertLess(end, len(frames) - 1)
        self.assertGreater(diagnostics["trimmedEndFrames"], 0)
        self.assertEqual(diagnostics["reason"], "target_tracking_quality_trim")


class RepRuleTest(unittest.TestCase):
    def test_reports_second_rep_depth_issue_with_time_range(self):
        frames = [
            pose_frame(0, landmarks_with_knee_angle(90)),
            pose_frame(1, landmarks_with_knee_angle(90)),
            pose_frame(2, landmarks_with_knee_angle(130)),
            pose_frame(3, landmarks_with_knee_angle(90)),
        ]
        events = [
            {"repIndex": 1, "poseKeyIndex": 0, "startTimeMs": 0, "endTimeMs": 120},
            {"repIndex": 2, "poseKeyIndex": 2, "startTimeMs": 200, "endTimeMs": 360},
            {"repIndex": 3, "poseKeyIndex": 3, "startTimeMs": 400, "endTimeMs": 520},
        ]

        issues = evaluate_rep_rules("barbell_squat", "squat", frames, events, "side_rear")
        depth_issues = [item for item in issues if item["code"] == "SQUAT_DEPTH_LIMITED"]

        self.assertEqual(len(depth_issues), 1)
        self.assertEqual(depth_issues[0]["repIndexes"], [2])
        self.assertEqual(depth_issues[0]["stage"], "keyPosition")
        self.assertEqual(depth_issues[0]["timeRangesMs"], [[200, 360]])

    def test_low_ankle_confidence_blocks_depth_issue(self):
        frames = [
            pose_frame(0, landmarks_with_knee_angle(130, confidence=0.2)),
        ]
        events = [{"repIndex": 1, "poseKeyIndex": 0, "startTimeMs": 0, "endTimeMs": 120}]

        issues = evaluate_rep_rules("barbell_squat", "squat", frames, events, "side_rear")

        self.assertFalse(any(item["code"] == "SQUAT_DEPTH_LIMITED" for item in issues))
        self.assertTrue(any(item["code"] == "LOW_CONFIDENCE_EVIDENCE" for item in issues))

    def test_hack_squat_side_view_uses_visible_knee_side(self):
        marks = landmarks_with_knee_angle(90)
        set_side_confidence(marks, "LEFT", 0.2)
        marks[int(LANDMARK.RIGHT_HIP)] = [0.65, 0.78, 0.0, 0.95]
        marks[int(LANDMARK.RIGHT_KNEE)] = [0.65, 0.76, 0.0, 0.95]
        frames = [pose_frame(0, marks)]
        events = [{"repIndex": 1, "poseKeyIndex": 0, "startTimeMs": 0, "endTimeMs": 120}]

        issues = evaluate_rep_rules("hack_squat", "squat", frames, events, "side")

        self.assertFalse(any(item["code"] == "LOW_CONFIDENCE_EVIDENCE" for item in issues))
        self.assertFalse(any(item["code"] == "HACK_SQUAT_DEPTH_LIMITED" for item in issues))

    def test_hack_squat_uses_hip_knee_depth_when_ankle_is_occluded_by_machine(self):
        marks = landmarks_with_knee_angle(90)
        set_side_confidence(marks, "LEFT", 0.2)
        marks[int(LANDMARK.RIGHT_HIP)] = [0.65, 0.78, 0.0, 0.95]
        marks[int(LANDMARK.RIGHT_KNEE)] = [0.65, 0.76, 0.0, 0.95]
        marks[int(LANDMARK.RIGHT_ANKLE)] = [0.90, 0.90, 0.0, 0.2]
        frames = [pose_frame(0, marks)]
        events = [{"repIndex": 1, "poseKeyIndex": 0, "startTimeMs": 0, "endTimeMs": 120}]

        issues = evaluate_rep_rules("hack_squat", "squat", frames, events, "side")

        self.assertFalse(any(item["code"] == "LOW_CONFIDENCE_EVIDENCE" for item in issues))
        self.assertFalse(any(item["code"] == "HACK_SQUAT_DEPTH_LIMITED" for item in issues))

    def test_side_bench_does_not_report_left_right_elbow_asymmetry(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angles(75, 150)),
        ]
        events = [{"repIndex": 1, "poseKeyIndex": 0, "startTimeMs": 0, "endTimeMs": 120}]

        issues = evaluate_rep_rules("bench_press", "press", frames, events, "side")

        self.assertFalse(any(item["code"] == "PRESS_ASYMMETRY" for item in issues))

    def test_bilateral_metric_requires_both_sides_confident(self):
        marks = landmarks_with_elbow_angles(75, 150, right_confidence=0.2)

        self.assertLess(metric_confidence(marks, "elbowAngle"), 0.55)

    def test_lat_pulldown_rep_rule_requires_elbow_below_90_degrees(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angle(145)),
            pose_frame(1, landmarks_with_elbow_angle(100)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("lat_pulldown", "pull", frames, events, "front_oblique")

        self.assertTrue(any(item["code"] == "LAT_PULLDOWN_RANGE_INCOMPLETE" for item in issues))

    def test_single_arm_hammer_row_rep_rule_requires_working_elbow_pull(self):
        frames = [
            pose_frame(0, landmarks_with_single_arm_hammer_row_shape(142, working_side="RIGHT")),
            pose_frame(1, landmarks_with_single_arm_hammer_row_shape(118, working_side="RIGHT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("single_arm_hammer_row", "pull", frames, events, "side_front")

        self.assertTrue(any(item["code"] == "HAMMER_ROW_RANGE_INCOMPLETE" for item in issues))
        self.assertFalse(any(item["code"] == "PRESS_ASYMMETRY" for item in issues))

    def test_plate_loaded_rear_leg_raise_rep_rule_uses_rear_leg_code_not_hip_thrust(self):
        frames = [
            pose_frame(0, landmarks_with_plate_loaded_rear_leg_raise_shape(116, working_side="RIGHT")),
            pose_frame(1, landmarks_with_plate_loaded_rear_leg_raise_shape(136, working_side="RIGHT")),
            pose_frame(2, landmarks_with_plate_loaded_rear_leg_raise_shape(116, working_side="RIGHT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 360,
        }]

        issues = evaluate_rep_rules("plate_loaded_rear_leg_raise", "hinge", frames, events, "side")

        self.assertTrue(any(item["code"] == "REAR_LEG_RAISE_EXTENSION_LIMITED" for item in issues))
        self.assertFalse(any(item["code"] == "HIP_THRUST_LOCKOUT_INCOMPLETE" for item in issues))

    def test_preacher_curl_rep_rule_requires_top_flexion(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angle(145)),
            pose_frame(1, landmarks_with_elbow_angle(118)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("preacher_curl", "isolation_elbow", frames, events, "side")

        self.assertTrue(any(item["code"] == "PREACHER_CURL_RANGE_INCOMPLETE" for item in issues))

    def test_romanian_deadlift_allows_video_standard_knee_bend_when_hip_hinge_is_deep(self):
        frames = [
            pose_frame(0, landmarks_with_hip_and_knee_angles(80, 90)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 0,
            "poseEndIndex": 0,
            "startTimeMs": 0,
            "endTimeMs": 120,
        }]

        issues = evaluate_rep_rules("romanian_deadlift", "hinge", frames, events, "side")

        self.assertFalse(any(item["code"] == "RDL_EXCESSIVE_KNEE_BEND" for item in issues))
        self.assertFalse(any(item["code"] == "LOW_CONFIDENCE_EVIDENCE" for item in issues))

    def test_back_extension_allows_clear_fold_and_neutral_return(self):
        frames = [
            pose_frame(0, landmarks_with_hip_angle(170)),
            pose_frame(1, landmarks_with_hip_angle(95)),
            pose_frame(2, landmarks_with_hip_angle(168)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 360,
        }]

        issues = evaluate_rep_rules("back_extension", "hinge", frames, events, "side")

        self.assertFalse(any(item["code"] == "BACK_EXTENSION_RANGE_LIMITED" for item in issues))
        self.assertFalse(any(item["code"] == "BACK_EXTENSION_TOP_SHORT" for item in issues))

    def test_back_extension_rep_rule_flags_limited_fold(self):
        frames = [
            pose_frame(0, landmarks_with_hip_angle(168)),
            pose_frame(1, landmarks_with_hip_angle(140)),
            pose_frame(2, landmarks_with_hip_angle(166)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 360,
        }]

        issues = evaluate_rep_rules("back_extension", "hinge", frames, events, "side")

        self.assertTrue(any(item["code"] == "BACK_EXTENSION_RANGE_LIMITED" for item in issues))

    def test_back_extension_rep_rule_flags_short_top_return(self):
        frames = [
            pose_frame(0, landmarks_with_hip_angle(170)),
            pose_frame(1, landmarks_with_hip_angle(95)),
            pose_frame(2, landmarks_with_hip_angle(132)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 360,
        }]

        issues = evaluate_rep_rules("back_extension", "hinge", frames, events, "side")

        self.assertTrue(any(item["code"] == "BACK_EXTENSION_TOP_SHORT" for item in issues))

    def test_y_raise_rep_rule_requires_high_y_top(self):
        frames = [
            pose_frame(0, landmarks_with_y_raise_shape(35, working_side="LEFT")),
            pose_frame(1, landmarks_with_y_raise_shape(96, working_side="LEFT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("y_raise", "isolation_shoulder", frames, events, "front")

        self.assertTrue(any(item["code"] == "Y_RAISE_TOP_RANGE_INCOMPLETE" for item in issues))

    def test_y_raise_rep_rule_rejects_flat_lateral_raise_path(self):
        frames = [
            pose_frame(0, landmarks_with_y_raise_shape(35, working_side="LEFT")),
            pose_frame(1, landmarks_with_y_raise_shape(126, working_side="LEFT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("y_raise", "isolation_shoulder", frames, events, "front")

        self.assertTrue(any(item["code"] == "Y_RAISE_PATH_TOO_FLAT" for item in issues))

    def test_y_raise_rep_rule_flags_bent_elbow(self):
        frames = [
            pose_frame(0, landmarks_with_y_raise_shape(35, working_side="LEFT")),
            pose_frame(1, landmarks_with_y_raise_shape(140, elbow_angle_degrees=118, working_side="LEFT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("y_raise", "isolation_shoulder", frames, events, "front")

        self.assertTrue(any(item["code"] == "Y_RAISE_ELBOW_BEND" for item in issues))

    def test_y_raise_rep_rule_does_not_require_bilateral_sync(self):
        frames = [
            pose_frame(0, landmarks_with_y_raise_shape(35, working_side="LEFT")),
            pose_frame(1, landmarks_with_y_raise_shape(140, working_side="LEFT")),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 1,
            "startTimeMs": 0,
            "endTimeMs": 240,
        }]

        issues = evaluate_rep_rules("y_raise", "isolation_shoulder", frames, events, "front")

        self.assertFalse(any(item["code"] == "Y_RAISE_ASYMMETRY" for item in issues))
        self.assertFalse(any(item["code"] == "Y_RAISE_TOP_RANGE_INCOMPLETE" for item in issues))


    def test_machine_chest_press_uses_visible_side_without_bilateral_failure(self):
        start = landmarks_with_elbow_angle(150)
        key = landmarks_with_elbow_angle(92)
        end = landmarks_with_elbow_angle(152)
        for marks in (start, key, end):
            for name in ("SHOULDER", "ELBOW", "WRIST"):
                marks[int(getattr(LANDMARK, f"RIGHT_{name}"))][3] = 0.1
        frames = [pose_frame(0, start), pose_frame(1, key), pose_frame(2, end)]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 800,
        }]

        issues = evaluate_rep_rules("machine_chest_press", "press", frames, events, "side_front")

        self.assertFalse(any(item["code"] == "LOW_CONFIDENCE_EVIDENCE" for item in issues))
        self.assertFalse(any(item["code"] == "MACHINE_CHEST_PRESS_RANGE_LIMITED" for item in issues))

    def test_machine_chest_press_flags_short_bottom_range(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angle(150)),
            pose_frame(1, landmarks_with_elbow_angle(128)),
            pose_frame(2, landmarks_with_elbow_angle(150)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 800,
        }]

        issues = evaluate_rep_rules("machine_chest_press", "press", frames, events, "front")

        self.assertTrue(any(item["code"] == "MACHINE_CHEST_PRESS_RANGE_LIMITED" for item in issues))

    def test_chest_supported_row_flags_short_pull(self):
        frames = [
            pose_frame(0, landmarks_with_elbow_angle(150)),
            pose_frame(1, landmarks_with_elbow_angle(128)),
            pose_frame(2, landmarks_with_elbow_angle(150)),
        ]
        events = [{
            "repIndex": 1,
            "poseStartIndex": 0,
            "poseKeyIndex": 1,
            "poseEndIndex": 2,
            "startTimeMs": 0,
            "endTimeMs": 800,
        }]

        issues = evaluate_rep_rules("chest_supported_row", "pull", frames, events, "side_front")

        self.assertTrue(any(item["code"] == "CHEST_SUPPORTED_ROW_RANGE_LIMITED" for item in issues))


class TargetSelectionTest(unittest.TestCase):
    def test_press_target_prefers_lying_lifter_over_standing_bystander(self):
        tracker = TargetTracker()
        standing = coco_candidate("standing", 0.98)
        lying = coco_candidate("lying", 0.78)

        selected = select_target_instance([standing, lying], 100, 100, "press", tracker)

        self.assertIsNotNone(selected)
        self.assertEqual(selected[2]["targetId"], 1)
        self.assertEqual(selected[2]["candidateCount"], 2)
        self.assertGreaterEqual(tracker.multi_person_frames, 1)

    def test_tracking_skips_far_smaller_candidate_instead_of_switching_to_bystander(self):
        tracker = TargetTracker(
            center=(0.58, 0.56),
            bbox=[0.42, 0.20, 0.72, 0.86],
            selected_index=0,
        )
        bystander = coco_candidate("standing", 0.70)

        selected = select_target_instance([bystander], 100, 100, "pull", tracker)

        self.assertIsNone(selected)
        self.assertEqual(tracker.target_lost_count, 1)
        self.assertEqual(tracker.rejected_distractor_count, 1)
        self.assertEqual(tracker.center, (0.58, 0.56))


class PoseBackendTest(unittest.TestCase):
    def test_strict_backend_never_falls_back_to_another_engine(self):
        calls = []

        def fail_rtmo(_video_path, _family, _fps, _step, *_extra):
            calls.append("rtmlib")
            raise RuntimeError("rtmo unavailable")

        def ok_mediapipe(_video_path, _family, _fps, _step, *_extra):
            calls.append("mediapipe")
            return [pose_frame(0, landmarks_with_knee_angle(90))]

        result = estimate_pose_frames(
            Path("missing.mp4"),
            "squat",
            30.0,
            3,
            preferred_backend="rtmlib",
            strict_backend=True,
            extractors={"rtmlib": fail_rtmo, "mediapipe": ok_mediapipe},
        )

        self.assertEqual(calls, ["rtmlib"])
        self.assertEqual(result.frames, [])
        self.assertEqual(result.diagnostics["poseBackend"], "none")

    def test_display_suppression_hides_face_landmarks(self):
        frame = pose_frame(0, landmarks_with_knee_angle(90))
        rendered = apply_display_landmark_suppression([frame], "lat_pulldown", "pull")[0]

        self.assertTrue(all(rendered.landmarks[index][3] == 0.0 for index in range(11)))
        self.assertGreater(rendered.landmarks[int(LANDMARK.LEFT_SHOULDER)][3], 0.0)

    def test_short_display_gap_is_recovered_without_mutating_analysis_frames(self):
        frames = [pose_frame(index, landmarks_with_knee_angle(90)) for index in range(5)]
        wrist = int(LANDMARK.RIGHT_WRIST)
        frames[2].landmarks[wrist][0] = 0.1
        frames[2].landmarks[wrist][3] = 0.05

        rendered = recover_short_display_landmark_gaps(frames, threshold=0.35, max_gap_frames=3)

        self.assertGreater(rendered[2].landmarks[wrist][3], 0.35)
        self.assertNotEqual(rendered[2].landmarks[wrist][0], 0.1)
        self.assertEqual(frames[2].landmarks[wrist][3], 0.05)

    def test_mmpose_failure_falls_back_to_mediapipe(self):
        expected = [pose_frame(0, landmarks_with_knee_angle(90))]

        def fail_mmpose(_video_path, _family, _fps, _step, *_extra):
            raise RuntimeError("mmpose missing")

        def ok_mediapipe(_video_path, _family, _fps, _step, *_extra):
            return expected

        result = estimate_pose_frames(
            Path("missing.mp4"),
            "squat",
            30.0,
            3,
            preferred_backend="mmpose",
            extractors={"mmpose": fail_mmpose, "mediapipe": ok_mediapipe},
        )

        self.assertEqual(result.frames, expected)
        self.assertEqual(result.diagnostics["poseBackend"], "mediapipe")
        self.assertIn("mmpose missing", result.diagnostics["poseBackendFallback"])


if __name__ == "__main__":
    unittest.main()
