import math
import unittest
from types import SimpleNamespace

from pose_compare import (
    _joint_delta_summary,
    _top_divergent_joints,
    angle_series,
    failed_pose_engine_comparison,
    joint_angle,
)


def landmarks_with_joint(angle_degrees):
    marks = [[0.5, 0.5, 0.0, 0.95] for _ in range(33)]
    marks[23] = [0.45, 0.55, 0.0, 0.95]
    marks[25] = [0.45, 0.75, 0.0, 0.95]
    theta = math.radians(angle_degrees)
    marks[27] = [0.45 + 0.18 * math.sin(theta), 0.75 - 0.18 * math.cos(theta), 0.0, 0.95]
    return marks


class PoseCompareTest(unittest.TestCase):
    def test_joint_angle_reads_mediapipe_landmark_layout(self):
        value = joint_angle(landmarks_with_joint(90), "left_knee")

        self.assertIsNotNone(value)
        self.assertGreater(value, 89)
        self.assertLess(value, 91)

    def test_angle_series_uses_frame_indexes(self):
        frames = [
            SimpleNamespace(frame_index=10, landmarks=landmarks_with_joint(90)),
            SimpleNamespace(frame_index=20, landmarks=landmarks_with_joint(135)),
        ]

        series = angle_series(frames)

        self.assertIn(10, series["left_knee"])
        self.assertIn(20, series["left_knee"])

    def test_failed_comparison_is_non_fatal_diagnostic_shape(self):
        payload = failed_pose_engine_comparison(
            primary_backend="rtmlib",
            primary_pose_coverage=0.8,
            primary_average_confidence=0.7,
            primary_frame_count=12,
            runtime_ms=15,
            error=RuntimeError("boom"),
        )

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["recommendation"], "keep_primary")
        self.assertEqual(payload["secondary"]["poseFrames"], 0)
        self.assertEqual(payload["error"]["type"], "RuntimeError")

    def test_delta_summary_reports_divergent_joint(self):
        primary = [SimpleNamespace(frame_index=1, landmarks=landmarks_with_joint(90))]
        secondary = [SimpleNamespace(frame_index=1, landmarks=landmarks_with_joint(135))]

        summary = _joint_delta_summary(primary, secondary)
        divergent = _top_divergent_joints(summary)

        self.assertEqual(summary["left_knee"]["count"], 1)
        self.assertGreater(summary["left_knee"]["meanAbsDelta"], 40)
        self.assertEqual(divergent[0]["joint"], "left_knee")

    def test_top_divergent_joints_uses_mean_when_medians_tie(self):
        divergent = _top_divergent_joints({
            "left_knee": {"count": 270, "meanAbsDelta": 1.43, "medianAbsDelta": 0.0, "p95AbsDelta": 8.67},
            "right_elbow": {"count": 239, "meanAbsDelta": 37.71, "medianAbsDelta": 0.0, "p95AbsDelta": 172.32},
        })

        self.assertEqual(divergent[0]["joint"], "right_elbow")


if __name__ == "__main__":
    unittest.main()
