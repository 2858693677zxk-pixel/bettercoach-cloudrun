import unittest

from motion_tracker_experiment import MotionSample, evaluate_action


def sample(index, elbow_angle, torso_lean=170):
    return MotionSample(
        frame_index=index,
        time_ms=index * 100,
        confidence=0.8,
        pose=None,
        angles={},
        posture={},
        metrics={
            "elbow_angle": elbow_angle,
            "hip_angle": 100,
            "knee_angle": 145,
            "torso_lean_2d": torso_lean,
            "body_lean": 0,
            "spine_curve": 10,
            "elbow_below_shoulder": 0.2,
            "forearm_tilt_ratio": 2,
        },
    )


class MotionTrackerExperimentTest(unittest.TestCase):
    def test_row_counts_smaller_elbow_cycles(self):
        values = [
            132, 132, 130, 128, 124, 118, 110, 104, 104, 108, 116, 124,
            130, 132, 131, 128, 122, 114, 106, 103, 104, 110, 118, 126,
            131, 132, 129, 122, 114, 106, 104, 106, 114, 123, 130,
        ]
        result = evaluate_action("row", [sample(i, value) for i, value in enumerate(values)], 1.0, 0.8)

        self.assertEqual(result["repCount"], 3)
        self.assertNotIn("MT_ROW_NO_FULL_REP", [item["code"] for item in result["issues"]])

    def test_lat_pulldown_requires_deeper_elbow_flexion(self):
        values = [
            150, 150, 146, 140, 125, 105, 88, 80, 80, 92, 115, 135,
            148, 150, 148, 140, 122, 100, 86, 78, 80, 96, 120, 140, 150,
        ]
        result = evaluate_action("lat_pulldown", [sample(i, value, 178) for i, value in enumerate(values)], 1.0, 0.8)

        self.assertEqual(result["repCount"], 2)
        self.assertNotIn("MT_LAT_NO_FULL_REP", [item["code"] for item in result["issues"]])


if __name__ == "__main__":
    unittest.main()
