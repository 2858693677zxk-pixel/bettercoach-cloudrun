import unittest

from stability_evaluation import aggregate_reports, cohens_kappa, evaluate_case


class StabilityEvaluationTest(unittest.TestCase):
    def test_case_reports_frame_and_boundary_metrics(self):
        result = {
            "frameJudgments": [
                {"timeMs": 0, "state": "stable"},
                {"timeMs": 100, "state": "stable"},
                {"timeMs": 200, "state": "unstable"},
                {"timeMs": 300, "state": "unstable"},
                {"timeMs": 400, "state": "stable"},
                {"timeMs": 500, "state": "unknown"},
            ],
            "judgmentSegments": [
                {"startTimeMs": 0, "endTimeMs": 200, "state": "stable"},
                {"startTimeMs": 200, "endTimeMs": 400, "state": "unstable"},
            ],
        }
        case = {
            "id": "machine-press-01",
            "actionType": "machine_chest_press",
            "view": "side",
            "consensusUnstableRangesMs": [[180, 420]],
            "annotators": [
                {"annotatorId": "coach-a", "unstableRangesMs": [[180, 420]]},
                {"annotatorId": "coach-b", "unstableRangesMs": [[200, 400]]},
            ],
        }

        report = evaluate_case(case, result)

        self.assertEqual(report["confusion"]["tp"], 2)
        self.assertEqual(report["confusion"]["fn"], 1)
        self.assertEqual(report["ignoredOrUnknownFrames"], 1)
        self.assertEqual(report["boundaries"]["meanOnsetErrorMs"], 20.0)
        self.assertEqual(report["boundaries"]["meanOffsetErrorMs"], 20.0)
        self.assertIsNotNone(report["coachAgreementKappa"])

    def test_aggregate_keeps_action_metrics_separate(self):
        first = evaluate_case(
            {"id": "a", "actionType": "machine_chest_press", "unstableRangesMs": [[100, 300]]},
            {
                "frameJudgments": [{"timeMs": 100, "state": "unstable"}],
                "judgmentSegments": [{"startTimeMs": 100, "endTimeMs": 300, "state": "unstable"}],
            },
        )
        second = evaluate_case(
            {"id": "b", "actionType": "barbell_squat", "unstableRangesMs": []},
            {
                "frameJudgments": [{"timeMs": 100, "state": "stable"}],
                "judgmentSegments": [],
            },
        )

        report = aggregate_reports([first, second])

        self.assertEqual(report["overall"]["videos"], 2)
        self.assertEqual(set(report["byAction"]), {"barbell_squat", "machine_chest_press"})

    def test_kappa_is_one_for_identical_labels(self):
        self.assertEqual(cohens_kappa([False, True, True], [False, True, True]), 1.0)


if __name__ == "__main__":
    unittest.main()
