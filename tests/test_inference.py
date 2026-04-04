import unittest

from inference import run_all_tasks, run_episode


class InferenceTests(unittest.TestCase):
    def test_heuristic_episode_runs(self) -> None:
        report = run_episode("easy", heuristic_only=True)

        self.assertEqual(report["scenario"], "easy")
        self.assertEqual(report["model_name"], "heuristic-baseline")
        self.assertIn("grade", report)
        self.assertGreaterEqual(report["grade"]["final_score"], 0.0)
        self.assertLessEqual(report["grade"]["final_score"], 1.0)

    def test_heuristic_all_tasks_runs(self) -> None:
        report = run_all_tasks(heuristic_only=True)

        self.assertEqual(report["runner"], "heuristic")
        self.assertEqual(len(report["tasks"]), 3)
        self.assertGreaterEqual(report["average_final_score"], 0.0)
        self.assertLessEqual(report["average_final_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
