import unittest
from unittest.mock import patch

from inference import run_all_tasks, run_episode, run_single_task


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

    def test_single_task_falls_back_to_heuristic_when_llm_setup_fails(self) -> None:
        with patch("inference.build_client", side_effect=RuntimeError("missing credentials")):
            report = run_single_task("easy", heuristic_only=False)

        self.assertEqual(report["scenario"], "easy")
        self.assertEqual(report["model_name"], "heuristic-baseline")
        self.assertIn("grade", report)
        self.assertGreaterEqual(report["grade"]["final_score"], 0.0)
        self.assertLessEqual(report["grade"]["final_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
