import unittest

from env.grader import grade_episode
from env.models import DecisionRecord
from env.tasks import load_scenario


class GraderTests(unittest.TestCase):
    def test_perfect_easy_episode_scores_one(self) -> None:
        scenario = load_scenario("easy")
        decisions = [
            DecisionRecord(
                ticket_id=ticket.id,
                department=ticket.ground_truth.department,
                priority=ticket.ground_truth.priority,
                action_type=ticket.ground_truth.action_type,
                handled_at=1,
                reward=0.45,
            )
            for ticket in scenario.initial_tickets
        ]

        grade = grade_episode(scenario, decisions, [])

        self.assertEqual(grade.routing_accuracy, 1.0)
        self.assertEqual(grade.priority_accuracy, 1.0)
        self.assertEqual(grade.sla_score, 1.0)
        self.assertEqual(grade.action_accuracy, 1.0)
        self.assertEqual(grade.final_score, 1.0)

    def test_missing_high_priority_ticket_hurts_sla(self) -> None:
        scenario = load_scenario("easy")
        non_urgent_decisions = [
            DecisionRecord(
                ticket_id=ticket.id,
                department=ticket.ground_truth.department,
                priority=ticket.ground_truth.priority,
                action_type=ticket.ground_truth.action_type,
                handled_at=1,
                reward=0.2,
            )
            for ticket in scenario.initial_tickets
            if ticket.id not in {"E006"}
        ]

        grade = grade_episode(scenario, non_urgent_decisions, [])

        self.assertLess(grade.sla_score, 1.0)
        self.assertLess(grade.final_score, 1.0)
        self.assertEqual(grade.total_high_priority, 4)
        self.assertEqual(grade.late_high_priority, 1)

    def test_wrong_labels_reduce_accuracy_metrics(self) -> None:
        scenario = load_scenario("easy")
        decisions = [
            DecisionRecord(
                ticket_id=ticket.id,
                department="general",
                priority="low",
                action_type="resolve",
                handled_at=3,
                reward=-0.2,
            )
            for ticket in scenario.initial_tickets
        ]

        grade = grade_episode(scenario, decisions, [])

        self.assertLess(grade.routing_accuracy, 1.0)
        self.assertLess(grade.priority_accuracy, 1.0)
        self.assertLess(grade.action_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
