import unittest

from env.models import Ticket
from env.tasks import load_scenario, scenario_names
from env.tickets import load_initial_tickets


class TaskLoadingTests(unittest.TestCase):
    def test_scenario_names_are_available(self) -> None:
        self.assertEqual(scenario_names(), ["easy", "medium", "hard"])

    def test_easy_scenario_has_ten_initial_tickets_without_arrivals(self) -> None:
        scenario = load_scenario("easy")
        self.assertEqual(scenario.difficulty, "easy")
        self.assertEqual(len(scenario.initial_tickets), 10)
        self.assertEqual(len(scenario.arrival_schedule), 0)

    def test_medium_and_hard_match_target_total_ticket_counts(self) -> None:
        medium = load_scenario("medium")
        hard = load_scenario("hard")
        self.assertGreater(len(medium.arrival_schedule), 0)
        self.assertGreater(len(hard.arrival_schedule), 0)
        self.assertEqual(
            len(medium.initial_tickets) + sum(len(wave.tickets) for wave in medium.arrival_schedule),
            15,
        )
        self.assertEqual(
            len(hard.initial_tickets) + sum(len(wave.tickets) for wave in hard.arrival_schedule),
            25,
        )

    def test_public_ticket_loader_strips_ground_truth(self) -> None:
        tickets = load_initial_tickets("easy")
        self.assertEqual(len(tickets), 10)
        self.assertIsInstance(tickets[0], Ticket)
        self.assertFalse(hasattr(tickets[0], "ground_truth"))
        self.assertIsNotNone(tickets[0].specialist_team)

    def test_specialist_team_is_present_in_scenarios(self) -> None:
        scenario = load_scenario("hard")
        self.assertIsNotNone(scenario.initial_tickets[0].specialist_team)
        self.assertIn(
            scenario.initial_tickets[0].specialist_team,
            {
                "payments_ops",
                "refunds",
                "subscription_ops",
                "account_access",
                "security",
                "product_bug",
                "platform_reliability",
                "sales_ops",
            },
        )


if __name__ == "__main__":
    unittest.main()
