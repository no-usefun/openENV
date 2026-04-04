import unittest

from env.models import Ticket
from env.tasks import load_scenario, scenario_names
from env.tickets import load_initial_tickets


class TaskLoadingTests(unittest.TestCase):
    def test_scenario_names_are_available(self) -> None:
        self.assertEqual(scenario_names(), ["easy", "medium", "hard"])

    def test_easy_scenario_has_no_arrivals(self) -> None:
        scenario = load_scenario("easy")
        self.assertEqual(scenario.difficulty, "easy")
        self.assertEqual(len(scenario.initial_tickets), 6)
        self.assertEqual(len(scenario.arrival_schedule), 0)

    def test_medium_and_hard_include_arrivals(self) -> None:
        medium = load_scenario("medium")
        hard = load_scenario("hard")
        self.assertGreater(len(medium.arrival_schedule), 0)
        self.assertGreater(len(hard.arrival_schedule), 0)

    def test_public_ticket_loader_strips_ground_truth(self) -> None:
        tickets = load_initial_tickets("easy")
        self.assertEqual(len(tickets), 6)
        self.assertIsInstance(tickets[0], Ticket)
        self.assertFalse(hasattr(tickets[0], "ground_truth"))


if __name__ == "__main__":
    unittest.main()
