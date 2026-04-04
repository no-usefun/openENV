import unittest

from env.core import TicketEnv
from env.models import Action


class EnvironmentTests(unittest.TestCase):
    def test_easy_reset_returns_expected_state(self) -> None:
        env = TicketEnv({"scenario_name": "easy"})

        state = env.reset()

        self.assertEqual(state.current_time, 0)
        self.assertEqual(state.step_number, 0)
        self.assertEqual(state.pending_count, 6)
        self.assertEqual(state.resolved_count, 0)
        self.assertEqual(state.current_ticket.id, "E006")

    def test_medium_arrivals_appear_after_two_steps(self) -> None:
        env = TicketEnv({"scenario_name": "medium"})
        state = env.reset()

        for _ in range(2):
            current_id = state.current_ticket.id
            labeled_ticket = next(ticket for ticket in env._pending_tickets if ticket.id == current_id)
            action = Action(
                ticket_id=labeled_ticket.id,
                department=labeled_ticket.ground_truth.department,
                priority=labeled_ticket.ground_truth.priority,
                action_type=labeled_ticket.ground_truth.action_type,
            )
            state, reward, done, info = env.step(action)

        self.assertEqual(state.current_time, 2)
        self.assertEqual(state.pending_count, 8)
        self.assertIn("M009", [ticket.id for ticket in state.pending_tickets])
        self.assertIn("M010", [ticket.id for ticket in state.pending_tickets])
        self.assertEqual(info["arrivals_added"], 2)

    def test_invalid_ticket_action_penalizes_and_advances_time(self) -> None:
        env = TicketEnv({"scenario_name": "easy"})
        env.reset()

        state, reward, done, info = env.step(
            Action(
                ticket_id="DOES_NOT_EXIST",
                department="general",
                priority="low",
                action_type="resolve",
            )
        )

        self.assertEqual(reward.step_score, -0.5)
        self.assertEqual(state.current_time, 1)
        self.assertEqual(state.step_number, 1)
        self.assertEqual(info["error"], "invalid_ticket")


if __name__ == "__main__":
    unittest.main()
