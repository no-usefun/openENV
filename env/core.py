import random
from typing import Tuple, Dict, Any
from .models import Observation, Action, Reward, Ticket


class TicketEnv:
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        random.seed(seed)

        self._tickets = []
        self._time = 0
        self._done = False

    # -------------------------
    # Reset
    # -------------------------
    def reset(self) -> Observation:
        self._time = 0
        self._done = False
        self._tickets = self._generate_tickets()

        return self.state()

    # -------------------------
    # Current State
    # -------------------------
    def state(self) -> Observation:
        return Observation(
            tickets=self._tickets,
            current_time=self._time
        )

    # -------------------------
    # Step Function
    # -------------------------
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            return self.state(), Reward(value=0.0), True, {}

        self._time += 1

        # find ticket
        ticket = next((t for t in self._tickets if t.id == action.ticket_id), None)

        if ticket is None:
            # invalid action
            return self.state(), Reward(value=-0.2), False, {"error": "invalid_ticket"}

        # compute reward (placeholder, refined later)
        reward = self._compute_reward(ticket, action)

        # remove processed ticket
        self._tickets = [t for t in self._tickets if t.id != ticket.id]

        # update waiting time for remaining tickets
        for t in self._tickets:
            t.time_waiting += 1

        # done condition
        if len(self._tickets) == 0 or self._time >= self.config["max_steps"]:
            self._done = True

        return self.state(), Reward(value=reward), self._done, {}

    # -------------------------
    # Reward Logic (initial)
    # -------------------------
    def _compute_reward(self, ticket: Ticket, action: Action) -> float:
        reward = 0.0

        # urgency-based reward
        if ticket.urgency >= 4 and action.priority == "high":
            reward += 0.2
        elif ticket.urgency <= 2 and action.priority == "low":
            reward += 0.1
        else:
            reward -= 0.1

        # delay penalty
        reward -= 0.02 * ticket.time_waiting

        return reward

    # -------------------------
    # Ticket Generator
    # -------------------------
    def _generate_tickets(self):
        tickets = []

        sample_texts = [
            "Payment failed but money deducted",
            "App crashes on login",
            "Need invoice for last purchase",
            "Unable to reset password",
            "Feature request for dashboard",
            "Refund not received",
            "Bug in latest update",
            "General inquiry about pricing"
        ]

        for i in range(self.config["num_tickets"]):
            tickets.append(
                Ticket(
                    id=i,
                    text=random.choice(sample_texts),
                    urgency=random.randint(1, 5),
                    customer_tier=random.choice(["free", "premium"]),
                    time_waiting=0
                )
            )

        return tickets