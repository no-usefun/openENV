from typing import Any, Dict, List, Optional, Tuple

from .grader import grade_episode
from .models import (
    Action,
    ArrivalWave,
    DecisionRecord,
    LabeledTicket,
    Observation,
    Reward,
    Scenario,
    Ticket,
)
from .tasks import clone_ticket_list, load_scenario, public_ticket_view


class TicketEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scenario_name = self.config.get("scenario_name", "easy")
        self.max_steps_override = self.config.get("max_steps")

        self._scenario: Optional[Scenario] = None
        self._pending_tickets: List[LabeledTicket] = []
        self._resolved_tickets: List[DecisionRecord] = []
        self._future_arrivals: List[ArrivalWave] = []
        self._time = 0
        self._step_number = 0
        self._done = False
        self._total_score = 0.0
        self._normalized_score_sum = 0.0

    def reset(self) -> Observation:
        self._scenario = load_scenario(self.scenario_name)
        self._time = self._scenario.start_time
        self._step_number = 0
        self._done = False
        self._total_score = 0.0
        self._normalized_score_sum = 0.0
        self._resolved_tickets = []
        self._pending_tickets = clone_ticket_list(
            self._scenario.initial_tickets,
            visible_at=self._time,
        )
        self._future_arrivals = sorted(
            self._scenario.arrival_schedule,
            key=lambda arrival: arrival.time,
        )
        if self.max_steps_override is not None:
            self._scenario.max_steps = int(self.max_steps_override)

        self._release_due_arrivals()
        return self.state()

    def state(self) -> Observation:
        visible_tickets = [
            public_ticket_view(ticket)
            for ticket in sorted(self._pending_tickets, key=_ticket_sort_key)
        ]
        return Observation(
            current_ticket=visible_tickets[0] if visible_tickets else None,
            pending_count=len(visible_tickets),
            resolved_count=len(self._resolved_tickets),
            current_time=self._time,
            step_number=self._step_number,
            pending_tickets=visible_tickets,
            resolved_tickets=list(self._resolved_tickets),
        )

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            return (
                self.state(),
                Reward(step_score=0.0, total_score=self._total_score),
                True,
                {"error": "episode_complete"},
            )

        ticket = next((item for item in self._pending_tickets if item.id == action.ticket_id), None)
        if ticket is None:
            arrivals_added = self._advance_time()
            self._step_number += 1
            step_score = -0.5
            self._normalized_score_sum += _normalize_step_score(step_score)
            self._total_score = _running_total_score(
                self._normalized_score_sum,
                self._step_number,
            )
            self._refresh_done_flag()
            return (
                self.state(),
                Reward(
                    step_score=step_score,
                    total_score=self._total_score,
                    breakdown={"invalid_action": -0.5},
                ),
                self._done,
                {
                    "error": "invalid_ticket",
                    "arrivals_added": arrivals_added,
                    "pending_count": len(self._pending_tickets),
                },
            )

        reward = self._compute_reward(ticket, action)
        self._normalized_score_sum += _normalize_step_score(reward.step_score)
        self._total_score = reward.total_score

        self._resolved_tickets.append(
            DecisionRecord(
                ticket_id=ticket.id,
                department=action.department,
                priority=action.priority,
                action_type=action.action_type,
                handled_at=self._time + 1,
                reward=reward.step_score,
            )
        )
        self._pending_tickets = [item for item in self._pending_tickets if item.id != ticket.id]

        arrivals_added = self._advance_time()
        self._step_number += 1
        self._refresh_done_flag()
        return (
            self.state(),
            reward,
            self._done,
            {
                "scenario_id": self._scenario.scenario_id,
                "arrivals_added": arrivals_added,
                "pending_count": len(self._pending_tickets),
                "resolved_count": len(self._resolved_tickets),
            },
        )

    def grade(self):
        if self._scenario is None:
            raise RuntimeError("Call reset() before grade().")

        return grade_episode(
            scenario=self._scenario,
            decisions=self._resolved_tickets,
            pending_tickets=self._pending_tickets,
        )

    def _compute_reward(self, ticket: LabeledTicket, action: Action) -> Reward:
        components = {
            "department": 0.0,
            "priority": 0.0,
            "action_type": 0.0,
            "ignored_urgent": 0.0,
            "delay": 0.0,
        }

        if action.department == ticket.ground_truth.department:
            components["department"] = 0.2
        else:
            components["department"] = -0.3

        if action.priority == ticket.ground_truth.priority:
            components["priority"] = 0.15

        if action.action_type == ticket.ground_truth.action_type:
            components["action_type"] = 0.1

        if self._is_ignoring_urgent_ticket(ticket):
            components["ignored_urgent"] = -0.5

        components["delay"] = -0.05 * self._time
        raw_step_score = round(sum(components.values()), 4)
        step_score = max(-1.0, min(1.0, raw_step_score))
        next_total_score = _running_total_score(
            self._normalized_score_sum + _normalize_step_score(step_score),
            self._step_number + 1,
        )

        return Reward(
            step_score=round(step_score, 4),
            total_score=next_total_score,
            breakdown=components,
        )

    def _is_ignoring_urgent_ticket(self, selected_ticket: LabeledTicket) -> bool:
        if selected_ticket.urgency >= 4:
            return False

        return any(ticket.urgency >= 4 for ticket in self._pending_tickets if ticket.id != selected_ticket.id)

    def _advance_time(self) -> int:
        self._time += 1

        for ticket in self._pending_tickets:
            ticket.time_waiting += 1

        return self._release_due_arrivals()

    def _release_due_arrivals(self) -> int:
        if not self._future_arrivals:
            return 0

        due_arrivals = [arrival for arrival in self._future_arrivals if arrival.time <= self._time]
        self._future_arrivals = [arrival for arrival in self._future_arrivals if arrival.time > self._time]

        for arrival in due_arrivals:
            self._pending_tickets.extend(
                clone_ticket_list(arrival.tickets, visible_at=arrival.time)
            )

        return sum(len(arrival.tickets) for arrival in due_arrivals)

    def _refresh_done_flag(self) -> None:
        if self._scenario is None:
            self._done = True
            return

        reached_max_steps = self._time >= self._scenario.max_steps
        no_more_work = not self._pending_tickets and not self._future_arrivals
        self._done = reached_max_steps or no_more_work


def _ticket_sort_key(ticket: Ticket) -> Tuple[int, int, str]:
    return (-ticket.urgency, -ticket.time_waiting, ticket.id)


def _normalize_step_score(step_score: float) -> float:
    return (step_score + 1.0) / 2.0


def _running_total_score(normalized_sum: float, step_count: int) -> float:
    if step_count <= 0:
        return 0.0
    return round(max(0.0, min(1.0, normalized_sum / step_count)), 4)
