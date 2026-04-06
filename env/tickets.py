from typing import List

from .models import Scenario, Ticket
from .tasks import load_scenario, scenario_names


def load_initial_tickets(name: str = "easy") -> List[Ticket]:
    scenario = load_scenario(name)
    return [Ticket.model_validate(ticket.model_dump()) for ticket in scenario.initial_tickets]


def load_ticket_scenario(name: str = "easy") -> Scenario:
    return load_scenario(name)


__all__ = ["load_initial_tickets", "load_ticket_scenario", "scenario_names"]
