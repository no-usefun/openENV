import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import LabeledTicket, Scenario, Ticket


TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def load_scenario(name: str = "easy") -> Scenario:
    scenario_path = TASKS_DIR / f"{name}.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Unknown scenario '{name}' at {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return Scenario.model_validate(payload)


def scenario_names() -> List[str]:
    discovered = {path.stem for path in TASKS_DIR.glob("*.json")}
    canonical = [name for name in ["easy", "medium", "hard"] if name in discovered]
    extras = sorted(discovered - set(canonical))
    return canonical + extras


def public_ticket_view(ticket: LabeledTicket) -> Ticket:
    return Ticket.model_validate(_model_to_dict(ticket))


def clone_ticket(ticket: LabeledTicket) -> LabeledTicket:
    return LabeledTicket.model_validate(_model_to_dict(ticket))


def clone_ticket_list(
    tickets: List[LabeledTicket], visible_at: Optional[int] = None
) -> List[LabeledTicket]:
    cloned = [clone_ticket(ticket) for ticket in tickets]
    if visible_at is not None:
        for ticket in cloned:
            ticket.visible_at = visible_at
    return cloned


def _model_to_dict(model) -> Dict:
    return model.model_dump()
