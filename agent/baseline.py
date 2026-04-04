from typing import Dict, Optional

from env.models import Action, Observation, Ticket


GENERAL_PATTERNS = [
    "how do i",
    "where do i",
    "can i ",
    "does ",
    "available on another plan",
    "support scheduled",
]

BILLING_PATTERNS = [
    "refund",
    "charge",
    "charged",
    "invoice",
    "payment ",
    " payment",
    "billed",
    "billing statement",
    "renewal",
    "subscription",
    "workspace still shows unpaid",
    "transaction still shows failed",
]

TECHNICAL_PATTERNS = [
    "cannot log in",
    "can't log in",
    "sign-in",
    "sign in",
    "login",
    "credentials",
    "two-factor",
    "2fa",
    "authentication",
    "access is blocked",
    "cannot complete sign-in",
    "appears suspended",
    "crash",
    "bug",
    "slow",
    "sync",
    "time out",
    "times out",
    "timeout",
    "500",
    "401",
    "webhook",
    "error",
]

REQUEST_INFO_PATTERNS = [
    "not sure",
    "do not know",
    "don't know",
    "which invoice",
    "which workspace",
    "whether the issue is limited",
    "if it is specific to",
]

ESCALATE_BILLING_PATTERNS = [
    "access is blocked",
    "whole team lost access",
    "workspace still shows unpaid",
    "transaction still shows failed",
]


def _matches_any(description: str, patterns: list[str]) -> bool:
    return any(pattern in description for pattern in patterns)


def choose_action(ticket: Ticket) -> Action:
    description = (ticket.description or "").lower()

    is_general = _matches_any(description, GENERAL_PATTERNS)
    is_technical = _matches_any(description, TECHNICAL_PATTERNS)
    is_billing = _matches_any(description, BILLING_PATTERNS)
    is_blocking_billing = (
        ("payment" in description or "charge" in description or "subscription" in description)
        and _matches_any(description, ESCALATE_BILLING_PATTERNS)
    )

    # Route based on the visible symptom, not the product name or noisy hint.
    if is_general and "refund" not in description and "invoice" not in description and "payment" not in description:
        department = "general"
    elif is_blocking_billing:
        department = "billing"
    elif is_technical:
        department = "technical"
    elif is_billing:
        department = "billing"
    else:
        department = ticket.category_hint

    if "cancelled unexpectedly" in description and "need clarification" in description and ticket.urgency <= 2:
        priority = "low"
    elif ticket.urgency >= 4:
        priority = "high"
    elif ticket.urgency >= 2:
        priority = "medium"
    else:
        priority = "low"

    if _matches_any(description, REQUEST_INFO_PATTERNS):
        action_type = "request_info"
    elif department == "general":
        action_type = "resolve"
    elif department == "billing":
        action_type = "escalate" if _matches_any(description, ESCALATE_BILLING_PATTERNS) else "resolve"
    else:
        action_type = "escalate"

    return Action(
        ticket_id=ticket.id,
        department=department,
        priority=priority,
        action_type=action_type,
    )


def choose_action_from_observation(
    observation: Observation,
    first_seen_at: Optional[Dict[str, int]] = None,
) -> Action:
    if observation.current_ticket is None or not observation.pending_tickets:
        raise RuntimeError("Cannot choose an action without pending tickets.")

    seen_times = first_seen_at or {}
    ranked = []
    for ticket in observation.pending_tickets:
        seen_at = seen_times.get(ticket.id, observation.current_time)
        predicted = choose_action(ticket)
        ranked.append((_ticket_rank(observation.current_time, seen_at, ticket, predicted), predicted))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _ticket_rank(current_time: int, seen_at: int, ticket: Ticket, action: Action) -> tuple[int, int, int, int, str]:
    seen_age = current_time - seen_at

    if action.priority == "high":
        # The env records handled_at after this step, so a high ticket is only still on time
        # if it was first seen at the current step or the immediately previous one.
        if seen_age <= 1:
            bucket = 4
            deadline_pressure = seen_age
        else:
            bucket = 3
            deadline_pressure = -99
    elif action.priority == "medium":
        bucket = 2
        deadline_pressure = -100
    else:
        bucket = 1
        deadline_pressure = -100

    return (
        bucket,
        deadline_pressure,
        ticket.urgency,
        ticket.time_waiting,
        ticket.id,
    )
