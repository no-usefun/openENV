from typing import Dict, Optional

from env.models import Action, Observation, Ticket


GENERAL_PATTERNS = [
    "how do i",
    "where do i",
    "where can i",
    "can i ",
    "does ",
    "available on another plan",
    "support scheduled",
    "is there a way to",
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
    "api key",
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
    "504",
    "webhook",
    "error",
    "crash",
    "crashes",
    "freeze",
    "freezes",
    "failing",
    "invalid",
]

REQUEST_INFO_PATTERNS = [
    "",
    "not sure",
    "do not know",
    "don't know",
    "which invoice",
    "which workspace",
    "whether the issue is limited",
    "if it is specific to",
    "maybe",
    "i think",
    "i guess",
    "just checking",
    "doesn't work",
    "doesnt work",
    "something is wrong",
    "issue with",
    "need help asap",
    "urgent help",
    "where is my money",
    "or maybe not",
    "could be correct",
    "only sometimes",
    "intermittently",
    "randomly",
    "i didn't change anything",
    "working yesterday not today",
    "can't access something",
    "cant access something",
    "payment issue i guess",
    "refund??",
    "just checking if everything is working fine",
]

ESCALATE_BILLING_PATTERNS = [
    "access is blocked",
    "whole team lost access",
    "workspace still shows unpaid",
    "transaction still shows failed",
    "payment failed",
    "money got deducted",
    "refund was promised",
    "haven't received it yet",
]

TECHNICAL_ESCALATE_PATTERNS = [
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
    "500",
    "401",
    "504",
    "webhook",
    "security",
    "outage",
    "nothing loads",
    "times out",
    "cannot access",
    "locked",
    "stopped working",
    "logged out",
    "sync between devices",
    "data sync",
    "uploads fail",
    "file uploads fail",
    "crashes when",
]

LOW_PRIORITY_BILLING_PATTERNS = [
    "refund",
    "invoice",
    "tax",
    "currency conversion",
]


def _matches_any(description: str, patterns: list[str]) -> bool:
    if "" in patterns and not description.strip():
        return True
    return any(pattern and pattern in description for pattern in patterns)


def _looks_vague(description: str) -> bool:
    words = [word for word in description.replace("?", " ").split() if word]
    if not words:
        return True
    if len(words) <= 4:
        return True
    return _matches_any(description, REQUEST_INFO_PATTERNS)


def choose_action(ticket: Ticket) -> Action:
    description = (ticket.description or "").lower()

    is_general = _matches_any(description, GENERAL_PATTERNS)
    is_technical = _matches_any(description, TECHNICAL_PATTERNS)
    is_billing = _matches_any(description, BILLING_PATTERNS)
    needs_info = _looks_vague(description)
    is_blocking_billing = (
        ("payment" in description or "charge" in description or "subscription" in description)
        and _matches_any(description, ESCALATE_BILLING_PATTERNS)
    )

    # Route based on the visible symptom, not the product name or noisy hint.
    if is_general and "refund" not in description and "payment" not in description:
        department = "general"
    elif is_blocking_billing:
        department = "billing"
    elif is_technical:
        department = "technical"
    elif is_billing:
        department = "billing"
    else:
        department = ticket.category_hint

    if department == "general" and is_general:
        priority = "low"
    elif (
        department == "billing"
        and ticket.urgency <= 2
        and _matches_any(description, LOW_PRIORITY_BILLING_PATTERNS)
    ):
        priority = "low"
    elif "cancelled unexpectedly" in description and "need clarification" in description and ticket.urgency <= 2:
        priority = "low"
    elif ticket.urgency >= 4:
        priority = "high"
    elif ticket.urgency >= 2:
        priority = "medium"
    else:
        priority = "low"

    if needs_info and not (
        (department == "billing" and _matches_any(description, ESCALATE_BILLING_PATTERNS))
        or (department == "technical" and _matches_any(description, TECHNICAL_ESCALATE_PATTERNS))
    ):
        action_type = "request_info"
    elif department == "general":
        action_type = "resolve"
    elif department == "billing":
        action_type = "escalate" if _matches_any(description, ESCALATE_BILLING_PATTERNS) else "resolve"
    else:
        action_type = "escalate" if _matches_any(description, TECHNICAL_ESCALATE_PATTERNS) else "resolve"

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
    constrained_mode = observation.current_time >= 8 and observation.pending_count >= 10
    ranked = []
    for ticket in observation.pending_tickets:
        seen_at = seen_times.get(ticket.id, observation.current_time)
        predicted = choose_action(ticket)
        ranked.append(
            (
                _ticket_rank(
                    observation.current_time,
                    seen_at,
                    ticket,
                    predicted,
                    constrained_mode=constrained_mode,
                ),
                predicted,
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _ticket_rank(
    current_time: int,
    seen_at: int,
    ticket: Ticket,
    action: Action,
    constrained_mode: bool = False,
) -> tuple[int, int, int, int, str]:
    seen_age = current_time - seen_at
    confidence = _prediction_confidence(ticket, action)

    if action.priority == "high":
        # The env records handled_at after this step, so a high ticket is only still on time
        # if it was first seen at the current step or the immediately previous one.
        if seen_age <= 1:
            bucket = 4
            deadline_pressure = seen_age
        else:
            bucket = 3
            deadline_pressure = -99
    elif constrained_mode:
        bucket = 1
        deadline_pressure = -100
    elif action.priority == "medium":
        bucket = 2
        deadline_pressure = -100
    else:
        bucket = 1
        deadline_pressure = -100

    return (
        bucket,
        confidence,
        deadline_pressure,
        ticket.urgency,
        ticket.time_waiting,
        ticket.id,
    )


def _prediction_confidence(ticket: Ticket, action: Action) -> int:
    description = (ticket.description or "").lower()
    confidence = 0

    if action.action_type == "request_info":
        confidence += 3 if _looks_vague(description) else 1
    elif action.department == "general":
        confidence += 3 if _matches_any(description, GENERAL_PATTERNS) else 1
    elif action.department == "billing":
        confidence += 3 if _matches_any(description, BILLING_PATTERNS) else 1
        if action.action_type == "escalate" and _matches_any(description, ESCALATE_BILLING_PATTERNS):
            confidence += 1
    elif action.department == "technical":
        confidence += 3 if _matches_any(description, TECHNICAL_PATTERNS) else 1
        if action.action_type == "escalate" and _matches_any(description, TECHNICAL_ESCALATE_PATTERNS):
            confidence += 1

    if action.priority == "low":
        confidence += 1

    if ticket.category_hint == action.department:
        confidence += 1

    return confidence
