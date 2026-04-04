from env.models import Action, Ticket


def choose_action(ticket: Ticket) -> Action:
    description = (ticket.description or "").lower()

    if any(word in description for word in ["charged", "invoice", "refund", "subscription", "bill", "payment"]):
        department = "billing"
    elif any(
        word in description
        for word in [
            "crash",
            "access",
            "login",
            "sso",
            "export",
            "bug",
            "api",
            "500",
            "401",
            "slow",
            "error",
            "throttling",
            "webhook",
        ]
    ):
        department = "technical"
    else:
        department = ticket.category_hint

    if ticket.urgency >= 4:
        priority = "high"
    elif ticket.urgency >= 2:
        priority = "medium"
    else:
        priority = "low"

    if any(phrase in description for phrase in ["not sure", "do not know", "don't know", "can someone check"]):
        action_type = "request_info"
    elif ticket.urgency >= 4 or department == "technical":
        action_type = "escalate"
    else:
        action_type = "resolve"

    return Action(
        ticket_id=ticket.id,
        department=department,
        priority=priority,
        action_type=action_type,
    )
