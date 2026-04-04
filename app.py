from env.core import TicketEnv
from env.models import Action, Ticket


def choose_action(ticket: Ticket) -> Action:
    description = (ticket.description or "").lower()

    if any(word in description for word in ["charged", "invoice", "refund", "subscription", "bill"]):
        department = "billing"
    elif any(word in description for word in ["crash", "access", "login", "sso", "export", "bug"]):
        department = "technical"
    else:
        department = ticket.category_hint

    if ticket.urgency >= 4:
        priority = "high"
    elif ticket.urgency >= 2:
        priority = "medium"
    else:
        priority = "low"

    if ticket.urgency >= 4 or department == "technical":
        action_type = "escalate"
    elif "not sure" in description or "can someone check" in description:
        action_type = "request_info"
    else:
        action_type = "resolve"

    return Action(
        ticket_id=ticket.id,
        department=department,
        priority=priority,
        action_type=action_type,
    )


def main() -> None:
    env = TicketEnv({"scenario_name": "easy"})

    state = env.reset()
    print("Scenario loaded with", state.pending_count, "tickets")

    done = False
    while not done:
        ticket = state.current_ticket
        if ticket is None:
            break
        action = choose_action(ticket)
        state, reward, done, info = env.step(action)
        print(
            "Handled",
            action.ticket_id,
            "step_score=",
            reward.step_score,
            "total_score=",
            reward.total_score,
            "done=",
            done,
        )
        if reward.breakdown:
            print("  reward breakdown:", reward.breakdown)

    grade = env.grade()
    print("Final score:", grade.final_score)
    print("Routing accuracy:", grade.routing_accuracy)
    print("Priority accuracy:", grade.priority_accuracy)
    print("SLA score:", grade.sla_score)
    print("Action accuracy:", grade.action_accuracy)


if __name__ == "__main__":
    main()
