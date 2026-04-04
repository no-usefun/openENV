from env.core import TicketEnv

config = {
    "num_tickets": 5,
    "max_steps": 10
}

env = TicketEnv(config)

state = env.reset()
print("Initial State:", state)

done = False

while not done:
    ticket = state.tickets[0]

    action = {
        "ticket_id": ticket.id,
        "department": "general",
        "priority": "low",
        "action_type": "resolve"
    }

    from env.models import Action
    action = Action(**action)

    state, reward, done, _ = env.step(action)
    print("Reward:", reward)